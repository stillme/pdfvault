"""Core orchestrator — convert() drives triage, extraction, assembly."""

from __future__ import annotations

import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Sequence

import pypdfium2 as pdfium

from pdfvault.assembler import assemble_markdown
from pdfvault.confidence import rescore_document
from pdfvault.config import Config, FigureMode, Tier
from pdfvault.document import Document
from pdfvault.enhancers.captions import (
    extract_figure_captions,
    extract_panel_references,
    insert_caption_text_blocks,
    match_captions_to_figures,
    remove_caption_text_blocks,
    sync_caption_alt_text,
)
from pdfvault.enhancers.cross_references import add_cross_references
from pdfvault.enhancers.figure_index import build_figure_index
from pdfvault.enhancers.figures import enhance_figures
from pdfvault.enhancers.math import convert_unicode_math, detect_math_regions, extract_equations_vlm
from pdfvault.enhancers.metadata import extract_metadata
from pdfvault.enhancers.references import parse_references
from pdfvault.enhancers.tables import enhance_table
from pdfvault.enhancers.superscripts import detect_superscripts
from pdfvault.enhancers.text_cleaner import clean_figure_text
from pdfvault.enhancers.unicode_normalizer import normalize_unicode_text
from pdfvault.extractors import get_available_extractors, get_extractor_by_name
from pdfvault.extractors.base import PageContent, RawFigure
from pdfvault.providers.base import VLMProvider
from pdfvault.providers.registry import get_provider
from pdfvault.triage.analyzer import analyze_page
from pdfvault.triage.router import select_engine, select_tier
from pdfvault.verifier import run_verify_loop

logger = logging.getLogger(__name__)


def _resolve_source(source: str | bytes) -> bytes:
    """Resolve a file path, URL, or raw bytes to PDF bytes."""
    if isinstance(source, bytes):
        return source

    if isinstance(source, str):
        # Check if it looks like a URL
        if source.startswith(("http://", "https://")):
            import httpx
            resp = httpx.get(source, follow_redirects=True, timeout=60)
            resp.raise_for_status()
            return resp.content

        # Treat as file path
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")
        return p.read_bytes()

    raise TypeError(f"Unsupported source type: {type(source)}")


def _count_pages(pdf_bytes: bytes) -> int:
    """Count pages using pypdfium2. Raises ValueError for invalid PDFs."""
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
    except Exception as e:
        raise ValueError(f"Invalid PDF: {e}") from e
    n = len(pdf)
    pdf.close()
    return n


def _merge_tables(primary: PageContent, table_source: PageContent) -> PageContent:
    """Merge tables from table_source into primary page content."""
    if table_source.tables and not primary.tables:
        primary = primary.model_copy(update={"tables": table_source.tables})
    return primary


def _get_vlm_provider(provider_string: str | None = None) -> VLMProvider | None:
    """Get a VLM provider, returning None if unavailable."""
    try:
        return get_provider(provider_string)
    except Exception:
        return None


def _render_page_image(pdf_bytes: bytes, page_number: int, dpi: int = 150) -> bytes | None:
    """Render a PDF page to PNG bytes for VLM enhancement."""
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        if page_number < 0 or page_number >= len(pdf):
            pdf.close()
            return None
        page = pdf[page_number]
        scale = dpi / 72
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        page.close()
        pdf.close()

        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def convert(
    source: str | bytes,
    *,
    tier: str | Tier = Tier.AUTO,
    figures: str | None = None,
    verify: bool = True,
    provider: str | None = None,
    equations: bool = True,
) -> Document:
    """Convert a PDF to a structured markdown Document.

    Args:
        source: File path, URL, or raw PDF bytes.
        tier: Processing tier — "fast", "standard", "deep", or "auto".
        figures: Figure handling mode (skip, caption, describe, extract).
        verify: Whether to run verification passes.
        provider: VLM provider override.
        equations: Run VLM equation extraction on math-heavy pages.
            Default ``True`` for backward compatibility. Set to ``False``
            for batch jobs on text-heavy corpora (biomedical, social
            sciences) where the math heuristic fires on incidental Greek
            letters and burns subscription quota for no real gain.

    Returns:
        A Document with markdown, sections, tables, figures, and metadata.
    """
    t0 = time.monotonic()

    # Resolve input to bytes
    pdf_bytes = _resolve_source(source)

    # Validate and count pages
    num_pages = _count_pages(pdf_bytes)

    # Resolve tier enum
    if isinstance(tier, str):
        tier = Tier(tier)

    # Build config
    config = Config(
        tier=tier,
        verify=verify,
        provider=provider,
        equations=equations,
    )
    if figures is not None:
        config = config.model_copy(update={"figures": figures})

    # Discover available engines
    available = get_available_extractors()
    available_names = [ext.name for ext in available]

    # Probe VLM provider once so the router can prefer VLM for scanned pages.
    try:
        routing_vlm = _get_vlm_provider(config.provider)
    except Exception:
        routing_vlm = None
    vlm_available = routing_vlm is not None

    # Process each page: triage -> select tier/engines -> extract
    all_pages: list[PageContent] = []

    for page_idx in range(num_pages):
        # Triage: analyze page complexity
        analysis = analyze_page(pdf_bytes, page_idx)

        # Select tier and engines for this page
        page_tier = select_tier(analysis, config.tier)
        engines = select_engine(
            page_tier,
            has_text_layer=analysis.has_text_layer,
            available_engines=available_names,
            has_tables=analysis.has_tables,
            is_scanned=analysis.is_scanned,
            vlm_available=vlm_available,
        )

        # Extract with primary engine
        primary_engine_name = engines[0] if engines else "pypdfium2"
        if primary_engine_name == "vlm" and routing_vlm is not None:
            from pdfvault.extractors.vlm_ext import VLMExtractor
            primary_ext = VLMExtractor(routing_vlm)
        else:
            primary_ext = get_extractor_by_name(primary_engine_name)
            if primary_ext is None:
                # Fallback
                primary_ext = get_extractor_by_name("pypdfium2")

        page_content = primary_ext.extract_page(pdf_bytes, page_idx)

        # If pdfplumber is a secondary engine, merge its tables
        if (
            len(engines) > 1
            and "pdfplumber" in engines
            and primary_engine_name != "pdfplumber"
        ):
            plumber_ext = get_extractor_by_name("pdfplumber")
            if plumber_ext is not None:
                plumber_page = plumber_ext.extract_page(pdf_bytes, page_idx)
                page_content = _merge_tables(page_content, plumber_page)

        all_pages.append(page_content)

    # Normalize Unicode artifacts (soft hyphens, ligatures, stray control
    # chars) on each page's text before anything downstream consumes it.
    # Doing this once here means the assembler, metadata extractor, caption
    # matcher, and bibliography parser all see clean text without each having
    # to re-normalize.
    all_pages = [
        page.model_copy(update={"text": normalize_unicode_text(page.text)})
        for page in all_pages
    ]

    # Extract bold headings and enhanced figures if PyMuPDF is available.
    # Without pymupdf, figure extraction degrades to whatever the primary
    # extractor produces (often nothing on Nature-style PDFs). We surface
    # this loudly because the silent failure mode used to drop scored
    # quality from ~95% to ~65% with no indication that anything was wrong.
    bold_headings: list[dict] | None = None
    pymupdf_warning: str | None = None
    try:
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        pymupdf_ext = PymupdfExtractor()

        # Bold heading detection for journals like Nature
        bold_headings = pymupdf_ext.extract_bold_headings(pdf_bytes)

        # Enhanced figure extraction with size filtering
        pymupdf_figures = pymupdf_ext.extract_figures(pdf_bytes)
        if pymupdf_figures:
            # Build a dict of page_number -> list[RawFigure] from PyMuPDF's
            # high-quality large figures (200x200+). These replace whatever
            # the primary extractor found (which may be tiny/spurious).
            pymupdf_by_page: dict[int, list[RawFigure]] = {}
            for fig_info in pymupdf_figures:
                fig_page = fig_info["page"]
                if fig_page < len(all_pages):
                    pymupdf_by_page.setdefault(fig_page, []).append(
                        RawFigure(
                            image_bytes=fig_info["image_bytes"],
                            caption=None,
                            bbox=None,
                        )
                    )
            for page_num, figs in pymupdf_by_page.items():
                all_pages[page_num] = all_pages[page_num].model_copy(
                    update={"figures": figs}
                )
    except ImportError:
        pymupdf_warning = (
            "pymupdf not installed: bold-heading detection and enhanced "
            "figure extraction are disabled. Install with "
            "`uv sync --extra pymupdf` (or `pip install pymupdf`)."
        )
        logger.warning(pymupdf_warning)

    # Assemble into Document
    doc = assemble_markdown(all_pages, bold_headings=bold_headings)

    # Surface the missing-pymupdf message into the document so callers see
    # it even if they aren't capturing the logger.
    if pymupdf_warning is not None:
        doc = doc.model_copy(update={
            "warnings": [*doc.warnings, pymupdf_warning],
        })

    # Enhance metadata: extract title, authors, DOI from assembled text
    full_text = "\n".join(p.text for p in all_pages)
    extracted_meta = extract_metadata(full_text, pages=num_pages)
    updated_meta = doc.metadata.model_copy(update={
        "title": extracted_meta.title or doc.metadata.title,
        "authors": extracted_meta.authors or doc.metadata.authors,
        "doi": extracted_meta.doi or doc.metadata.doi,
    })
    doc = doc.model_copy(update={"metadata": updated_meta})

    # Extract captions before figure-leak cleanup so complete legends are
    # preserved even when cleanup strips statistical legend language from prose.
    captions = extract_figure_captions(doc.markdown)
    if captions:
        doc = doc.model_copy(update={
            "markdown": remove_caption_text_blocks(doc.markdown, captions),
        })

    # Clean figure-leak text (axis labels, gene names from figures)
    doc = doc.model_copy(update={
        "markdown": clean_figure_text(doc.markdown),
    })

    if captions and doc.figures:
        matched_figures = match_captions_to_figures(doc.figures, captions)
        synced_markdown = sync_caption_alt_text(doc.markdown, matched_figures, captions)
        synced_markdown = insert_caption_text_blocks(synced_markdown, matched_figures, captions)
        doc = doc.model_copy(update={
            "figures": matched_figures,
            "markdown": synced_markdown,
        })

    # Store panel references as metadata (available via doc internals)
    _panel_refs = extract_panel_references(doc.markdown)

    # Superscript detection: wrap inline citation refs and affiliations
    doc = doc.model_copy(update={
        "markdown": detect_superscripts(doc.markdown),
    })

    # Math enhancement (all tiers): convert Unicode math symbols to LaTeX.
    # Honour ``equations=False`` here too — callers that opt out expect plain
    # prose, and ``$...\alpha$`` wrappers leaking into figure legends are
    # unrenderable for downstream consumers.
    if config.equations:
        doc = doc.model_copy(update={
            "markdown": convert_unicode_math(doc.markdown),
        })

    # VLM enhancement for STANDARD and DEEP tiers
    effective_tier_enum = config.tier
    if effective_tier_enum == Tier.AUTO and num_pages > 0:
        analysis0 = analyze_page(pdf_bytes, 0)
        effective_tier_enum = select_tier(analysis0, Tier.AUTO)

    if effective_tier_enum in (Tier.STANDARD, Tier.DEEP):
        vlm = _get_vlm_provider(config.provider)
        if vlm is not None:
            # Enhance figures if mode is DESCRIBE or EXTRACT
            if config.figures in (FigureMode.DESCRIBE, FigureMode.EXTRACT):
                doc = doc.model_copy(update={
                    "figures": enhance_figures(
                        doc.figures,
                        mode=config.figures,
                        provider=vlm,
                        output_dir=config.output_dir,
                    ),
                })

            # Enhance low-confidence tables
            enhanced_tables = []
            for table in doc.tables:
                page_image = _render_page_image(pdf_bytes, table.page)
                enhanced_tables.append(
                    enhance_table(table, provider=vlm, page_image=page_image)
                )
            if enhanced_tables:
                doc = doc.model_copy(update={"tables": enhanced_tables})

            # VLM equation extraction on math-heavy pages. The math
            # heuristic fires on any page with a few Greek letters or
            # ``=`` characters, so on text-heavy corpora it can spawn
            # one VLM call per page for no real gain. Disable via
            # ``equations=False`` for batch jobs that don't need LaTeX.
            all_equations = list(doc.equations)
            if config.equations:
                for page_idx_eq in range(num_pages):
                    page_text = all_pages[page_idx_eq].text
                    if not detect_math_regions(page_text):
                        continue
                    page_image = _render_page_image(pdf_bytes, page_idx_eq)
                    if page_image is None:
                        continue
                    try:
                        eqs = extract_equations_vlm(
                            page_image=page_image,
                            page_text=page_text,
                            provider=vlm,
                            page_number=page_idx_eq,
                        )
                        all_equations.extend(eqs)
                    except Exception as exc:
                        msg = f"equation extraction failed on page {page_idx_eq}: {exc}"
                        logger.warning(msg)
                        doc.warnings.append(msg)
            if all_equations:
                doc = doc.model_copy(update={"equations": all_equations})

    # Agentic verify-correct loop for DEEP tier
    if effective_tier_enum == Tier.DEEP and config.verify:
        vlm_verify = _get_vlm_provider(config.provider)
        if vlm_verify is not None:
            corrected_pages = list(all_pages)
            verify_warnings: list[str] = []
            for i, page in enumerate(corrected_pages):
                page_image = _render_page_image(pdf_bytes, page.page_number)
                if page_image is None:
                    continue

                page_num = page.page_number

                def _record(summary, _page_num=page_num):
                    if (
                        summary.skipped_no_match
                        or summary.skipped_ambiguous
                    ):
                        verify_warnings.append(
                            f"verify page {_page_num}: "
                            f"{summary.applied} applied, "
                            f"{summary.skipped_no_match} no-match, "
                            f"{summary.skipped_ambiguous} ambiguous"
                        )

                def _record_error(explanation, _page_num=page_num):
                    # One line per failed page. The explanation already
                    # carries the provider's HTTP status / model id so a
                    # reader can spot a deprecated-model 404 immediately.
                    verify_warnings.append(
                        f"verify page {_page_num}: {explanation}"
                    )

                corrected_md, confidence = run_verify_loop(
                    page_image,
                    page.text,
                    vlm_verify,
                    max_rounds=config.max_verify_rounds,
                    on_patch_summary=_record,
                    on_error=_record_error,
                )
                corrected_pages[i] = page.model_copy(update={
                    "text": corrected_md,
                    "confidence": max(page.confidence, confidence),
                })
            # Reassemble with corrected pages
            all_pages = corrected_pages
            doc = assemble_markdown(all_pages, bold_headings=bold_headings)

            # Re-extract metadata lost during reassembly
            full_text_corrected = "\n".join(p.text for p in all_pages)
            meta = extract_metadata(full_text_corrected, doc.metadata.pages)
            updated_meta = doc.metadata.model_copy(update={
                "title": meta.title or doc.metadata.title,
                "authors": meta.authors or doc.metadata.authors,
                "doi": meta.doi or doc.metadata.doi,
            })
            doc = doc.model_copy(update={"metadata": updated_meta})

            if verify_warnings:
                doc = doc.model_copy(update={
                    "warnings": [*doc.warnings, *verify_warnings],
                })

    doc = doc.model_copy(update={
        "figure_index": build_figure_index(doc.markdown, doc.figures),
    })

    # Bibliography parsing: structured Reference entries for the
    # ``References`` / ``Bibliography`` section. Runs before cross-reference
    # linking so downstream consumers see populated ``doc.bibliography``.
    doc = doc.model_copy(update={
        "bibliography": parse_references(doc.markdown),
    })

    # Cross-reference linking: turn "Fig. 3", "Section 4.2", "[12]" into
    # markdown links pointing at the matching figure / heading / bibliography
    # entry. Runs after figure_index is built so figure anchors are available.
    doc = doc.model_copy(update={
        "markdown": add_cross_references(doc.markdown, doc),
    })

    # Rescore page confidences based on actual extraction results.
    # Replaces the crude text-length heuristic with content-aware scoring:
    # figure pages get credit for extracted images, text pages for prose quality.
    page_texts = [p.text for p in all_pages]
    doc = rescore_document(doc, page_texts)

    # Stamp tier and timing
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    effective_tier = tier.value if isinstance(tier, Tier) else str(tier)
    # For auto tier, report the tier that was actually used for page 0
    if config.tier == Tier.AUTO and num_pages > 0:
        analysis0 = analyze_page(pdf_bytes, 0)
        effective_tier = select_tier(analysis0, Tier.AUTO).value

    doc = doc.model_copy(update={
        "processing_time_ms": elapsed_ms,
        "tier_used": effective_tier,
        "engine_used": primary_engine_name,
    })

    return doc


def convert_batch(
    sources: Sequence[str | bytes],
    *,
    tier: str | Tier = Tier.AUTO,
    figures: str | None = None,
    verify: bool = True,
    provider: str | None = None,
) -> list[Document]:
    """Convert multiple PDFs sequentially.

    Args:
        sources: List of file paths, URLs, or raw PDF bytes.
        tier: Processing tier for all documents.
        figures: Figure handling mode.
        verify: Whether to run verification passes.
        provider: VLM provider override.

    Returns:
        List of Documents, one per source.
    """
    results: list[Document] = []
    for source in sources:
        doc = convert(
            source,
            tier=tier,
            figures=figures,
            verify=verify,
            provider=provider,
        )
        results.append(doc)
    return results
