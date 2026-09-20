"""PyMuPDF extractor (optional dependency)."""

from __future__ import annotations

import re

try:
    import pymupdf
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

from pdfvault.extractors.base import ExtractionResult, PageContent, RawFigure

# Pattern to reject author-like lines (e.g. "John Smith, Jane Doe, Bob Lee")
_AUTHOR_RE = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)+")
# Pattern for superscript annotations or footnote markers
_SUPERSCRIPT_RE = re.compile(r"^\d+[,\s\d]*$")

# --- Vector-figure detection ------------------------------------------------
#
# Modern review journals (Nature Reviews, eLife illustrated reviews,
# many Cell-press graphical abstracts) render figures as PDF vector
# paths rather than embedded raster images. ``page.get_images()``
# returns an empty list for such pages even though the figure is
# clearly visible. Rendering the page area where vector drawings live
# recovers the figure as a PNG so the rest of the pipeline (caption
# matching, VLM description) can treat it like any raster figure.

# Below this drawing count a "page with vectors" is just decorative
# rules / box borders, not a real figure region. Tuned against
# Nat Rev Genetics where real figure pages have 1500-2500 drawings
# and unrelated pages have 20-40.
_MIN_VECTOR_DRAWINGS_PER_PAGE = 100

# Vertical gap (PDF points) that separates two distinct figures on
# the same page. A 30pt gap (~10mm) is more than typical inter-line
# spacing but smaller than the gap between a figure and surrounding
# body text. Used to cluster drawings into per-figure regions.
_FIGURE_VERTICAL_GAP_PT = 30.0

# Minimum bbox area (square points) for a region to count as a figure.
# Single rule lines / decorative boxes can produce small clusters of
# drawings that we don't want to render.
_MIN_FIGURE_AREA_PT2 = 20_000

#: A vector cluster is rejected once this much of its area is covered by the
#: page's text blocks. Measured on real papers: genuine vector figures score
#: 0.00-0.05 (their axis labels are drawn as outlined paths, and even real
#: text labels cover only a sliver), whereas body-text bands and journal
#: cover blocks score 0.40-1.00. 0.35 sits in the gap with margin on both
#: sides.
_MAX_TEXT_COVERAGE = 0.35

#: Banner geometry. Inline link glyphs and rule bars cluster into wide, flat
#: strips that carry almost no text of their own. Real figures measured
#: 400-611pt tall; these strips measured 40-170pt.
_BANNER_MIN_ASPECT = 6.0
_BANNER_MAX_HEIGHT_PT = 120.0

# Pixmap render DPI for vector figure regions. 150 DPI matches the
# resolution our VLM page renderer uses elsewhere — high enough for
# Sonnet/Haiku to read axis labels, low enough not to bloat output.
_VECTOR_FIGURE_DPI = 150


def _cluster_drawings_by_y(
    drawings: list[dict], gap: float = _FIGURE_VERTICAL_GAP_PT,
) -> list[tuple[float, float, float, float]]:
    """Group page drawings into bounding boxes by vertical proximity.

    Each returned bbox is ``(x0, y0, x1, y1)`` covering one cluster of
    drawings whose y-ranges overlap or are within ``gap`` points of
    each other. The caller renders one image per bbox, so a page with
    Figure 1 at the top and Figure 2 at the bottom yields two
    figures rather than one giant figure spanning empty space.
    """
    rects: list[tuple[float, float, float, float]] = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        # Normalise — pymupdf rects expose .x0/.y0/.x1/.y1 attributes.
        try:
            r = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        except Exception:
            continue
        # Reject zero-area degenerate rects (path control points etc.)
        if r[2] <= r[0] or r[3] <= r[1]:
            continue
        rects.append(r)
    if not rects:
        return []

    # Sort by y0 ascending, then sweep merging anything within `gap`.
    rects.sort(key=lambda r: r[1])
    clusters: list[list[tuple[float, float, float, float]]] = [[rects[0]]]
    for r in rects[1:]:
        last_cluster = clusters[-1]
        cluster_y1 = max(c[3] for c in last_cluster)
        if r[1] - cluster_y1 <= gap:
            last_cluster.append(r)
        else:
            clusters.append([r])

    bboxes: list[tuple[float, float, float, float]] = []
    for cluster in clusters:
        x0 = min(c[0] for c in cluster)
        y0 = min(c[1] for c in cluster)
        x1 = max(c[2] for c in cluster)
        y1 = max(c[3] for c in cluster)
        if (x1 - x0) * (y1 - y0) >= _MIN_FIGURE_AREA_PT2:
            bboxes.append((x0, y0, x1, y1))
    return bboxes


def _extract_vector_figures(page, page_idx: int) -> list[dict]:
    """Render dense vector-drawing regions on ``page`` as PNG figures.

    Skips pages with too few drawings (decorative rules don't count).
    Returns the same dict shape as the raster path so the caller can
    treat both uniformly.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    if len(drawings) < _MIN_VECTOR_DRAWINGS_PER_PAGE:
        return []

    bboxes = _cluster_drawings_by_y(drawings)
    if not bboxes:
        return []

    figures: list[dict] = []
    matrix = pymupdf.Matrix(_VECTOR_FIGURE_DPI / 72, _VECTOR_FIGURE_DPI / 72)
    text_rects = _text_block_rects(page)
    for bbox in bboxes:
        if not _is_figure_region(bbox, text_rects):
            continue
        try:
            clip = pymupdf.Rect(*bbox)
            pix = page.get_pixmap(matrix=matrix, clip=clip)
            png = pix.tobytes("png")
        except Exception:
            continue
        figures.append({
            "page": page_idx,
            "image_bytes": png,
            "width": pix.width,
            "height": pix.height,
        })
    return figures


def _text_block_rects(page) -> list:
    """Bounding boxes of the page's text blocks, for figure/text separation."""
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return []
    rects = []
    for block in blocks:
        try:
            rect = pymupdf.Rect(block[:4])
        except Exception:
            continue
        if not rect.is_empty:
            rects.append(rect)
    return rects


def _text_coverage(bbox, text_rects: list) -> float:
    """Fraction of ``bbox`` covered by text blocks, clamped to 1.

    Blocks can overlap, so the summed intersection is an upper bound on the
    true covered area. That bias is safe here: it only ever makes a region
    look *more* like text, and the decision margin is wide.
    """
    region = pymupdf.Rect(*bbox)
    area = abs(region.get_area())
    if area <= 0:
        return 1.0
    covered = 0.0
    for rect in text_rects:
        overlap = rect & region
        if not overlap.is_empty:
            covered += abs(overlap.get_area())
    return min(covered / area, 1.0)


def _is_figure_region(bbox, text_rects: list) -> bool:
    """Whether a vector cluster is a figure rather than page furniture.

    The fallback clusters *every* vector drawing on a page, so inline link
    glyphs drag in the surrounding paragraph and a masthead's rule bars drag
    in the article title. Rendering those produced pictures of body text that
    downstream caption matching then labelled with a real figure's legend.
    """
    region = pymupdf.Rect(*bbox)
    if _text_coverage(bbox, text_rects) >= _MAX_TEXT_COVERAGE:
        return False
    height = abs(region.height)
    if (
        height < _BANNER_MAX_HEIGHT_PT
        and height > 0
        and abs(region.width) / height >= _BANNER_MIN_ASPECT
    ):
        return False
    return True


def _is_bold_span(span: dict) -> bool:
    """Return whether a PyMuPDF text span is bold/semibold/heavy/black weight.

    Recognises both the explicit bold flag (PyMuPDF flag bit 4) and a wide
    range of font-name conventions used by scientific publishers:

    - "bold"/"semibold" — most common
    - "heavy"/"black" — Cell-press papers use HelveticaNeue-Heavy for
      section headings; Adobe-Acrobat-encoded Cell papers expose this same
      face as ``AdvPSHN-H`` (the trailing ``-H`` suffix) which we also treat
      as a heavy/bold weight.
    """
    flags = span.get("flags", 0)
    raw_font = span.get("font", "")
    font_name = raw_font.lower()
    if flags & (1 << 4):
        return True
    if "bold" in font_name or "semibold" in font_name:
        return True
    if "heavy" in font_name or "black" in font_name:
        return True
    # Trailing weight suffix used by Adv* / Type1-encoded Cell-press fonts
    # e.g. "AdvPSHN-H" (Heavy), "AdvPSHN-B" (Bold). We accept "-H" or "-B"
    # at the end of the (case-sensitive) font name, but only when the rest
    # of the name is not a regular weight already handled above.
    if raw_font.endswith(("-H", "-B")) and not font_name.endswith(("-r", "-l", "-m")):
        return True
    return False


def _leading_bold_text(spans: list[dict]) -> str:
    """Return the contiguous bold text at the start of a line.

    Tolerates short non-bold "bridge" spans between bold runs:

    - Whitespace-only spans (commonly emitted by PyMuPDF as
      ``GlyphLessFont`` separators between adjacent bold spans).
    - Single-glyph symbol spans (e.g. the ★/✦ in ``STAR★METHODS``,
      which Cell-press encodes as a short span in a symbol font that
      lives between two bold ``HelveticaNeue-Heavy`` words).

    Without these allowances, multi-word bold headings such as
    ``OVERCOMING FUNCTIONAL CHALLENGES`` and ``STAR★METHODS`` would be
    truncated to just the first word.
    """
    parts: list[str] = []
    pending: list[str] = []
    for span in spans:
        text = span.get("text", "")
        if not text:
            continue
        if _is_bold_span(span):
            # bold span: flush any pending bridge spans, then append
            if pending:
                parts.extend(pending)
                pending = []
            parts.append(text)
            continue
        if not parts:
            # haven't started a bold run yet → not a leading-bold line
            break
        # We have an in-progress bold run and a non-bold span.
        # Treat as a bridge if it's whitespace OR a short symbol/punct token.
        stripped = text.strip()
        if stripped == "":
            pending.append(text)
            continue
        # Allow short symbolic glyphs (1-3 chars, no alphanumerics) — these
        # are typically font-encoded ornaments between adjacent bold words.
        if len(stripped) <= 3 and not any(c.isalnum() for c in stripped):
            pending.append(text)
            continue
        # Real non-bold text — bold run ends here, drop pending bridges.
        break
    return "".join(parts).strip()


class PymupdfExtractor:
    """Extractor backed by PyMuPDF (fitz) — fast text + image extraction."""

    def __init__(self) -> None:
        if not _PYMUPDF_AVAILABLE:
            raise ImportError(
                "pymupdf is not installed. "
                "Install it with: pip install pymupdf"
            )

    @property
    def name(self) -> str:
        return "pymupdf"

    @property
    def capabilities(self) -> list[str]:
        return ["text", "images", "tables"]

    def extract(self, pdf_bytes: bytes) -> ExtractionResult:
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Invalid PDF: {e}") from e

        pages = []
        for i in range(len(doc)):
            content = self._extract_page(doc, i)
            pages.append(content)
        doc.close()
        return ExtractionResult(pages=pages, engine=self.name)

    def extract_page(self, pdf_bytes: bytes, page_number: int) -> PageContent:
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Invalid PDF: {e}") from e

        if page_number < 0 or page_number >= len(doc):
            doc.close()
            raise ValueError(f"Page {page_number} out of range (0-{len(doc) - 1})")

        content = self._extract_page(doc, page_number)
        doc.close()
        return content

    def _extract_page(self, doc: "pymupdf.Document", page_idx: int) -> PageContent:
        page = doc[page_idx]
        text = page.get_text() or ""

        # Extract embedded images
        figures: list[RawFigure] = []
        for img_ref in page.get_images(full=True):
            xref = img_ref[0]
            try:
                img_data = doc.extract_image(xref)
                figures.append(RawFigure(
                    image_bytes=img_data.get("image"),
                    caption=None,
                    bbox=None,
                ))
            except Exception:
                pass

        confidence = 0.85 if len(text) > 50 else 0.3

        return PageContent(
            page_number=page_idx,
            text=text,
            tables=[],
            figures=figures,
            confidence=confidence,
        )

    def extract_bold_headings(self, pdf_bytes: bytes) -> list[dict]:
        """Extract bold text lines that are likely section headings.

        Returns list of dicts: {"text": str, "page": int, "font_size": float}

        Two-pass approach:
        1. First pass: determine the dominant body text font size
        2. Second pass: collect bold text that is LARGER than body text
        This prevents author names, table headers, and inline bold from
        being detected as headings.
        """
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Invalid PDF: {e}") from e

        # Pass 1: find dominant body text size (most common non-bold font size)
        size_counts: dict[float, int] = {}
        for page_idx in range(min(len(doc), 10)):  # sample first 10 pages
            page = doc[page_idx]
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        flags = span.get("flags", 0)
                        is_bold = bool(flags & (1 << 4))
                        font_name = span.get("font", "")
                        if not is_bold and "Bold" not in font_name:
                            size = round(span.get("size", 0), 1)
                            text = span.get("text", "").strip()
                            if len(text) > 10:  # only count substantial text
                                size_counts[size] = size_counts.get(size, 0) + len(text)

        body_size = max(size_counts, key=size_counts.get) if size_counts else 10.0

        # Pass 2: collect bold headings larger than body text
        headings: list[dict] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue

                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    line_text = _leading_bold_text(spans)

                    if len(line_text) < 3 or len(line_text) > 80:
                        continue

                    first_span = spans[0]
                    if not _is_bold_span(first_span):
                        continue

                    font_size = float(first_span.get("size", 0.0))

                    # Key filter: must be larger than body text OR same size
                    # but with a heading-like font (e.g., Nature uses same-size bold)
                    # Allow same-size bold only if the text looks like a heading
                    # (not a URL, not an author list, not a figure caption)
                    if font_size < body_size - 0.5:
                        continue  # smaller than body = definitely not a heading
                    if font_size > body_size * 2:
                        continue  # title text, not a section heading

                    if not line_text[0].isupper():
                        continue

                    if _SUPERSCRIPT_RE.match(line_text):
                        continue

                    if _AUTHOR_RE.match(line_text):
                        continue

                    # Reject URLs
                    if "http" in line_text.lower() or "www." in line_text.lower():
                        continue

                    # Reject figure/table captions (start with "Fig." or "Table")
                    if re.match(r"^(Fig\.|Figure|Table|Extended Data)", line_text):
                        continue

                    # Reject lines with commas that look like author affiliations
                    if "," in line_text and line_text.count(",") >= 2:
                        continue

                    words = line_text.split()

                    # Reject person-name-like lines (2-3 capitalized words, no
                    # section keywords) — catches single author names like
                    # "William El Sayed" at near-body-size bold
                    if (font_size <= body_size + 1.0
                            and 2 <= len(words) <= 4
                            and all(w[0].isupper() for w in words if w[0].isalpha())
                            and not any(w.lower() in {
                                "abstract", "introduction", "methods", "results",
                                "discussion", "conclusions", "references",
                                "acknowledgements", "online", "content",
                            } for w in words)):
                        # Likely a person name, not a heading
                        continue

                    # Reject lines with email-like patterns or special chars
                    if "✉" in line_text or "@" in line_text:
                        continue

                    if len(words) == 1 and len(line_text) < 4:
                        continue

                    # For same-size-as-body bold, be stricter: require short text
                    # (real headings at body size are typically 2-6 words)
                    if font_size <= body_size + 0.5 and len(words) > 8:
                        continue

                    headings.append({
                        "text": line_text,
                        "page": page_idx,
                        "font_size": font_size,
                    })

        total_pages = len(doc)
        doc.close()

        # Filter out running headers: same heading text appearing on most
        # pages (e.g. "Article", "Review", "Letter" on Cell-press papers).
        if total_pages >= 4 and headings:
            from collections import defaultdict
            pages_by_text: dict[str, set[int]] = defaultdict(set)
            for h in headings:
                pages_by_text[h["text"]].add(h["page"])
            running_headers = {
                txt for txt, pages in pages_by_text.items()
                if len(pages) >= 3 and len(pages) / total_pages >= 0.5
            }
            if running_headers:
                headings = [h for h in headings if h["text"] not in running_headers]

        return headings

    def extract_figures(
        self, pdf_bytes: bytes, min_width: int = 200, min_height: int = 200,
        max_per_page: int | None = 1,
    ) -> list[dict]:
        """Extract significant images from the PDF.

        Returns list of dicts: ``{"page": int, "image_bytes": bytes,
        "width": int, "height": int}``.

        Two passes:

        1. Raster images via ``page.get_images()`` — the historical
           behaviour; filters by ``min_width`` / ``min_height`` and
           caps at ``max_per_page`` per page so composite-panel A/B/C
           images don't each count as a separate figure.

        2. Vector-graphics fallback for pages where raster extraction
           produces nothing. Many modern review journals (Nature
           Reviews Genetics, eLife illustrated reviews) render figures
           as PDF vector paths instead of embedded raster images, so
           ``get_images()`` returns an empty list. We instead cluster
           the page's vector drawings into bounding-box regions by
           vertical gap, render each cluster via
           ``page.get_pixmap(clip=bbox)`` at 150 DPI, and emit those
           PNGs as figures. Caption matching downstream pairs them
           with the right ``Fig. N | ...`` line by page order, so no
           wiring change is needed.
        """
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Invalid PDF: {e}") from e

        figures: list[dict] = []
        seen_xrefs: set[int] = set()
        pages_with_raster: set[int] = set()

        for page_idx in range(len(doc)):
            page = doc[page_idx]

            for img_ref in page.get_images(full=True):
                xref = img_ref[0]

                # Deduplicate across pages (same image can appear on multiple pages)
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    img_data = doc.extract_image(xref)
                    width = img_data.get("width", 0)
                    height = img_data.get("height", 0)
                    image_bytes = img_data.get("image")

                    if width >= min_width and height >= min_height and image_bytes:
                        figures.append({
                            "page": page_idx,
                            "image_bytes": image_bytes,
                            "width": width,
                            "height": height,
                        })
                        pages_with_raster.add(page_idx)
                except Exception:
                    pass

        # Limit to largest N images per page to avoid counting sub-panels
        # (panels A, B, C of a composite figure) as separate figures.
        if max_per_page is not None:
            by_page: dict[int, list[dict]] = {}
            for fig in figures:
                by_page.setdefault(fig["page"], []).append(fig)
            figures = []
            for page_key in sorted(by_page.keys()):
                page_figs = sorted(
                    by_page[page_key],
                    key=lambda f: f["width"] * f["height"],
                    reverse=True,
                )
                figures.extend(page_figs[:max_per_page])

        # Vector-graphics fallback for pages with no qualifying raster.
        for page_idx in range(len(doc)):
            if page_idx in pages_with_raster:
                continue
            page = doc[page_idx]
            vector_figs = _extract_vector_figures(page, page_idx)
            figures.extend(vector_figs)

        doc.close()
        figures.sort(key=lambda f: f["page"])
        return figures
