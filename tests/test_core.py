"""Tests for the core convert() orchestrator."""

import pytest
from pdfvault import convert, Document
from pdfvault.config import Tier


def test_convert_file_path(sample_pdf_path):
    doc = convert(str(sample_pdf_path))
    assert isinstance(doc, Document)
    assert doc.metadata.pages == 2
    assert "Introduction" in doc.markdown
    assert doc.processing_time_ms > 0


def test_convert_bytes(sample_pdf_bytes):
    doc = convert(sample_pdf_bytes)
    assert isinstance(doc, Document)
    assert doc.metadata.pages == 2


def test_convert_fast_tier(sample_pdf_bytes):
    doc = convert(sample_pdf_bytes, tier="fast")
    assert doc.tier_used == "fast"


def test_convert_extracts_sections(sample_pdf_bytes):
    doc = convert(sample_pdf_bytes, tier="fast")
    section_titles = [s.title for s in doc.sections]
    assert any("Introduction" in t for t in section_titles) or any("Results" in t for t in section_titles)


def test_convert_extracts_tables(sample_pdf_bytes):
    doc = convert(sample_pdf_bytes, tier="fast")
    assert len(doc.tables) >= 1
    assert "Condition" in doc.tables[0].headers


def test_convert_has_confidence(sample_pdf_bytes):
    doc = convert(sample_pdf_bytes, tier="fast")
    assert doc.confidence > 0
    assert len(doc.page_confidences) == 2


def test_convert_invalid_input():
    with pytest.raises(ValueError):
        convert(b"not a pdf")


def test_convert_file_not_found():
    with pytest.raises(FileNotFoundError):
        convert("/nonexistent/file.pdf")


def test_convert_save_markdown(sample_pdf_bytes, tmp_path):
    doc = convert(sample_pdf_bytes, tier="fast")
    out = tmp_path / "output.md"
    doc.save_markdown(str(out))
    content = out.read_text()
    assert len(content) > 100


def test_convert_standard_tier_with_mock_vlm(sample_pdf_bytes):
    from unittest.mock import MagicMock, patch
    from pdfvault.extractors.pypdfium_ext import PypdfiumExtractor
    from pdfvault.extractors.pdfplumber_ext import PdfplumberExtractor
    mock_provider = MagicMock()
    mock_provider.name = "gemini"
    mock_provider.complete_sync.return_value = "A figure showing experimental results."
    # Exclude marker so the test only exercises VLM logic, not the surya ML model
    with patch("pdfvault.core._get_vlm_provider", return_value=mock_provider), \
         patch("pdfvault.extractors.get_available_extractors",
               return_value=[PypdfiumExtractor(), PdfplumberExtractor()]):
        doc = convert(sample_pdf_bytes, tier="standard", figures="describe")
        assert doc.tier_used == "standard"


def test_convert_standard_without_vlm_falls_back(sample_pdf_bytes):
    from unittest.mock import patch
    from pdfvault.extractors.pypdfium_ext import PypdfiumExtractor
    from pdfvault.extractors.pdfplumber_ext import PdfplumberExtractor
    # Exclude marker (ML-based) so test exercises fallback logic, not marker
    with patch("pdfvault.core._get_vlm_provider", return_value=None), \
         patch("pdfvault.extractors.get_available_extractors",
               return_value=[PypdfiumExtractor(), PdfplumberExtractor()]):
        doc = convert(sample_pdf_bytes, tier="standard")
        assert doc.metadata.pages == 2
        assert len(doc.markdown) > 100


# --- Bug 4: Deep tier preserves title after reassembly ---

def test_deep_tier_preserves_title(sample_pdf_bytes):
    import json
    from unittest.mock import MagicMock, patch
    from pdfvault.extractors.pypdfium_ext import PypdfiumExtractor
    from pdfvault.extractors.pdfplumber_ext import PdfplumberExtractor
    import pdfvault
    mock_provider = MagicMock()
    mock_provider.name = "test"
    mock_provider.complete_sync.return_value = json.dumps({"status": "pass", "confidence": 0.95, "corrections": []})
    # Exclude marker so test exercises deep tier verification logic, not marker
    with patch("pdfvault.core._get_vlm_provider", return_value=mock_provider), \
         patch("pdfvault.extractors.get_available_extractors",
               return_value=[PypdfiumExtractor(), PdfplumberExtractor()]):
        doc = pdfvault.convert(sample_pdf_bytes, tier="deep")
        assert doc.metadata.title is not None, "Title should be preserved after deep tier reassembly"


def _generate_math_pdf() -> bytes:
    """Generate a minimal PDF whose page text contains math symbols."""
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Math Paper", styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Introduction", styles["Heading1"]),
        Paragraph(
            "We study the equation ∇ · (κ∇u) = f on Ω.",
            styles["Normal"],
        ),
        Paragraph(
            "For all x ∈ Ω we require u ≤ M and ∇u ≠ 0.",
            styles["Normal"],
        ),
    ]
    doc.build(story)
    return buf.getvalue()


def test_convert_surfaces_equation_extraction_failure():
    """When VLM equation extraction raises, the failure must surface in
    doc.warnings without crashing the pipeline."""
    from unittest.mock import MagicMock, patch
    import pdfvault

    pdf_bytes = _generate_math_pdf()

    mock_provider = MagicMock()
    mock_provider.name = "test"

    def side_effect(prompt, image=None, **kwargs):
        if "mathematical equations" in prompt:
            raise RuntimeError("VLM API unavailable")
        return ""

    mock_provider.complete_sync.side_effect = side_effect

    with patch("pdfvault.core._get_vlm_provider", return_value=mock_provider):
        doc = pdfvault.convert(pdf_bytes, tier="standard", verify=False)

    # Pipeline did not crash and produced a document
    assert isinstance(doc, Document)
    # Other content is still present
    assert len(doc.markdown) > 0
    assert doc.metadata.pages >= 1
    # The failure surfaced as a warning
    assert len(doc.warnings) >= 1
    assert any("equation extraction failed on page" in w for w in doc.warnings)
    assert any("VLM API unavailable" in w for w in doc.warnings)


def test_convert_equations_false_skips_vlm_equation_calls():
    """``equations=False`` must short-circuit the per-page math VLM loop.
    Lean-mode batch jobs on biomedical corpora can't afford a VLM call
    per page that happens to mention a Greek letter."""
    from unittest.mock import MagicMock, patch
    import pdfvault

    pdf_bytes = _generate_math_pdf()
    mock_provider = MagicMock()
    mock_provider.name = "test"
    mock_provider.complete_sync.return_value = ""

    with patch("pdfvault.core._get_vlm_provider", return_value=mock_provider):
        doc = pdfvault.convert(
            pdf_bytes, tier="standard", verify=False, equations=False,
        )

    assert isinstance(doc, Document)
    # No per-page equation prompts were issued. The figure / table paths
    # may still call complete_sync; the strict assertion is that no
    # prompt mentioning equation extraction fired.
    for call in mock_provider.complete_sync.call_args_list:
        prompt = call.args[0] if call.args else call.kwargs.get("prompt", "")
        assert "mathematical equations" not in prompt, (
            f"equations=False must suppress VLM math calls, got: {prompt[:80]}"
        )
    # And no equation-extraction warning surfaced (because none were tried).
    assert not any("equation extraction failed" in w for w in doc.warnings)


def test_convert_equations_true_default_still_runs_math_pipeline():
    """Backwards-compat check: omitting ``equations`` keeps the old
    behaviour and the math VLM loop fires on math-heavy pages."""
    from unittest.mock import MagicMock, patch
    import pdfvault

    pdf_bytes = _generate_math_pdf()
    mock_provider = MagicMock()
    mock_provider.name = "test"
    mock_provider.complete_sync.return_value = ""

    with patch("pdfvault.core._get_vlm_provider", return_value=mock_provider):
        pdfvault.convert(pdf_bytes, tier="standard", verify=False)

    saw_math_prompt = any(
        "mathematical equations" in (
            (call.args[0] if call.args else call.kwargs.get("prompt", ""))
        )
        for call in mock_provider.complete_sync.call_args_list
    )
    assert saw_math_prompt, "default equations=True must still issue math prompts"


def test_convert_equations_false_leaves_unicode_math_unwrapped():
    """``equations=False`` must also skip the Unicode→LaTeX rewrite. Batch
    consumers that opt out of equations expect plain prose; ``$...\\alpha$``
    markup leaking into figure captions is unrenderable downstream."""
    import pdfvault

    doc = pdfvault.convert(_generate_math_pdf(), tier="fast", verify=False, equations=False)

    assert "$" not in doc.markdown
    assert "\\nabla" not in doc.markdown


def test_convert_equations_default_still_wraps_unicode_math_on_fast_tier():
    import pdfvault

    doc = pdfvault.convert(_generate_math_pdf(), tier="fast", verify=False)

    assert "$" in doc.markdown
