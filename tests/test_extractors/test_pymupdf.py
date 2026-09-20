"""Tests for PyMuPDF extractor."""
import pytest

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def test_leading_bold_text_stops_at_first_regular_span():
    from pdfvault.extractors.pymupdf_ext import _leading_bold_text

    spans = [
        {"text": "Spatial transcriptomics data processing.", "flags": 16, "font": "SemiBold"},
        {"text": " Raw Visium datasets were", "flags": 4, "font": "Regular"},
    ]

    assert _leading_bold_text(spans) == "Spatial transcriptomics data processing."


@pytest.mark.skipif(not HAS_PYMUPDF, reason="pymupdf not installed")
class TestPymupdfExtractor:
    def test_name(self):
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        ext = PymupdfExtractor()
        assert ext.name == "pymupdf"

    def test_extract(self, sample_pdf_bytes):
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        ext = PymupdfExtractor()
        result = ext.extract(sample_pdf_bytes)
        assert result.engine == "pymupdf"
        assert len(result.pages) == 2
        assert "Introduction" in result.pages[0].text

    def test_extract_page(self, sample_pdf_bytes):
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        ext = PymupdfExtractor()
        page = ext.extract_page(sample_pdf_bytes, 1)
        assert "Discussion" in page.text

    def test_extract_bold_headings(self, sample_pdf_bytes):
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        ext = PymupdfExtractor()
        headings = ext.extract_bold_headings(sample_pdf_bytes)
        assert isinstance(headings, list)
        # Sample PDF uses reportlab Heading1 style which should be detected
        titles = [h["text"] for h in headings]
        assert any("Introduction" in t for t in titles) or any("Results" in t for t in titles)

    def test_extract_bold_headings_have_page_and_font_size(self, sample_pdf_bytes):
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        ext = PymupdfExtractor()
        headings = ext.extract_bold_headings(sample_pdf_bytes)
        for h in headings:
            assert "text" in h
            assert "page" in h
            assert "font_size" in h
            assert isinstance(h["page"], int)
            assert isinstance(h["font_size"], float)

    def test_extract_figures(self, sample_pdf_bytes):
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        ext = PymupdfExtractor()
        figures = ext.extract_figures(sample_pdf_bytes)
        assert isinstance(figures, list)
        # Sample PDF may or may not have large images, but shouldn't error

    def test_extract_figures_max_per_page_one(self, sample_pdf_bytes):
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        ext = PymupdfExtractor()
        # With max_per_page=1, each page contributes at most one figure
        figures = ext.extract_figures(sample_pdf_bytes, max_per_page=1)
        assert isinstance(figures, list)
        pages_seen = [f["page"] for f in figures]
        assert len(pages_seen) == len(set(pages_seen)), (
            "max_per_page=1 should yield at most one figure per page"
        )

    def test_extract_figures_max_per_page_none_returns_all(self, sample_pdf_bytes):
        from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
        ext = PymupdfExtractor()
        # max_per_page=None disables the per-page limit
        unrestricted = ext.extract_figures(sample_pdf_bytes, max_per_page=None)
        restricted = ext.extract_figures(sample_pdf_bytes, max_per_page=1)
        # Restricted should have no more figures than unrestricted
        assert len(restricted) <= len(unrestricted)


# --- Vector-graphics figure detection ---------------------------------------


def _make_vector_figure_pdf() -> bytes:
    """Build a PDF whose only figure is a dense cluster of vector paths.

    Modern review journals (Nat Rev Genetics) render figures as PDF
    vector drawings rather than embedded raster images. ``get_images``
    returns nothing for those pages but the figure is clearly visible.
    The fixture mirrors that shape — a page with title text, then a
    box of >>100 small line segments, then no images.
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 14)
    c.drawString(72, 720, "Figure 1 | Schematic of a complex pathway.")

    # Render ~600 short vector lines in a dense rectangle. This puts us
    # well above the 100-drawing threshold without producing any
    # raster image.
    for i in range(20):
        for j in range(30):
            x = 80 + j * 12
            y = 400 + i * 8
            c.line(x, y, x + 8, y + 6)
    c.showPage()

    # Add a second page with nothing on it — the fallback should NOT
    # invent a figure here.
    c.drawString(72, 720, "Plain text on a follow-on page.")
    c.showPage()

    c.save()
    return buf.getvalue()


def test_vector_figures_extracted_when_no_raster():
    """A page with no raster image but many vector drawings produces a
    figure via the vector-fallback path."""
    from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
    pdf_bytes = _make_vector_figure_pdf()
    ext = PymupdfExtractor()
    figures = ext.extract_figures(pdf_bytes)
    assert len(figures) >= 1, "vector-only figure page should produce ≥ 1 figure"
    fig = figures[0]
    assert fig["page"] == 0
    assert fig["image_bytes"][:8] == b"\x89PNG\r\n\x1a\n", "vector figures render as PNG"
    assert fig["width"] >= 100 and fig["height"] >= 100


def test_vector_fallback_does_not_invent_figures_on_blank_pages():
    """The second page in the fixture has only a single text line — the
    drawing count is well below the threshold and no figure should be
    produced."""
    from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
    pdf_bytes = _make_vector_figure_pdf()
    figures = PymupdfExtractor().extract_figures(pdf_bytes)
    blank_page_figures = [f for f in figures if f["page"] == 1]
    assert blank_page_figures == [], "no vector figures expected on blank page"


def test_vector_fallback_skips_pages_with_qualifying_raster():
    """When a page already contains a raster image at or above the
    min_width / min_height threshold the vector fallback must NOT add
    a duplicate figure for the same page — that's the regression we'd
    see on Nature-style papers where every figure-bearing page also
    has a real raster."""
    from io import BytesIO
    from PIL import Image
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from pdfvault.extractors.pymupdf_ext import PymupdfExtractor

    # Build a 400x300 blue rectangle as a raster image.
    img = Image.new("RGB", (400, 300), color=(40, 70, 200))
    img_buf = BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 720, "Page with both a raster and many vectors.")
    c.drawImage(ImageReader(img_buf), 72, 360, width=400, height=300)
    # Lots of vector lines too — the fallback would otherwise fire.
    for i in range(20):
        for j in range(30):
            c.line(80 + j * 6, 100 + i * 4, 80 + j * 6 + 4, 100 + i * 4 + 3)
    c.showPage()
    c.save()

    figures = PymupdfExtractor().extract_figures(buf.getvalue())
    assert len(figures) == 1, (
        f"page with qualifying raster should produce exactly one figure, "
        f"got {len(figures)}"
    )


def _make_text_band_with_icons_pdf() -> bytes:
    """A page of body prose whose only vector marks are inline link icons.

    Journals such as eLife draw a small vector glyph beside every inline
    citation link. Those glyphs push the page over the vector-drawing
    threshold, and clustering then spans the whole paragraph, so the
    fallback renders a picture of body text and calls it a figure.
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 10)
    for row in range(26):
        y = 700 - row * 10
        c.drawString(72, y, "inserting a LacZ cassette into the Slap gene, beta-galactosidase activity was")
        # Six tiny link glyphs per line, sitting inside the text band.
        for k in range(6):
            x = 90 + k * 60
            c.rect(x, y, 4, 4, stroke=1, fill=0)
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_title_block_pdf() -> bytes:
    """A journal cover block: title, authors, a logo and rule bars.

    Roughly half of the region is text and the rest is branding, which
    is why a pure text-coverage rule alone does not settle it.
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    # Logo: a block of small filled squares, plus full-width rule bars
    # above and below the title so the cluster spans the page.
    for i in range(12):
        for j in range(10):
            c.rect(72 + j * 5, 730 + i * 4, 4, 3, stroke=0, fill=1)
    c.rect(72, 772, 468, 2, stroke=0, fill=1)
    c.rect(72, 648, 468, 2, stroke=0, fill=1)
    c.setFont("Helvetica", 16)
    c.drawString(150, 740, "Slap restricts oncogenic Src-family kinase")
    c.drawString(150, 720, "signaling to maintain colonic homeostasis")
    c.setFont("Helvetica", 9)
    for row in range(6):
        c.drawString(150, 700 - row * 11, "Dana Naim, Zouheir Houhou, Florent Cauchois, Kevin Espie, Valerie Simon")
    c.showPage()
    c.save()
    return buf.getvalue()


def test_vector_fallback_rejects_a_body_text_band():
    from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
    figures = PymupdfExtractor().extract_figures(_make_text_band_with_icons_pdf())
    assert figures == [], "a paragraph of prose with inline link glyphs is not a figure"


def test_vector_fallback_rejects_a_title_block():
    from pdfvault.extractors.pymupdf_ext import PymupdfExtractor
    figures = PymupdfExtractor().extract_figures(_make_title_block_pdf())
    assert figures == [], "the article title block is not a figure"
