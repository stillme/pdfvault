"""Tests for pdfplumber table extractor."""

from io import BytesIO
from unittest.mock import patch

import pytest
from pdfvault.extractors.pdfplumber_ext import (
    PdfplumberExtractor,
    _is_degenerate_table,
    _normalize_layout_whitespace,
    _sanitize_cell,
    _table_confidence,
)


def _make_two_column_pdf() -> bytes:
    """Build a synthetic 2-column PDF with non-overlapping columns.

    Left column is taller than right (uneven heights) so the two columns
    cannot trivially be read row-by-row — column-aware extractors must
    walk top-to-bottom of column 1 then column 2.
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter
    left_x = 50.0
    right_x = page_w / 2 + 30.0
    top_y = page_h - 60.0
    # Short content so columns do not horizontally overlap.
    for i in range(20):
        y = top_y - i * 18.0
        c.drawString(left_x, y, f"LEFT-{i:02d} left col")
    # Right column has fewer entries so the bottom rows are left-only —
    # this guarantees columns must be read in column-major order.
    for i in range(8):
        y = top_y - i * 18.0
        c.drawString(right_x, y, f"RIGHT-{i:02d} right col")
    c.save()
    return buf.getvalue()


def test_extract_finds_table(sample_pdf_bytes):
    ext = PdfplumberExtractor()
    result = ext.extract(sample_pdf_bytes)
    assert result.engine == "pdfplumber"
    page1 = result.pages[0]
    assert len(page1.tables) >= 1


def test_table_has_headers(sample_pdf_bytes):
    ext = PdfplumberExtractor()
    result = ext.extract(sample_pdf_bytes)
    table = result.pages[0].tables[0]
    assert "Condition" in table.headers
    assert "Mean" in table.headers


def test_table_has_rows(sample_pdf_bytes):
    ext = PdfplumberExtractor()
    result = ext.extract(sample_pdf_bytes)
    table = result.pages[0].tables[0]
    assert len(table.rows) >= 3


def test_table_markdown_format(sample_pdf_bytes):
    ext = PdfplumberExtractor()
    result = ext.extract(sample_pdf_bytes)
    table = result.pages[0].tables[0]
    assert "|" in table.markdown
    assert "---" in table.markdown


def test_page_without_table(sample_pdf_bytes):
    ext = PdfplumberExtractor()
    result = ext.extract(sample_pdf_bytes)
    page2 = result.pages[1]
    assert len(page2.tables) == 0


def test_text_extraction(sample_pdf_bytes):
    ext = PdfplumberExtractor()
    result = ext.extract(sample_pdf_bytes)
    assert "Introduction" in result.pages[0].text


def test_invalid_pdf():
    ext = PdfplumberExtractor()
    with pytest.raises(ValueError, match="Invalid PDF"):
        ext.extract(b"not a pdf")


def test_table_confidence_clean_table():
    headers = ["Condition", "Mean", "SD", "p-value"]
    rows = [
        ["Control", "12.3", "2.1", "-"],
        ["Treatment A", "18.7", "3.4", "0.002"],
        ["Treatment B", "15.1", "2.8", "0.041"],
        ["Treatment C", "16.0", "3.0", "0.030"],
        ["Treatment D", "17.2", "2.9", "0.020"],
    ]
    assert _table_confidence(headers, rows) >= 0.8


def test_table_confidence_high_padding():
    headers = ["A", "B", "C", "D", "E"]
    # 5 cols x 5 rows = 25 row cells, plus 5 header cells = 30 total.
    # 18 empty out of 30 = 60% padding.
    rows = [
        ["x", "", "", "", ""],
        ["", "y", "", "", ""],
        ["", "", "z", "", ""],
        ["", "", "", "w", ""],
        ["", "", "", "", "v"],
    ]
    assert _table_confidence(headers, rows) < 0.7


def test_table_confidence_ragged_columns():
    headers = ["A", "B", "C", "D"]
    rows = [
        ["1", "2", "3", "4"],
        ["1", "2"],
        ["1", "2", "3"],
        ["1"],
    ]
    assert _table_confidence(headers, rows) < 0.7


def test_table_confidence_one_row():
    headers = ["A", "B", "C"]
    rows = [["1", "2", "3"]]
    assert _table_confidence(headers, rows) < 0.7


def test_table_confidence_numeric_header():
    # Header looks like data (all numeric) — likely a misparsed first row.
    headers = ["1.0", "2.5", "3.7", "4.2"]
    rows = [
        ["a", "b", "c", "d"],
        ["e", "f", "g", "h"],
        ["i", "j", "k", "l"],
    ]
    assert _table_confidence(headers, rows) < 0.8


def test_pdfplumber_uses_layout_mode(sample_pdf_bytes):
    """The extractor must call extract_text(layout=True).

    Without layout=True pdfplumber ignores in-column line order. The
    column-aware path crops per-column and then calls extract_text with
    layout=True on each crop, so the spy still observes layout=True.
    """
    ext = PdfplumberExtractor()
    captured: dict = {}
    import pdfplumber.page as plumber_page
    real = plumber_page.Page.extract_text

    def spy(self, *args, **kwargs):
        # Record kwargs from the first text extraction call.
        captured.setdefault("kwargs", kwargs)
        return real(self, *args, **kwargs)

    with patch.object(plumber_page.Page, "extract_text", spy):
        ext.extract(sample_pdf_bytes)

    assert captured.get("kwargs", {}).get("layout") is True


def test_pdfplumber_two_column_separates_columns_completely():
    """Column-aware extraction returns the entire left column before any
    right-column text — not the row-by-row interleave that plain
    layout=True produces. Without this, ``faecal micro-`` from the end
    of column 1 lands on the same physical line as ``biota transplant``
    from the start of column 2 and the cleaner can't recover."""
    pdf_bytes = _make_two_column_pdf()
    ext = PdfplumberExtractor()
    page = ext.extract_page(pdf_bytes, 0)
    text = page.text

    # All left-col tokens present.
    assert "LEFT-00" in text and "LEFT-19" in text
    # All right-col tokens present.
    assert "RIGHT-00" in text and "RIGHT-07" in text

    # Strong invariant: every LEFT marker appears before every RIGHT
    # marker. This is what fails on Nature-style two-column papers
    # under plain layout=True.
    last_left = max(
        text.find(f"LEFT-{i:02d}") for i in range(20)
        if text.find(f"LEFT-{i:02d}") >= 0
    )
    first_right = min(
        text.find(f"RIGHT-{i:02d}") for i in range(8)
        if text.find(f"RIGHT-{i:02d}") >= 0
    )
    assert last_left < first_right, (
        f"Column order broken: last LEFT at {last_left}, "
        f"first RIGHT at {first_right} — RIGHT must come after all LEFT"
    )

    # Within each column, markers appear in numerical order.
    left_positions = [text.find(f"LEFT-{i:02d}") for i in range(20)]
    assert left_positions == sorted(left_positions)
    right_positions = [text.find(f"RIGHT-{i:02d}") for i in range(8)]
    assert right_positions == sorted(right_positions)


def test_pdfplumber_single_column_unchanged():
    """Single-column pages must still extract cleanly — the column
    detector should not invent boundaries on a normal narrative page."""
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(
            "This is a single column document with sufficient prose "
            "to exercise the detector. " * 10,
            styles["Normal"],
        ),
    ]
    doc.build(story)

    ext = PdfplumberExtractor()
    page = ext.extract_page(buf.getvalue(), 0)
    assert "single column document" in page.text
    # The whole paragraph should appear as one block — no column-break
    # artefact should bisect the prose.
    assert page.text.count("single column document") >= 5


def test_normalize_layout_whitespace_collapses_padding():
    """Layout mode pads with runs of spaces/blank lines — the cleanup
    must collapse them without destroying semantic spacing."""
    raw = "alpha     beta\ngamma   delta\n\n\n\nepsilon zeta\n"
    out = _normalize_layout_whitespace(raw)
    # Runs of >=3 spaces collapsed to one
    assert "alpha beta" in out
    assert "gamma delta" in out
    # Runs of >=3 blank lines collapsed to one blank line
    assert "\n\n\n" not in out
    # Single space and single blank line preserved
    assert "epsilon zeta" in out


def test_normalize_layout_whitespace_keeps_short_runs():
    """Two-space runs must be preserved (could be sentence spacing)."""
    raw = "End of sentence.  Next sentence starts."
    out = _normalize_layout_whitespace(raw)
    assert out == raw


# --- Phase 2 fixes: degenerate-table filter and cell sanitization -------

def test_is_degenerate_table_all_empty():
    """All-empty grids (pdfplumber's bordered-figure false positives)
    must be flagged as degenerate. On the Nature gut paper this filters
    out 7 spurious 'tables' that were figure panels with detected
    border lines."""
    assert _is_degenerate_table(["", "", ""], [["", "", ""], ["", "", ""]])
    assert _is_degenerate_table([""], [[""]])


def test_is_degenerate_table_single_fragment():
    """A 1-column 'table' with only one non-empty cell is just an
    enclosed text fragment, not a real table."""
    assert _is_degenerate_table([""], [["only data"]])
    assert _is_degenerate_table(["only header"], [[""]])


def test_is_degenerate_table_keeps_real_tables():
    """A real table — multiple columns and multiple non-empty cells —
    must NOT be flagged as degenerate."""
    headers = ["Gene", "Fold change", "p-value"]
    rows = [["TP53", "2.1", "0.001"], ["BRCA1", "1.7", "0.01"]]
    assert not _is_degenerate_table(headers, rows)


def test_is_degenerate_table_keeps_sparse_real_table():
    """A multi-column table with a few non-empty cells (e.g. STAR
    Methods reagent list with sparse columns) must be kept — VLM can
    still recover structure from it."""
    headers = ["Reagent", "Source", "Identifier"]
    rows = [
        ["FITC anti-CD8", "BD Biosciences", "Cat#553031"],
        ["", "", ""],  # one empty padding row
    ]
    assert not _is_degenerate_table(headers, rows)


def test_sanitize_cell_collapses_newlines():
    """Multi-line cells (common in Cancer Cell tables) must collapse
    to a single line — a literal newline inside a markdown ``|`` cell
    breaks the row and corrupts every cell after it."""
    out = _sanitize_cell("Novartis\nPharmaceuticals")
    assert out == "Novartis Pharmaceuticals"
    assert "\n" not in out


def test_sanitize_cell_collapses_all_whitespace_runs():
    """Runs of internal whitespace (tabs, multiple spaces) collapse
    to one space; leading/trailing whitespace is stripped."""
    assert _sanitize_cell("  a   b\t\tc \n d  ") == "a b c d"


def test_sanitize_cell_escapes_pipes():
    """A literal ``|`` inside cell text would terminate the markdown
    cell early; escape it so the row stays intact."""
    out = _sanitize_cell("0.5 | 0.7")
    assert "\\|" in out
    assert out == "0.5 \\| 0.7"


def test_sanitize_cell_handles_empty():
    assert _sanitize_cell("") == ""
    assert _sanitize_cell("   ") == ""


def _make_multiline_cell_table_pdf() -> bytes:
    """Build a 3-col bordered table where some cells contain
    multi-line text — emulates the Cancer Cell paper, where pdfplumber
    returns cells like ``Novartis\\nPharmaceuticals``.

    The synthetic PDF draws table grid lines and writes one cell with
    text spanning two physical lines; a correct extractor must then
    flatten the two lines into one markdown cell."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter

    # 3 columns x 3 rows (header + 2 data rows). Each cell is 120pt wide
    # x 40pt tall. The middle column on the second data row holds a
    # two-line value to exercise the newline-in-cell case.
    col_w = 120.0
    row_h = 40.0
    x0 = 60.0
    y_top = page_h - 120.0
    n_cols, n_rows = 3, 3

    # Draw grid (so pdfplumber detects this as a table).
    for r in range(n_rows + 1):
        y = y_top - r * row_h
        c.line(x0, y, x0 + n_cols * col_w, y)
    for col in range(n_cols + 1):
        x = x0 + col * col_w
        c.line(x, y_top, x, y_top - n_rows * row_h)

    # Headers row
    headers = ["Gene", "Source", "Notes"]
    for col, h in enumerate(headers):
        c.drawString(x0 + col * col_w + 5, y_top - 25, h)

    # Data row 1: single-line cells
    row1 = ["TP53", "Sanger", "ok"]
    for col, v in enumerate(row1):
        c.drawString(x0 + col * col_w + 5, y_top - 65, v)

    # Data row 2: middle cell is split across two physical lines
    # within the same bordered cell — pdfplumber will return this as
    # ``Novartis\nPharmaceuticals`` style.
    c.drawString(x0 + 0 * col_w + 5, y_top - 105, "BRCA1")
    c.drawString(x0 + 1 * col_w + 5, y_top - 95, "Novartis")
    c.drawString(x0 + 1 * col_w + 5, y_top - 115, "Pharmaceuticals")
    c.drawString(x0 + 2 * col_w + 5, y_top - 105, "yes")

    c.save()
    return buf.getvalue()


def test_extractor_drops_degenerate_tables():
    """End-to-end: a synthetic PDF that has an empty bordered region
    (e.g. a figure panel with detected lines) must not yield a table
    in the output. Without this filter, low-confidence empty tables
    slip into ``doc.tables`` and trigger one wasted VLM call each."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter

    # Draw an empty bordered 2x2 grid — no text inside. pdfplumber's
    # default settings detect this as a table because of the lines.
    col_w = 120.0
    row_h = 40.0
    x0, y_top = 80.0, page_h - 200.0
    for r in range(3):
        c.line(x0, y_top - r * row_h, x0 + 2 * col_w, y_top - r * row_h)
    for col in range(3):
        c.line(x0 + col * col_w, y_top, x0 + col * col_w, y_top - 2 * row_h)
    c.drawString(72, 72, "body text outside the empty grid")
    c.save()

    ext = PdfplumberExtractor()
    page = ext.extract_page(buf.getvalue(), 0)
    # All extracted tables (if any) must have at least one non-empty
    # cell — i.e. the empty grid must have been filtered out.
    for t in page.tables:
        all_cells = list(t.headers) + [c for row in t.rows for c in row]
        assert any(cell.strip() for cell in all_cells), (
            "Empty grid leaked through degenerate-table filter"
        )


def test_extractor_flattens_multiline_cells():
    """End-to-end: a multi-line cell ``Novartis\\nPharmaceuticals`` must
    appear in the markdown as a single line. Otherwise the row is
    broken across two physical lines and downstream markdown parsers
    see a corrupted table."""
    pdf_bytes = _make_multiline_cell_table_pdf()
    ext = PdfplumberExtractor()
    page = ext.extract_page(pdf_bytes, 0)
    assert page.tables, "expected at least one table from synthetic PDF"
    table = page.tables[0]

    # The multi-line value must appear as a single space-joined cell
    # both on the Table object and in the rendered markdown.
    flat_cells = list(table.headers) + [c for row in table.rows for c in row]
    multiline_cell = next(
        (c for c in flat_cells if "Novartis" in c and "Pharmaceuticals" in c),
        None,
    )
    assert multiline_cell is not None, (
        f"Expected a Novartis/Pharmaceuticals cell, got: {flat_cells}"
    )
    assert "\n" not in multiline_cell, (
        f"Cell must be flattened — got: {multiline_cell!r}"
    )
    assert "Novartis Pharmaceuticals" in multiline_cell

    # The rendered markdown row must be on a single physical line.
    md_lines = table.markdown.splitlines()
    container = [ln for ln in md_lines
                 if "Novartis" in ln and "Pharmaceuticals" in ln]
    assert container, (
        "Novartis and Pharmaceuticals were split across markdown lines — "
        f"markdown was:\n{table.markdown}"
    )


def test_table_confidence_all_empty_bottoms_out():
    """A near-100%-empty grid must score below the VLM enhancer
    threshold's 'maybe re-process' band — it should bottom out so the
    confidence signal correctly reflects that there is nothing to
    correct. Defensive layer below ``_is_degenerate_table``."""
    headers = ["", "", "", ""]
    rows = [["", "", "", ""]] * 4
    # Fully empty: ``_is_degenerate_table`` would catch it, but if it
    # ever leaked through, confidence must be ~0, not ~0.5.
    assert _table_confidence(headers, rows) <= 0.1


def _make_small_font_pdf(font_size: float = 7.5) -> bytes:
    """Single-column PDF set in legend-sized type.

    Nature / Cell figure legends are set at 7-8pt where a word gap is about
    2pt — below pdfplumber's default ``x_tolerance`` of 3pt, so every space
    is swallowed unless the tolerance scales with the font.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", font_size)
    lines = [
        "Fig. 2 | Blocking BMP signaling in vitro does not affect YAP target gene",
        "expression. a Schematic of the epithelial cell compartment in antral glands.",
        "b Organoid treatment schematic. Scale bar: 100 um.",
    ]
    from reportlab.pdfbase.pdfmetrics import stringWidth

    # Typeset journals position each word by offset rather than emitting a
    # space glyph, so draw the words individually with a legend-sized gap.
    gap = 0.28 * font_size
    for i, line in enumerate(lines):
        x = 50.0
        y = 700 - i * (font_size + 2)
        for word in line.split(" "):
            c.drawString(x, y, word)
            x += stringWidth(word, "Helvetica", font_size) + gap
    c.save()
    return buf.getvalue()


def test_small_font_word_gaps_are_preserved():
    text = PdfplumberExtractor().extract_page(_make_small_font_pdf(), 0).text
    assert "Blocking BMP signaling in vitro does not affect" in text
    assert "BlockingBMPsignaling" not in text


def _make_two_column_pdf_with_two_column_legend() -> bytes:
    """Two-column body above a two-column figure legend in smaller type.

    Nature-style pages set the legend in two columns beneath the figure and
    the body in two columns beneath that. Cropping the whole page into two
    columns reads left-legend, left-body, right-legend, right-body — cutting
    the legend in half. The legend band has to be read before the body band.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter
    left_x, right_x, top_y = 50.0, page_w / 2 + 30.0, page_h - 60.0
    legend_left = [
        "Fig. 1 | Downregulation of BMP signaling induces the",
        "regenerative state. a Confocal images of antral tissue",
    ]
    legend_right = [
        "from uninfected mice. b Quantification of KI67 positive",
        "cells per gland (n = 3 mice per group). Scale bars: 100 um.",
    ]
    c.setFont("Helvetica", 7.5)
    for i, (l, r) in enumerate(zip(legend_left, legend_right)):
        c.drawString(left_x, top_y - i * 9.5, l)
        c.drawString(right_x, top_y - i * 9.5, r)
    c.setFont("Helvetica", 9)
    body_top = top_y - 4 * 9.5 - 14
    for i in range(20):
        c.drawString(left_x, body_top - i * 12.0, f"LEFT-{i:02d} body text in the left column")
        c.drawString(right_x, body_top - i * 12.0, f"RIGHT-{i:02d} body text in the right column")
    c.save()
    return buf.getvalue()


def test_two_column_legend_is_read_as_one_band_before_the_body():
    text = PdfplumberExtractor().extract_page(_make_two_column_pdf_with_two_column_legend(), 0).text
    flat = " ".join(text.split())
    legend = ("Fig. 1 | Downregulation of BMP signaling induces the regenerative state. "
              "a Confocal images of antral tissue from uninfected mice. b Quantification of KI67 positive "
              "cells per gland (n = 3 mice per group). Scale bars: 100 um.")
    assert legend in flat
    assert flat.index("LEFT-00") < flat.index("LEFT-19") < flat.index("RIGHT-00") < flat.index("RIGHT-19")
