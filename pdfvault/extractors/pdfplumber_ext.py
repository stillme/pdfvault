"""pdfplumber-based table extractor (MIT core)."""

from __future__ import annotations

import io
import re

import pdfplumber

from pdfvault.extractors.base import ExtractionResult, PageContent, RawTable

_NUMERIC_RE = re.compile(r"^[\s\-+()$%]*\d[\d.,\s%]*[\s\-+()$%]*$")


def _is_degenerate_table(headers: list[str], rows: list[list[str]]) -> bool:
    """Return True if a 'table' is junk that should be dropped entirely.

    pdfplumber's default ``extract_tables`` flags any region with
    detected horizontal/vertical lines as a table — including bordered
    figure panels, decorative boxes, and column rules in two-column
    layouts. These produce all-empty cell grids that have no
    information value but still count toward the VLM enhancement
    budget (any table with ``confidence < CONFIDENCE_THRESHOLD``
    triggers a VLM call). On the Nature gut paper this single filter
    eliminates 7 spurious VLM calls per document.

    A table is considered degenerate when:
    * Every cell (header + data rows) is empty/whitespace, OR
    * It has fewer than 2 columns AND fewer than 2 non-empty cells —
      i.e. effectively a single text fragment that pdfplumber
      happened to enclose in border-detected lines.
    """
    all_cells = list(headers) + [c for row in rows for c in row]
    non_empty = [c for c in all_cells if c.strip()]
    if not non_empty:
        return True
    if len(headers) < 2 and len(non_empty) < 2:
        return True
    return False


# Cell-text sanitization. Markdown tables are line-oriented: a literal
# newline inside a ``|`` cell breaks the row and corrupts every cell
# after it. A pipe character likewise terminates the cell early. Both
# happen in the wild on multi-line cells (e.g. Cancer Cell tables that
# pdfplumber returns as ``Novartis\nPharmaceuticals``). We collapse
# all internal whitespace to a single space and escape literal pipes
# so downstream markdown parsers (and KG ingest pipelines) see
# well-formed rows.
def _sanitize_cell(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned.replace("|", "\\|")


def _table_confidence(headers: list[str], rows: list[list[str]]) -> float:
    """Score a table 0-1 from padding ratio, column variance, header sanity, and shape."""
    if not headers or not rows:
        return 0.2
    n_cols = len(headers)
    score = 1.0
    if n_cols < 2:
        score -= 0.4
    if len(rows) < 2:
        score -= 0.4
    # Column variance: rows that don't match header width
    ragged = sum(1 for r in rows if len(r) != n_cols)
    if ragged:
        score -= 0.15 + min(0.35, 0.35 * ragged / max(1, len(rows)))
    # Padding ratio: empty cells / total cells (across header + rows).
    # An almost-entirely-empty grid (>=95% padding) should bottom out
    # hard rather than land at ~0.5 where it still trips the VLM
    # enhancer threshold (see ``enhancers/tables.CONFIDENCE_THRESHOLD``).
    # Truly all-empty grids are dropped upstream by
    # ``_is_degenerate_table``; this clause is the defensive layer for
    # near-empty grids (e.g. a 4x5 with two stray cells of OCR noise)
    # that would otherwise burn a VLM call for no useful gain.
    cells = [c for r in [headers] + rows for c in r]
    if cells:
        empty_ratio = sum(1 for c in cells if not c.strip()) / len(cells)
        if empty_ratio >= 0.95:
            score -= 0.9
        else:
            score -= min(0.5, empty_ratio * 0.7)
    # Header sanity: numeric-looking header cells suggest data-as-header
    non_empty_headers = [h for h in headers if h.strip()]
    if non_empty_headers:
        numeric = sum(1 for h in non_empty_headers if _NUMERIC_RE.match(h.strip()))
        if numeric / len(non_empty_headers) > 0.5:
            score -= 0.3
    return max(0.0, min(1.0, score))


# Collapse the heavy whitespace padding introduced by pdfplumber's
# ``layout=True`` mode while preserving semantic spacing. Layout mode pads
# columns with runs of spaces and inserts blank lines for vertical gaps; we
# only flatten obvious noise (3+ spaces -> 1 space, 3+ blank lines -> 1 blank
# line) so downstream cleaners see something close to normal prose.
_RUN_OF_SPACES_RE = re.compile(r" {3,}")
_RUN_OF_BLANK_LINES_RE = re.compile(r"(?:[ \t]*\n){3,}")


# pdfplumber starts a new word when the gap between two glyphs exceeds
# ``x_tolerance`` (default 3pt, absolute). Typeset journals position words by
# offset rather than emitting space glyphs, and at legend sizes (7-8pt) a word
# gap is only ~2pt — so the default glued every word of every Nature / Cell
# figure legend together ("BlockingBMPsignalinginvitro"). Scaling the
# tolerance to the glyph size keeps body text intact and restores legend spaces.
WORD_GAP_TOLERANCE_RATIO = 0.15


def _extract_layout_text(page) -> str:
    """``extract_text(layout=True)`` with a font-relative word-gap tolerance."""
    return page.extract_text(layout=True, x_tolerance_ratio=WORD_GAP_TOLERANCE_RATIO) or ""


def _normalize_layout_whitespace(text: str) -> str:
    if not text:
        return text
    text = _RUN_OF_SPACES_RE.sub(" ", text)
    text = _RUN_OF_BLANK_LINES_RE.sub("\n\n", text)
    # Strip trailing spaces left at the end of lines by layout padding.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


# --- Column-aware text extraction ---------------------------------------
#
# ``layout=True`` preserves spatial padding but doesn't actually split a
# physical line that spans multiple columns. On Nature- / Cell-style
# 2-column journals the columns are close enough that words from the
# right column glue onto words from the left column on every line. The
# downstream cost of that is brutal:
#
# * "faecal microbiota transplant" gets split as "faecal micro-" (end
#   of left column) + "biota transplant" (start of right column),
#   producing a dangling hyphen the cleaner can't rejoin.
# * Bibliography entries from facing columns merge onto one line, so
#   the parser sees ``[3] Hickey ... Nature 41. Xu ...`` and treats
#   the second author block as ref-3's continuation.
# * Heatmap axis labels in figure regions leak into the text stream
#   alongside body prose.
#
# The fix here detects column boundaries from the x-distribution of
# words, crops each column's bbox, extracts text per-column with
# ``layout=True`` to preserve in-column line order, then concatenates
# top-to-bottom column-by-column. arXiv-style wide-gap layouts are also
# handled — the threshold tolerates either.

# Minimum horizontal gap (in PDF points) that counts as a column
# boundary. Nature's tight two-column body produces gaps of ~10pt;
# arXiv-style papers run ~30-50pt. 8pt is above any inter-word slack
# we've measured and below the narrowest real column gap on the
# corpus. Single-column pages produce zero empty runs in the text
# region, so a lower threshold doesn't cause false positives there.
_MIN_COLUMN_GAP_PT = 8.0
# Minimum word count on a page before we trust the gap-detection at all.
# Sparse pages (title pages, blank back matter) don't have enough data
# to cluster reliably.
_COLUMN_MIN_WORDS = 60
# Each side of a candidate boundary must hold at least this fraction of
# the page's words. Without this, a single-column page with a sparse
# strip near one margin produces a "boundary" that bisects narrative
# text instead of separating real columns. 20% is permissive enough to
# cover end-of-section pages where one column is much shorter than the
# other (very common on Nature/Cell layouts).
_MIN_COLUMN_FILL_FRACTION = 0.20


def _detect_column_boundaries(page) -> list[float]:
    """Return x-coordinates of column splits on ``page``, ascending.

    Empty list means the page is single-column (or too sparse to split).

    Spatial-occupancy approach: build a 1pt-resolution histogram of
    which x-coordinates have any word covering them, then find the
    longest contiguous empty interval inside the text-bearing region.
    A real column gap is a vertical strip that NO word crosses — single-
    column pages don't have one because line wrapping fills the page
    width. The midpoint-sort approach is fooled by sparse line wraps.
    """
    try:
        words = page.extract_words()
    except Exception:
        return []
    if len(words) < _COLUMN_MIN_WORDS:
        return []

    page_width = int(round(float(page.width)))
    if page_width <= 0:
        return []

    # 1pt-resolution occupancy: covered[x] = True if any word's bbox
    # spans the integer column x. Words can have fractional bboxes;
    # rounding outward (floor x0, ceil x1) avoids missing tight gaps.
    import math
    covered = [False] * (page_width + 1)
    for w in words:
        x0 = max(0, int(math.floor(w["x0"])))
        x1 = min(page_width, int(math.ceil(w["x1"])))
        for x in range(x0, x1 + 1):
            covered[x] = True

    # Bound the search to the text-bearing region — anything outside is
    # margin and would produce a huge spurious gap.
    try:
        text_min = max(0, int(math.floor(min(w["x0"] for w in words))))
        text_max = min(page_width, int(math.ceil(max(w["x1"] for w in words))))
    except ValueError:
        return []
    if text_max - text_min < 2 * _MIN_COLUMN_GAP_PT:
        return []

    # Walk the text-bearing range, find the longest run of False.
    longest_run = 0
    longest_start = -1
    longest_end = -1
    run_start = -1
    for x in range(text_min, text_max + 1):
        if not covered[x]:
            if run_start < 0:
                run_start = x
            run_len = x - run_start + 1
            if run_len > longest_run:
                longest_run = run_len
                longest_start = run_start
                longest_end = x
        else:
            run_start = -1

    if longest_run < _MIN_COLUMN_GAP_PT:
        return []

    boundary = (longest_start + longest_end) / 2.0

    # Both columns must actually contain text. Sparse stragglers near
    # one margin can produce a "boundary" that bisects narrative prose.
    left_count = sum(1 for w in words if (w["x0"] + w["x1"]) / 2 < boundary)
    right_count = len(words) - left_count
    threshold = int(len(words) * _MIN_COLUMN_FILL_FRACTION)
    if left_count < threshold or right_count < threshold:
        return []

    return [boundary]


def _extract_text_with_columns(page) -> str:
    """Extract page text, splitting at detected column boundaries.

    Single-column pages fall back to the plain ``layout=True`` extract.
    Multi-column pages are cropped per-column and concatenated in
    reading order (left to right). Within each column we still use
    ``layout=True`` so vertical line order is preserved.

    Pages are first cut into horizontal *bands* wherever the dominant
    type size changes between two-column regions (see
    ``_type_size_bands``). Nature-style pages set a figure legend in two
    columns of 7-8pt type above two columns of body text; cropping the
    whole page into columns reads left-legend, left-body, right-legend,
    right-body and cuts the legend in half. Reading band by band keeps
    the legend contiguous and ahead of the body.
    """
    boundaries = _detect_column_boundaries(page)
    if not boundaries:
        return _normalize_layout_whitespace(_extract_layout_text(page))

    bands = _type_size_bands(page, boundaries[0])
    if len(bands) <= 1:
        return _columns_text(page, boundaries)

    band_texts: list[str] = []
    for top, bottom in bands:
        try:
            region = page.crop((0.0, top, float(page.width), bottom))
        except Exception:
            continue
        # A short legend band rarely has enough words for its own column
        # detection; the page-level boundary applies to it just the same.
        region_boundaries = _detect_column_boundaries(region) or boundaries
        text = _columns_text(region, region_boundaries)
        if text.strip():
            band_texts.append(text)
    return "\n\n".join(band_texts)


def _columns_text(page, boundaries: list[float]) -> str:
    """Crop ``page`` at ``boundaries`` and concatenate the columns left to right."""
    # bbox is (x0, top, x1, bottom). Build [0, b1, b2, ..., width].
    edges = [0.0] + list(boundaries) + [float(page.width)]
    column_texts: list[str] = []
    for i in range(len(edges) - 1):
        bbox = (edges[i], float(page.bbox[1]), edges[i + 1], float(page.bbox[3]))
        try:
            cropped = page.crop(bbox)
            text = _extract_layout_text(cropped)
        except Exception:
            text = ""
        text = _normalize_layout_whitespace(text)
        if text.strip():
            column_texts.append(text)
    return "\n\n".join(column_texts)


# Lines whose ``top`` differs by less than this are the same visual line.
_LINE_TOP_TOLERANCE_PT = 2.0
# A band must have at least this many lines *and* text on both sides of the
# column boundary; anything smaller (a heading, a stray label) is folded into
# its neighbour so a left-column subheading cannot chop the right column.
_MIN_BAND_LINES = 2


def _type_size_bands(page, boundary: float) -> list[tuple[float, float]]:
    """Return (top, bottom) bands where the dominant glyph size changes.

    Empty list (or a single band) means the page reads fine as one block.
    """
    try:
        chars = [c for c in page.chars if str(c.get("text", "")).strip()]
    except Exception:
        return []
    if not chars:
        return []

    chars.sort(key=lambda c: (float(c["top"]), float(c["x0"])))
    lines: list[dict] = []
    for ch in chars:
        top, bottom = float(ch["top"]), float(ch["bottom"])
        size = round(float(ch.get("size") or 0.0) * 2) / 2
        center_x = (float(ch["x0"]) + float(ch["x1"])) / 2
        if lines and abs(top - lines[-1]["top"]) <= _LINE_TOP_TOLERANCE_PT:
            line = lines[-1]
            line["bottom"] = max(line["bottom"], bottom)
            line["sizes"][size] = line["sizes"].get(size, 0) + 1
        else:
            line = {"top": top, "bottom": bottom, "sizes": {size: 1}, "left": False, "right": False}
            lines.append(line)
        if center_x < boundary:
            line["left"] = True
        else:
            line["right"] = True
    for line in lines:
        line["size"] = max(line["sizes"], key=line["sizes"].get)

    runs: list[dict] = []
    for line in lines:
        if runs and runs[-1]["size"] == line["size"]:
            runs[-1]["lines"].append(line)
        else:
            runs.append({"size": line["size"], "lines": [line]})

    def qualifies(run: dict) -> bool:
        return (
            len(run["lines"]) >= _MIN_BAND_LINES
            and any(l["left"] for l in run["lines"])
            and any(l["right"] for l in run["lines"])
        )

    bands: list[dict] = []
    for run in runs:
        if bands and not qualifies(run):
            bands[-1]["lines"].extend(run["lines"])
        else:
            bands.append(run)
    if len(bands) > 1 and not qualifies(bands[0]):
        bands[1]["lines"] = bands[0]["lines"] + bands[1]["lines"]
        bands.pop(0)
    if len(bands) <= 1:
        return []

    page_top, page_bottom = float(page.bbox[1]), float(page.bbox[3])
    edges = [page_top]
    for prev, nxt in zip(bands, bands[1:]):
        prev_bottom = max(l["bottom"] for l in prev["lines"])
        next_top = min(l["top"] for l in nxt["lines"])
        edges.append((prev_bottom + next_top) / 2)
    edges.append(page_bottom)
    return list(zip(edges, edges[1:]))


class PdfplumberExtractor:
    @property
    def name(self) -> str:
        return "pdfplumber"

    @property
    def capabilities(self) -> list[str]:
        return ["text", "tables"]

    def extract(self, pdf_bytes: bytes) -> ExtractionResult:
        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        except Exception as e:
            raise ValueError(f"Invalid PDF: {e}") from e

        pages = []
        for i, page in enumerate(pdf.pages):
            content = self._extract_page(page, i)
            pages.append(content)
        pdf.close()
        return ExtractionResult(pages=pages, engine=self.name)

    def extract_page(self, pdf_bytes: bytes, page_number: int) -> PageContent:
        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        except Exception as e:
            raise ValueError(f"Invalid PDF: {e}") from e

        if page_number < 0 or page_number >= len(pdf.pages):
            pdf.close()
            raise ValueError(f"Page {page_number} out of range")

        content = self._extract_page(pdf.pages[page_number], page_number)
        pdf.close()
        return content

    def _extract_page(self, page, page_idx: int) -> PageContent:
        # Column-aware extraction. Detects multi-column layouts via the
        # x-distribution of word midpoints, then extracts each column
        # bbox independently so right-column text doesn't glue onto
        # left-column lines. Falls back transparently to plain
        # ``layout=True`` extraction on single-column pages.
        text = _extract_text_with_columns(page)

        tables = []
        raw_tables = page.extract_tables() or []

        for raw_table in raw_tables:
            if not raw_table or len(raw_table) < 2:
                continue

            # Sanitize each cell as we read it so headers/rows stored
            # on the Table object are themselves clean — downstream
            # consumers (KG ingest, table comparators) read the
            # ``rows``/``headers`` lists directly, not just the
            # rendered markdown.
            headers = [_sanitize_cell(str(cell or "")) for cell in raw_table[0]]
            rows = [
                [_sanitize_cell(str(cell or "")) for cell in row]
                for row in raw_table[1:]
            ]

            # Drop pdfplumber's bordered-region false positives —
            # see ``_is_degenerate_table`` for rationale.
            if _is_degenerate_table(headers, rows):
                continue

            markdown = self._table_to_markdown(headers, rows)
            tables.append(RawTable(
                markdown=markdown, headers=headers, rows=rows,
                confidence=_table_confidence(headers, rows),
            ))

        confidence = 0.8 if len(text) > 50 else 0.3

        return PageContent(
            page_number=page_idx, text=text, tables=tables, figures=[], confidence=confidence,
        )

    def _table_to_markdown(self, headers: list[str], rows: list[list[str]]) -> str:
        if not headers:
            return ""
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        data_lines = []
        for row in rows:
            padded = row + [""] * (len(headers) - len(row))
            data_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
        return "\n".join([header_line, sep_line] + data_lines)
