"""Tests for figure caption extraction and matching."""

from pdfvault.document import Figure
from pdfvault.enhancers.captions import (
    extract_figure_captions,
    extract_panel_references,
    insert_caption_text_blocks,
    match_captions_to_figures,
    remove_caption_text_blocks,
    sync_caption_alt_text,
)


def test_extract_nature_style_captions():
    text = "Fig. 1 | Mapping the spatial landscape of the intestine reveals regional and shared expression."
    captions = extract_figure_captions(text)
    assert len(captions) >= 1
    assert captions[0]["fig_num"] == 1
    assert "Mapping" in captions[0]["caption"]


def test_extract_standard_captions():
    text = "Figure 2. Results of the analysis showing significant differences."
    captions = extract_figure_captions(text)
    assert len(captions) >= 1
    assert captions[0]["fig_num"] == 2


def test_extract_colon_separator():
    text = "Fig. 5: Heatmap of gene expression across cell types."
    captions = extract_figure_captions(text)
    assert len(captions) >= 1
    assert captions[0]["fig_num"] == 5
    assert "Heatmap" in captions[0]["caption"]


def test_extract_extended_data():
    text = "Extended Data Fig. 3 | Additional validation experiments."
    captions = extract_figure_captions(text)
    assert len(captions) >= 1
    assert captions[0]["is_extended"] is True
    assert captions[0]["fig_num"] == 3


def test_extract_multiple_captions():
    text = (
        "Fig. 1 | First figure caption here.\n"
        "Some text in between.\n"
        "Fig. 2 | Second figure caption here."
    )
    captions = extract_figure_captions(text)
    assert len(captions) == 2
    assert captions[0]["fig_num"] == 1
    assert captions[1]["fig_num"] == 2


def test_extract_panel_references():
    text = "As shown in Fig. 3a, the results were consistent with Fig. 4c,d."
    refs = extract_panel_references(text)
    assert len(refs) >= 2
    assert refs[0]["fig_num"] == 3
    assert "a" in refs[0]["panels"]
    assert refs[1]["fig_num"] == 4


def test_extract_extended_data_panel_references():
    text = "The validation is shown in Extended Data Fig. 2a-c."
    refs = extract_panel_references(text)
    assert len(refs) == 1
    assert refs[0]["fig_num"] == 2
    assert refs[0]["is_extended"] is True
    assert set(refs[0]["panels"]) == {"a", "b", "c"}


def test_panel_range():
    text = "Fig. 2a\u2013c shows the progression."
    refs = extract_panel_references(text)
    assert len(refs) >= 1
    assert set(refs[0]["panels"]) == {"a", "b", "c"}


def test_panel_range_hyphen():
    text = "Fig. 7a-d displays the results."
    refs = extract_panel_references(text)
    assert len(refs) >= 1
    assert set(refs[0]["panels"]) == {"a", "b", "c", "d"}


def test_panel_context():
    text = "The data in Fig. 1a demonstrates the effect clearly."
    refs = extract_panel_references(text)
    assert len(refs) >= 1
    assert "demonstrates" in refs[0]["context"]


def test_no_captions_in_plain_text():
    text = "This is just regular text without any figure references."
    captions = extract_figure_captions(text)
    assert len(captions) == 0


def test_no_panel_refs_in_plain_text():
    text = "This is just regular text without any figure references."
    refs = extract_panel_references(text)
    assert len(refs) == 0


def test_match_captions_to_figures():
    figures = [
        Figure(id="fig1", page=0),
        Figure(id="fig2", page=1),
        Figure(id="fig3", page=2),
    ]
    captions = [
        {"fig_num": 1, "caption": "First caption.", "is_extended": False},
        {"fig_num": 2, "caption": "Second caption.", "is_extended": False},
    ]
    result = match_captions_to_figures(figures, captions)
    assert result[0].caption == "First caption."
    assert result[1].caption == "Second caption."
    assert result[2].caption is None  # no caption for fig3


def test_match_uses_page_order_and_replaces_next_page_placeholders():
    figures = [
        Figure(id="fig1", page=1, image_base64="small"),
        Figure(id="fig2", page=2, image_base64="small"),
        Figure(id="fig3", page=10, image_base64="large" * 100),
        Figure(id="fig4", page=11, image_base64="large" * 100),
    ]
    captions = [
        {"fig_num": 1, "caption": "Main one.", "is_extended": False},
        {"fig_num": 2, "caption": "Main two.", "is_extended": False},
        {"fig_num": 1, "caption": "See next page for caption.", "is_extended": True},
        {"fig_num": 1, "caption": "Extended one.", "is_extended": True},
        {"fig_num": 2, "caption": "See next page for caption.", "is_extended": True},
        {"fig_num": 2, "caption": "Extended two.", "is_extended": True},
    ]

    result = match_captions_to_figures(figures, captions)

    assert [f.caption for f in result] == [
        "Main one.",
        "Main two.",
        "Extended one.",
        "Extended two.",
    ]


def test_sync_caption_alt_text_includes_figure_labels():
    figures = [
        Figure(id="fig1", page=1),
        Figure(id="fig2", page=10),
    ]
    captions = [
        {"fig_num": 1, "caption": "Main caption.", "is_extended": False},
        {"fig_num": 1, "caption": "Extended caption.", "is_extended": True},
    ]
    markdown = "![Figure 1](fig1)\n\n![Figure 2](fig2)"

    result = sync_caption_alt_text(markdown, figures, captions)

    assert "![Fig. 1 | Main caption.](fig1)" in result
    assert "![Extended Data Fig. 1 | Extended caption.](fig2)" in result


def test_sync_caption_alt_text_handles_backslash_letter_in_caption():
    """A caption containing ``\\B``, ``\\d``, etc. used to crash with
    ``re.PatternError: bad escape \\B`` because the alt text was passed
    as the ``repl`` argument of ``re.sub`` and Python's regex engine
    interprets backslash-letter sequences as group references in repl
    strings. See PR A — captions regex crash."""
    figures = [Figure(id="fig1", page=1)]
    # Caption has a literal backslash-B. Real PDFs hit this with math
    # captions like ``\\Beta`` or LaTeX residue.
    captions = [
        {"fig_num": 1, "caption": r"Math caption with \B in it.", "is_extended": False},
    ]
    markdown = "![Figure 1](fig1)"

    # Must not raise.
    result = sync_caption_alt_text(markdown, figures, captions)

    assert r"\B" in result


def test_remove_caption_text_blocks_uses_extracted_line_range():
    markdown = (
        "Body before\n"
        "Fig. 1 | Full caption title.\n"
        "a, Panel text.\n"
        "\n"
        "Body after"
    )
    captions = extract_figure_captions(markdown)

    result = remove_caption_text_blocks(markdown, captions)

    assert "Fig. 1 |" not in result
    assert "a, Panel text." not in result
    assert "Body before" in result
    assert "Body after" in result


def test_insert_caption_text_blocks_uses_full_caption_after_marker():
    figures = [Figure(id="fig1", page=1)]
    captions = [
        {
            "fig_num": 1,
            "caption": (
                "Main title. a, Panel text. Processed using ImageJ and "
                "representative of n = 5 biological replicates. Scale bar, 1,000 um."
            ),
            "is_extended": False,
        },
    ]
    markdown = "![Figure 1](fig1)"

    result = insert_caption_text_blocks(markdown, figures, captions)

    assert "![Figure 1](fig1)" in result
    assert "Fig. 1 | Main title. a, Panel text." in result
    assert "Processed using ImageJ" in result
    assert "Scale bar, 1,000 um." in result


def test_insert_caption_text_blocks_moves_sentence_interruptions():
    figures = [Figure(id="fig1", page=1)]
    captions = [{"fig_num": 1, "caption": "Main title. a, Panel text.", "is_extended": False}]
    markdown = (
        "To understand transcriptional\n"
        "![Figure 1](fig1)\n"
        "signatures, we mapped TFs.\n"
        "\n"
        "Next paragraph."
    )

    result = insert_caption_text_blocks(markdown, figures, captions)

    assert "transcriptional\nsignatures" in result
    assert "signatures, we mapped TFs.\n\n![Figure 1](fig1)" in result


def test_insert_caption_text_blocks_moves_adjacent_sentence_interruptions():
    figures = [Figure(id="fig3", page=3), Figure(id="fig4", page=4)]
    captions = [
        {"fig_num": 3, "caption": "Third title. a, Panel text.", "is_extended": False},
        {"fig_num": 4, "caption": "Fourth title. a, Panel text.", "is_extended": False},
    ]
    markdown = (
        "We identified\n"
        "![Figure 3](fig3)\n"
        "![Figure 4](fig4)\n"
        "subsets of each lineage.\n\n"
        "Next paragraph."
    )

    result = insert_caption_text_blocks(markdown, figures, captions)

    assert "We identified\nsubsets of each lineage." in result
    assert "subsets of each lineage.\n\n![Figure 3](fig3)" in result
    assert result.index("![Figure 3](fig3)") < result.index("![Figure 4](fig4)")
    assert "Fig. 3 | Third title." in result
    assert "Fig. 4 | Fourth title." in result


def test_insert_caption_text_blocks_rechecks_cascaded_interruptions():
    figures = [Figure(id="fig3", page=3), Figure(id="fig4", page=4)]
    captions = [
        {"fig_num": 3, "caption": "Third title. a, Panel text.", "is_extended": False},
        {"fig_num": 4, "caption": "Fourth title. a, Panel text.", "is_extended": False},
    ]
    markdown = (
        "distribution of cell types and identified\n"
        "![Figure 3](fig3)\n"
        "differential enrichment across this axis.\n"
        "We identified\n"
        "![Figure 4](fig4)\n"
        "subsets of each lineage.\n\n"
        "Next paragraph."
    )

    result = insert_caption_text_blocks(markdown, figures, captions)

    assert "identified\ndifferential enrichment across this axis." in result
    assert "We identified\nsubsets of each lineage." in result
    assert "differential enrichment across this axis.\n\n![Figure 3](fig3)" in result
    assert "subsets of each lineage.\n\n![Figure 4](fig4)" in result
    assert result.index("![Figure 3](fig3)") < result.index("![Figure 4](fig4)")


def test_match_skips_already_captioned():
    figures = [
        Figure(id="fig1", caption="Existing caption", page=0),
        Figure(id="fig2", page=1),
    ]
    captions = [
        {"fig_num": 1, "caption": "New caption.", "is_extended": False},
    ]
    result = match_captions_to_figures(figures, captions)
    # fig1 already had a caption, so the new one goes to fig2
    assert result[0].caption == "Existing caption"
    assert result[1].caption == "New caption."


def test_match_empty_figures():
    result = match_captions_to_figures([], [{"fig_num": 1, "caption": "Test.", "is_extended": False}])
    assert result == []


def test_match_empty_captions():
    figures = [Figure(id="fig1", page=0)]
    result = match_captions_to_figures(figures, [])
    assert result[0].caption is None


def test_caption_stops_at_cross_reference_to_another_figure():
    """Column-cropped text has no blank line between a legend and the body
    prose below it. Body prose cites *other* figures in parentheses; a legend
    for Fig. 2 does not, so that is where the legend ends."""
    md = "\n".join([
        "Fig. 2 | Blocking BMP signaling. a Schematic of the model. b Organoid treatment schematic.",
        "This effect was completely inhibited by BMP2 (Fig. 3c and Supplementary Fig. 5e).",
        "Treating organoids with LPS showed similar results.",
    ])
    caps = extract_figure_captions(md)
    assert len(caps) == 1
    assert caps[0]["caption"] == "Blocking BMP signaling. a Schematic of the model. b Organoid treatment schematic."
    assert caps[0]["end_line"] == 1


def test_caption_stops_at_running_footer():
    md = "\n".join([
        "Fig. 4 | Stromal cells promote epithelial YAP activation.",
        "Nature Communications | (2026) 17:9510 https://doi.org/10.1038/s41467-026-77520-1",
        "Screening our in vivo transcriptome data for differentially expressed genes",
    ])
    caps = extract_figure_captions(md)
    assert caps[0]["caption"] == "Stromal cells promote epithelial YAP activation."


def test_caption_is_capped_at_max_length_on_a_sentence_boundary():
    from pdfvault.enhancers.captions import MAX_CAPTION_CHARS

    body = ["Fig. 1 | Downregulation of BMP signaling induces a regenerative state."]
    body += [f"Body sentence number {i} continues the results without any blank line." for i in range(200)]
    caps = extract_figure_captions("\n".join(body))
    caption = caps[0]["caption"]
    assert len(caption) <= MAX_CAPTION_CHARS
    assert caption.endswith(".")
    assert caption.startswith("Downregulation of BMP signaling")


def test_caption_cut_before_other_figure_citation_within_one_line():
    """Paragraph reflow can merge legend and body into one physical line;
    the boundary must still be found inside the line."""
    md = ("Fig. 2 | Blocking BMP signaling. a Schematic of the model. b Organoid treatment schematic. "
          "This effect was completely inhibited by BMP2 (Fig. 3c and Supplementary Fig. 5e). Treating organoids showed similar results.")
    caps = extract_figure_captions(md)
    assert caps[0]["caption"] == "Blocking BMP signaling. a Schematic of the model. b Organoid treatment schematic."


def test_caption_cut_before_running_footer_within_one_line():
    md = ("Fig. 4 | Stromal cells promote epithelial YAP activation. Scale bars: 100 µm. "
          "Nature Communications | (2026) 17:9510 https://doi.org/10.1038/s41467-026-77520-1 Screening our data")
    caps = extract_figure_captions(md)
    assert caps[0]["caption"] == "Stromal cells promote epithelial YAP activation. Scale bars: 100 µm."


def test_caption_keeps_references_to_its_own_figure_and_supplementary_figures():
    md = "Fig. 2 | Quantification of panel (Fig. 2a) as in (Supplementary Fig. 5d). n = 3 mice per group."
    caps = extract_figure_captions(md)
    assert caps[0]["caption"] == "Quantification of panel (Fig. 2a) as in (Supplementary Fig. 5d). n = 3 mice per group."


def test_caption_continues_across_blank_line_when_mid_sentence():
    """A two-column legend arrives as left column, blank line, right column.
    A legend never ends mid-sentence, so an unterminated caption keeps
    going past the blank line; a terminated one still stops there."""
    md = "\n".join([
        "Fig. 1 | Downregulation of BMP signaling. a, b Confocal images of KI67 (a) and",
        "",
        "GSII (b) in antral tissue (n = 3 mice per group). Scale bars: 100 µm.",
        "",
        "We also analyzed the Lgr5+ stem cell signature and found that it was unchanged.",
    ])
    caps = extract_figure_captions(md)
    assert caps[0]["caption"] == (
        "Downregulation of BMP signaling. a, b Confocal images of KI67 (a) and "
        "GSII (b) in antral tissue (n = 3 mice per group). Scale bars: 100 µm."
    )
    assert caps[0]["end_line"] == 3


def test_caption_skips_running_footer_and_continues_when_mid_sentence():
    md = "\n".join([
        "Fig. 1 | Downregulation of BMP signaling. a GSEA of uninfected Bmpr1a KO vs WT mice, for",
        "https://doi.org/10.1038/s41467-026-77520-1",
        "",
        "YAP target gene signature (h) (n = 2 mice per group). Scale bars: 100 µm.",
        "",
        "We also analyzed the Lgr5+ stem cell signature.",
    ])
    caps = extract_figure_captions(md)
    assert caps[0]["caption"] == (
        "Downregulation of BMP signaling. a GSEA of uninfected Bmpr1a KO vs WT mice, for "
        "YAP target gene signature (h) (n = 2 mice per group). Scale bars: 100 µm."
    )
    assert "doi.org" not in caps[0]["caption"]


def test_caption_backs_up_to_sentence_boundary_when_body_line_stops_it():
    md = "\n".join([
        "Fig. 2 | Blocking BMP signaling. a Schematic of the model. Organoids grown in standard medium",
        "strongly upregulated cytokines upon treatment (Fig. 3c and Supplementary Fig. 5e).",
        "Treating organoids with LPS showed similar results.",
    ])
    caps = extract_figure_captions(md)
    assert caps[0]["caption"] == "Blocking BMP signaling. a Schematic of the model."
