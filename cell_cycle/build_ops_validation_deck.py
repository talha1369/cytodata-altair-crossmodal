#!/usr/bin/env python
"""'OPS validation experiments' slides for the 2026 SBI2/CytoData hackathon wind-down deck, built on the deck's own
template (layout TITLE_1: green bars, swoosh; titles Roboto 28 pt in the theme's dark grey; body Arial). The template's
own slides are removed afterwards (scripts/clean.py), so the file holds only these slides, ready for File > Import slides.
usage: python build_ops_validation_deck.py TEMPLATE.pptx FIGS_DIR LB_GUIDES_DIR ALL_MEASURED_CSV LOUISE_DENSITY_PNG OUT.pptx"""
import sys
import numpy as np, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

TPL, FIGS, LBG, ALLM, LDEN, OUT = sys.argv[1:7]
GREY, MUTED, BLUE, GREEN, ORANGE = RGBColor(0x59, 0x59, 0x59), RGBColor(0x8C, 0x8C, 0x8C), RGBColor(0x3D, 0x85, 0xC6), RGBColor(0x6A, 0xA8, 0x4F), RGBColor(0xE6, 0x91, 0x38)
prs = Presentation(TPL); LAYOUT = [l for l in prs.slide_layouts if l.name == "TITLE_1"][0]
n_template = len(prs.slides)
T = pd.read_csv(ALLM).set_index("gene"); RAND = T[T.set == "calibration"]
r2n, r4n = RAND.measured_d_pct_2N.median(), RAND.measured_d_pct_4N.median()
G = pd.read_csv(f"{LBG}/gene_level_all_cells.csv").set_index("gene")
H = pd.read_csv(f"{LBG}/guide_heterogeneity_two_nulls.csv").set_index("gene")
CAL = pd.read_csv(f"{LBG}/permutation_test_on_ntc_guide_sets.csv").iloc[:, 0].values
L = pd.read_csv(f"{LBG}/louise_guides_vs_measured_dna.csv")


def text(slide, x, y, w, h, paras, size=14, color=GREY, bold_first=False, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = anchor; tf.margin_left = tf.margin_right = Inches(.05)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(8)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run(); r.text = t; f = r.font; f.name = "Arial"; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", False)
            f.italic = o.get("italic", False); f.color.rgb = o.get("color", color)
    return tb


def new_slide(title, notes):
    s = prs.slides.add_slide(LAYOUT)
    tb = s.shapes.add_textbox(Emu(321075), Emu(157600), Inches(8.6), Inches(.75)); tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = title; r.font.name = "Roboto"; r.font.size = Pt(24); r.font.color.rgb = GREY
    s.notes_slide.notes_text_frame.text = notes
    return s


def bar_chart(slide, x, y, w, h, cats, vals, fmt, colors, title=None, maxv=None):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series("s", vals)
    ch = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.has_legend = False; ch.font.name = "Arial"; ch.font.size = Pt(12); ch.font.color.rgb = GREY
    if title:
        ch.has_title = True; ch.chart_title.text_frame.text = title
        tr = ch.chart_title.text_frame.paragraphs[0].runs[0].font; tr.size = Pt(13); tr.bold = False; tr.color.rgb = GREY; tr.name = "Arial"
    else:
        ch.has_title = False
    ca, va = ch.category_axis, ch.value_axis; ca.reverse_order = True; ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    va.visible = False; va.has_major_gridlines = False; va.minimum_scale = 0
    if maxv: va.maximum_scale = maxv
    pl = ch.plots[0]; pl.gap_width = 45; pl.has_data_labels = True; dl = pl.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(13); dl.font.color.rgb = GREY
    for pt, c in zip(pl.series[0].points, colors):
        pt.format.fill.solid(); pt.format.fill.fore_color.rgb = c
    return ch


# 1. the clock, read against measured DNA
s = new_slide("What each part of the 4i clock contains",
              "Before we used the 4i clock, we checked it against something measured directly: the DNA stain. "
              "For about 35,600 control cells we measured DNA content from DAPI in three imaging rounds and called each cell 2N, S or 4N. "
              "The plot shows the share of each DNA state along the clock. Cells with doubled DNA sit between 0.5 and 0.75, "
              "and the last quarter, 0.75 to 1, is almost entirely cells that have just divided and are back to one genome copy. "
              "The first quarter labels (early G1, late G1, S, G2/M) were therefore shifted by about a quarter turn. "
              "That explained a first-day puzzle: knockouts expected to stop cells in G1, such as CDK7, seemed to pile up in G2/M. "
              "Those were G1 cells just after division. Note that the clock itself uses DNA as an input, so this checks the labels, not the clock.")
s.shapes.add_picture(f"{FIGS}/v1_clock_contents.png", Inches(.45), Inches(1.0), width=Inches(5.9))
text(s, 6.55, 1.25, 2.5, 3.6, [[("76%", {"size": 30, "bold": True, "color": BLUE})], "of cells at 0.5 to 0.75 have doubled DNA (4N)",
                                [("0.75 to 1", {"size": 22, "bold": True, "color": GREEN})], "is cells just after division, back to 2N",
                                [("So the first labels were a quarter turn off.", {"italic": True})]], size=14)

# 2. phase contrast reads DNA state
s = new_slide("Phase contrast reads DNA state like the antibody panel",
              "Next question: can label-free phase contrast read the cycle? We trained models on control cells to predict each cell's DNA state "
              "from phase-contrast features, and scored them on image tiles they never saw. Phase contrast separates 4N from 2N cells with an AUC of 0.92; "
              "the 4i antibody panel (pRb, Rb and p21) reaches 0.89, and chance is 0.5. Gradient-boosted trees on the raw features beat a linear model on "
              "principal components. On the right: a model trained only on control cells reads knockout cells as well as control cells "
              "(CCNB1, MCM3 and CENPC, 1,353 cells with measured DNA; 0.963 against 0.968), so it does not break on perturbed cells. "
              "These transfer numbers come from a lighter laptop run (10,000 control cells, 1,740 features), so they sit a little below the full model.")
bar_chart(s, .45, 1.05, 4.7, 2.6, ["phase contrast", "4i antibodies (pRb, Rb, p21)", "random"], [.92, .89, .50], "0.00", [BLUE, GREEN, RGBColor(0xBF, 0xBF, 0xBF)],
          "AUC, 4N vs 2N cells, held-out tiles", 1.05)
bar_chart(s, 5.35, 1.05, 3.3, 2.6, ["control cells", "knockout cells"], [.968, .963], "0.000", [BLUE, BLUE], "trained on controls, tested on", 1.1)
text(s, .5, 3.85, 7.4, 1.3, ["AUC: the chance that a random 4N cell scores higher than a random 2N cell (0.5 is chance, 1 is perfect).",
                              [("Any brightfield or phase-contrast screen can get a cell-cycle readout without spending a fluorescence channel.", {"bold": True})]], size=13)

# 3. knockout shifts confirmed by measured DNA
both, hgbo, ridgeo, neither = [(int(T[T.called_by == k].confirmed.sum()), int((T.called_by == k).sum())) for k in ["both", "HistGB only", "ridge only", "neither"]]
s = new_slide("Predicted shifts hold up against measured DNA",
              "We then predicted the cycle distribution of all 1,000 knockouts from phase contrast, flagged those that differ from controls, "
              f"and measured DNA directly in {len(T)} knockouts: every flagged one, a random set, and the team's list of expected arrests. "
              f"Left: predicted against measured change in the share of 4N cells. They agree with a rank correlation of 0.80. "
              "Replication genes such as MTBP, MCM7 and POLA2 gain 4N cells; ribosome and translation genes such as EIF6 lose them, as expected for cells that stall before copying DNA. "
              f"Right: how often a flagged knockout is confirmed by measured DNA at 5% false discovery rate. Flagged by both models: {both[0]} of {both[1]}. "
              f"Only by gradient boosting: {hgbo[0]} of {hgbo[1]}, weaker but well above the {neither[0]} of {neither[1]} knockouts flagged by neither. "
              f"Only by the linear model: {ridgeo[0]} of {ridgeo[1]}. The extra measurements for 111 knockouts were made this week on a laptop "
              "from the public OPS0077 images, with the same rules as the original measurement.")
s.shapes.add_picture(f"{FIGS}/v3_predicted_vs_measured.png", Inches(.45), Inches(.95), height=Inches(4.2))
bar_chart(s, 5.5, 1.1, 3.6, 2.6, [f"both models ({both[0]}/{both[1]})", f"gradient boosting only ({hgbo[0]}/{hgbo[1]})", f"linear model only ({ridgeo[0]}/{ridgeo[1]})", f"neither, baseline ({neither[0]}/{neither[1]})"],
          [100 * a / b for a, b in [both, hgbo, ridgeo, neither]], '0"%"', [BLUE, BLUE, RGBColor(0xBF, 0xBF, 0xBF), RGBColor(0xBF, 0xBF, 0xBF)],
          "confirmed by measured DNA", 75)
text(s, 5.55, 3.75, 2.3, 1.0, ["FDR 5% against control groups of the same size."], size=11, color=MUTED)

# 4. the right reference: random knockouts
s = new_slide("Judge expected arrests against random knockouts",
              "The team listed 60 genes expected to stop cells in G1, S or G2/M. Here each dot is one knockout with measured DNA, as change against non-cutting control guides. "
              f"Grey on the left: 30 knockouts picked at random. They shift too: median +{r2n:.1f} points of 2N cells and {r4n:.1f} points of 4N cells. "
              "Cutting DNA alone seems to hold some cells back; A549 cells have working p53, and p21 RNA rises in 636 of 1,000 knockouts in the CROP-seq data. "
              "So the dashed line, not zero, is the fair reference. Against it, the S-phase list stands out (about 8 points more 4N cells than random knockouts). "
              "The G1 list gains 2N cells exactly as much as random knockouts, and the G2/M list does not gain 4N cells at all. "
              "Literature expectations do not always hold in one cell line and about 89 hours of Cas9, and a direct readout is needed to tell.")
s.shapes.add_picture(f"{FIGS}/v4_arrest_vs_random.png", Inches(.45), Inches(.95), width=Inches(8.0))
text(s, .5, 4.3, 7.0, .8, [[("Dashed line: median of 30 random knockouts. S-phase list passes; G1 and G2/M lists look like random knockouts.", {"bold": True})]], size=13)

# 5. Brian's panels against measured DNA (all cells of each gene)
rows = [("AURKB", "cytokinesis failure: G2", "G2/M +7.1 (Z 3.2)"), ("POLA1", "fork stalling: S or G2/M", "G2/M +10.4 (Z 5.4)"),
        ("POLA2", "fork stalling: S or G2/M", "G2/M +14.7 (Z 6.0)"), ("CDK7", "G1 restriction point: G1", "early G1 +8.5 (Z 2.9)"),
        ("BUB1B", "checkpoint loss: G1 (slippage)", "early G1 +6.3 (Z 2.1)"), ("CCNB1", "G2 block, back to G1", "no clear change")]
s = new_slide("Brian's updated panels against measured DNA",
              "Brian redrew his knockout panels on the corrected clock. For each gene we measured DNA in every cell of that knockout, not a sample, "
              "and compared it with non-cutting controls and with random knockouts. AURKB, POLA1 and POLA2 gain 4N cells, as their panels show, "
              "well above random knockouts. CDK7 gains 2N cells, again above random knockouts, so the G1 restriction-point expectation holds for it. "
              "BUB1B gains early-G1 cells on the clock, but its DNA does not move beyond what cutting alone does; slippage out of mitosis would leave G1 cells with doubled DNA, "
              "so a shorter mitosis is the more likely reading. On Brian's literature note for POLA1 and POLA2 (S-phase build-up, survivors in G2/M): "
              "measured DNA shows mostly 4N cells and little extra S (+1 and +3 points), so in this window the survivors dominate. "
              "CCNB1 shows no clear shift on the clock, but against random knockouts it does gain 4N cells, and two of its guides gain 11 to 14 points (next slide). "
              "There is no RNA sign of CCNB2 compensation in CROP-seq (rank 387 of 997 knockouts), though compensation could still act on protein.")
tbl = s.shapes.add_table(len(rows) + 1, 5, Inches(.45), Inches(.95), Inches(8.0), Inches(3.0)).table
hdr = ["knockout", "Brian's expectation", "clock shift (panel)", "measured DNA vs controls", "vs random knockouts"]
for j, wdt in enumerate([0.95, 2.15, 1.75, 1.75, 1.4]): tbl.columns[j].width = Inches(wdt)
for i, row in enumerate([hdr] + [list(r) + [None, None] for r in rows]):
    if i > 0:
        g = row[0]; d = G.loc[g] if g in G.index else T.loc[g]
        d2, d4 = (d.d_pct_2N, d.d_pct_4N) if g in G.index else (d.measured_d_pct_2N, d.measured_d_pct_4N)
        n = int(d.n) if g in G.index else int(d.n_measured)
        key = "2N" if g in ("CDK7", "BUB1B") else "4N"; v = d2 if key == "2N" else d4; ref = r2n if key == "2N" else r4n
        row[3] = f"{key} {v:+.1f} (n={n})"; row[4] = f"{key} {v - ref:+.1f}"
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = str(val); c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xF2, 0xF2, 0xF2) if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
        c.margin_left = c.margin_right = Inches(.06); c.margin_top = c.margin_bottom = Inches(.03)
        for p in c.text_frame.paragraphs:
            for r in p.runs: r.font.name = "Arial"; r.font.size = Pt(12); r.font.bold = i == 0; r.font.color.rgb = GREY
text(s, .5, 4.1, 7.2, 1.0, ["Clock shifts read from Brian's panels (against matched control groups). DNA: change in % of cells, all cells of each knockout; "
                             f"random knockouts: median of 30 ({r2n:+.1f} 2N, {r4n:+.1f} 4N)."], size=12, color=MUTED)

# 6. guides
cal = 100 * np.mean(CAL < .05)
B = pd.read_csv(f"{LBG}/per_guide_all_cells.csv").set_index("short")
c1, c2, c3 = B.loc["ATATTTGC"], B.loc["GTCAGACC"], B.loc["TGAGGCCA"]
s = new_slide("Measured DNA reproduces Louise's CCNB1 guide split",
              "Louise found that the four guides of CCNB1, MCM3 and CENPC behave differently on the clock (her plot, left). "
              "We measured DNA in every cell of these knockouts. Her per-guide clock shifts agree with measured DNA across her 12 guides "
              f"(G1 against 2N, rank correlation 0.9; G2/M against 4N, 0.7; her score uses DNA, so this is consistency rather than independent proof). "
              f"For CCNB1, the two guides that add G2/M cells on her clock also gain 4N cells in DNA: ATATTTGC {c1.d_pct_4N:+.0f} points and GTCAGACC {c2.d_pct_4N:+.0f}, "
              f"while TGAGGCCA gains 2N cells instead ({c3.d_pct_2N:+.0f}). So CCNB1 likely does have the classic G2/M phenotype, diluted in the gene average by guides that work less well. "
              f"We also checked the guide test. The usual test shuffles cells among a gene's guides; applied to sets of four non-targeting control guides it calls a difference in {cal:.0f}% of sets instead of 5%, "
              "because control guides differ from each other too, for example when cells grow in patches. Against control guides of the same size, the guides clearly differ only in AURKB and POLA2, "
              "two genes with a real phenotype that some guides show strongly and others not at all, as uneven knockout efficiency would. "
              "Take-home: where an expected arrest is weak, look at the guides one by one, and compare them with control guides, not with shuffled cells.")
s.shapes.add_picture(LDEN, Inches(.4), Inches(.95), width=Inches(5.6))
text(s, .4, 3.4, 5.6, .35, ["Louise: cells of each guide along the clock (grey: controls). Top: stain score; bottom: phase contrast."], size=10, color=MUTED)
text(s, .4, 3.8, 5.8, 1.4, [f"CCNB1 guides, measured change in % 4N cells: ATATTTGC {c1.d_pct_4N:+.0f}, GTCAGACC {c2.d_pct_4N:+.0f}, CATCAGAG {B.loc['CATCAGAG'].d_pct_4N:+.0f}, TGAGGCCA {c3.d_pct_4N:+.0f}.",
                            [(f"The usual shuffled-cell test calls {cal:.0f}% of control-guide sets different (5% expected); compare guides with control guides instead.", {"bold": True})]], size=12)
hrows = [["gene", "p, shuffled cells", "p, vs control guides"]] + [[g, f"{r.p_cell_permutation:.3f}", f"{r.p_vs_ntc_guides:.3f}"]
                                                                 for g, r in H.loc[[g for g in ["CCNB1", "MCM3", "CENPC", "AURKB", "POLA1", "POLA2", "BUB1B", "CDK7"] if g in H.index]].iterrows()]
tb = s.shapes.add_table(len(hrows), 3, Inches(6.25), Inches(.95), Inches(2.9), Inches(.25 * len(hrows))).table
for j, wdt in enumerate([.8, 1.05, 1.05]): tb.columns[j].width = Inches(wdt)
for i, row in enumerate(hrows):
    tb.rows[i].height = Inches(.25 if i else .42)
    for j, val in enumerate(row):
        c = tb.cell(i, j); c.text = str(val); c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xF2, 0xF2, 0xF2) if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
        c.margin_left = c.margin_right = Inches(.04); c.margin_top = c.margin_bottom = Inches(.01)
        for p in c.text_frame.paragraphs:
            for r in p.runs: r.font.name = "Arial"; r.font.size = Pt(10); r.font.bold = i == 0; r.font.color.rgb = GREY
text(s, 6.25, 3.35, 2.9, .5, ["Do a gene's 4 guides differ in measured DNA? (all cells)"], size=10, color=MUTED)

# keep only the new slides (template slides are dropped; scripts/clean.py removes their parts afterwards)
ids = prs.slides._sldIdLst
for sid in list(ids)[:n_template]:
    prs.part.drop_rel(sid.rId); ids.remove(sid)
prs.save(OUT); print("wrote", OUT, len(prs.slides), "slides")
