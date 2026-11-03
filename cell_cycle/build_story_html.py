#!/usr/bin/env python
"""Build the team results page (one self-contained HTML, figures embedded) that tells the cell-cycle story in order.
usage: python build_story_html.py REPORTS_DIR EXTRA_DIR OUT.html"""
import sys, base64, os, html

REP, EXTRA, OUT = sys.argv[1:4]
GH = "https://github.com/cytodata/2026_sbi2_hackathon_resources/blob/talha-cycle-modalities"


def img(path, alt):
    p = path if os.path.exists(path) else os.path.join(EXTRA, os.path.basename(path))
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(alt)}" loading="lazy"><figcaption>{alt}</figcaption></figure>'


def nb(name, label): return f'<a class="nb" href="{GH}/{name}" target="_blank" rel="noopener">{label}</a>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head); tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def chip(kind, text): return f'<span class="chip {kind}">{text}</span>'


R = lambda *p: os.path.join(REP, *p)
page = f"""<title>Cell Cycle Heterogeneity Findings</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* layout: one reading column, figures at full column width, numbered sections because the story is a sequence */
:root {{
  --bg: #f5f7f8; --panel: #ffffff; --fg: #15232c; --muted: #56656f; --line: #d5dde2;
  --accent: #0d6a86; --accent-soft: #e2f0f4;
  --ok: #2f7d4f; --ok-soft: #e3f2e8; --warn: #9a6400; --warn-soft: #fbf0d9; --open: #4f5d75; --open-soft: #e7eaf0;
  --display: "Source Serif 4", Georgia, "Times New Roman", serif;
  --body: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f171c; --panel: #16222a; --fg: #e3eaee; --muted: #9aabb5; --line: #2a3a44;
  --accent: #5bb8d6; --accent-soft: #15323d; --ok: #6cc391; --ok-soft: #173326; --warn: #e2b04f; --warn-soft: #382a10; --open: #a9b6cc; --open-soft: #222b38; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #0f171c; --panel: #16222a; --fg: #e3eaee; --muted: #9aabb5; --line: #2a3a44;
  --accent: #5bb8d6; --accent-soft: #15323d; --ok: #6cc391; --ok-soft: #173326; --warn: #e2b04f; --warn-soft: #382a10; --open: #a9b6cc; --open-soft: #222b38; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 17px/1.6 var(--body); }}
.wrap {{ max-width: 960px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 40px 80px; }}
header h1 {{ font: 650 clamp(30px, 4.4vw, 44px)/1.15 var(--display); margin: 0 0 12px; text-wrap: balance; }}
header .sub {{ color: var(--muted); max-width: 68ch; margin: 0; }}
.eyebrow {{ font: 600 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin-bottom: 14px; }}
.keys {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 28px 0 8px; }}
.key {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; min-width: 0; }}
.key b {{ display: block; font: 600 26px/1.1 var(--mono); color: var(--accent); font-variant-numeric: tabular-nums; }}
.key span {{ font-size: 14.5px; color: var(--muted); }}
nav.toc {{ margin: 28px 0 8px; display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 15px; }}
nav.toc a {{ color: var(--fg); text-decoration: none; border-bottom: 1px solid var(--line); }}
nav.toc a:hover, nav.toc a:focus-visible {{ border-color: var(--accent); color: var(--accent); outline: none; }}
section {{ margin-top: 56px; display: grid; gap: 14px; }}
section > h2 {{ font: 650 27px/1.25 var(--display); margin: 0; text-wrap: balance; display: flex; gap: 12px; align-items: baseline; }}
section > h2 .n {{ font: 600 14px/1 var(--mono); color: var(--accent); }}
h3 {{ font: 600 18px/1.3 var(--body); margin: 10px 0 0; }}
p, li {{ max-width: 72ch; }}
p {{ margin: 0; }}
ul {{ margin: 0; padding-left: 22px; display: grid; gap: 6px; }}
.answer {{ background: var(--accent-soft); border-radius: 8px; padding: 14px 18px; max-width: none; }}
.answer strong {{ color: var(--accent); }}
figure {{ margin: 6px 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
figure img {{ display: block; width: 100%; height: auto; border-radius: 4px; background: #fff; }}
figcaption {{ font-size: 14px; color: var(--muted); padding: 8px 4px 2px; }}
.tw {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }}
table {{ border-collapse: collapse; width: 100%; font-size: 15px; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font: 600 12.5px/1.3 var(--mono); text-transform: uppercase; letter-spacing: .04em; color: var(--muted); background: var(--bg); }}
tr:last-child td {{ border-bottom: 0; }}
td.num, .num {{ font-family: var(--mono); }}
.chip {{ display: inline-block; font: 600 11.5px/1 var(--mono); letter-spacing: .05em; text-transform: uppercase; padding: 5px 8px; border-radius: 99px; vertical-align: 2px; margin-right: 6px; }}
.chip.ok {{ color: var(--ok); background: var(--ok-soft); }} .chip.warn {{ color: var(--warn); background: var(--warn-soft); }} .chip.open {{ color: var(--open); background: var(--open-soft); }}
a {{ color: var(--accent); }}
a.nb {{ font: 600 13.5px/1 var(--mono); text-decoration: none; border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px; display: inline-block; background: var(--panel); }}
a.nb:hover, a.nb:focus-visible {{ border-color: var(--accent); outline: none; }}
.nbs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.note {{ font-size: 15px; color: var(--muted); }}
footer {{ margin-top: 64px; font-size: 14px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 16px; }}
@media (prefers-reduced-motion: no-preference) {{ html {{ scroll-behavior: smooth; }} }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">SBI2 / CytoData hackathon · Altair OPS0077 and Element A549</div>
  <h1>Cell Cycle Heterogeneity Findings</h1>
  <p class="sub">Can we read the cell cycle from morphology, and does removing it make perturbed cell populations less heterogeneous?
  This page collects the results in the order of the argument. Each section gives the answer first, then the evidence, then the notebook to open.</p>
  <div class="keys">
    <div class="key"><b>AUC 0.92</b><span>label-free phase contrast separates 4N from 2N cells (held-out), as well as the 4i antibody panel (0.89)</span></div>
    <div class="key"><b>2 to 9%</b><span>of single-cell variance follows the cell cycle, in every modality we measured</span></div>
    <div class="key"><b>1 to 5%</b><span>smaller spread of perturbed populations after removing the cycle; separation from controls unchanged</span></div>
    <div class="key"><b>81 of 1,000</b><span>knockouts change the cell-cycle distribution; only 1 of them changes nothing else</span></div>
    <div class="key"><b>rho 0.78</b><span>predicted vs measured (DAPI) change in 4N cells across 158 knockouts, slope 1.02</span></div>
  </div>
  <nav class="toc" aria-label="Sections">
    <a href="#clock">1 · The clock</a><a href="#readout">2 · Reading it</a><a href="#align">3 · Aligning scores</a><a href="#knockouts">4 · Knockouts</a>
    <a href="#regress">5 · Regressing it out</a><a href="#element">6 · Element</a><a href="#cropseq">7 · CROP-seq</a><a href="#status">8 · Status</a>
  </nav>
</header>

<section id="clock">
  <h2><span class="n">1</span>What each part of the 4i clock contains</h2>
  <p class="answer"><strong>Answer.</strong> The clock (<code>cc_phase</code>, 0 = start of G1) runs G1, then S, then G2/M, and wraps back to G1. Measured DNA puts the 4N (G2/M) cells at 0.5 to 0.75, and the last quarter (0.75 to 1) is almost entirely 2N cells after division. Labelling the quarters "Early G1 / Late G1 / S / G2/M" therefore shifts every label by about one quarter.</p>
  {chip("warn", "correction")}<span class="note">This explains the G1 puzzle in item 6: a G1 accumulation adds cells to the 0.75 to 1 quarter, which the figure labels as "G2/M". CDK7 is an example: its "+10.8% G2/M" is a gain of 2N cells.</span>
  {table(["clock range", "2N", "S", "4N", "median pRb (z)", "what it is"], [
      ["0 to 0.25", "100%", "0%", "0%", "+0.6 then -0.4", "G1"],
      ["0.25 to 0.375", "90%", "10%", "0%", "-0.7", "late G1, entering S"],
      ["0.375 to 0.5", "15%", "84%", "1%", "-0.2", "S"],
      ["0.5 to 0.625", "0%", "13%", "87%", "+0.5", "G2"],
      ["0.625 to 0.75", "3%", "70%", "28%", "+0.9", "mitosis and division (intermediate DNA)"],
      ["0.75 to 1", "64 to 100%", "0 to 36%", "0%", "+0.7 then -0.6", "early G1 after division"]], "")}
  <p class="note">35,633 NTC cells with reliable DNA. 2N &lt; 0.25 &le; S &le; 0.65 &lt; 4N (log2 DNA relative to the 2N peak). The DNA-only clock (<code>cc_phase_dna</code>) has the same layout.</p>
  {img(R("phase_to_cell_cycle", "cc_clock_morphology.png"), "Left: NTC cells on the measured clock, coloured by DNA. Middle: DNA, pRb and p21 around the measured clock. Right: the same markers around the clock predicted from phase contrast (solid) and the measured clock (dashed).")}
  <div class="nbs">{nb("16_phase_to_cell_cycle.ipynb", "notebook 16")}</div>
</section>

<section id="readout">
  <h2><span class="n">2</span>How well each modality reads the cell cycle</h2>
  <p class="answer"><strong>Answer.</strong> Label-free phase contrast reads the cycle as well as the 4i antibody panel: both predict the clock to about 30 degrees, and phase contrast separates measured DNA state slightly better. In Element, imaging without the DNA stain reads the RNA clock about as well as the RNA clock itself separates DNA state.</p>
  {table(["dataset", "cell-cycle score", "balanced accuracy, 2N / S / 4N", "AUC, 4N vs 2N"], [
      ["OPS0077", "phase contrast (predicted, held out)", "0.57", "0.92"],
      ["OPS0077", "4i protein panel (predicted, held out)", "0.53", "0.89"],
      ["OPS0077", "4i markers-only clock (measured, no DNA)", "0.33", "0.65"],
      ["OPS0077", "4i clocks built with DNA (circular)", "0.87", "1.00"],
      ["Element", "RNA clock (cycleVI, measured)", "0.46", "0.74"],
      ["Element", "imaging without the DNA stain (predicted)", "0.42", "0.71"],
      ["Element", "scanpy G1/S/G2M calls", "0.33", "0.63"],
      ["both", "random angle", "0.33", "0.50"]])}
  <p class="note">The common truth is measured DNA content in both datasets. The phase-contrast rows use our ridge model on PCA-100.</p>
  <h3>A better phase-contrast model</h3>
  <p>Brian's gradient-boosted trees on the raw phase features predict the clock to <b>21.8 degrees</b> (held out by tile), against 29.3 for our ridge on PCA-100. Most of the gain needs the raw features: on PCA-100 the same trees reach only 28.3, and a single-target neural network 26.8. How missing values are handled does not matter (21.7 vs 21.8). Predicting the DNA x pRb stages directly reaches balanced accuracy 0.59 (chance 0.25), but S phase stays hard (recall 0.24).</p>
  {table(["model (35,633 NTC cells, 5 folds by tile)", "median error, DNA clock", "median error, final clock"], [
      ["gradient boosting, raw features (Brian)", "21.7 deg", "22.3 deg"],
      ["gradient boosting, raw features, native missing values", "21.8 deg", "22.2 deg"],
      ["neural network, single target, PCA-100", "26.8 deg", "27.3 deg"],
      ["gradient boosting, PCA-100", "28.3 deg", "29.1 deg"],
      ["ridge, PCA-100 (ours so far)", "29.3 deg", "29.6 deg"]])}
  {img(R("cell_cycle_by_modality", "modality_score_quality.png"), "How well each score separates measured DNA state. Grey bars: scores built from DNA (circular).")}
  {img(R("phase_to_cell_cycle", "phase_cycle_result.png"), "Phase contrast predicts DNA content (held-out R2 0.48) and separates G1 from G2/M (AUC 0.94).")}
  <div class="nbs">{nb("16_phase_to_cell_cycle.ipynb", "notebook 16")}{nb("19_cell_cycle_by_modality.ipynb", "notebook 19")}</div>
</section>

<section id="align">
  <h2><span class="n">3</span>Aligning the cell-cycle scores across regimes</h2>
  <p class="answer"><strong>Answer.</strong> Within OPS0077 the DNA-based 4i clocks are one clock. The markers-only clock runs the same way but is rotated by 142 degrees, and its stored origin is arbitrary. Across datasets only the S to G2/M arc aligns (OPS0077 to the Element RNA clock: +240 degrees, from DNA). The G1 arc cannot be aligned with the shared features we have, so we compare datasets through DNA-content classes and not through angles.</p>
  {img(R("cell_cycle_alignment", "cc_align_4_regime_agreement.png"), "Agreement between the OPS0077 scoring regimes after the best rotation.")}
  <div class="nbs">{nb("17_align_cell_cycle_regimes.ipynb", "notebook 17")}</div>
</section>

<section id="knockouts">
  <h2><span class="n">4</span>Which knockouts shift the cell cycle</h2>
  <p class="answer"><strong>Answer.</strong> 81 of the 1,000 OPS0077 knockouts change the predicted cell-cycle distribution. DNA replication and mitosis genes gain 4N cells (CDC45 +10.5 points, MTBP +14.3, KIF23 +11.8). Ribosome, translation and chaperonin genes lose them (POLR1B -8.5, RPL and RPS genes, CCT complex), as expected for an arrest before S. The predicted change in 4N cells agrees with measured pRb on the same cells (rho 0.37).</p>
  <h3>Comparing fairly with NTC</h3>
  <ul>
    <li>Knockout cells are never in training, so NTC must be scored out of fold too, and all NTC cells must be used (a third of them have no measured clock but the knockout cells were never filtered that way).</li>
    <li>The null is NTC guides pooled to the knockout's cell count. With this reference CDK7 and CCNB1 shift by about +3 to +4 points (not significant), not +7 to +11.</li>
    <li>For CCNB1 the 4N gain is real: measured DAPI gives +4.3 points.</li>
  </ul>
  <h3>Checked against measured DNA</h3>
  <p>For 111 knockouts (81 predicted shifters and 30 calibration knockouts) we measured DNA from DAPI. <b>The predicted change in 4N cells matches the measured change: Spearman 0.76, slope 1.01.</b> 42 of the 81 predicted shifts are confirmed at FDR 5%, 41 of them in the predicted direction. Most are losses of 4N cells (ribosome biogenesis, proteasome, chaperone and splicing genes such as HSPA9, IMP4, PSMD4, SRSF3); the confirmed gains are replication genes (MTBP +17.0, MCM7 +12.2, CDC45 +8.4, POLA2 +7.9 points). 7 of the 30 calibration knockouts also shift, always in the predicted direction, so the library misses some real shifts (the largest is the mitotic kinesin KIF11, +19.5 points).</p>
  <p>Adding the other 47 arrest-list genes (158 knockouts, not selected on a predicted shift) keeps the agreement (Spearman 0.64 for the 47 alone, 0.78 over all 158). Measured, the arrest list behaves as follows: the S group gains 4N cells (median +4.7 points), the G1 group gains 2N cells (+6.1, mostly not significant one by one), and <b>the G2/M group loses 4N cells (-2.5), the opposite of a G2/M arrest</b>. Only CDC45 gains. These knockouts are not hidden as binucleate cells either (10.0% vs 10.4% in NTC; only KIF23 is clearly higher, 31%). So phase contrast was right not to see a G2/M arrest for them.</p>
  {img(R("ko_cell_cycle", "ko_measured_vs_predicted_dna_all.png"), "Left: predicted vs measured change in 4N cells for 158 knockouts (large points: significant by measured DNA; blue: the 47 arrest-list genes added second). Right: measured DNA content of the largest shifters against NTC.")}
  <h3>With the better model</h3>
  <p>Rerunning the library with gradient boosting finds <b>194</b> knockouts with a changed cycle distribution (73 of the 81 ridge hits among them). Agreement with measured pRb is unchanged (rho 0.38), and so is the heterogeneity result (spread -2.75%, separation from NTC unchanged). Most hits gain cells in "G1 after division" and lose G2 cells; the largest group of new hits is ribosome, translation and splicing genes. The extra hits are a more sensitive call under the same null, and a few of them (olfactory, taste-receptor and keratin-associated genes) are probably not expressed in A549.</p>
  <p><b>Against measured DNA the gradient-boosting predictions are also better: Spearman 0.84 vs 0.78 over the 158 measured knockouts</b>, and better in the subsets not chosen from either model (arrest list 0.70 vs 0.64, calibration 0.71 vs 0.63). The 8 hits only ridge calls confirm 0 times; 3 of the 10 measured hits only gradient boosting calls confirm, and all 10 move in the predicted direction. The other 111 of its 121 extra hits are not measured yet. A 4N classifier trained on NTC reads knockout cells as well as NTC cells (CCNB1, MCM3 and CENPC, 1,353 cells: AUC 0.963 vs 0.968), so the model does not break on perturbed cells.</p>
  <h3>The arrest list against its expected arrest phase</h3>
  <p>S-phase genes behave as expected: 8 of 9 gain 4N cells (p = 3e-5 vs the rest of the library). The G1 and G2/M groups do not differ from the library. Several genes look misassigned (RRM1, POLE and RPA2 are replication genes listed under G1; CCNB1 is the mitotic cyclin listed under G1; CDC45 is listed under G2/M).</p>
  {img(R("ko_cell_cycle", "arrest_phase_test.png"), "Predicted change in 4N cells for the arrest list, by expected arrest phase, against the rest of the library.")}
  <h3>Brian's item 6: are G1-arrested cells misclassified?</h3>
  <ul>
    <li>{chip("ok", "explained")}The main reason is the quarter labels (section 1): G1 cells after division sit at 0.75 to 1.</li>
    <li>{chip("ok", "tested")}<b>The hypertrophy idea is not supported.</b> Within NTC the largest 2N cells are over-called as 4N (15.6% vs 3.4% of all 2N cells). But measured-2N knockout cells, although larger than NTC 2N cells in 148 of 158 knockouts (about 17% more area), are not called 4N more often (median excess -1.0 point; no relation with size, rho 0.09; the 26 G1-list knockouts range from -3.4 to +3.2 points). Brian's DNA floor changes no knockout by more than 0.63 points, and size corrections lower agreement with measured DNA (0.86 to 0.74 and 0.67).</li>
  </ul>
  {img(R("ko_cell_cycle", "g1_hypertrophy_check_all.png"), "Left: four phase models against measured DNA across knockouts (standard and DNA floor overlap). Middle: size of measured-2N cells vs excess false 4N calls, per knockout (red: G1 list). Right: false 4N calls among NTC 2N cells.")}
  <div class="nbs">{nb("18_ko_cell_cycle_heterogeneity.ipynb", "notebook 18")}</div>
</section>

<section id="regress">
  <h2><span class="n">5</span>Regressing the cell cycle out: does heterogeneity go down?</h2>
  <p class="answer"><strong>Answer.</strong> The cycle is a few percent of the variance in every modality. It concentrates in the top principal components, which is why those tighten by 10 to 30%, but over the whole space it is 3 to 7%. Removing it shrinks perturbed populations by about 1 to 5% and does not change how well they separate from controls.</p>
  {table(["modality", "control variance removed", "spread of perturbed populations", "separation from controls (median AUC)"], [
      ["OPS0077 phase contrast, 1,000 knockouts", "8.0%", "-3.8%", "0.572 to 0.572"],
      ["OPS0077 4i protein panel, 1,000 knockouts", "8.7%", "-5.5%", "0.585 to 0.578"],
      ["Element imaging, measured RNA clock", "2.9%", "-0.6%", "0.567 to 0.569"],
      ["Element RNA (DRVI), measured RNA clock", "6.4%", "-3.2%", "0.558 to 0.556"]])}
  {img(R("cell_cycle_by_modality", "modality_heterogeneity.png"), "Left: change in spread of each perturbed population (blue: cycle removed, grey: shuffled cycle). Right: separation from controls before and after.")}
  <h3>Diego's item 7, extended</h3>
  <p>Diego found that regressing <code>cc_phase</code> out of NTC embeddings tightens the top PCs (4i: 28 / 26 / 33%; phase contrast: 9 / 19 / 9%). We reproduce this when features are z-scored and the cycle is fitted flexibly (4i: 25 / 31%, phase contrast: 11 / 14%; total over 50 PCs 7.4% and 4.4%). Two additions:</p>
  <ul>
    <li><b>About half of the 4i effect comes from the score itself.</b> <code>cc_phase</code> is built from the 4i markers and principal components of the pRb, Rb and p21 features. Removing measured DNA content alone, which is independent of both embeddings, gives 10% on the top 4i PCs instead of 19%. For phase contrast the score is independent of the features, and the effect is real but smaller.</li>
    <li><b>Knockouts tighten like NTC on the top PCs.</b> With up to 200 cells from each of the 1,000 knockouts and Brian's predicted clock, pooled knockout cells tighten by 16 / 22 / 6% on PC1 to PC3, against 20 / 22 / 6% for NTC with the same clock. On PC5, PC7 and PC9 they tighten less.</li>
    <li><b>Removing the cycle barely moves knockouts toward NTC.</b> Their distance from NTC in PC1 to PC10 (corrected for sampling noise) shrinks by a median of about 5%, nearly the same for cycle-shifting knockouts (6.9%) and the rest (4.9%). Replication knockouts keep most of their distance (MCM7 76%, POLA2 83%, POLE 84%). <b>Knockout phenotypes in phase contrast are mostly not cell-cycle composition.</b></li>
    <li><b>Which phase features carry the cycle.</b> 168 of 1,726 features lose more than 10% of their variance; the strongest are nuclear and nucleolar phase intensity, lowest in late G1 and highest from late G2 into division. Size features are under-represented.</li>
  </ul>
  {img(R("cell_cycle_by_modality", "phase_per_pc_by_covariate.png"), "Phase contrast, z-scored features, 48 clock bins: within-field variance removed per PC by each cycle covariate, and for pooled knockout cells (purple).")}
  {img(os.path.join(EXTRA, "diego_per_pc_by_covariate.png"), "Within-field variance removed per PC (NTC), for each cell-cycle covariate. Robust scaling and two harmonics, the conservative setting.")}
  {table(["setting (NTC, within-field variance)", "4i PC1 / PC2 / PC3", "phase PC1 / PC2 / PC3", "total, 50 PCs (4i / phase)"], [
      ["Diego", "28 / 26 / 33%", "9 / 19 / 9%", "about 4% (features)"],
      ["z-score, 48 clock bins, cc_phase", "25 / 32 / 0%", "12 / 14 / 4%", "7.4% / 4.5%"],
      ["z-score, 48 bins, DNA content only", "24 / 7 / 0%", "10 / 6 / 2%", "3.4% / 2.9%"],
      ["robust + clip, 2 harmonics, cc_phase", "14 / 14 / 17%", "8 / 8 / 0%", "5.0% / 2.6%"]])}
  <h3>The three cycle-normalisation questions</h3>
  <ul>
    <li><b>Field and well heterogeneity:</b> not reduced. With a measured clock, the between-well and between-field share of control morphology does not change (Element wells 1.67% before, 1.72% after). A reduction seen with a morphology-predicted cycle is circular.</li>
    <li><b>Finding perturbations that change the cycle distribution:</b> yes (81 knockouts in OPS0077; TNFa and CTNNB1 in Element with the measured RNA clock).</li>
    <li><b>Telling them apart from perturbations that change the cells:</b> yes. Splitting each morphology shift into a cycle-mix part and a matched-cycle part shows 790 knockouts change the cells, 80 change both, 1 changes only the cycle mix (HSPA14), 129 change neither.</li>
  </ul>
  {img(R("ko_cell_cycle", "library_cycle_decomposition.png"), "1,000 knockouts: change in cycle distribution vs change of the cells at a matched cycle state.")}
  <div class="nbs">{nb("18_ko_cell_cycle_heterogeneity.ipynb", "notebook 18")}{nb("19_cell_cycle_by_modality.ipynb", "notebook 19")}</div>
</section>

<section id="element">
  <h2><span class="n">6</span>Element: the RNA clock mapped to morphology</h2>
  <p class="answer"><strong>Answer.</strong> Element imaging reads the RNA cell-cycle clock to a median of 37.5 degrees in held-out wells (95,868 control cells; a shuffled clock gives 77.9). The DNA stain carries most of it, the phospho-antibody panel adds some, and mitochondria, membrane, actin and shape do no better than the IgG negative controls. Imaging fails on one stretch, about 135 to 255 degrees, where cells have just divided: their DNA is back at 2N but their RNA still reads G2/M.</p>
  <ul>
    <li><b>Largest cycle phenotypes.</b> 319 of 990 features change by more than 0.5 SD reproducibly (none of the 194 IgG features). The biggest sit around division: phospho-p38 and phospho-ERK intensity rise (up to +1.4 SD), DNA-stain and pERK texture change, and nuclei shrink.</li>
    <li><b>Perturbations.</b> Morphology recovers the largest measured shift (CTNNB1 knockout: -3.5 and -5.1 points on the G2/M arc, predicted -2.3 and -5.0) but makes false calls (TGFBR2 knockout, TNFa). It ranks the size of shifts (rho 0.79 over 11 contrasts) but not reliably their direction.</li>
    <li><b>Correlates in OPS0077.</b> DNA-stain texture changes from 2N to S to 4N the same way in both datasets (94% same sign over 52 features). Granularity does not transfer. In OPS0077 the best phase-contrast markers of DNA state are intensity extremes of phase objects (effect size up to 1.64, not size-adjusted).</li>
  </ul>
  {img(R("element_cycle_morphology", "fig_A_error_by_arc.png"), "Element: error of the imaging-predicted clock along the RNA clock, with the scanpy phase mix and DNA content of each arc.")}
  {img(R("element_cycle_morphology", "fig_B_profiles.png"), "Element: the most reproducible features that change along the RNA clock (SD from the well median).")}
  <div class="nbs">{nb("20_element_cycle_morphology.ipynb", "notebook 20")}</div>
</section>

<section id="cropseq">
  <h2><span class="n">7</span>The CROP-seq knockouts</h2>
  <p class="answer"><strong>Answer.</strong> The public CROP-seq data are knockout-level only. Scored with cell-cycle gene programmes learned on Element single cells, only 14 of 997 knockouts move a programme strongly (SMARCD1 most), the classic arrest genes are flat, and agreement with phase contrast is weak (rho about 0.1).</p>
  {img(R("ko_cell_cycle", "crop_to_phase_overview.png"), "CROP-seq cell-cycle programmes against phase-contrast predictions, per knockout.")}
  <div class="nbs">{nb("18_ko_cell_cycle_heterogeneity.ipynb", "notebook 18")}</div>
</section>

<section id="status">
  <h2><span class="n">8</span>What is running and what is next</h2>
  <ul>
    <li>{chip("ok", "done")}Measured DNA for 111 knockouts and Brian's hypertrophy test (section 4); multimodal file for all 111 at <code>/srv/team/data/crossmodal/ops0077_ko/OPS0077_KOcyc_Multimodal_4i_Phase_DNA.h5ad</code>.</li>
    <li>{chip("ok", "done")}Brian's model vs ours on the same folds, and the library rerun with it (section 2 and 4).</li>
    <li>{chip("ok", "done")}Element: the RNA clock mapped to morphology, and its correlates in OPS0077 (section 6).</li>
    <li>{chip("ok", "done")}Measured DNA for the remaining 47 arrest-list genes: the mitotic and ESCRT knockouts do not arrest in G2/M (section 4).</li>
    <li>{chip("open", "next")}Relabel the clock quarters in the per-gene figures (section 1) and redo them with an out-of-fold NTC reference.</li>
  </ul>
</section>

<footer>Built from the notebooks in cytodata/2026_sbi2_hackathon_resources (branch talha-cycle-modalities). Numbers are held-out estimates unless marked. Data: Biohub OPS Explorer (OPS0077), Element A549 multiomics on the team VM, Altair CROP-seq knockout-level table.</footer>
</div>
"""
open(OUT, "w").write(page)
print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB")
