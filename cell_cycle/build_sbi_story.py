#!/usr/bin/env python
"""One plain-language page for people outside the hackathon (SBI conference): the cell-cycle story of the Altair team,
built from the repo (cytodata/2026_sbi2_hackathon_resources, notebooks 15 to 20, Diego's branch) and the team document.
Each numbered section is one slide: a claim as the title, what we did, what we found, why it matters, one figure.
usage: python build_sbi_story.py REPORTS_DIR FIGS_DIR NEWBATCH_DIR OUT.html
FIGS_DIR holds figures from the team document (brian_cdk7_panel.png, diego_*_pcs.png, tim_dapi_histogram.png);
NEWBATCH_DIR holds the outputs of ko_dna_hgb_only_stats.py (optional; the section says so when it is missing)."""
import sys, os, io, base64, html, math
import numpy as np, pandas as pd
from PIL import Image

REP, FIGS, NEW, OUT = sys.argv[1:5]
GH = "https://github.com/cytodata/2026_sbi2_hackathon_resources"


def img(path, alt, cap, maxw=1500):
    im = Image.open(path).convert("RGB")
    if im.width > maxw: im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return (f'<figure><img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{html.escape(alt)}" '
            f'width="{im.width}" height="{im.height}" loading="lazy"><figcaption>{cap}</figcaption></figure>')


R = lambda *p: os.path.join(REP, *p); F = lambda n: os.path.join(FIGS, n)


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head); tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def clock_svg():
    cx = cy = 200; RO, RI = 128, 88

    def pt(r, f):
        a = 2 * math.pi * f - math.pi / 2; return cx + r * math.cos(a), cy + r * math.sin(a)

    def arc(r, f0, f1):
        (x0, y0), (x1, y1) = pt(r, f0), pt(r, f1)
        return f'M{x0:.1f} {y0:.1f} A{r} {r} 0 {1 if f1 - f0 > .5 else 0} 1 {x1:.1f} {y1:.1f}'

    def label(r, f, lines, cls):
        x, y = pt(r, f); anchor = "start" if x > cx + 8 else "end" if x < cx - 8 else "middle"
        y0 = y - (len(lines) - 1) * 8 + 4
        return (f'<text x="{x:.1f}" y="{y0:.1f}" text-anchor="{anchor}" class="{cls}">'
                + "".join(f'<tspan x="{x:.1f}" dy="{0 if i == 0 else 16}"{" class=dna" if i else ""}>{t}</tspan>' for i, t in enumerate(lines)) + "</text>")

    seg = [(0, .25, "g1", ["G1", "2N"]), (.25, .375, "g1s", ["G1 to S", "2N"]), (.375, .5, "s", ["S", "copying DNA"]),
           (.5, .625, "g2", ["G2", "4N"]), (.625, .75, "m", ["Mitosis", "dividing"]), (.75, 1, "pd", ["G1 after division", "2N"])]
    old = [(0, .25, "Early G1"), (.25, .5, "Late G1"), (.5, .75, "S"), (.75, 1, "G2/M")]
    g = 0.004
    out = [f'<path d="{arc(RO, a + g, b - g)}" class="ring {c}"/>' for a, b, c, _ in seg]
    out += [label(RO + 26, (a + b) / 2, t, "lab") for a, b, _, t in seg]
    out += [f'<path d="{arc(RI, a + g, b - g)}" class="old"/>' for a, b, _ in old]
    out += [label(RI - 22, (a + b) / 2, [t], "oldlab") for a, b, t in old]
    x0, y0 = pt(RO + 16, 0); x1, y1 = pt(RI - 10, 0)
    out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" class="tick"/>')
    out.append(f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" class="ctr">one cell cycle</text>'
               f'<text x="{cx}" y="{cy + 16}" text-anchor="middle" class="ctr2">0 at the top, clockwise</text>')
    return ('<svg viewBox="-75 -6 550 412" role="img" aria-labelledby="clk-t clk-d"><title id="clk-t">The cell-cycle clock</title>'
            '<desc id="clk-d">Outer ring, from measured DNA: 0 to 0.25 G1 (2N), 0.25 to 0.375 G1 to S, 0.375 to 0.5 S, 0.5 to 0.625 G2 (4N), '
            '0.625 to 0.75 mitosis, 0.75 to 1 G1 after division (2N). Inner dashed ring: the quarter labels Early G1, Late G1, S, G2/M '
            'used at first, about one quarter off.</desc>' + "".join(out) + "</svg>")


# ---- the newest result: measured DNA for the knockouts only the gradient-boosting model calls
nb_ok = os.path.exists(os.path.join(NEW, "confirmation_by_model_call.csv"))
if nb_ok:
    S = pd.read_csv(os.path.join(NEW, "confirmation_by_model_call.csv")).set_index("called_by")
    T = pd.read_csv(os.path.join(NEW, "all_measured_knockouts.csv"))
    B = pd.read_csv(os.path.join(NEW, "hgb_only_measured_dna.csv"))
    from scipy.stats import spearmanr
    n_all = len(T); rho_h = spearmanr(T.d_pct_4N_pred_hgb, T.measured_d_pct_4N)[0]; rho_r = spearmanr(T.d_pct_4N_pred_ridge, T.measured_d_pct_4N)[0]
    c = lambda k: (int(S.loc[k, "confirmed"]), int(S.loc[k, "n"]), S.loc[k, "confirmed"] / S.loc[k, "n"] * 100)
    both, hgbo, ridgeo, neither = c("both"), c("HistGB only"), c("ridge only"), c("neither")
    sign_b = 100 * (pd.Series(B.d_pct_4N_pred_hgb).apply(lambda v: v > 0) == (B.measured_d_pct_4N > 0)).mean()
NEWFIG = os.path.join(NEW, "hgb_only_measured_dna.png")
# ---- guide-level test on measured DNA (ko_guide_dna.py outputs, copied next to the new batch)
gd_ok = os.path.exists(os.path.join(NEW, "guide_heterogeneity_measured_dna.csv"))
if gd_ok:
    GH_ = pd.read_csv(os.path.join(NEW, "guide_heterogeneity_measured_dna.csv")); PG = pd.read_csv(os.path.join(NEW, "per_guide_measured_dna.csv"))
    g_n, g_p05, g_q05 = len(GH_), int((GH_.p_guides_differ < .05).sum()), int((GH_.q_guides_differ < .05).sum())
    g_ng, g_min = int(GH_.n_guides.median()), int(GH_.min_guide_n.median())
    cfg = PG[PG.confirmed == True]; byg = cfg.groupby("gene").same_direction.agg(["sum", "size"])
    g_same, g_tot, g_allk, g_k = int(cfg.same_direction.sum()), len(cfg), int((byg["sum"] == byg["size"]).sum()), len(byg)
# ---- Brian's V2 panels (numbers read from his figures) next to measured DNA
BRIAN_V2 = [("AURKB", "cytokinesis failure: G2", "G2/M +7.1 (Z 3.2)"), ("POLA1", "fork stalling: S or G2/M", "G2/M +10.4 (Z 5.4)"),
            ("POLA2", "fork stalling: S or G2/M", "G2/M +14.7 (Z 6.0)"), ("CCNB1", "G2 block, back to G1", "no clear change (G2/M +2.1, Z 1.0)"),
            ("BUB1B", "checkpoint loss, slippage: G1", "early G1 +6.3 (Z 2.1)"),
            ("CDK7", "G1 restriction point: G1", "early G1 +8.5 (Z 2.9)")]
# all cells of Brian's and Louise's genes (lb_guides.py outputs, copied next to the new batch); else the ~200-cell sample
lb_ok = os.path.exists(os.path.join(NEW, "gene_level_all_cells.csv"))
if lb_ok:
    LG = pd.read_csv(os.path.join(NEW, "gene_level_all_cells.csv")).set_index("gene"); LH = pd.read_csv(os.path.join(NEW, "guide_heterogeneity_two_nulls.csv")).set_index("gene")
    LB = pd.read_csv(os.path.join(NEW, "per_guide_all_cells.csv")).set_index("short")
    LCAL = 100 * np.mean(pd.read_csv(os.path.join(NEW, "permutation_test_on_ntc_guide_sets.csv")).iloc[:, 0].values < .05)
    LL = pd.read_csv(os.path.join(NEW, "louise_guides_vs_measured_dna.csv"))
    from scipy.stats import spearmanr as _sp
    l_g1, l_g2m = _sp(LL["G1"], LL["d_pct_2N"])[0], _sp(LL["G2/M"], LL["d_pct_4N"])[0]
    RND = T[T.set == "calibration"]; r2_, r4_ = RND.measured_d_pct_2N.median(), RND.measured_d_pct_4N.median()


def dna_txt(g_):
    if lb_ok and g_ in LG.index:
        r = LG.loc[g_]; return f"4N {r.d_pct_4N:+.1f}, 2N {r.d_pct_2N:+.1f} (all {int(r.n)} cells; vs random knockouts 4N {r.d_pct_4N - r4_:+.1f}, 2N {r.d_pct_2N - r2_:+.1f})"
    if not nb_ok or g_ not in set(T.gene): return "not measured"
    r = T.set_index("gene").loc[g_]
    return f"4N {r.measured_d_pct_4N:+.1f}, 2N {r.measured_d_pct_2N:+.1f}" + (" (confirmed)" if r.confirmed else " (not significant)")

page = f"""<title>Is It Just the Cell Cycle?</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,560;1,6..72,400&family=Public+Sans:wght@400;700&display=swap">
<style>
/* layout: hero = the question, the short answer and the measured clock (to scale); then one numbered section per slide in a
   68ch reading column with full-width figures; numbers hang in the margin on wide screens because the slides are a sequence.
   colour: cool slide-grey ground, DAPI blue accent (also the 4N arc); the clock arcs carry the DNA-state colours. */
:root {{
  --bg: #f3f5f7; --surface: #ffffff; --ink: #16212a; --muted: #56646f; --rule: #cbd4db;
  --accent: #2b44b0; --accent-soft: #e3e7f6; --mat: #ffffff;
  --g1: #2e8a72; --g1s: #7b8a37; --s: #c2841a; --g2: #2b44b0; --m: #ad3f86; --pd: #86c0b0; --old: #8a96a0;
  --display: "Newsreader", "Iowan Old Style", Georgia, "Times New Roman", serif;
  --body: "Public Sans", "Segoe UI", Helvetica, Arial, sans-serif;
  --code: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0d1319; --surface: #141c24; --ink: #e3e9ee; --muted: #9babb6; --rule: #2a3744;
  --accent: #93a3ff; --accent-soft: #1c2444; --mat: #ffffff;
  --g1: #5cc2a5; --g1s: #b2be62; --s: #e6ad4a; --g2: #93a3ff; --m: #e07db9; --pd: #3f7568; --old: #6f7d88; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #0d1319; --surface: #141c24; --ink: #e3e9ee; --muted: #9babb6; --rule: #2a3744;
  --accent: #93a3ff; --accent-soft: #1c2444; --mat: #ffffff;
  --g1: #5cc2a5; --g1s: #b2be62; --s: #e6ad4a; --g2: #93a3ff; --m: #e07db9; --pd: #3f7568; --old: #6f7d88; color-scheme: dark }}
body {{ background: var(--bg); color: var(--ink); font: 17px/1.62 var(--body); }}
.wrap {{ max-width: 1040px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 44px 88px; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }}
.hero {{ display: grid; gap: 28px 48px; grid-template-columns: minmax(0, 1fr); align-items: start; }}
@media (min-width: 880px) {{ .hero {{ grid-template-columns: minmax(0, 1.05fr) minmax(0, .95fr); }} }}
.hero > * {{ min-width: 0; }}
.src {{ font-size: 15px; color: var(--muted); margin: 0 0 10px; }}
h1 {{ font: 560 clamp(38px, 6vw, 64px)/1.02 var(--display); letter-spacing: -.015em; margin: 0 0 18px; text-wrap: balance; }}
.q {{ font: 400 clamp(19px, 2.1vw, 22px)/1.45 var(--display); margin: 0 0 18px; max-width: 40ch; }}
.short {{ margin: 0 0 24px; max-width: 58ch; }}
.short b {{ color: var(--accent); }}
.findings {{ margin: 0; border-top: 1.5px solid var(--ink); }}
.findings > div {{ display: grid; grid-template-columns: 6.6em minmax(0, 1fr); gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--rule); align-items: baseline; }}
.findings dt {{ font: 560 22px/1.15 var(--display); font-variant-numeric: tabular-nums lining-nums; color: var(--accent); }}
.findings dd {{ margin: 0; font-size: 15.5px; line-height: 1.5; min-width: 0; }}
.clock {{ margin: 0; }}
.clock svg {{ display: block; width: 100%; max-width: 520px; height: auto; margin: 0 auto; }}
.clock figcaption {{ max-width: 52ch; margin: 6px auto 0; }}
.ring {{ fill: none; stroke-width: 26; }}
.ring.g1 {{ stroke: var(--g1); }} .ring.g1s {{ stroke: var(--g1s); }} .ring.s {{ stroke: var(--s); }}
.ring.g2 {{ stroke: var(--g2); }} .ring.m {{ stroke: var(--m); }} .ring.pd {{ stroke: var(--pd); }}
.old {{ fill: none; stroke: var(--old); stroke-width: 2; stroke-dasharray: 3 5; }}
.tick {{ stroke: var(--ink); stroke-width: 1.5; }}
.clock text {{ fill: var(--ink); font-family: var(--body); }}
.clock .lab {{ font-size: 15px; font-weight: 700; }}
.clock .lab .dna {{ font-weight: 400; fill: var(--muted); font-size: 13px; }}
.clock .oldlab {{ font-size: 13px; font-style: italic; fill: var(--muted); font-family: var(--display); }}
.clock .ctr {{ font-size: 15px; font-weight: 700; }}
.clock .ctr2 {{ font-size: 12.5px; fill: var(--muted); }}
nav.toc {{ margin: 40px 0 0; padding: 14px 0; border-block: 1px solid var(--rule); display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: 15.5px; }}
nav.toc a {{ color: var(--ink); text-decoration: none; display: inline-flex; gap: 7px; align-items: baseline; }}
nav.toc a span {{ font: 560 15px/1 var(--display); color: var(--accent); font-variant-numeric: lining-nums; }}
nav.toc a:hover {{ color: var(--accent); }}
section {{ margin-top: 76px; display: grid; gap: 16px; }}
section > * {{ min-width: 0; }}
section > h2 {{ font: 560 clamp(26px, 3.2vw, 34px)/1.18 var(--display); margin: 0; text-wrap: balance; position: relative; max-width: 30ch; }}
section > h2 .n {{ display: block; font: 400 17px/1 var(--display); font-style: italic; color: var(--accent); margin-bottom: 8px; }}
@media (min-width: 1180px) {{ section > h2 .n {{ position: absolute; left: -64px; top: .12em; margin: 0; font: 400 36px/1 var(--display); font-style: normal; }} }}
.lede {{ font: 400 20px/1.5 var(--display); max-width: 60ch; margin: 0; }}
section > h3 {{ font: 560 21px/1.25 var(--display); margin: 18px 0 0; text-wrap: balance; }}
.wdw {{ display: grid; gap: 10px; max-width: 68ch; }}
.wdw p {{ margin: 0; }}
.wdw b {{ font-weight: 700; }}
p, li {{ max-width: 68ch; }}
p {{ margin: 0; }}
ul {{ margin: 0; padding-left: 22px; display: grid; gap: 8px; }}
li::marker {{ color: var(--muted); }}
code {{ font: 14.5px var(--code); background: var(--accent-soft); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }}
figure {{ margin: 6px 0; }}
figure img {{ display: block; width: 100%; height: auto; background: var(--mat); padding: 8px; box-sizing: border-box; border: 1px solid var(--rule); border-radius: 3px; }}
figcaption {{ font-size: 14.5px; color: var(--muted); padding-top: 8px; max-width: 72ch; }}
.two {{ display: grid; gap: 16px; grid-template-columns: minmax(0, 1fr); }}
@media (min-width: 760px) {{ .two {{ grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }} }}
.tw {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 15px; font-variant-numeric: tabular-nums; border-top: 1.5px solid var(--ink); border-bottom: 1.5px solid var(--ink); }}
th, td {{ padding: 8px 14px 8px 0; text-align: left; vertical-align: top; }}
th {{ font-weight: 700; font-size: 14px; border-bottom: 1px solid var(--ink); }}
td {{ border-bottom: 1px solid var(--rule); }}
tr:last-child td {{ border-bottom: 0; }}
.gloss {{ display: grid; gap: 0; margin: 0; border-top: 1.5px solid var(--ink); }}
.gloss > div {{ display: grid; grid-template-columns: minmax(0, 13em) minmax(0, 1fr); gap: 16px; padding: 9px 0; border-bottom: 1px solid var(--rule); }}
@media (max-width: 560px) {{ .gloss > div {{ grid-template-columns: minmax(0, 1fr); gap: 2px; }} }}
.gloss dt {{ font-weight: 700; }}
.gloss dd {{ margin: 0; }}
.take {{ counter-reset: t; list-style: none; padding: 0; display: grid; gap: 14px; }}
.take li {{ counter-increment: t; display: grid; grid-template-columns: 2.2em minmax(0, 1fr); font: 400 19px/1.5 var(--display); }}
.take li::before {{ content: counter(t); color: var(--accent); font-weight: 560; }}
a {{ color: var(--accent); text-underline-offset: 3px; }}
.note {{ font-size: 15px; color: var(--muted); }}
footer {{ margin-top: 80px; font-size: 14.5px; color: var(--muted); border-top: 1.5px solid var(--ink); padding-top: 16px; max-width: 80ch; display: grid; gap: 8px; }}
@media (prefers-reduced-motion: no-preference) {{ html {{ scroll-behavior: smooth; }} }}
</style>
<div class="wrap">
<header>
  <div class="hero">
    <div>
      <p class="src">SBI2 CytoData hackathon, November 2026. Altair team: cell-cycle state in image-based screens.</p>
      <h1>Is It Just the Cell Cycle?</h1>
      <p class="q">Cells that carry the same genetic perturbation still look very different from one another. Part of that is where each cell is in its division cycle.</p>
      <p class="short">We asked two things. Can we read the cell cycle from label-free phase-contrast images? And if we remove it, do perturbed cell populations become more alike? <b>The first answer is yes. The second is: only a little.</b> The cycle explains a few percent of single-cell variation, and knockout phenotypes are mostly something else.</p>
      <dl class="findings">
        <div><dt>0.92</dt><dd>how well phase contrast tells cells with doubled DNA (4N) from cells before copying (2N), on held-out images. The antibody panel scores 0.89. (1 is perfect, 0.5 is chance.)</dd></div>
        <div><dt>{f"{rho_h:.2f}" if nb_ok else "0.84"}</dt><dd>agreement between the cycle shift predicted from phase contrast and the shift measured with a DNA stain, across {n_all if nb_ok else 158} knockouts (rank correlation).</dd></div>
        <div><dt>3 to 9%</dt><dd>of single-cell variation follows the cell cycle, in every imaging and RNA readout we tested.</dd></div>
        <div><dt>0.6 to 5.5%</dt><dd>tighter perturbed populations after the cycle is removed. How well knockouts separate from controls does not change.</dd></div>
      </dl>
    </div>
    <figure class="clock">
      {clock_svg()}
      <figcaption>The cell-cycle clock we used, drawn to scale. Each cell gets a position from 0 to 1 on a circle. The outer ring shows what cells at each position actually contain, from a DNA stain: one genome copy (2N), copying (S) or two copies (4N). The dashed inner ring shows the labels used at first, which were about a quarter turn off (slide 3).</figcaption>
    </figure>
  </div>
  <nav class="toc" aria-label="Slides">
    <a href="#why"><span>1</span>Why it matters</a><a href="#data"><span>2</span>Data</a><a href="#clocks"><span>3</span>Building the clock</a>
    <a href="#phase"><span>4</span>Label-free readout</a><a href="#align"><span>5</span>Across datasets</a><a href="#knockouts"><span>6</span>Knockouts</a>
    <a href="#arrest"><span>7</span>Expected arrests</a><a href="#regress"><span>8</span>Removing the cycle</a><a href="#element"><span>9</span>Imaging and RNA</a>
    <a href="#take"><span>10</span>Take-home</a><a href="#team"><span>11</span>Team and data</a><a href="#glossary">Glossary</a><a href="#deck">Deck outline</a>
  </nav>
</header>

<section id="why">
  <h2><span class="n">1</span>A cell's place in its cycle changes how it looks</h2>
  <p class="lede">Optical pooled screens image hundreds of thousands of cells, each carrying one CRISPR knockout. If a knockout changes how many cells sit at each cycle stage, its image profile mixes two things: "this gene changes the cycle" and "this gene changes the cell".</p>
  <div class="wdw">
    <p><b>The cell cycle in one line.</b> A cell grows with one copy of its genome (G1, called 2N), copies its DNA (S), holds two copies (G2, called 4N) and divides (mitosis), giving two G1 cells. Size, DNA amount and nuclear shape all change along the way.</p>
    <p><b>The hackathon goal.</b> Use measured ground truth (proteins, DNA, RNA) to learn a cell-cycle readout from images, then remove it from perturbation profiles and test whether perturbed populations become less heterogeneous.</p>
    <p><b>Why it matters.</b> If the cycle drives much of the spread, correcting for it would sharpen every phenotype in a screen. If it does not, the cycle is better used as a readout in its own right.</p>
  </div>
</section>

<section id="data">
  <h2><span class="n">2</span>Three datasets, each with a different ground truth</h2>
  <p class="lede">All three use A549 lung cancer cells. Two measure images and a cell-cycle ground truth in the same cells; the third has RNA averaged per knockout.</p>
  {table(["dataset", "what was perturbed", "images", "cell-cycle ground truth"], [
      ["OPS0077 optical pooled screen (CZ Biohub OPS Explorer)", "1,000 genes knocked out with CRISPR, plus non-targeting controls: 52,189 control cells, 591,136 knockout cells", "phase contrast (label-free), and 4i cyclic immunofluorescence on the same cells", "DNA stain (DAPI) and antibodies for pRb, Rb and p21"],
      ["Element Biosciences multiomics", "TNF-alpha and gene knockouts; 206,091 cells", "fluorescence: DNA stain, phospho-antibodies, organelle and membrane stains", "single-cell RNA from the same cells"],
      ["CROP-seq (Altair)", "the same 1,000 knockouts", "none", "RNA, averaged per knockout"]])}
  <p class="note">The phase-contrast image and the 4i image of each OPS0077 cell are matched by position, so every cell has a label-free view and a measured cycle state.</p>
</section>

<section id="clocks">
  <h2><span class="n">3</span>We built cell-cycle clocks from the ground truth, then checked what each part contains</h2>
  <p class="lede">A clock turns each cell's markers into a position on a circle. The labels on that circle have to be checked against something measured, because the circle has no natural starting point.</p>
  <div class="wdw">
    <p><b>What we did.</b> In OPS0077, Louise fitted a closed curve through DNA content, pRb, p21 and the phosphorylated fraction of Rb, which gives each control cell a position from 0 to 1 (35,633 of 52,220 control cells have all four inputs). Position 0 is the lowest-DNA part of the loop, DNA peaks at about 0.55 to 0.6, and p21 is lowest around S. In Element, a model of the RNA (cycleVI) gives each cell an angle. We then read out the DNA content of the cells at each position.</p>
    <p><b>What we found.</b> The doubled-DNA cells (4N) sit at 0.5 to 0.75, and the last quarter, 0.75 to 1, is almost entirely cells that have just divided and are back to one copy. Labelling the four quarters "Early G1, Late G1, S, G2/M" shifted every label by about a quarter.</p>
    <p><b>Why it matters.</b> This explained a puzzle from the first day. Knockouts expected to stop cells in G1 seemed to pile up in "G2/M" (CDK7 showed +10.8% "G2/M"). Those were G1 cells after division, so the knockouts did what was expected.</p>
  </div>
  {img(R("phase_to_cell_cycle", "cc_clock_morphology.png"), "Control cells on the measured clock", "Left: control cells placed on the measured clock, coloured by DNA content. Middle: DNA, pRb and p21 around the clock. Right: the same markers around the clock predicted from phase contrast (solid) and the measured clock (dashed).")}
  {img(F("brian_cdk7_panel.png"), "CDK7 knockout against controls", "The first-day view of one knockout (CDK7, red) against controls (grey): stage percentages, the distribution along the clock and the polar clock. The stage labels on the left panel are the quarter labels that turned out to be a quarter turn off.")}
</section>

<section id="phase">
  <h2><span class="n">4</span>Label-free phase contrast reads the cycle about as well as the antibody panel</h2>
  <p class="lede">Trained on control cells and tested on images it never saw, phase contrast separates 4N from 2N cells with a score of 0.92, against 0.89 for the 4i antibody panel.</p>
  <div class="wdw">
    <p><b>What we did.</b> We trained models that predict each cell's measured clock position and DNA state from its phase-contrast features, and scored them on held-out image tiles. Brian's model (gradient-boosted trees on the raw features) was compared with ours (a linear model on 100 principal components) on the same folds.</p>
    <p><b>What we found.</b> Gradient boosting places cells on the clock with a typical error of 22 degrees out of 360, against 29 for the linear model (random guessing gives 90). Louise's independent circular model reaches a mean error of 0.105 turns (38 degrees) on cells it never saw and 0.113 turns on a whole held-out well. It also works on knockout cells: a model trained only on controls separates 4N from 2N in knockout cells as well as in control cells (0.963 against 0.968).</p>
    <p><b>Why it matters.</b> Any screen with brightfield or phase-contrast images can get a cell-cycle readout without spending a fluorescence channel on it.</p>
  </div>
  {img(R("cell_cycle_by_modality", "modality_score_quality.png"), "How well each score separates DNA state", "How well each cell-cycle score separates measured DNA state (2N, S, 4N), in each dataset and imaging mode. Grey bars are scores built from the DNA stain itself, so they are expected to score highly.")}
</section>

<section id="align">
  <h2><span class="n">5</span>Clocks from different datasets only partly line up</h2>
  <p class="lede">Within one dataset the clocks agree. Between the protein-based clock (OPS0077) and the RNA-based clock (Element) only the stretch from DNA copying to division can be matched.</p>
  <div class="wdw">
    <p><b>What we did.</b> We rotated and compared the clocks using features measured the same way in both datasets: DNA content, cell and nuclear size, and cell shape.</p>
    <p><b>What we found.</b> The DNA-based OPS0077 clocks agree within 4 to 6 degrees. Across datasets, DNA lines up the S to G2/M arc, but size and shape do not agree on the rotation, and no combination beats a random-rotation control. The G1 part cannot be aligned with what both datasets share.</p>
    <p><b>A second route: transfer a model.</b> A forest trained on Element DNA-stain features (DNA amount, nuclear size and texture) to predict the RNA clock reaches 53 degrees error on held-out Element wells. Applied to OPS0077 cells and compared with their 4i clock, it gets the same quarter for 34.6% of cells (25% by chance). Keeping only the 56 features whose distributions match between the datasets raises this to 49% and lowers the median disagreement from 53.6 to 41.6 degrees. The agreement is again strongest from S to G2.</p>
    <p><b>Matching features without their names.</b> Daniel's next step was to match features through anchors rather than names. Two things help. First, ranking every feature within its own well removes the scale differences between instruments: the 88 name-matched features then reach 51.3% same quarter and 40.3 degrees, without any filter. Second, we described each feature by how it behaves across a grid of DNA content and nuclear size, ranked within each well; both datasets measure these the same way. We then paired each Element feature with the OPS0077 feature that behaves most alike, whatever its name. This anchor matching reaches 55.4% same quarter and 37.9 degrees (+4.1 points, 95% interval 3.4 to 4.8, on the same held-out cells). Same-named features do behave alike (median correlation 0.94), but so do many others, so behaviour confirms a match rather than finding a unique partner. The 91 anchor matches land on only 19 OPS0077 features, and only 4 keep their name. Texture measured at a fixed pixel distance on two microscopes with different pixel sizes is one likely reason names mislead.</p>
    <p><b>Why it matters.</b> To compare cell-cycle effects across platforms, use DNA-content classes (2N, S, 4N), which are measured identically, rather than clock angles.</p>
  </div>
  <div class="two">
    {img(R("cell_cycle_alignment", "cc_align_4_regime_agreement.png"), "Agreement between OPS0077 clock versions", "Agreement between the OPS0077 clock versions after the best rotation. The DNA-based versions agree closely; the markers-only version runs the same way but is rotated.", 1100)}
    {img(F("element_to_biohub_filtered.png"), "Element-trained model on OPS0077 cells", "An Element-trained DNA-feature model applied to OPS0077 cells (56 distribution-matched features): rows are the OPS0077 4i clock, columns the transferred prediction, in twelve 30-degree bins. The diagonal is clearest around 0.42 to 0.58, the S to G2 stretch.", 1100)}
  </div>
</section>

<section id="knockouts">
  <h2><span class="n">6</span>Phase contrast finds the knockouts that shift the cycle, and a DNA stain confirms them</h2>
  <p class="lede">Across all 1,000 knockouts, the phase-contrast readout flags genes whose cells pile up at a different cycle stage. When we measured DNA directly, the predicted shifts held.</p>
  <div class="wdw">
    <p><b>What we did.</b> For every knockout we compared its predicted cycle distribution with control cells, using groups of control cells of the same size as a fair reference. Then we measured DNA content from the DAPI stain in {n_all if nb_ok else 158} knockouts: the predicted shifters, a random set without a predicted shift, the 60 genes the team expected to cause an arrest{", and the extra knockouts only the gradient-boosting model flags" if nb_ok else ""}.</p>
    <p><b>What we found.</b> The predicted change in 4N cells matches the measured change (rank correlation {f"{rho_h:.2f} for gradient boosting and {rho_r:.2f} for the linear model, over {n_all} knockouts" if nb_ok else "0.84 for gradient boosting and 0.78 for the linear model"}). DNA replication genes gain 4N cells (MTBP +17 points, MCM7 +12, CDC45 +8). Ribosome, proteasome, chaperone and splicing genes lose them, as expected for cells that stall before copying their DNA.</p>
    {"<p><b>Newest check.</b> Gradient boosting flags " + f"{both[1] + hgbo[1]}" + " knockouts and the linear model " + f"{both[1] + ridgeo[1]}" + ". We have now measured DNA in every flagged knockout. Of those both models flag, " + f"{both[0]} of {both[1]} ({both[2]:.0f}%)" + " are confirmed by measured DNA. Of those only gradient boosting flags, " + f"{hgbo[0]} of {hgbo[1]} ({hgbo[2]:.0f}%)" + " are confirmed, a weaker but real set. Of those only the linear model flags, " + f"{ridgeo[0]} of {ridgeo[1]}" + " are confirmed. As a baseline, " + f"{neither[0]} of {neither[1]} ({neither[2]:.0f}%)" + " knockouts that neither model flags shift anyway.</p>" if nb_ok else ""}
    <p><b>Why it matters.</b> A label-free screen can be mined for cell-cycle regulators, and the hits hold up against a direct DNA measurement.</p>
  </div>
  {img(NEWFIG if nb_ok else R("ko_cell_cycle", "ko_measured_vs_predicted_dna_all.png"), "Predicted vs measured change in 4N cells", (f"Left: predicted against measured change in 4N cells for {n_all} knockouts, coloured by which model flags them (large points: confirmed by measured DNA). Right: share of each group confirmed by measured DNA." if nb_ok else "Predicted against measured change in 4N cells for 158 knockouts (large points: significant by measured DNA)."))}
</section>

<section id="arrest">
  <h2><span class="n">7</span>Some expected arrests did not happen, and phase contrast was right about that</h2>
  <p class="lede">The team listed 60 genes expected to stop cells in G1, S or G2/M. Measured DNA agrees with the S-phase group. The G1 and G2/M groups look no different from random knockouts.</p>
  <div class="wdw">
    <p><b>The right reference.</b> Control cells carry guides that cut nowhere. Cutting itself shifts cells: 30 randomly chosen knockouts gain 2N cells (median +6.1 points) and lose 4N cells (-3.3 points), and 28 of the 30 gain 2N. In the CROP-seq RNA, p21 rises in 636 of 1,000 knockouts. A549 cells have working p53, so a mild stop after DNA cutting is the likely cause. An arrest call should therefore beat random knockouts, not only non-cutting controls.</p>
    <p><b>What we found.</b> S-phase genes gain 4N cells (median +4.7 points, about 8 points above random knockouts). The G1 group gains 2N cells (21 of 26 genes, median +6.1 points), exactly as much as random knockouts; only RPTOR stands out on its own. The mitotic and checkpoint genes expected to pile up in G2/M lose 4N cells (median -2.5 points), close to random knockouts rather than above them. They are not hiding as cells with two nuclei either (10.0% against 10.4% in controls); only KIF23 clearly is (31%).</p>
    <p><b>A hypothesis we tested.</b> Brian proposed that G1-arrested cells keep growing until they look like G2 cells. Knockout cells with one genome copy are indeed larger than control cells in 148 of 158 knockouts, but they are not called 4N more often, and a DNA floor on the 4N call changes no knockout by more than 0.63 points.</p>
    <p><b>Why it matters.</b> Literature expectations for a gene do not always hold in a given cell line and time window (here about 89 hours of Cas9). A direct readout is needed to know.</p>
  </div>
  {img(F("arrest_vs_random.png"), "Arrest lists against random knockouts", "Each dot is one knockout with measured DNA, as change against non-cutting control guides. Grey: 30 random knockouts; dashed line: their median, the fair reference. The S-phase list gains 4N cells; the G1 and G2/M lists look like random knockouts.")}
  {img(R("ko_cell_cycle", "g1_hypertrophy_check_all.png"), "Testing the cell-growth hypothesis", "Left: four versions of the phase model against measured DNA across knockouts. Middle: size of one-copy knockout cells against excess 4N calls, per knockout (red: the G1 list). Right: wrong 4N calls among control cells, by size.")}
  <h3>Brian's updated panels</h3>
  <div class="wdw">
    <p><b>What we did.</b> Brian revised the expected outcome for each group of genes and redrew the knockout panels on the corrected clock labels (4N at 0.5 to 0.75, just-divided cells at 0.75 to 1), each against groups of control cells of the same size. We put the measured DNA of the same knockouts next to them, from every cell of each knockout, with the change beyond random knockouts.</p>
    <p><b>What we found.</b> The replication and cytokinesis genes move as expected, and measured DNA moves the same way. Brian's newest panel, CDK7 for the G1 restriction point, gains early-G1 cells, and measured DNA agrees: 2N +10.1 points, about 4 above random knockouts. CCNB1 shows no clear shift on the clock, and its guides disagree with each other (next part). BUB1B gains early-G1 cells as expected, but its DNA distribution does not change. Slippage out of mitosis would leave G1 cells with doubled DNA, so the gain more likely reflects a shorter mitosis once the checkpoint is gone.</p>
  </div>
  <div class="wdw">
    <p><b>Brian's follow-up question, checked in the RNA.</b> Could CCNB2 make up for the loss of CCNB1? In the CROP-seq data, CCNB2 RNA in CCNB1-knockout cells changes no more than in a typical knockout (rank 387 of 997). This gives no RNA evidence for compensation, though compensation could still act at the protein level. CCNB1's own RNA is not lower either (log2 fold change +0.06), but Cas9 cuts often leave RNA levels intact, so this does not show whether the knockout worked. p21 RNA, a stress and arrest signal, rises in most knockouts (636 of 1,000), consistent with a general response to cutting. It rises more in CCNB1 (rank 77) and falls in CENPC.</p>
  </div>
  {table(["knockout", "Brian's expectation", "clock shift vs controls (points)", "measured DNA vs controls (points)"], [[g_, e_, c_, dna_txt(g_)] for g_, e_, c_ in BRIAN_V2])}
  <div class="two">
    {img(F("brian_v2_pola2.png"), "POLA2 knockout against controls", "POLA2 (replication fork stalling): stage percentages against the matched control null, the distribution along the clock and the polar clock. Cells pile up at 0.5 to 0.65, the doubled-DNA stretch.", 1100)}
    {img(F("brian_v2_cdk7.png"), "CDK7 knockout against controls", "CDK7 (G1 restriction point), same panels. More early-G1 cells and fewer G2/M cells on the clock; measured DNA agrees in direction (2N +10.1, 4N -6.3 points), about 4 points of 2N beyond random knockouts.", 1100)}
  </div>
  <h3>Guides of the same gene can disagree</h3>
  <div class="wdw">
    <p><b>What Louise found.</b> Each gene is targeted by four guide RNAs. In CCNB1, MCM3 and CENPC the expected arrest is not clear at the gene level, but the guides of each gene differ (p = 0.016, 0.046 and 0.027). Two CCNB1 guides add about 15 points of G2/M cells while another adds 12 points of G1 cells, so the gene average hides both. The guide ranking holds up in two separate sets of cells scored by different readouts (G1 fraction, rank correlation 0.82). All three genes also carry more just-divided cells than controls (20 to 26% against 13%). That shift is shared rather than gene-specific, and may be a general response to cutting.</p>
    {("<p><b>The same question on measured DNA.</b> For the " + f"{g_n}" + " newly measured knockouts (median " + f"{g_ng}" + " guides each, " + f"{g_min}" + " cells in the smallest), we compared the guides' 2N, S and 4N mix with control guides of the same sizes, so ordinary guide-to-guide variation is part of the reference. The guides differ more than control guides in " + f"{g_p05}" + " knockouts at p &lt; 0.05 (about " + f"{round(0.05 * g_n)}" + " expected by chance) and in " + f"{g_q05}" + " after correcting for testing many knockouts. Among the confirmed knockouts, " + f"{g_same} of {g_tot}" + " guides move 4N cells in the knockout's direction, and all guides agree in " + f"{g_allk} of {g_k}" + " knockouts. With about 50 cells per guide, this test only sees large differences between guides.</p>") if gd_ok else ""}
    {("<p><b>Every cell of Louise's three genes, measured.</b> We measured DNA in all cells of CCNB1, MCM3, CENPC and Brian's panel genes (3,072 usable cells). Louise's per-guide clock shifts agree with measured DNA across her 12 guides (rank correlation " + f"{l_g1:.1f}" + " for G1 against 2N and " + f"{l_g2m:.1f}" + " for G2/M against 4N; her score uses DNA, so this shows consistency rather than independent proof). The CCNB1 split is real in DNA: the two guides that add G2/M cells on her clock gain 4N cells (ATATTTGC " + f"{LB.loc['ATATTTGC'].d_pct_4N:+.0f}" + " and GTCAGACC " + f"{LB.loc['GTCAGACC'].d_pct_4N:+.0f}" + " points), while TGAGGCCA gains 2N cells (" + f"{LB.loc['TGAGGCCA'].d_pct_2N:+.0f}" + "). CCNB1 likely has the classic G2/M phenotype, diluted in the gene average by guides that work less well. The same two MCM3 guides she flagged gain 4N cells (" + f"{LB.loc['CAGTATGA'].d_pct_4N:+.0f}" + " and " + f"{LB.loc['ATATCCAC'].d_pct_4N:+.0f}" + ").</p><p><b>A check on the guide test.</b> The usual test shuffles cells among a gene's guides. Applied to sets of four non-targeting control guides, where no difference is expected, it calls a difference in " + f"{LCAL:.0f}" + "% of sets instead of 5%, because control guides differ from each other too. Compared with control guides of the same size, the guides clearly differ only in AURKB (p " + f"{LH.loc['AURKB'].p_vs_ntc_guides:.3f}" + ") and POLA2 (p " + f"{LH.loc['POLA2'].p_vs_ntc_guides:.3f}" + "), two genes whose real phenotype some guides show strongly and others barely, as uneven knockout efficiency would.</p>") if lb_ok else ""}
    <p><b>Why it matters.</b> For most knockouts the guides agree, so gene-level calls are sound. Where a gene's expected arrest is missing, as for CCNB1, check its guides one by one: guide efficiency varies, a gene average can dilute a real effect, and guide differences should be judged against control guides, not shuffled cells.</p>
  </div>
  {img(F("louise_guide_density.png"), "Cells of each guide along the cycle, CCNB1, MCM3 and CENPC", "Louise's density of cells along the clock for each guide (coloured) and for control cells (thick grey), from the stain score (top, cells with all markers and DNA) and the phase-contrast score (bottom, all cells). In CCNB1, guide TGAGGCCA piles up just after division (near 1) while ATATTTGC and GTCAGACC peak in the doubled-DNA stretch (about 0.6). Shaded bands mark the clock sectors.")}
</section>

<section id="regress">
  <h2><span class="n">8</span>Removing the cycle tightens a few main axes but barely changes heterogeneity</h2>
  <p class="lede">The cycle concentrates in the strongest axes of variation, so those shrink noticeably. Over all features it is a few percent, and perturbed populations stay almost as varied as before.</p>
  <div class="wdw">
    <p><b>What we did.</b> Diego removed the measured clock position from control-cell profiles and compared the variation within each image field before and after. We repeated this on cells from all 1,000 knockouts and on Element, and asked how much each knockout's shift comes from changing the cycle mix versus changing the cells at the same cycle stage.</p>
    <p><b>What we found.</b> The top principal components (the main axes of variation) tighten by 26 to 33% in 4i and 9 to 19% in phase contrast, but a typical single feature loses about 1% of its variance (median ratio 0.989 in 4i, 0.993 in phase contrast; a tail of a few hundred features loses 10 to 35%), and differences between fields and wells barely change. Knockout populations become 0.6 to 5.5% tighter and separate from controls just as well as before. Of 1,000 knockouts, 790 change the cells at a matched cycle stage, 80 change both, and only 1 acts through the cycle mix alone.</p>
    <p><b>Why it matters.</b> Correcting for the cycle is cheap and tightens the dominant axes, but it is not the main source of single-cell heterogeneity. Field effects, clonal patches and local cell density matter more.</p>
  </div>
  <div class="two">
    {img(F("diego_4i_pcs.png"), "Within-field variance per principal component, 4i", "4i features: variation within fields for the top 10 principal components, before (grey) and after (orange) removing the clock position.", 1100)}
    {img(F("diego_phase_pcs.png"), "Within-field variance per principal component, phase contrast", "Phase-contrast features, same comparison. The cycle mostly affects PC1 to PC3.", 1100)}
  </div>
  {img(R("ko_cell_cycle", "library_cycle_decomposition.png"), "Cycle mix versus matched-cycle change", "1,000 knockouts: how much each changes the cycle distribution (horizontal) against how much it changes the cells at the same cycle stage (vertical).")}
</section>

<section id="element">
  <h2><span class="n">9</span>With RNA as ground truth, imaging reads the cycle except just after division</h2>
  <p class="lede">In the Element data, images predict the RNA-based clock to a typical error of 37.5 degrees in wells the model never saw (a shuffled clock gives 77.9).</p>
  <div class="wdw">
    <p><b>What we found.</b> The DNA stain carries most of the signal, and the phospho-antibodies add some. Images fail on one stretch, just after division: the DNA is back to one copy while the RNA still looks like G2/M. The largest image changes around the cycle are at division, where phospho-p38 and phospho-ERK rise and nuclei shrink.</p>
    <p><b>Knockout-level RNA.</b> The CROP-seq data, averaged per knockout, carry little cell-cycle signal: 14 of 997 knockouts move a cell-cycle gene programme strongly, and agreement with the imaging readout is weak (rank correlation about 0.1). Against measured DNA in 269 knockouts the RNA cell-cycle programmes do no better (rank correlation at most 0.07; cross-validated R<sup>2</sup> -0.02 against 0.73 for phase contrast), and adding them to phase contrast does not help (0.73). Averaged per knockout, the RNA does not carry the cycle shift that the images and the DNA stain show.</p>
    <p><b>Why it matters.</b> RNA and images measure different moments of the same cycle. Single-cell paired data are what make a cell-level comparison possible.</p>
  </div>
  {img(R("element_cycle_morphology", "fig_A_error_by_arc.png"), "Imaging error along the RNA clock", "How far imaging misplaces cells, along the RNA clock. The error rises sharply for cells that have just divided.")}
</section>

<section id="take">
  <h2><span class="n">10</span>Take-home messages</h2>
  <ol class="take">
    <li>Label-free phase contrast is a usable cell-cycle readout, confirmed against a DNA stain in hundreds of knockouts.</li>
    <li>The cell cycle is a small part of single-cell variation (3 to 9%). Removing it tightens a few main axes and leaves knockout phenotypes and field effects mostly in place.</li>
    <li>The cycle is more useful as a readout: which perturbations shift it, and what they change in cells at the same stage.</li>
    <li>Anchor any learned clock to measured DNA, and compare datasets through DNA-content classes rather than angles.</li>
    <li>Judge knockouts against random knockouts as well as non-cutting controls: cutting alone moves about 6% of cells into 2N.</li>
  </ol>
  <div class="wdw">
    <p><b>Limits.</b> One cell line and one screen; a measured clock for about two thirds of control cells; predicted, not measured, clocks for most knockout cells; across datasets only the S to G2/M stretch aligns.</p>
    <p><b>Next.</b> Every knockout either model flags now has measured DNA. Next: more cells per guide for the knockouts whose guides disagree, the 4i markers for knockout cells (Diego's questions 2 and 3), more anchors for the cross-dataset transfer (cell shape, other stains) and a check on other random splits, and image embeddings (Cell-DINO) as the phase-contrast input.</p>
  </div>
</section>

<section id="team">
  <h2><span class="n">11</span>Team, data and code</h2>
  {table(["who", "contribution"], [
      ["Louise Morlot", "the circular 4i cell-cycle score from DNA, pRb, Rb and p21, its phase-contrast prediction, and the per-guide knockout analysis (ops0077_cell_cycle/ in the repository)"],
      ["Tim Treis", "the Element RNA cell-cycle clock (cycleVI) and the Element multiomics object"],
      ["Brian Feng", "per-knockout cycle panels (first and updated versions), the gradient-boosting phase model and the cell-growth hypothesis"],
      ["Diego Camacho", "the cycle-normalisation questions and the within-field variance analysis"],
      ["Daniel Siegismund", "aligning cell-cycle scores across datasets, with Tim, Louise and Talha; the Element-to-OPS0077 DNA-feature transfer"],
      ["Talha Rehman", "phase contrast to cell cycle, alignment, the knockout library and measured DNA, Element and CROP-seq mapping (notebooks 16 to 20)"]])}
  <p class="note">Organisers and infrastructure: Loan Vulliard, Nikita Moshkov and the SBI2 CytoData hackathon team. Data: CZ Biohub OPS Explorer (OPS0077), Element Biosciences A549 multiomics, Altair CROP-seq. Code and notebooks: <a href="{GH}" target="_blank" rel="noopener">github.com/cytodata/2026_sbi2_hackathon_resources</a>; Louise's cell-cycle score and per-guide analysis are in <a href="{GH}/tree/main/ops0077_cell_cycle" target="_blank" rel="noopener">ops0077_cell_cycle/</a>.</p>
</section>

<section id="glossary">
  <h2>Words used on this page</h2>
  <dl class="gloss">
    <div><dt>2N, S, 4N</dt><dd>One genome copy (G1), copying DNA, two copies (G2 and mitosis). Measured from the total DNA stain in the nucleus.</dd></div>
    <div><dt>Phase contrast</dt><dd>A label-free microscope image that shows cell and nuclear outlines and texture without any stain.</dd></div>
    <div><dt>4i</dt><dd>Iterative indirect immunofluorescence imaging: the same cells are stained, imaged and stripped repeatedly to read many antibodies.</dd></div>
    <div><dt>pRb, Rb, p21</dt><dd>Phosphorylated Rb rises when a cell commits to divide; p21 holds cells back in G1.</dd></div>
    <div><dt>Optical pooled screen</dt><dd>Many CRISPR knockouts grown together in one well; each cell's knockout is read from a barcode in the image.</dd></div>
    <div><dt>Guide RNA</dt><dd>The 20-letter sequence that directs Cas9 to a gene. Each gene here has four, and they can differ in how well they knock the gene out.</dd></div>
    <div><dt>Non-targeting control</dt><dd>Cells with a guide RNA that cuts nowhere; the reference for every knockout.</dd></div>
    <div><dt>Held out</dt><dd>Tested on image tiles or wells the model never saw during training.</dd></div>
    <div><dt>AUC</dt><dd>The chance that a random 4N cell gets a higher score than a random 2N cell. 0.5 is chance, 1 is perfect.</dd></div>
    <div><dt>Rank correlation</dt><dd>Spearman rho: how well two lists agree in order, from -1 to 1.</dd></div>
    <div><dt>Confirmed</dt><dd>Significant against groups of control cells of the same size, after correcting for testing many knockouts (false discovery rate 5%).</dd></div>
    <div><dt>Principal component</dt><dd>One of the main axes along which cells vary in a feature space.</dd></div>
    <div><dt>Removing the cycle</dt><dd>Subtracting, from every feature, the part that follows clock position.</dd></div>
  </dl>
</section>

<section id="deck">
  <h2>Deck outline</h2>
  <p class="lede">One slide per numbered section. The title is the claim; the figure is the one shown in that section.</p>
  {table(["slide", "title", "figure", "say"], [
      ["0", "Is it just the cell cycle?", "the clock (top of page)", "Cells with the same knockout differ; is the cycle why?"],
      ["1", "A cell's place in its cycle changes how it looks", "none, or a sketch of G1, S, G2, M", "The goal: learn the cycle from images, remove it, re-test heterogeneity."],
      ["2", "Three datasets, three ground truths", "the data table", "Images plus a measured cycle in the same cells."],
      ["3", "Check what each part of a clock contains", "clock figure; CDK7 panel", "The first labels were a quarter off; that explained the G1 puzzle."],
      ["4", "Phase contrast reads the cycle", "score-quality bars", "0.92 vs 0.89; 22 degrees; works on knockout cells."],
      ["5", "Clocks only partly align across datasets", "regime agreement; Element-to-OPS0077 transfer", "Transfer gets 49% of quarters (25% chance); anchor-matched features, names unused, 55%; compare platforms by DNA classes."],
      ["6", "Phase contrast finds cycle-shifting knockouts", "predicted vs measured DNA", f"rho {rho_h:.2f} over {n_all} knockouts; replication up, ribosome down." if nb_ok else "rho 0.84 over 158 knockouts; replication up, ribosome down."],
      ["7", "Some expected arrests did not happen", "growth test; Brian's POLA2 and AURKB panels; Louise's guide plot", "Cutting alone adds about 6 points of 2N, so compare with random knockouts; S-phase genes pass, G1 and G2/M lists do not; guides mostly agree, CCNB1's do not."],
      ["8", "Removing the cycle barely changes heterogeneity", "Diego's PC bars; decomposition", "Top axes tighten; populations 0.6 to 5.5% tighter; 790 of 1,000 change the cells."],
      ["9", "Imaging and RNA disagree just after division", "error along the RNA clock", "37.5 degrees; the gap is just-divided cells."],
      ["10", "Take-home", "none", "Use the cycle as a readout, anchored to DNA."],
      ["11", "Team and data", "none", "Credits and the repository."]])}
</section>

<footer>
  <p>Numbers are from held-out tests unless marked. Built from the notebooks and reports in the hackathon repository (notebooks 15 to 20 and the team document).</p>
</footer>
</div>
"""
open(OUT, "w").write(page)
print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB")
