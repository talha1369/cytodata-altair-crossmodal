#!/usr/bin/env python
"""Slide figures for the "OPS validation experiments" part of the hackathon wind-down deck (16:9, Arial, deck grey).
usage: python ops_validation_figs.py NTC_DAPI_H5AD OPS_CLOCK_CSV ALL_MEASURED_CSV ARREST_LIST_CSV OUT_DIR"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

NTC, CLK, ALLM, ARR, OUT = sys.argv[1:6]; os.makedirs(OUT, exist_ok=True)
GREY, LIGHT = "#595959", "#BFBFBF"
COL = {"2N": "#6AA84F", "S": "#E69138", "4N": "#3D85C6"}
plt.rcParams.update({"font.family": "Arial", "font.size": 15, "text.color": GREY, "axes.labelcolor": GREY, "xtick.color": GREY,
                     "ytick.color": GREY, "axes.edgecolor": LIGHT, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})

# 1. what the measured clock contains: DNA state of control cells along Louise's 4i clock
o = ad.read_h5ad(NTC, backed="r").obs; rel = o["dna_reliable"].astype(str) == "True"
c = pd.read_csv(CLK, index_col=0)["cc_phase_4i"]; d = o.loc[rel & o.index.isin(c.index), "dna_rel_R234"].astype(float)
ph = c.loc[d.index].values; st = np.digitize(d.values, [.25, .65 + 1e-12]); b = np.minimum((ph * 20).astype(int), 19)
frac = np.array([[np.mean(st[b == i] == k) * 100 for k in range(3)] for i in range(20)]); x = (np.arange(20) + .5) / 20
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.stackplot(x, frac.T, colors=[COL["2N"], COL["S"], COL["4N"]], alpha=.9, labels=["2N (one genome copy)", "S (copying)", "4N (two copies)"])
for q, lab_old, lab_new in [(0, "Early G1", "G1"), (.25, "Late G1", "G1 to S"), (.5, "S", "4N: G2"), (.75, "G2/M", "after division, 2N")]:
    ax.axvline(q, color="white", lw=1.5)
    ax.text(q + .125, 104, lab_old, ha="center", va="bottom", fontsize=13, color="#A6A6A6", style="italic")
    ax.text(q + .125, 113, lab_new, ha="center", va="bottom", fontsize=14, color=GREY, weight="bold")
ax.text(-.01, 104, "first labels", ha="right", va="bottom", fontsize=12, color="#A6A6A6", style="italic")
ax.text(-.01, 113, "DNA stain says", ha="right", va="bottom", fontsize=12, color=GREY)
ax.set_xlim(0, 1); ax.set_ylim(0, 100); ax.set_xlabel("position on the 4i clock (0 to 1)"); ax.set_ylabel("% of control cells")
ax.legend(loc="lower center", bbox_to_anchor=(.5, -.42), ncol=3, frameon=False, fontsize=13)
fig.savefig(f"{OUT}/v1_clock_contents.png", bbox_inches="tight"); plt.close(fig)
print("v1: cells", len(d), "| 4N share by quarter", [round(np.mean(st[(ph >= q) & (ph < q + .25)] == 2) * 100) for q in (0, .25, .5, .75)])

# 3. predicted vs measured change in % 4N across measured knockouts
T = pd.read_csv(ALLM)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.axhline(0, color=LIGHT, lw=1); ax.axvline(0, color=LIGHT, lw=1); ax.axline((0, 0), slope=1, color=LIGHT, lw=1, ls="--")
for conf, col, s, lab in [(False, "#BFBFBF", 22, "not significant"), (True, "#3D85C6", 46, "confirmed by measured DNA")]:
    t = T[T.confirmed == conf]; ax.scatter(t.d_pct_4N_pred_hgb, t.measured_d_pct_4N, s=s, c=col, edgecolor="white", lw=.4, label=f"{lab} ({len(t)})")
r = spearmanr(T.d_pct_4N_pred_hgb, T.measured_d_pct_4N)[0]
ax.text(.03, .97, f"{len(T)} knockouts\nrank correlation {r:.2f}", transform=ax.transAxes, va="top", fontsize=15)
for g in ["MTBP", "MCM7", "POLA2", "EIF6"]:
    if g in set(T.gene):
        rr = T.set_index("gene").loc[g]; ax.annotate(g, (rr.d_pct_4N_pred_hgb, rr.measured_d_pct_4N), xytext=(6, -4), textcoords="offset points", fontsize=12)
ax.set_xlabel("predicted from phase contrast\n(change in % 4N cells)"); ax.set_ylabel("measured with the DNA stain\n(change in % 4N cells)")
ax.legend(loc="upper left", bbox_to_anchor=(0, .84), frameon=False, fontsize=12)
fig.savefig(f"{OUT}/v3_predicted_vs_measured.png", bbox_inches="tight"); plt.close(fig)
print("v3: rho", round(r, 3))

# 4. arrest-list groups against random knockouts (the cutting reference)
A = pd.read_csv(ARR)[["Gene_Symbol", "Primary_Arrest_Phase"]].rename(columns={"Gene_Symbol": "gene"})
M = T.merge(A, on="gene", how="left"); grp = {"random knockouts": M[M.set == "calibration"]}
for k, lab in [("G1 Phase", "expected G1 arrest"), ("S Phase", "expected S arrest"), ("G2/M Phase", "expected G2/M arrest")]:
    grp[lab] = M[M.Primary_Arrest_Phase == k]
gc = {"random knockouts": "#A6A6A6", "expected G1 arrest": COL["2N"], "expected S arrest": COL["S"], "expected G2/M arrest": COL["4N"]}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
rng = np.random.default_rng(0)
for ax, col, ttl in [(axs[0], "measured_d_pct_2N", "change in % 2N cells"), (axs[1], "measured_d_pct_4N", "change in % 4N cells")]:
    ax.axhline(0, color=GREY, lw=1); ax.text(3.55, 0, "non-cutting\ncontrols", va="center", fontsize=11)
    ref = np.median(grp["random knockouts"][col]); ax.axhline(ref, color="#A6A6A6", lw=1.5, ls="--")
    for i, (k, g) in enumerate(grp.items()):
        v = g[col].values; ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=30, color=gc[k], alpha=.85, edgecolor="white", lw=.4)
        ax.plot([i - .3, i + .3], [np.median(v)] * 2, color=GREY, lw=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{k}\n(n={len(g)})" for k, g in zip(["random\nknockouts", "G1\nlist", "S\nlist", "G2/M\nlist"], grp.values())], fontsize=13); ax.set_title(ttl, fontsize=16, color=GREY)
    ax.set_xlim(-.6, 3.5)
axs[0].set_ylabel("points vs non-cutting controls")
fig.tight_layout(); fig.savefig(f"{OUT}/v4_arrest_vs_random.png", bbox_inches="tight"); plt.close(fig)
for k, g in grp.items():
    print("v4:", k, len(g), "median d2N", round(g.measured_d_pct_2N.median(), 1), "d4N", round(g.measured_d_pct_4N.median(), 1))
