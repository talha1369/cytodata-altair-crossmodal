#!/usr/bin/env python
"""Figures for notebook 18 from the tables written by cycle_norm_questions.py (NULL_REPS=2000) and ko_cycle_library.py.
usage: python ko_cycle_figures.py CYCLE_NORM_DIR KO_LIBRARY_DIR OUT_DIR"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker

CN, KL, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
ARMS = ["no cycle normalisation", "regress out cycle", "reweight to common cycle distribution", "regress out SHUFFLED cycle (control)"]
COL = ["0.35", "tab:blue", "tab:green", "0.8"]

# Q1: between-group share of control morphology variance above the sampling floor
H = pd.read_csv(f"{CN}/q1_field_well_heterogeneity.csv")
SHORT = {"OPS0077 (predicted cycle, all NTC)": "OPS0077\npredicted cycle", "OPS0077 (MEASURED 4i clock, labelled NTC)": "OPS0077\nmeasured 4i clock",
         "Element baseline controls (MEASURED RNA clock)": "Element baseline\nmeasured RNA clock", "Element TNFa controls (MEASURED RNA clock)": "Element TNFa\nmeasured RNA clock"}
H["label"] = H.dataset.map(SHORT) + "\n" + H.grouping.str.replace("field (tile, >=30 cells)", "fields", regex=False) + " (" + H.n_groups.astype(str) + ")"
labels = list(dict.fromkeys(H.label)); x = np.arange(len(labels)); w = .2
fig, ax = plt.subplots(figsize=(15, 5))
for k, (arm, c) in enumerate(zip(ARMS, COL)):
    v = [H[(H.label == l) & (H.arm == arm)].excess_pct.values[0] for l in labels]
    ax.bar(x + (k - 1.5) * w, v, w, color=c, label=arm)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("% of control morphology variance\nbetween groups, above sampling noise")
ax.set_title("Q1: does normalising by the cell cycle reduce field / well heterogeneity?"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/q1_field_well_heterogeneity.png", dpi=110); plt.close()

# Q2/Q3: library-wide decomposition and heterogeneity
R = pd.read_csv(f"{KL}/ko_library_cycle.csv")
cls_col = {"cells change, cycle distribution unchanged": "tab:blue", "both": "tab:red", "cycle distribution only": "tab:orange", "neither": "0.75"}
fig, ax = plt.subplots(1, 3, figsize=(19, 5.3))
for c_, col in cls_col.items():
    r = R[R["class"] == c_]
    if len(r): ax[0].scatter(r.cycle_TV, r.within_vs_null_median, s=12, c=col, label=f"{c_} ({len(r)})", alpha=.8)
for g in R[R["class"] == "both"].nlargest(10, "cycle_TV").gene:
    r = R[R.gene == g].iloc[0]; ax[0].annotate(g, (r.cycle_TV, r.within_vs_null_median), fontsize=7)
ax[0].set_xlabel("change in predicted cycle distribution (total variation vs NTC)"); ax[0].set_ylabel("change of the cells at a matched cycle state\n(x median of NTC-guide null)")
ax[0].set_yscale("log"); ax[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[0].legend(fontsize=8); ax[0].set_title("Q2/Q3: 1,000 knockouts, cycle mix vs the cells themselves")
b = R[R["class"] != "neither"]
ax[1].hist(np.clip(b.cycle_share_of_shift, -.5, 1), bins=40, color="tab:blue")
ax[1].axvline(0, color="k", lw=.6); ax[1].set_xlabel("share of the morphology shift explained by the cycle shift")
ax[1].set_ylabel("knockouts"); ax[1].set_title(f"median {b.cycle_share_of_shift.median():.2f} (knockouts with any effect)")
ax[2].scatter(R.auc_before, R.auc_after, s=8, c=np.where(R["class"] == "both", "tab:red", "0.5"))
ax[2].plot([.45, 1], [.45, 1], "k--", lw=.7); ax[2].set_xlabel("knockout vs NTC separation (AUC), before"); ax[2].set_ylabel("after regressing out the predicted cycle")
ax[2].set_title(f"heterogeneity: spread {R.spread_change_pct.median():+.1f}%, AUC {R.auc_change.median():+.3f} (medians)")
plt.tight_layout(); plt.savefig(f"{OUT}/library_cycle_decomposition.png", dpi=110); plt.close()

# Element: measured RNA clock
E = pd.read_csv(f"{CN}/q23_element_decomposition.csv")
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8)); y = np.arange(len(E))
ax[0].barh(y, E.cycle_TV, color=np.where(E.cycle_beyond_null, "tab:red", "0.6")); ax[0].scatter(E.null_max_cycle_TV, y, marker="|", s=200, c="k", label="largest null value")
ax[0].set_yticks(y); ax[0].set_yticklabels(E.contrast, fontsize=8); ax[0].set_xlabel("change in measured RNA-clock distribution (TV)"); ax[0].legend(fontsize=8)
ax[0].set_title("Element: does the perturbation change the cycle mix?")
ax[1].barh(y - .2, E.cycle_part, .4, color="tab:orange", label="cycle-mix part"); ax[1].barh(y + .2, E.within_cycle_part, .4, color="tab:blue", label="matched-cycle part")
ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xscale("log"); ax[1].set_xlabel("morphology shift (PCA-50 distance, log scale)"); ax[1].legend(fontsize=8)
ax[1].set_title("...and is the morphology change explained by it?")
plt.tight_layout(); plt.savefig(f"{OUT}/element_cycle_decomposition.png", dpi=110); plt.close()
print("DONE")
