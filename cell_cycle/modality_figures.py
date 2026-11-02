#!/usr/bin/env python
"""Figures for notebook 19 from modality_cycle.py outputs.  usage: python modality_figures.py IN_DIR OUT_DIR"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
Q = pd.read_csv(f"{IN}/q1_score_quality.csv"); H = pd.read_csv(f"{IN}/q2_heterogeneity_by_group.csv"); S = pd.read_csv(f"{IN}/q2_heterogeneity_summary.csv")

# (1) score quality vs measured DNA state
q = Q.dropna(subset=["DNA_state_balanced_acc"]).copy()
q["circular"] = q.score.str.contains("uses DNA|DNA stain\\)|measured; uses", regex=True)
fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
for a_, ds in zip(ax, ["OPS0077", "Element"]):
    d = q[q.dataset == ds].sort_values("DNA_state_balanced_acc"); y = np.arange(len(d))
    a_.barh(y - .2, d.DNA_state_balanced_acc, .4, color=np.where(d.circular, "0.75", "tab:blue"), label="balanced accuracy, 2N / S / 4N")
    a_.barh(y + .2, d.AUC_4N_vs_2N, .4, color=np.where(d.circular, "0.55", "tab:orange"), label="AUC, 4N vs 2N")
    a_.axvline(1 / 3, color="k", ls=":", lw=.8); a_.axvline(.5, color="k", ls="--", lw=.6)
    a_.set_yticks(y); a_.set_yticklabels([s.replace(": predicted clock", " (predicted)") for s in d.score], fontsize=8); a_.set_xlim(0, 1.02)
    a_.set_title(f"{ds}: does the score separate measured DNA state?\n(grey = score built from DNA, circular)"); a_.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/modality_score_quality.png", dpi=110); plt.close()

# (2) heterogeneity before / after, per modality
h = H.dropna(subset=["spread_change_pct"]).copy(); h["cycle_used"] = h.cycle_used.fillna("own predicted clock").str.replace("modality's own", "own").str.replace("imaging's own", "own"); mods = list(dict.fromkeys(h.modality.str.replace(r" \[.*\]", "", regex=True) + " | " + h.cycle_used))
h["key"] = h.modality.str.replace(r" \[.*\]", "", regex=True) + " | " + h.cycle_used
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
for i, k in enumerate(mods):
    d = h[h.key == k]; x = i + np.random.default_rng(0).uniform(-.25, .25, len(d))
    ax[0].scatter(x, d.spread_change_pct, s=6 if len(d) > 50 else 30, alpha=.5 if len(d) > 50 else .9, c="tab:blue")
    ax[0].scatter(x, d.spread_change_shuffled_pct, s=6 if len(d) > 50 else 30, alpha=.3, c="0.6")
    ax[0].hlines(d.spread_change_pct.median(), i - .3, i + .3, color="k", lw=2)
    ax[1].scatter(d.auc_before, d.auc_after, s=6 if len(d) > 50 else 40, alpha=.5 if len(d) > 50 else .9, label=k)
ax[0].axhline(0, color="k", lw=.5); ax[0].set_xticks(range(len(mods))); ax[0].set_xticklabels([m.replace(" | ", "\n").replace("OPS0077 ", "OPS0077\n").replace("Element ", "Element\n").replace(" (pRb, Rb, p21)", "")
                       .replace(" (all features)", "").replace(" (DRVI embedding)", " (DRVI)") for m in mods], fontsize=8)
ax[0].set_ylabel("change in spread of each perturbed population (%)\nblue = cycle removed, grey = shuffled cycle"); ax[0].set_title("does removing the cycle reduce within-population heterogeneity?")
ax[1].plot([.45, 1], [.45, 1], "k--", lw=.7); ax[1].set_xlabel("separation from controls (AUC), before"); ax[1].set_ylabel("after removing the cycle")
ax[1].legend(fontsize=7); ax[1].set_title("separation from controls, before vs after")
plt.tight_layout(); plt.savefig(f"{OUT}/modality_heterogeneity.png", dpi=110); plt.close()
print("DONE")
