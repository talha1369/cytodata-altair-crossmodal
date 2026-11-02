#!/usr/bin/env python
"""Do the predicted cell-cycle shifts of the 60 likely-arrest knockouts follow their expected arrest phase?

Expected phase: column Primary_Arrest_Phase of the arrest list ("Genes expected to affect cell cycle.csv", sheet 2 of the
summary workbook): G1 (26 genes), S (9), G2/M (25). Expectation: G1 arrest -> fewer 4N cells; S or G2/M arrest -> more.
Readout: predicted change in % 4N cells per knockout (ko_library_cycle.csv), compared with the rest of the library
(Mann-Whitney). usage: python arrest_phase_test.py ARREST_LIST.csv KO_LIBRARY_CYCLE.csv OUT_DIR"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

A_, K_, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
A = pd.read_csv(A_); A["gene"] = A.Gene_Symbol.astype(str).str.upper()
R = pd.read_csv(K_); M = A.merge(R, on="gene", how="left"); lib = R[~R.gene.isin(M.gene)]
rows = []
for ph in ["G1 Phase", "S Phase", "G2/M Phase"]:
    v = M[M.Primary_Arrest_Phase == ph]
    rows.append(dict(expected_arrest_phase=ph, n=len(v), expected_4N_change="down" if ph == "G1 Phase" else "up",
                     median_predicted_d_pct_4N=round(v.d_pct_4N_pred.median(), 2), fraction_4N_up=round(float((v.d_pct_4N_pred > 0).mean()), 2),
                     n_significant_cycle_shift=int((v.q_cycle < .05).sum()), p_vs_rest_of_library=float(f"{mannwhitneyu(v.d_pct_4N_pred, lib.d_pct_4N_pred).pvalue:.2g}")))
rows.append(dict(expected_arrest_phase="rest of the library", n=len(lib), median_predicted_d_pct_4N=round(lib.d_pct_4N_pred.median(), 2),
                 fraction_4N_up=round(float((lib.d_pct_4N_pred > 0).mean()), 2), n_significant_cycle_shift=int((lib.q_cycle < .05).sum())))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/arrest_phase_test.csv", index=False); print(T.to_string(index=False))
M[["gene", "Gene_Name", "Primary_Arrest_Phase", "n", "q_cycle", "d_pct_4N_pred", "measured_pRb_shift_log2"]].sort_values(["Primary_Arrest_Phase", "d_pct_4N_pred"]).round(3).to_csv(f"{OUT}/arrest_phase_by_gene.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
groups = [("rest of the library", lib), *[(ph, M[M.Primary_Arrest_Phase == ph]) for ph in ["G1 Phase", "S Phase", "G2/M Phase"]]]
for i, (nm, d) in enumerate(groups):
    x = i + np.random.default_rng(0).uniform(-.18, .18, len(d)); ax.scatter(x, d.d_pct_4N_pred, s=8 if i == 0 else 22, c="0.75" if i == 0 else ["tab:blue", "tab:green", "tab:red"][i - 1], alpha=.6 if i == 0 else .9)
    ax.hlines(d.d_pct_4N_pred.median(), i - .3, i + .3, color="k", lw=2)
    if i:
        for _, r in d[d.q_cycle < .05].iterrows(): ax.annotate(r.gene, (i + .2, r.d_pct_4N_pred), fontsize=7)
ax.axhline(0, color="k", lw=.5); ax.set_xticks(range(4)); ax.set_xticklabels([f"{n}\n(n={len(d)})" for n, d in groups])
ax.set_ylabel("predicted change in % 4N cells (phase contrast)"); ax.set_title("arrest list: predicted 4N change by expected arrest phase")
plt.tight_layout(); plt.savefig(f"{OUT}/arrest_phase_test.png", dpi=110)
