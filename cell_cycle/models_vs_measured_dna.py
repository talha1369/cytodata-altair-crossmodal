"""Which cycle model (ridge PCA-100 or HistGB) better predicts the MEASURED DAPI shift of the knockouts we measured?
usage: python models_vs_measured_dna.py REPORT_DIR   (reads ko_measured_vs_predicted_dna_all.csv + compare_ridge_vs_hgb.csv)
Writes models_vs_measured_dna.csv (per knockout) and prints rho/slope per model and set, and confirmation by which model calls it."""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
d = sys.argv[1]
M = pd.read_csv(f"{d}/ko_measured_vs_predicted_dna_all.csv"); C = pd.read_csv(f"{d}/compare_ridge_vs_hgb.csv")
k = M.merge(C[["gene", "d_pct_4N_pred_ridge", "d_pct_4N_pred_hgb", "sig_ridge", "sig_hgb"]], on="gene", how="left")
k["called_by"] = np.select([k.sig_ridge & k.sig_hgb, k.sig_hgb & ~k.sig_ridge, k.sig_ridge & ~k.sig_hgb], ["both", "hgb_only", "ridge_only"], "neither")
for m in ["ridge", "hgb"]:
    p = k[f"d_pct_4N_pred_{m}"]
    print(f"{m}: rho {spearmanr(p, k.measured_d_pct_4N)[0]:.3f} slope {np.polyfit(p, k.measured_d_pct_4N, 1)[0]:.2f} n {len(k)} | "
          + " ".join(f"{s} {spearmanr(g[f'd_pct_4N_pred_{m}'], g.measured_d_pct_4N)[0]:.2f}" for s, g in k.groupby("set")))
for c, g in k.groupby("called_by"):
    print(f"{c}: n {len(g)} confirmed {int(g.measured_significant.sum())} "
          f"hgb sign agrees {np.mean(np.sign(g.d_pct_4N_pred_hgb) == np.sign(g.measured_d_pct_4N)):.2f}")
print("hgb_only in library:", int((C.sig_hgb & ~C.sig_ridge).sum()), "| measured:", int((k.called_by == "hgb_only").sum()))
k.to_csv(f"{d}/models_vs_measured_dna.csv", index=False)
