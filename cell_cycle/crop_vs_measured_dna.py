#!/usr/bin/env python
"""Open items of the hackathon doc, built on measured DNA for 269 knockouts.
1. A cell-cycle representation from CROP-seq expression: do the knockout-level cycle-programme scores (S, G2/M, G1 gene sets,
   their differences, and the deconvolved % 4N) track the MEASURED change in % 4N and % 2N? Compared with the phase-contrast
   prediction on the same knockouts, and whether RNA adds to phase contrast (5-fold cross-validated R2 of linear models).
2. Brian's "G1 restriction point" group of the arrest list: measured DNA for each gene.
usage: python crop_vs_measured_dna.py ALL_MEASURED_CSV CROP_PROGRAMMES_CSV ARREST_LIST_CSV OUT_DIR"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

ALLM, CROP, ARR, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
T = pd.read_csv(ALLM); P = pd.read_csv(CROP)
D = T.merge(P, on="gene", how="inner")
print(f"{len(D)} of {len(T)} measured knockouts have CROP-seq programme scores")
rows = []
for c in ["z_S", "z_G2M", "z_G1", "z_G2M_minus_G1", "z_S_minus_G1", "z_proliferation", "d_pct_4N", "d_pct_4N_pred_hgb", "d_pct_4N_pred_ridge"]:
    for tgt in ["measured_d_pct_4N", "measured_d_pct_2N"]:
        r, p = spearmanr(D[c], D[tgt], nan_policy="omit"); rows.append(dict(predictor=c, target=tgt, spearman=r, p=p, n=int(D[[c, tgt]].dropna().shape[0])))
R = pd.DataFrame(rows); R.round(4).to_csv(f"{OUT}/crop_programmes_vs_measured_dna.csv", index=False)
print(R.pivot(index="predictor", columns="target", values="spearman").round(3).to_string())

cv = KFold(5, shuffle=True, random_state=0); y = D.measured_d_pct_4N.values
def r2(cols):
    X = D[cols].values; ok = np.isfinite(X).all(1) & np.isfinite(y)
    pr = cross_val_predict(LinearRegression(), X[ok], y[ok], cv=cv); return 1 - np.sum((y[ok] - pr) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
rna = ["z_S", "z_G2M", "z_G1", "z_proliferation"]
M = pd.DataFrame([dict(model="phase contrast (gradient boosting)", cv_r2=r2(["d_pct_4N_pred_hgb"])),
                  dict(model="CROP-seq programmes", cv_r2=r2(rna)),
                  dict(model="phase contrast + CROP-seq", cv_r2=r2(["d_pct_4N_pred_hgb"] + rna))])
M.round(4).to_csv(f"{OUT}/cv_r2_measured_4N.csv", index=False); print("\n5-fold cross-validated R2 for measured change in % 4N:\n", M.round(3).to_string(index=False))
conf = D[D.confirmed]
for c in ["z_S_minus_G1", "z_G2M_minus_G1"]:
    r, p = spearmanr(conf[c], conf.measured_d_pct_4N); print(f"confirmed knockouts only (n {len(conf)}): {c} vs measured d%4N rho {r:.2f} p {p:.3f}")

A = pd.read_csv(ARR); g1 = A[A.Predicted_Mechanism.astype(str).str.contains("G1/S Restriction", case=False)]
G = g1[["Gene_Symbol", "Gene_Name"]].rename(columns={"Gene_Symbol": "gene"}).merge(T, on="gene", how="left")
G = G.merge(P[["gene", "z_G1", "z_S_minus_G1"]], on="gene", how="left")
cols = ["gene", "Gene_Name", "n_measured", "measured_d_pct_2N", "measured_d_pct_S", "measured_d_pct_4N", "q_2N_pooled", "q_4N_pooled", "confirmed", "z_G1"]
G[cols].round(3).to_csv(f"{OUT}/g1_restriction_point_measured_dna.csv", index=False)
print(f"\nBrian's G1 restriction point group: {len(G)} genes, {int(G.n_measured.notna().sum())} measured")
print(G[cols].round(2).to_string(index=False))
m = G[G.n_measured.notna()]
print(f"gain 2N cells: {int((m.measured_d_pct_2N > 0).sum())} of {len(m)}; confirmed: {int(m.confirmed.sum())}; median d%2N {m.measured_d_pct_2N.median():+.1f}, d%4N {m.measured_d_pct_4N.median():+.1f}")
