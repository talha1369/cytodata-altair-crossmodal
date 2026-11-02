#!/usr/bin/env python
"""Map the CROP-seq cell-cycle representation (crop_cycle_sets.py programme scores; crop_cycle_deconv.py composition as a
secondary) to phase-contrast morphology (ko_cycle_library.py),
knockout by knockout, across the 1,000 Altair knockouts (CROP-seq cells and imaged cells are different cells; the KO is the link).

Route 1  per cell, no RNA used for training: the NTC-trained morphology models predict each KO cell's cycle state; the KO's
         predicted change in % 4N cells is compared with its RNA S / G2M / G1 programme scores.
Route 2  gene level, trained on RNA: ridge from the KO's mean phase profile (PCA-100, minus NTC) to its RNA cycle representation,
         10-fold cross-validated across KOs, with a permuted-label null.
Strong essential-gene KOs move every readout, so agreement is also reported after partialling out overall perturbation
strength in each modality (morphology: total centroid shift; RNA: mean |DESeq2 stat| over all genes).
usage: python map_crop_to_phase.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, h5py, s3fs
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "crop_to_phase"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)

C = pd.read_csv("crop_cycle/crop_cycle_programmes.csv"); C["z_max_abs"] = C[["z_S", "z_G2M", "z_G1"]].abs().max(1)
M = pd.read_csv("ko_library/ko_library_cycle.csv")
P = "ops-explorer-public/leonetti_ops/ops_data_portal_submission/CROPseq/2026_05_28_CropSeq_June2025_perturbation_lvl_ops_8-24update.h5ad"
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
with fs.open(P, "rb") as f, h5py.File(f, "r") as c:
    co = read_elem(c["obs"]); st = c["layers"]["stat"][:].astype(float)
C = C.merge(pd.DataFrame({"gene": co["gene_symbol"].astype(str).str.upper().values, "rna_strength": np.nanmean(np.abs(st), 1)}), on="gene", how="left")
J = C.merge(M, on="gene", suffixes=("_rna", "_morph"))
print(f"knockouts in both CROP-seq and OPS0077: {len(J)}", flush=True)


def partial_spearman(x, y, covs):
    """Spearman of x and y after regressing the ranks of the covariates out of both ranks"""
    R = lambda v: rankdata(v); Z = np.c_[np.ones(len(x)), *[R(c) for c in covs]]
    rx = R(x) - Z @ np.linalg.lstsq(Z, R(x), rcond=None)[0]; ry = R(y) - Z @ np.linalg.lstsq(Z, R(y), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


rows = []
pairs = [("RNA G2M programme z  vs  morphology-predicted d%4N", "z_G2M", "d_pct_4N_pred"),
         ("RNA proliferation z (S+G2M)  vs  morphology-predicted d%4N", "z_proliferation", "d_pct_4N_pred"),
         ("RNA G1 programme z  vs  morphology-predicted d%4N", "z_G1", "d_pct_4N_pred"),
         ("RNA strongest programme |z|  vs  morphology cycle shift (TV)", "z_max_abs", "cycle_TV_morph"),
         ("RNA proliferation z  vs  measured pRb (4i)", "z_proliferation", "measured_pRb_shift_log2"),
         ("RNA composition deconvolution d%4N (secondary)  vs  morphology d%4N", "d_pct_4N", "d_pct_4N_pred")]
for name, a, b in pairs:
    d = J[[a, b, "total_shift", "rna_strength"]].dropna()
    rho, p = spearmanr(d[a], d[b])
    rows.append(dict(comparison=name, n=len(d), spearman=round(rho, 3), p=float(f"{p:.2g}"),
                     partial_spearman_given_strength=round(partial_spearman(d[a].values, d[b].values, [d.total_shift.values, d.rna_strength.values]), 3)))
strong = J[J.z_max_abs > 4]
if len(strong):
    rows.append(dict(comparison="KOs with a strong RNA programme shift (|z|>4): % also significant in morphology (cycle or 4N, FDR 5%)", n=len(strong),
                     spearman=np.nan, p=np.nan, partial_spearman_given_strength=np.nan,
                     fraction=round(float(((strong.q_cycle < .05) | (strong.q_4N < .05)).mean()), 3)))
    rows.append(dict(comparison="same, all other KOs", n=int((J.z_max_abs <= 4).sum()), spearman=np.nan, p=np.nan, partial_spearman_given_strength=np.nan,
                     fraction=round(float(((J.q_cycle < .05) | (J.q_4N < .05))[J.z_max_abs <= 4].mean()), 3)))
S1 = pd.DataFrame(rows); print(S1.to_string(index=False), flush=True)

# ------------------------------------------------ route 2: gene-level ridge from the mean phase profile
z = np.load("ko_library/ko_library_mean_profiles.npz", allow_pickle=True); prof = pd.DataFrame(z["profiles"], index=z["genes"])
G = J.set_index("gene"); X = prof.loc[G.index].values
targets = ["z_S", "z_G2M", "z_G1", "z_proliferation", "d_pct_4N"]


def cv_pred(X, y):
    out = np.zeros(len(y))
    for a, b in KFold(10, shuffle=True, random_state=0).split(X):
        out[b] = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(X[a], y[a]).predict(X[b])
    return out


rows2 = []; preds = {}
for t in targets:
    y = G[t].values.astype(float); ok = np.isfinite(y); yp = cv_pred(X[ok], y[ok]); preds[t] = (y[ok], yp)
    r2 = 1 - np.sum((y[ok] - yp) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2); rho = spearmanr(y[ok], yp)[0]
    null = [spearmanr(yy := RNG.permutation(y[ok]), cv_pred(X[ok], yy))[0] for _ in range(50)]
    rows2.append(dict(target=t, n=int(ok.sum()), cv_R2=round(r2, 3), cv_spearman=round(rho, 3), null_95th=round(float(np.percentile(null, 95)), 3),
                      partial_spearman_given_strength=round(partial_spearman(y[ok], yp, [G.total_shift.values[ok], G.rna_strength.values[ok]]), 3)))
S2 = pd.DataFrame(rows2); print("\ngene-level route (phase profile -> RNA cycle representation, 10-fold CV across KOs):\n", S2.to_string(index=False), flush=True)
S1.to_csv(f"{OUT}/route1_per_cell_agreement.csv", index=False); S2.to_csv(f"{OUT}/route2_gene_level_cv.csv", index=False)
J.to_csv(f"{OUT}/crop_vs_phase_by_ko.csv", index=False)

cols = ["gene", "z_S", "z_G2M", "z_G1", "cycle_TV_morph", "q_cycle", "d_pct_4N_pred", "q_4N", "auc_before", "auc_after", "measured_pRb_shift_log2"]
print("\nKOs with the strongest RNA cycle programme shift, and what phase morphology says:\n", J.sort_values("z_max_abs", ascending=False).head(15)[cols].round(3).to_string(index=False))
print("\nKOs with the strongest morphology cycle shift, and what CROP-seq RNA says:\n", J[J.q_cycle < .05].sort_values("cycle_TV_morph", ascending=False).head(15)[cols].round(3).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
st_ = J.z_max_abs > 4
ax[0].scatter(J.loc[~st_, "z_proliferation"], J.loc[~st_, "d_pct_4N_pred"], s=10, c="0.7", label="RNA programmes |z|<=4")
ax[0].scatter(J.loc[st_, "z_proliferation"], J.loc[st_, "d_pct_4N_pred"], s=18, c="tab:red", label="RNA programme |z|>4")
for g_ in pd.concat([J.loc[st_, "gene"], J.nlargest(4, "cycle_TV_morph").gene]).unique():
    r_ = J[J.gene == g_].iloc[0]; ax[0].annotate(g_, (r_.z_proliferation, r_.d_pct_4N_pred), fontsize=7)
ax[0].axhline(0, color="k", lw=.5); ax[0].axvline(0, color="k", lw=.5); ax[0].legend(fontsize=8)
ax[0].set_xlabel("CROP-seq proliferation programme z (S + G2M)"); ax[0].set_ylabel("change in % 4N predicted from phase morphology")
ax[0].set_title(f"route 1, per cell (rho {S1.spearman.iloc[1]:.2f})")
y_, yp_ = preds["z_proliferation"]; ax[1].scatter(y_, yp_, s=10, c="0.4")
ax[1].set_xlabel("CROP-seq proliferation programme z"); ax[1].set_ylabel("cross-validated prediction from phase profile")
ax[1].set_title(f"route 2, gene level (rho {S2.cv_spearman.iloc[3]:.2f})")
ax[2].barh(S2.target, S2.cv_spearman, color="tab:blue", label="cross-validated rho"); ax[2].barh(S2.target, S2.null_95th, color="none", edgecolor="k", label="permuted 95th")
ax[2].set_xlabel("Spearman, phase profile -> RNA cycle representation"); ax[2].legend(fontsize=8); ax[2].set_title("which parts of the RNA cycle state phase reads")
plt.tight_layout(); plt.savefig(f"{OUT}/crop_to_phase_overview.png", dpi=105)
print("DONE")
