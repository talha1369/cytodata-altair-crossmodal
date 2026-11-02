#!/usr/bin/env python
"""Per-knockout cell-cycle programme scores from the Altair CROP-seq DE statistics, with gene sets defined on single cells.

Gene sets: the 397 strongly cycling genes of Element (A549 single cells, measured RNA clock; crop_cycle_deconv.py), grouped by
the clock bin where each gene peaks, using what Element's bins mean (scanpy phase calls and DNA content):
  S-peak     peak at 285-15 deg     G2M-peak   peak at 45-135 deg     G1-peak   peak at 195-255 deg
Score per KO and set: mean DESeq2 Wald stat over the set (KO's own target gene excluded), as a z-score against 1,000 random
gene sets matched to the set's expression (baseMean deciles). Reliability: each set split into two random halves, z-scores
correlated across KOs (all KOs, and KOs with a strong signal).
The composition deconvolution in crop_cycle_deconv.py was not reproducible across gene halves (rho 0.05), so these programme
scores are the representation carried forward.
usage: python crop_cycle_sets.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, h5py, s3fs
from scipy.stats import spearmanr, pearsonr
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "crop_cycle"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0); NNULL = 1000

prof = pd.read_csv(f"{OUT}/element_cycling_gene_profiles.csv", index_col=0)
peak = np.argmax(np.log2(prof[[f"bin{i}" for i in range(12)]].values + .05), 1)
SETS = {"S": np.isin(peak, [9, 10, 11, 0]), "G2M": np.isin(peak, [1, 2, 3, 4]), "G1": np.isin(peak, [6, 7, 8])}
P = "ops-explorer-public/leonetti_ops/ops_data_portal_submission/CROPseq/2026_05_28_CropSeq_June2025_perturbation_lvl_ops_8-24update.h5ad"
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
with fs.open(P, "rb") as f, h5py.File(f, "r") as c:
    cv = read_elem(c["var"]); co = read_elem(c["obs"]); st = c["layers"]["stat"][:].astype(float); bm = c["layers"]["baseMean"][:].astype(float)
sym = cv["gene_symbol"].astype(str).str.upper().values; col = {s: i for i, s in enumerate(sym)}
ko = co["gene_symbol"].astype(str).str.upper().values; role = co["role"].astype(str).values
keep_rows = (role == "targeting") & (np.isfinite(st).sum(1) > 1000)
st, ko = st[keep_rows], ko[keep_rows]
expr = np.nanmean(bm, 0); dec = np.digitize(expr, np.nanpercentile(expr, np.arange(10, 100, 10)))
cycling_cols = {col[s] for s in prof.index if s in col}
pool = {d: np.array([j for j in np.where(dec == d)[0] if j not in cycling_cols]) for d in range(10)}   # null genes: non-cycling, same decile


def set_z(cols):
    """z of the mean Wald stat over cols, per KO, vs expression-matched random sets; the KO's own gene is masked"""
    cols = np.asarray(cols); S = st[:, cols].copy()
    for i, g in enumerate(ko):
        if g in col and col[g] in cols: S[i, np.where(cols == col[g])[0]] = np.nan
    obs = np.nanmean(S, 1); counts = np.bincount(dec[cols], minlength=10)
    null = np.empty((len(ko), NNULL))
    for r in range(NNULL):
        rc = np.concatenate([RNG.choice(pool[d], counts[d], replace=False) for d in range(10) if counts[d]])
        null[:, r] = np.nanmean(st[:, rc], 1)
    return (obs - null.mean(1)) / null.std(1)


R = pd.DataFrame({"gene": ko})
rel = []
for name, m in SETS.items():
    cols = np.array([col[s] for s in prof.index[m] if s in col]); R[f"z_{name}"] = set_z(cols)
    h = RNG.permutation(len(cols)); a, b = set_z(cols[h[: len(cols) // 2]]), set_z(cols[h[len(cols) // 2:]])
    strong = np.abs(R[f"z_{name}"]) > 4
    rel.append(dict(programme=name, n_genes=len(cols), split_half_pearson_all=round(pearsonr(a, b)[0], 3), split_half_spearman_all=round(spearmanr(a, b)[0], 3),
                    n_strong_KOs=int(strong.sum()), split_half_pearson_strong=round(pearsonr(a[strong], b[strong])[0], 3) if strong.sum() > 5 else np.nan))
R["z_proliferation"] = R[["z_S", "z_G2M"]].mean(1)
R["z_G2M_minus_G1"] = R.z_G2M - R.z_G1; R["z_S_minus_G1"] = R.z_S - R.z_G1
REL = pd.DataFrame(rel); REL.to_csv(f"{OUT}/crop_programme_reliability.csv", index=False)
print("programme scores: reliability (two random halves of each gene set, across KOs):\n", REL.to_string(index=False))
print("\ncorrelation between programmes across KOs (Spearman):\n", R[["z_S", "z_G2M", "z_G1"]].corr("spearman").round(2).to_string())
D = pd.read_csv(f"{OUT}/crop_cycle_representation.csv")[["gene", "d_pct_G1", "d_pct_S", "d_pct_G2M", "d_pct_4N", "cycle_TV", "q"]].rename(columns={"q": "q_deconv"})
R = R.merge(D, on="gene", how="left")
R.to_csv(f"{OUT}/crop_cycle_programmes.csv", index=False)
for k in ["z_G2M", "z_S", "z_G1"]:
    print(f"\nlowest {k}:\n", R.nsmallest(6, k)[["gene", "z_S", "z_G2M", "z_G1"]].round(2).to_string(index=False))
    print(f"highest {k}:\n", R.nlargest(6, k)[["gene", "z_S", "z_G2M", "z_G1"]].round(2).to_string(index=False))
print("\nKOs with |z| > 4 in at least one programme:", int((R[["z_S", "z_G2M", "z_G1"]].abs() > 4).any(axis=1).sum()), "of", len(R))
for g in ["CCNB1", "CDK7", "AURKB", "CDC45", "RRM1", "POLA1", "MCM7", "PLK1", "CDK1", "RPS29", "SMARCD1"]:
    if (R.gene == g).any(): print(f"  {g:8s}", R.loc[R.gene == g, ["z_S", "z_G2M", "z_G1"]].round(2).to_dict("records")[0])
print("DONE")
