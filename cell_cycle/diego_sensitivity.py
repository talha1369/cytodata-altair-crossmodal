#!/usr/bin/env python
"""Why do our per-PC reductions (build_on_team.py) come out about half of Diego's? Same readout (within-field variance of PC1-PC10,
NTC, PCA axes fit before regression), varying
  scaling     robust (10-90 quantile range) + clip at 5   |   standard z-score (NaN -> median)
  regression  linear in cc_phase (not circular) | harmonics 1-2 | 1-4 | 1-8 | 24 clock-bin means | 48 clock-bin means
  covariate   cc_phase (Diego's)  |  DNA content only (linear | cubic | 24 / 48 bin means)
usage: python diego_sensitivity.py [OUT_DIR]"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

OUT = sys.argv[1] if len(sys.argv) > 1 else "build_on_team"
os.makedirs(OUT, exist_ok=True)


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
L = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")]; L = L[~L.index.duplicated()]


def load(fn):
    a = ad.read_h5ad(find(fn))
    if "cell_key" in a.obs: key = a.obs["cell_key"].astype(str).values
    elif {"well", "segmentation_id"} <= set(a.obs.columns): key = (a.obs["well"].astype(str) + "|" + a.obs["segmentation_id"].astype("int64").astype(str)).values
    else: key = np.array(["|".join(s.split("|")[:2]) for s in a.obs_names.astype(str)])
    a.obs_names = key; a = a[~a.obs_names.duplicated()]; a = a[a.obs_names.isin(L.index)]
    X = np.asarray(a.X, np.float32); keep = np.isfinite(X).mean(0) >= .85
    return pd.DataFrame(X[:, keep], index=a.obs_names), (a.obs["tile_pheno"].astype(str) if "tile_pheno" in a.obs else None)


ph, tile = load("OPS0077_NTC_phase_OrganelleProfiler.h5ad"); fi, _ = load("OPS0077_NTC_OrganelleProfiler_pRb_Rb_p21.h5ad")
common = ph.index.intersection(fi.index); tile = tile.loc[common].values if hasattr(tile, "loc") else pd.Series(tile, index=ph.index).loc[common].values
Q = L.loc[common]; cc = Q["cc_phase"].astype(float).values; dna = Q["dna_rel_R234"].astype(float).values
EMB = {"phase contrast": ph.loc[common].values, "4i panel": fi.loc[common].values}


def harm(t, k): return np.column_stack([f(2 * np.pi * j * t) for j in range(1, k + 1) for f in (np.sin, np.cos)])


def bins(v, n, circular):
    edges = np.linspace(0, 1, n + 1) if circular else np.nanquantile(v, np.linspace(0, 1, n + 1))
    b = np.clip(np.digitize(v, edges[1:-1]), 0, n - 1); return np.eye(n)[b][:, 1:]          # one-hot, first bin as reference


def zs(v): return (v - np.nanmean(v)) / np.nanstd(v)


DES = {("cc_phase", "linear (not circular)"): cc[:, None], ("cc_phase", "harmonics 1-2"): harm(cc, 2), ("cc_phase", "harmonics 1-4"): harm(cc, 4),
       ("cc_phase", "harmonics 1-8"): harm(cc, 8), ("cc_phase", "24 bin means"): bins(cc, 24, True), ("cc_phase", "48 bin means"): bins(cc, 48, True),
       ("DNA content only", "linear"): zs(dna)[:, None], ("DNA content only", "cubic"): np.c_[zs(dna), zs(dna) ** 2, zs(dna) ** 3],
       ("DNA content only", "24 bin means"): bins(dna, 24, False), ("DNA content only", "48 bin means"): bins(dna, 48, False)}


def wfv(Z):
    D = pd.DataFrame(Z); return (D - D.groupby(tile).transform("mean")).var().values


rows = []
for en, X in EMB.items():
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    for sn, scaler in [("robust + clip (ours)", lambda A: np.clip(RobustScaler(quantile_range=(10, 90)).fit_transform(A), -5, 5)), ("standard z-score", lambda A: StandardScaler().fit_transform(A))]:
        Xs = scaler(Xi); Xs = np.nan_to_num(Xs); Z = PCA(50, random_state=0).fit_transform(Xs); v0 = wfv(Z)
        for (cov, reg), A in DES.items():
            A1 = np.c_[A, np.ones(len(A))]; B = np.linalg.lstsq(A1, Z, rcond=None)[0]; Zr = Z - A @ B[:-1]; red = 100 * (1 - wfv(Zr) / v0)
            rows.append(dict(embedding=en, scaling=sn, covariate=cov, regression=reg, PC1=round(red[0], 1), PC2=round(red[1], 1), PC3=round(red[2], 1),
                             top3_mean=round(red[:3].mean(), 1), max_of_PC1_10=round(red[:10].max(), 1), total_50_PCs=round(100 * (1 - wfv(Zr).sum() / v0.sum()), 2)))
        print(en, sn, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/diego_sensitivity.csv", index=False)
print("\nDiego's numbers: 4i PC1-3 = 28 / 26 / 33 %, phase contrast 9 / 19 / 9 %\n")
print(R.to_string(index=False))
print("DONE")
