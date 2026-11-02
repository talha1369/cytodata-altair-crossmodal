#!/usr/bin/env python
"""Does the NTC reference in hackathon_4i_inference_notebook_bf.ipynb carry an in-sample bias?

Replicates that notebook's model (StandardScaler on phase features with NaN -> 0, MultiOutput HistGradientBoostingRegressor
max_iter=150 on cos/sin of cc_phase_dna, all NTC cells with a DNA clock as training data) and compares quarter compositions
("Early_G1", "Late_G1", "S", "G2M" = quarters of cc_phase) for
  NTC predicted in-sample (as in the notebook: the trained model scores the NTC cells it was trained on)
  NTC predicted out of fold (folds grouped by imaging tile; unlabelled NTC cells always out of sample)
and the CCNB1 / POLA1 shifts against each reference. usage: python check_bf_insample.py [OUT_DIR]"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold

OUT = sys.argv[1] if len(sys.argv) > 1 else "bf_check"
os.makedirs(OUT, exist_ok=True)
ST = ["Early_G1", "Late_G1", "S", "G2M"]


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


def model(): return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, random_state=42))


def prep(Z): return np.nan_to_num(np.asarray(Z, np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def phase(p): return (np.arctan2(p[:, 1], p[:, 0]) % (2 * np.pi)) / (2 * np.pi)


def comp(x): return pd.Series(100 * np.bincount(np.minimum((np.mod(x, 1) * 4).astype(int), 3), minlength=4) / len(x), index=ST)


L = pd.read_csv(find("cell_cycle_scores.csv")).dropna(subset=["cc_phase_dna"]).drop_duplicates("cell_key").set_index("cell_key")
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str); ph = ph[~ph.obs_names.duplicated()]
X = prep(ph.X); lab = ph.obs_names.isin(L.index); tile = ph.obs["tile_pheno"].astype(str).values
t = 2 * np.pi * L.loc[ph.obs_names[lab], "cc_phase_dna"].astype(float).values; Y = np.c_[np.cos(t), np.sin(t)]
print(f"NTC {len(X):,} cells, {lab.sum():,} with a DNA clock", flush=True)

sc = StandardScaler().fit(X[lab]); full = model().fit(sc.transform(X[lab]), Y)
ins = phase(full.predict(sc.transform(X)))                       # the notebook's NTC reference
oof = np.full(len(X), np.nan); li = np.where(lab)[0]
for k, (a, b) in enumerate(GroupKFold(5).split(X, groups=tile)):
    tr = a[lab[a]]; s_ = StandardScaler().fit(X[tr]); pos = np.searchsorted(li, tr)
    oof[b] = phase(model().fit(s_.transform(X[tr]), Y[pos]).predict(s_.transform(X[b]))); print(f"fold {k + 1}/5", flush=True)
meas = t / (2 * np.pi)
err = lambda p: float(np.median(np.abs((p - meas + .5) % 1 - .5)) * 360)
print(f"median error on labelled NTC: in-sample {err(ins[lab]):.1f} deg | out-of-fold {err(oof[lab]):.1f} deg", flush=True)

ko = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r"); g = ko.obs["gene"].astype(str).str.upper().values
rows = {"measured clock, labelled NTC": comp(meas), "NTC in-sample (notebook reference)": comp(ins), "NTC out-of-fold": comp(oof),
        "NTC in-sample, labelled cells only": comp(ins[lab]), "NTC out-of-fold, labelled cells only": comp(oof[lab])}
for gene in ["CCNB1", "POLA1", "CDK7"]:
    idx = np.where(g == gene)[0]; Xk = prep(ko[idx].to_memory().X); rows[f"{gene} KO (n={len(idx)})"] = comp(phase(full.predict(sc.transform(Xk))))
T = pd.DataFrame(rows).T.round(1); print("\n% of cells per quarter:\n", T.to_string(), flush=True)
D = []
for gene in ["CCNB1", "POLA1", "CDK7"]:
    k_ = [r for r in rows if r.startswith(gene)][0]
    D.append(pd.Series(rows[k_] - rows["NTC in-sample (notebook reference)"], name=f"{gene} vs in-sample NTC (notebook)"))
    D.append(pd.Series(rows[k_] - rows["NTC out-of-fold"], name=f"{gene} vs out-of-fold NTC (fair)"))
D = pd.DataFrame(D).round(1); print("\nshift, percentage points:\n", D.to_string())
T.to_csv(f"{OUT}/bf_reference_compositions.csv"); D.to_csv(f"{OUT}/bf_knockout_shifts.csv")
print("DONE")
