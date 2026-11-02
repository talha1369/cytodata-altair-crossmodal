#!/usr/bin/env python
"""A joint DNA + markers cell-cycle predictor from phase-contrast morphology (OPS0077 NTC), compared with separate models.

Targets (same cells, held-out by tile, same folds and front-end as phase_predicts_regimes.py):
  clocks       4i DNA (cc_phase_dna), 4i markers (cc_phase_markers), final cc_phase    -> sin/cos
  components   DNA content, pRb, p21, phospho-Rb fraction (z-scored)
Predictors (all on PCA-100 of robust-scaled phase features):
  single        one ridge per target (the notebook-17 baseline)
  reduced-rank  one ridge for all 10 targets, fitted values projected on their top-r directions (r by inner CV)
  components    predict the 4 components jointly, then read every clock from the predicted state with a k-nearest-neighbour
                map learned on MEASURED components -> clock in the training fold (the clocks are curves through these components)
  shared MLP    one multi-output network (64 hidden units) for all 10 targets
Also: do the predicted DNA and markers clocks keep the measured relation (same direction, rotated ~142 deg)?
usage: python joint_cycle_predictor.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

OUT = sys.argv[1] if len(sys.argv) > 1 else "joint_cycle"
os.makedirs(OUT, exist_ok=True)
RA = np.logspace(-2, 5, 15)


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
Q = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")].copy()
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str)
ph = ph[~ph.obs_names.duplicated()]; com = ph.obs_names[ph.obs_names.isin(Q.index)]; ph = ph[com]; Q = Q.loc[com]
present = np.isfinite(np.asarray(ph.X)).mean(0) >= 0.85
X = np.asarray(ph.X, np.float32)[:, present]; groups = ph.obs["tile_pheno"].astype(str).values
CLK = {"4i DNA": "cc_phase_dna", "4i markers": "cc_phase_markers", "final cc_phase": "cc_phase"}
COMP = ["z_dna", "z_pRb", "z_p21", "z_phos"]
TH = {k: 2 * np.pi * Q[c].astype(float).values for k, c in CLK.items()}
Y = np.c_[Q[COMP].astype(float).values, *[np.c_[np.sin(t), np.cos(t)] for t in TH.values()]]   # 4 + 6 targets
ok = np.isfinite(Y).all(1); X, Y, groups, Q = X[ok], Y[ok], groups[ok], Q[ok]; TH = {k: v[ok] for k, v in TH.items()}
n = len(Y); print(f"{n:,} cells with all 10 targets | {X.shape[1]} features | {len(set(groups)):,} tiles", flush=True)
ys = StandardScaler().fit(Y[:, :4])
PRED = {m: np.full((n, 10), np.nan) for m in ["single", "reduced-rank", "components", "shared MLP"]}
rank_used = []

for fold, (a, b) in enumerate(GroupKFold(5).split(X, groups=groups)):
    imp = SimpleImputer(strategy="median").fit(X[a]); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(X[a]))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5)
    pca = PCA(100, random_state=0).fit(f(X[a])); A, B = pca.transform(f(X[a])), pca.transform(f(X[b]))
    # single
    for j in range(10): PRED["single"][b, j] = RidgeCV(alphas=RA).fit(A, Y[a, j]).predict(B)
    # reduced rank: inner split by tile to choose the rank
    ga = groups[a]; ia, ib = next(GroupKFold(4).split(A, groups=ga))
    alpha = RidgeCV(alphas=RA).fit(A, Y[a]).alpha_
    def rrr(Atr, Ytr, Ate, r):
        m = Ridge(alpha=alpha).fit(Atr, Ytr); F = m.predict(Atr); _, _, Vt = np.linalg.svd(F - F.mean(0), full_matrices=False)
        Pm = Vt[:r].T @ Vt[:r]; return (m.predict(Ate) - F.mean(0)) @ Pm + F.mean(0)
    sc_r = {r: np.mean((rrr(A[ia], Y[a][ia], A[ib], r) - Y[a][ib]) ** 2) for r in (2, 3, 4, 6, 8, 10)}
    r_best = min(sc_r, key=sc_r.get); rank_used.append(r_best); PRED["reduced-rank"][b] = rrr(A, Y[a], B, r_best)
    # components route
    Cb = np.c_[[RidgeCV(alphas=RA).fit(A, Y[a, j]).predict(B) for j in range(4)]].T
    knn = KNeighborsRegressor(50, weights="distance").fit(ys.transform(Y[a, :4]), Y[a, 4:])
    PRED["components"][b, :4] = Cb; PRED["components"][b, 4:] = knn.predict(ys.transform(Cb))
    # shared MLP (targets standardised)
    yt = StandardScaler().fit(Y[a]); mlp = MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True, random_state=0, max_iter=300).fit(A, yt.transform(Y[a]))
    PRED["shared MLP"][b] = yt.inverse_transform(mlp.predict(B))
    print(f"fold {fold + 1}/5 | reduced rank {r_best}", flush=True)


def ang(P, k):
    j = 4 + 2 * list(CLK).index(k); return np.arctan2(P[:, j], P[:, j + 1])


def err(t, p): return np.degrees(np.abs(np.angle(np.exp(1j * (t - p)))))


def align(x, y):
    best = max(((np.mean(np.cos(f_ * x + s_ - y)), f_, s_) for f_ in (1, -1) for s_ in np.linspace(0, 2 * np.pi, 180, endpoint=False)))
    sc_, f_, s_ = best; return f_, round(float(np.degrees(s_)) % 360), round(float(np.median(err(f_ * x + s_, y))), 1)


rows = []
for m, P in PRED.items():
    r = dict(predictor=m)
    for k in CLK: r[f"{k}: median err (deg)"] = round(float(np.median(err(TH[k], ang(P, k)))), 1)
    for j, c in enumerate(["DNA", "pRb", "p21", "phospho-Rb"]): r[f"{c} R2"] = round(r2_score(Y[:, j], P[:, j]), 3)
    fdir, rot, med = align(ang(P, "4i markers"), ang(P, "4i DNA")); r["pred markers->DNA: direction"] = fdir; r["rotation deg"] = rot; r["median err after rotation"] = med
    rows.append(r)
fdir, rot, med = align(TH["4i markers"], TH["4i DNA"])
rows.append({"predictor": "MEASURED (reference)", "pred markers->DNA: direction": fdir, "rotation deg": rot, "median err after rotation": med})
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/joint_vs_single.csv", index=False)
print("\nheld-out by tile (random guess = 90 deg; reduced ranks used per fold:", rank_used, ")")
print(T.T.to_string())
# where on the clock does joint help? error by arc of the final clock
arc = np.floor(np.degrees(TH["final cc_phase"]) % 360 / 30).astype(int)
E = pd.DataFrame({f"{m} | {k}": pd.Series(err(TH[k], ang(P, k))).groupby(arc).median() for m, P in PRED.items() for k in CLK})
E.index = [f"{30 * i}-{30 * i + 30}" for i in E.index]; E.round(1).to_csv(f"{OUT}/joint_error_by_arc.csv")
print("\nmedian error by arc of the final clock (deg):\n", E.round(0).to_string())
print("DONE")
