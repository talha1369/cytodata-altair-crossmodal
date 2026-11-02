#!/usr/bin/env python
"""Which phase-contrast -> cell-cycle model to use: ours (ridge on PCA-100) vs Brian's (HistGradientBoosting on all raw phase
features, hackathon_4i_inference_notebook_bf.ipynb) and variants, all on the same held-out folds (OPS0077 NTC, grouped by tile).

Clock targets: final cc_phase and cc_phase_dna (Brian's target), as sin/cos.
Stage target: Brian's DNA x pRb staging, applied to MEASURED DNA (dna_rel_R234; 2N < 0.25 <= S <= 0.65 < 4N) and pRb
  (Early G1 = 2N and pRb at or below its 35th percentile, Late G1 = 2N above it, S, G2M = 4N), predicted directly by a classifier.
Also: do quarters of the measured clock (the labels used in Brian's figures) match the DNA x pRb stages?
usage: python model_compare.py [OUT_DIR]"""
import os, sys, time
import numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

OUT = sys.argv[1] if len(sys.argv) > 1 else "model_compare"
os.makedirs(OUT, exist_ok=True)
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
STG = ["Early_G1", "Late_G1", "S", "G2M"]


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", ".", os.environ.get("DATA_DIR", ".")]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
Q = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")].copy()
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str); ph = ph[~ph.obs_names.duplicated()]
com = ph.obs_names[ph.obs_names.isin(Q.index)]; ph = ph[com]; Q = Q.loc[com]
Xraw = np.asarray(ph.X, np.float32); groups = ph.obs["tile_pheno"].astype(str).values
present = np.isfinite(Xraw).mean(0) >= .85
TH = {"final cc_phase": 2 * np.pi * Q["cc_phase"].astype(float).values, "cc_phase_dna": 2 * np.pi * Q["cc_phase_dna"].astype(float).values}
dna = Q["dna_rel_R234"].astype(float).values
prb = Q["z_pRb"].astype(float).values                     # pRb (z-scored mean intensity, cell_cycle_scores.csv); only its 35th percentile is used
two_n = dna < .25; thr = np.nanpercentile(prb[two_n], 35)
stage = np.select([two_n & (prb <= thr), two_n & (prb > thr), (dna >= .25) & (dna <= .65), dna > .65], STG, "NA")
ok = np.isfinite(TH["final cc_phase"]) & np.isfinite(TH["cc_phase_dna"]) & (stage != "NA")
Xraw, groups, stage, dna = Xraw[ok], groups[ok], stage[ok], dna[ok]; TH = {k: v[ok] for k, v in TH.items()}
log(f"{len(Xraw):,} cells | {Xraw.shape[1]} raw features ({int(present.sum())} with >=85% finite) | stages {pd.Series(stage).value_counts().to_dict()}")

# do quarters of the measured clocks match the DNA x pRb stages?
for k, t in TH.items():
    qs = np.array(STG)[np.minimum((np.mod(t / (2 * np.pi), 1) * 4).astype(int), 3)]
    cm = pd.crosstab(pd.Series(stage, name="DNA x pRb stage"), pd.Series(qs, name=f"quarter of measured {k}"), normalize="index").reindex(index=STG, columns=STG).round(2)
    log(f"measured {k} quarters vs measured DNA x pRb stage (row = stage, fraction of cells; agreement {np.mean(qs == stage):.2f}):\n{cm.to_string()}")
    cm.to_csv(f"{OUT}/quarters_vs_stage_{k.replace(' ', '_')}.csv")


def front_pca(Xa, Xb):
    imp = SimpleImputer(strategy="median").fit(Xa[:, present]); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xa[:, present]))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z[:, present])), -5, 5); p = PCA(100, random_state=0).fit(f(Xa))
    return p.transform(f(Xa)), p.transform(f(Xb))


def front_bf(Xa, Xb):                                     # Brian: NaN -> 0, StandardScaler, all features
    z = lambda Z: np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0); s = StandardScaler().fit(z(Xa)); return s.transform(z(Xa)), s.transform(z(Xb))


MODELS = {
    "ridge, PCA-100 (ours)": ("pca", lambda: "ridge"),
    "HistGB, raw features NaN->0 (Brian)": ("bf", lambda: HistGradientBoostingRegressor(max_iter=150, random_state=42)),
    "HistGB, raw features (>=85% finite), native NaN": ("raw", lambda: HistGradientBoostingRegressor(max_iter=150, random_state=42)),
    "HistGB, PCA-100": ("pca", lambda: HistGradientBoostingRegressor(max_iter=300, random_state=42)),
    "MLP single-target, PCA-100": ("pca", lambda: "mlp"),
}
P = {(m, k): np.full(len(Xraw), np.nan) for m in MODELS for k in TH}
SP = {m: np.empty(len(Xraw), dtype=object) for m in ["logistic, PCA-100", "HistGB classifier, raw features (>=85% finite), native NaN"]}
for fold, (a, b) in enumerate(GroupKFold(5).split(Xraw, groups=groups)):
    Ap, Bp = front_pca(Xraw[a], Xraw[b]); Ab, Bb = front_bf(Xraw[a], Xraw[b]); Ar, Br = Xraw[a][:, present], Xraw[b][:, present]
    F = {"pca": (Ap, Bp), "bf": (Ab, Bb), "raw": (Ar, Br)}
    for m, (fr, mk) in MODELS.items():
        A_, B_ = F[fr]
        for k, t in TH.items():
            if mk() == "ridge":
                s_ = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(A_, np.sin(t[a])).predict(B_); c_ = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(A_, np.cos(t[a])).predict(B_)
            elif mk() == "mlp":
                Y = np.c_[np.sin(t[a]), np.cos(t[a])]; mm = MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True, random_state=0, max_iter=300).fit(A_, Y)
                s_, c_ = mm.predict(B_).T
            else:
                s_ = mk().fit(A_, np.sin(t[a])).predict(B_); c_ = mk().fit(A_, np.cos(t[a])).predict(B_)
            P[(m, k)][b] = np.arctan2(s_, c_)
        log(f"fold {fold + 1}: {m} done")
    SP["logistic, PCA-100"][b] = LogisticRegression(C=.1, max_iter=3000).fit(Ap, stage[a]).predict(Bp)
    SP["HistGB classifier, raw features (>=85% finite), native NaN"][b] = HistGradientBoostingClassifier(max_iter=200, random_state=42).fit(Ar, stage[a]).predict(Br)


def err(t, p): return np.degrees(np.abs(np.angle(np.exp(1j * (t - p)))))


def circ_r(a, b):
    a0, b0 = a - np.angle(np.mean(np.exp(1j * a))), b - np.angle(np.mean(np.exp(1j * b)))
    return float(np.sum(np.sin(a0) * np.sin(b0)) / np.sqrt(np.sum(np.sin(a0) ** 2) * np.sum(np.sin(b0) ** 2)))


rows = []
for (m, k), p in P.items():
    e = err(TH[k], p); qs = np.array(STG)[np.minimum((np.mod(p / (2 * np.pi), 1) * 4).astype(int), 3)]
    rows.append(dict(model=m, target=k, median_err_deg=round(float(np.median(e)), 1), mean_err_deg=round(float(np.mean(e)), 1),
                     circ_r=round(circ_r(TH[k], p), 3), within_45=round(float(np.mean(e < 45)), 3),
                     quarter_vs_DNAxpRb_stage_bal_acc=round(balanced_accuracy_score(stage, qs), 3)))
C = pd.DataFrame(rows).sort_values(["target", "median_err_deg"]); C.to_csv(f"{OUT}/model_compare_clock.csv", index=False)
log("clock prediction, held-out by tile:\n" + C.to_string(index=False))
srows = []
for m, s in SP.items():
    cm = pd.crosstab(pd.Series(stage, name="true"), pd.Series(s, name="pred"), normalize="index").reindex(index=STG, columns=STG).round(2)
    srows.append(dict(model=m, balanced_accuracy=round(balanced_accuracy_score(stage, s), 3), **{f"recall_{c}": cm.loc[c, c] for c in STG}))
    cm.to_csv(f"{OUT}/stage_confusion_{m.split(',')[0].replace(' ', '_')}.csv")
S = pd.DataFrame(srows); S.to_csv(f"{OUT}/model_compare_stage.csv", index=False); log("DNA x pRb stage, predicted directly (held-out by tile):\n" + S.to_string(index=False))
log("DONE")
