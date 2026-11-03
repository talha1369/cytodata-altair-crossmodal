#!/usr/bin/env python
"""Does a phase-contrast -> DNA model trained on NTC cells read knockout cells as well as it reads held-out NTC cells?
Target = measured 4N vs 2N (DNA log2 rel. 2N peak: 2N < 0.25, 4N > 0.65; S cells left out of the classification metrics).
Models (same as model_compare.py): ridge/logistic on PCA-100 of in-fold imputed + robust-scaled features (ours) and HistGB on
the raw features with NaN->0 (Brian's). NTC: 5-fold GroupKFold by tile (held-out). Knockouts: model fit on all NTC.
Reports per population: AUC 4N vs 2N, predicted vs measured %4N (calibration), and per-gene shift vs NTC.
Light mode for the laptop: N_NTC random reliable-DNA NTC cells (default 10,000) and only the features finite in >= 85% of
them (HistGB is unchanged by this in model_compare.py: 21.8 vs 21.7 deg), so models are somewhat weaker than the full ones;
the NTC-vs-knockout comparison is still like for like.
usage: python ko_transfer_check.py NTC_H5AD KO_H5AD OUT_DIR [N_NTC]"""
import os, sys, time
import numpy as np, pandas as pd, anndata as ad, h5py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

NTC, KO, OUT = sys.argv[1:4]; N_NTC = int(sys.argv[4]) if len(sys.argv) > 4 else 10000; os.makedirs(OUT, exist_ok=True)
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def load(path, n=None):
    o = ad.read_h5ad(path, backed="r").obs.copy()
    keep = np.flatnonzero((o.dna_reliable.astype(str) == "True").values & np.isfinite(o.dna_content_log2.astype(float).values))
    if n and n < len(keep): keep = np.sort(np.random.default_rng(0).choice(keep, n, replace=False))
    with h5py.File(path, "r") as f:
        D = f["obsm/X_phase_raw"]; X = np.vstack([D[keep[i:i + 2000]] for i in range(0, len(keep), 2000)]).astype(np.float32)
    return o.iloc[keep].reset_index(), X


on, Xn = load(NTC, N_NTC); ok, Xk = load(KO)
pres = np.isfinite(Xn).mean(0) >= .85; Xn, Xk = Xn[:, pres], Xk[:, pres]; pres = np.ones(Xn.shape[1], bool)
dn, dk = on.dna_content_log2.astype(float).values, ok.dna_content_log2.astype(float).values
yn, yk = (dn > .65).astype(int), (dk > .65).astype(int)
log(f"NTC reliable-DNA cells used {len(on):,} (4N {yn.mean():.3f}) | knockout cells {len(ok):,} (4N {yk.mean():.3f}) | features {Xn.shape[1]}")
groups = on.tile_pheno.astype(str).values


def ridge_fit(Xtr, ytr):
    imp = SimpleImputer(strategy="median").fit(Xtr[:, pres]); rs = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xtr[:, pres]))
    g = lambda X: np.clip(rs.transform(imp.transform(X[:, pres])), -5, 5); pca = PCA(100, random_state=0).fit(g(Xtr))
    m = LogisticRegression(C=.1, max_iter=3000).fit(pca.transform(g(Xtr)), ytr)
    return lambda X: m.predict_proba(pca.transform(g(X)))[:, 1]


def hgb_fit(Xtr, ytr):
    z0 = lambda X: np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    m = HistGradientBoostingClassifier(max_iter=150, random_state=42).fit(z0(Xtr), ytr)
    return lambda X: m.predict_proba(z0(X))[:, 1]


P = {}
for name, fit in [("ridge", ridge_fit), ("hgb", hgb_fit)]:
    oof = np.full(len(yn), np.nan)
    for tr, te in GroupKFold(5).split(Xn, yn, groups): oof[te] = fit(Xn[tr], yn[tr])(Xn[te])
    P[name] = (oof, fit(Xn, yn)(Xk)); log(f"{name} done")

rows, genes = [], []
two_four_n, two_four_k = (dn < .25) | (dn > .65), (dk < .25) | (dk > .65)
for name, (pn, pk) in P.items():
    thr = np.quantile(pn, 1 - yn.mean())  # threshold that calls the NTC 4N fraction on held-out NTC
    rows.append(dict(model=name, population="NTC (held-out)", n=len(yn), auc_4N_vs_2N=roc_auc_score(yn[two_four_n], pn[two_four_n]),
                     measured_pct_4N=100 * yn.mean(), predicted_pct_4N=100 * (pn > thr).mean(), mean_prob=pn.mean()))
    for gname, gi in [("all knockouts", np.ones(len(yk), bool))] + [(g, (ok.gene == g).values) for g in sorted(ok.gene.unique())]:
        sel = two_four_k & gi
        rows.append(dict(model=name, population=gname, n=int(gi.sum()), auc_4N_vs_2N=roc_auc_score(yk[sel], pk[sel]) if yk[sel].min() < yk[sel].max() else np.nan,
                         measured_pct_4N=100 * yk[gi].mean(), predicted_pct_4N=100 * (pk[gi] > thr).mean(), mean_prob=pk[gi].mean()))
R = pd.DataFrame(rows)
ntc_ref = R[R.population == "NTC (held-out)"].set_index("model")
R["measured_shift_vs_NTC"] = R.measured_pct_4N - R.model.map(ntc_ref.measured_pct_4N)
R["predicted_shift_vs_NTC"] = R.predicted_pct_4N - R.model.map(ntc_ref.predicted_pct_4N)
R.round(4).to_csv(f"{OUT}/ko_transfer_check.csv", index=False)
pd.DataFrame({"cell_key": ok.iloc[:, 0], "gene": ok.gene, "dna": dk, "p4N_ridge": P["ridge"][1], "p4N_hgb": P["hgb"][1]}).to_csv(f"{OUT}/ko_transfer_percell.csv", index=False)
print(R.round(3).to_string(index=False))
