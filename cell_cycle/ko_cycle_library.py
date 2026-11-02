#!/usr/bin/env python
"""Cell-cycle normalisation across ALL OPS0077 knockouts (1,000 genes, phase-contrast morphology).

Models trained on NTC cells only (phase PCA-100, folds grouped by tile; all NTC cells are the reference, out-of-fold):
  cycle angle    ridge -> sin/cos of the 4i cc_phase
  4N class       logistic -> measured DNA content above the 2N/4N midpoint (dna_rel_R234 > 0.65), a dataset-independent currency
Per knockout:
  cycle_TV, d_pct_4N_pred         predicted change in cycle distribution and in % 4N cells vs NTC
  cycle_part / within_cycle_part  decomposition of the morphology shift (cycle_norm_questions.py), null = NTC guides pooled to n
  spread / AUC before vs after    regressing the predicted cycle out (harmonics 1-2, coefficients fit on NTC), + shuffled control
  measured_pRb_shift_log2         4i pRb on the same knockout cells (independent validation)
Writes ko_library_cycle.csv and ko_library_mean_profiles.npz (per-KO mean PCA profile, for the gene-level route).
usage: python ko_cycle_library.py [OUT_DIR]
"""
import os, sys, time
import numpy as np, pandas as pd, anndata as ad, h5py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

OUT = sys.argv[1] if len(sys.argv) > 1 else "ko_library"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
NBIN = 12; REPS = int(os.environ.get("NULL_REPS", 2000)); T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


def fit_transform(Xtr, k=100):
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xtr))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5)
    pca = PCA(k, random_state=0).fit(f(Xtr))
    return lambda Z: np.vstack([pca.transform(f(Z[i:i + 50000])) for i in range(0, len(Z), 50000)]).astype(np.float32)


def fit_angle(Z, t):
    rs = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.sin(t)); rc = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.cos(t))
    return lambda W: np.mod(np.arctan2(rs.predict(W), rc.predict(W)), 2 * np.pi) / (2 * np.pi)


def fit_4n(Z, y):
    m = LogisticRegression(C=.1, max_iter=3000).fit(Z, y); return lambda W: m.predict_proba(W)[:, 1]


def harm(p):
    t = 2 * np.pi * p; return np.c_[np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)]


def bins(p): return np.floor(np.mod(p, 1) * NBIN).astype(int) % NBIN


def hist(p):
    h = np.bincount(bins(p), minlength=NBIN).astype(float); return h / h.sum()


def decompose(Zg, pg, Zr, pr):
    D = Zg.mean(0) - Zr.mean(0); hs, ht = hist(pr), hist(pg); r = np.where(hs > 0, ht / np.maximum(hs, 1e-12), 0)
    mrw = np.average(Zr, axis=0, weights=r[bins(pr)]); C = mrw - Zr.mean(0); W = Zg.mean(0) - mrw
    return dict(cycle_TV=float(.5 * np.abs(ht - hs).sum()), total_shift=float(np.linalg.norm(D)), cycle_part=float(np.linalg.norm(C)),
                within_cycle_part=float(np.linalg.norm(W)), cycle_share_of_shift=float(C @ D / max(D @ D, 1e-12)))


def bh(pv):
    pv = np.asarray(pv, float); o = np.argsort(pv); m = len(pv); q = np.empty(m)
    q[o] = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]; return np.minimum(q, 1)


# ------------------------------------------------ NTC models
L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
L = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")]
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str)
ph = ph[~ph.obs_names.duplicated()]
lab = ph.obs_names.isin(L.index)
present = np.isfinite(np.asarray(ph.X)[lab]).mean(0) >= 0.85
Xn = np.asarray(ph.X, np.float32)[:, present]
tile = ph.obs["tile_pheno"].astype(str).values; guide = ph.obs["sgRNA"].astype(str).values
L = L.loc[ph.obs_names[lab]]; th = 2 * np.pi * L["cc_phase"].astype(float).values; y4 = (L["dna_rel_R234"].astype(float).values > .65).astype(int)
oof = np.full(len(Xn), np.nan); oof4 = np.full(len(Xn), np.nan); li = np.where(lab)[0]; pos = {c: i for i, c in enumerate(li)}
for a, b in GroupKFold(5).split(Xn, groups=tile):
    tr = a[lab[a]]; tri = [pos[i] for i in tr]; T_ = fit_transform(Xn[tr]); Ztr = T_(Xn[tr]); Zb = T_(Xn[b])
    oof[b] = fit_angle(Ztr, th[tri])(Zb); oof4[b] = fit_4n(Ztr, y4[tri])(Zb)
T = fit_transform(Xn[lab]); Zlab = T(Xn[lab]); P = fit_angle(Zlab, th); P4 = fit_4n(Zlab, y4); Zn = T(Xn)
auc4 = roc_auc_score(y4, oof4[lab])
log(f"NTC {len(Zn):,} cells ({lab.sum():,} labelled) | out-of-fold 4N AUC {auc4:.3f} | NTC % 4N measured {100 * y4.mean():.1f}, "
    f"predicted (all NTC, out-of-fold) {100 * np.nanmean(oof4):.1f}")
B = np.linalg.lstsq(np.c_[harm(oof), np.ones(len(oof))], Zn, rcond=None)[0][:4]
psh = RNG.permutation(oof); Bs = np.linalg.lstsq(np.c_[harm(psh), np.ones(len(psh))], Zn, rcond=None)[0][:4]
Zn_c, Zn_s = Zn - harm(oof) @ B, Zn - harm(psh) @ Bs

# ------------------------------------------------ all knockout cells, transformed in chunks
f = h5py.File("OPS0077_allko_phase.h5ad", "r"); ko_obs = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r").obs.copy()
Xd = f["X"]; Zk = np.vstack([T(Xd[i:i + 50000][:, present]) for i in range(0, Xd.shape[0], 50000)]); f.close()
pk = P(Zk); p4k = P4(Zk); gene = ko_obs["gene"].astype(str).str.upper().values
log(f"knockout cells {len(Zk):,} | genes {len(np.unique(gene))}")
Zk_c = Zk - harm(pk) @ B; Zk_s = Zk - harm(RNG.permutation(pk)) @ Bs

# ------------------------------------------------ nulls from NTC guides pooled to n cells
gidx = pd.Series(np.arange(len(Zn))).groupby(guide).apply(lambda s: s.values).to_dict(); gl = list(gidx)


def pool(n):
    RNG.shuffle(gl); ix = []
    for g_ in gl:
        ix.extend(gidx[g_].tolist())
        if len(ix) >= n: break
    return np.array(ix[:n])


def null_stats(n):
    out = []
    for _ in range(REPS):
        ix = pool(n); m = np.ones(len(Zn), bool); m[ix] = False
        d = decompose(Zn[ix], oof[ix], Zn[m], oof[m]); d["d4"] = 100 * (oof4[ix].mean() - oof4[m].mean()); out.append(d)
    return pd.DataFrame(out)


def spread(Z): return float(np.mean(np.linalg.norm(Z - Z.mean(0), axis=1)))


def separation(Zg, Zref):
    n = min(len(Zg), 3000); idx = RNG.choice(len(Zref), n, replace=False)
    X = np.r_[Zg[:n], Zref[idx]]; y = np.r_[np.ones(n), np.zeros(n)]; pp = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        pp[b] = LogisticRegression(C=.05, max_iter=2000).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return roc_auc_score(y, pp)


# AUC null: NTC guide pools of typical KO size vs the remaining NTC, before / after
auc_null = []
for _ in range(60):
    ix = pool(600); m = np.ones(len(Zn), bool); m[ix] = False
    auc_null.append((separation(Zn[ix], Zn[m]), separation(Zn_c[ix], Zn_c[m])))
auc_null = np.array(auc_null); thr_b, thr_a = np.percentile(auc_null[:, 0], 95), np.percentile(auc_null[:, 1], 95)
log(f"AUC of NTC-vs-NTC guide pools (n=600): before median {np.median(auc_null[:, 0]):.3f} (95th {thr_b:.3f}) | after {np.median(auc_null[:, 1]):.3f} (95th {thr_a:.3f})")

rows, nulls, prof = [], {}, {}
mu_n = Zn.mean(0); ntc4 = 100 * float(np.nanmean(oof4))
groups = list(pd.Series(np.arange(len(gene))).groupby(gene).apply(lambda s: s.values).items())
if os.environ.get("MAX_GENES"): groups = groups[:int(os.environ["MAX_GENES"])]        # quick test mode
for k_, (g_, ix) in enumerate(groups):
    key = max(int(round(len(ix) / 100) * 100), 100)
    if key not in nulls: nulls[key] = null_stats(key); log(f"  null for n={key} ready")
    nd = nulls[key]; r = decompose(Zk[ix], pk[ix], Zn, oof); d4 = 100 * p4k[ix].mean() - ntc4
    r.update(gene=g_, n=len(ix), d_pct_4N_pred=d4,
             p_cycle=(np.sum(nd.cycle_TV >= r["cycle_TV"]) + 1) / (REPS + 1),
             p_within=(np.sum(nd.within_cycle_part >= r["within_cycle_part"]) + 1) / (REPS + 1),
             p_4N=(np.sum(np.abs(nd.d4) >= abs(d4)) + 1) / (REPS + 1),
             within_vs_null_median=r["within_cycle_part"] / nd.within_cycle_part.median(),
             spread_before=spread(Zk[ix]), spread_after=spread(Zk_c[ix]), spread_shuffled=spread(Zk_s[ix]),
             auc_before=separation(Zk[ix], Zn), auc_after=separation(Zk_c[ix], Zn_c), auc_shuffled=separation(Zk_s[ix], Zn_s))
    rows.append(r); prof[g_] = Zk[ix].mean(0) - mu_n
    if k_ % 100 == 0: log(f"  {k_} genes done")
R = pd.DataFrame(rows)
for c_ in ["cycle", "within", "4N"]: R[f"q_{c_}"] = bh(R[f"p_{c_}"])
R["class"] = np.select([(R.q_cycle < .05) & (R.q_within >= .05), (R.q_cycle >= .05) & (R.q_within < .05), (R.q_cycle < .05) & (R.q_within < .05)],
                       ["cycle distribution only", "cells change, cycle distribution unchanged", "both"], "neither")
R["spread_change_pct"] = 100 * (R.spread_after / R.spread_before - 1); R["spread_change_shuffled_pct"] = 100 * (R.spread_shuffled / R.spread_before - 1)
R["auc_change"] = R.auc_after - R.auc_before; R["separates_before"] = R.auc_before > thr_b; R["separates_after"] = R.auc_after > thr_a

# measured pRb on the same knockout cells
a = ad.read_h5ad("raw/all_cells_fluor_prb.h5ad", backed="r")
j = list(a.var_names).index("cp_single_object_pRb_cell_Intensity_MeanIntensity")
mp = pd.DataFrame({"gene": a.obs["gene"].astype(str).str.upper().values, "prb": np.asarray(a[:, j].to_memory().X).ravel()}).dropna()
base = np.log2(mp.loc[mp.gene == "NTC", "prb"].median())
R["measured_pRb_shift_log2"] = R.gene.map(np.log2(mp.groupby("gene")["prb"].median()) - base)
R.to_csv(f"{OUT}/ko_library_cycle.csv", index=False)
np.savez_compressed(f"{OUT}/ko_library_mean_profiles.npz", genes=np.array(list(prof)), profiles=np.vstack(list(prof.values())))

log("\nclasses (FDR 5%):\n" + R["class"].value_counts().to_string())
log(f"KOs with a significant predicted %4N change: {int((R.q_4N < .05).sum())} (up {int(((R.q_4N < .05) & (R.d_pct_4N_pred > 0)).sum())})")
log("median share of the morphology shift explained by the cycle shift, by class:\n" + R.groupby("class")["cycle_share_of_shift"].median().round(3).to_string())
log(f"heterogeneity: median spread change {R.spread_change_pct.median():+.2f}% (shuffled {R.spread_change_shuffled_pct.median():+.3f}%) | "
    f"median AUC change {R.auc_change.median():+.4f} | KOs separating from NTC before {int(R.separates_before.sum())}, after {int(R.separates_after.sum())}")
from scipy.stats import spearmanr
ok = R.measured_pRb_shift_log2.notna()
log(f"validation: predicted d%4N vs measured pRb shift across {ok.sum()} KOs, rho {spearmanr(R.d_pct_4N_pred[ok], R.measured_pRb_shift_log2[ok])[0]:+.2f}")
cols = ["gene", "n", "cycle_TV", "q_cycle", "d_pct_4N_pred", "within_vs_null_median", "q_within", "cycle_share_of_shift", "auc_before", "auc_after"]
log("\nlargest cycle-distribution changers:\n" + R[R.q_cycle < .05].sort_values("cycle_TV", ascending=False).head(15)[cols].round(3).to_string(index=False))
log("\nlargest AUC drop after removing the cycle:\n" + R.nsmallest(10, "auc_change")[cols].round(3).to_string(index=False))
log("DONE")
