#!/usr/bin/env python
"""Per modality: (1) how well its cell-cycle score separates cell-cycle state, and (2) whether regressing the cell cycle out
WITHIN that modality reduces the heterogeneity of perturbed populations.

Common truth for (1): measured DNA-content state (2N / S / 4N), available in both datasets.
  score quality = cross-validated balanced accuracy of the 3 DNA states from the score (sin/cos of the angle, harmonics 1-2),
  plus AUC 4N vs 2N. Scores built with DNA content are flagged (circular by construction).
OPS0077 modalities: phase contrast, 4i protein panel (pRb, Rb, p21 features + their segmentations), DAPI (NTC only).
  Each modality's own score = its held-out prediction of the final cc_phase (folds by tile, trained on NTC).
Element modalities: imaging (all features; and without the DNA stain), RNA (Tim's DRVI embedding; cycleVI's cycle-free latent
  as a model-based reference). Measured RNA clock = cycle_position_deg; imaging score = leave-one-well-out prediction of it.
(2) heterogeneity: perturbed populations vs controls, in each modality's PCA space, before / after removing the cell cycle
  (harmonics 1-2 of the cycle, coefficients fit on controls), with a shuffled-cycle control:
  spread = mean distance to the population centroid; separation = held-out logistic AUC vs controls.
  OPS0077: all 1,000 knockouts (up to 200 cells each), cycle = the modality's own prediction (knockout cells have no measured clock).
  Element: TNFa and 5 knockouts x 2 conditions, cycle = the MEASURED RNA clock, and for imaging also its own prediction.
usage: python modality_cycle.py [OUT_DIR]"""
import os, sys, time
import numpy as np, pandas as pd, anndata as ad, h5py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "modality_cycle"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0); T0 = time.time(); MAXC = int(os.environ.get("MAX_CELLS", 200))
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


def read_rows(dset, idx, cols=None, block=50000):
    """rows idx (any order, may contain NaN = missing) of a dense on-disk matrix, read in contiguous blocks"""
    idx = np.asarray(idx, float); out = np.full((len(idx), dset.shape[1] if cols is None else int(np.sum(cols))), np.nan, np.float32)
    ok = np.isfinite(idx); ii = idx[ok].astype(np.int64); dest = np.where(ok)[0]; o = np.argsort(ii); ii, dest = ii[o], dest[o]
    for s0 in range(int(ii.min()), int(ii.max()) + 1, block):
        lo, hi = np.searchsorted(ii, [s0, s0 + block])
        if hi > lo:
            B = dset[s0:min(s0 + block, dset.shape[0])]; B = B if cols is None else B[:, cols]; out[dest[lo:hi]] = B[ii[lo:hi] - s0]
    return out


def fit_transform(Xtr, k=50):
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xtr))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5); pca = PCA(min(k, Xtr.shape[1]), random_state=0).fit(f(Xtr))
    return lambda Z: np.vstack([pca.transform(f(Z[i:i + 50000])) for i in range(0, len(Z), 50000)]).astype(np.float32)


def fit_angle(Z, t):
    rs = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.sin(t)); rc = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.cos(t))
    return lambda W: np.mod(np.arctan2(rs.predict(W), rc.predict(W)), 2 * np.pi)


def harm(t): return np.c_[np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)]


def coef(Z, t): return np.linalg.lstsq(np.c_[harm(t), np.ones(len(t))], Z, rcond=None)[0][:4]


def var_explained(Z, t):
    R = Z - harm(t) @ coef(Z, t); return 100 * (1 - np.trace(np.cov(R.T)) / np.trace(np.cov(Z.T)))


def dna_state(d, lo, hi): return np.select([d < lo, d > hi], ["2N", "4N"], "S")


def score_quality(theta, state):
    F = harm(theta); y = state; m = y != "NA"
    pr = cross_val_predict(LogisticRegression(max_iter=2000), F[m], y[m], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    cls = np.array(sorted(set(y[m]))); bacc = balanced_accuracy_score(y[m], cls[pr.argmax(1)])
    k = np.isin(y[m], ["2N", "4N"]); auc = roc_auc_score(y[m][k] == "4N", pr[k][:, list(cls).index("4N")])
    return round(float(bacc), 3), round(float(auc), 3)


def spread(Z): return float(np.mean(np.linalg.norm(Z - Z.mean(0), axis=1)))


def separation(Zg, Zr):
    n = min(len(Zg), 3000); idx = RNG.choice(len(Zr), min(n, len(Zr)), replace=False); n = min(n, len(idx))
    X = np.r_[Zg[RNG.choice(len(Zg), n, replace=False)], Zr[idx]]; y = np.r_[np.ones(n), np.zeros(n)]
    pp = cross_val_predict(LogisticRegression(C=.05, max_iter=2000), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, pp)


def heterogeneity(name, Zc, tc, groups, Zp, tp, label_ref, n_null=40, null_groups=None):
    """Zc/tc: control cells + their cycle; Zp/tp/groups: perturbed cells, cycle, labels. Returns per-group rows + summary."""
    B = coef(Zc, tc); ts = RNG.permutation(tc); Bs = coef(Zc, ts)
    Zc_c, Zc_s = Zc - harm(tc) @ B, Zc - harm(ts) @ Bs
    Zp_c, Zp_s = Zp - harm(tp) @ B, Zp - harm(RNG.permutation(tp)) @ Bs
    rows = []
    for g_ in pd.unique(groups):
        m = groups == g_
        rows.append(dict(modality=name, group=g_, n=int(m.sum()), spread_before=spread(Zp[m]), spread_after=spread(Zp_c[m]), spread_shuffled=spread(Zp_s[m]),
                         auc_before=separation(Zp[m], Zc), auc_after=separation(Zp_c[m], Zc_c)))
    R = pd.DataFrame(rows); R["spread_change_pct"] = 100 * (R.spread_after / R.spread_before - 1); R["spread_change_shuffled_pct"] = 100 * (R.spread_shuffled / R.spread_before - 1)
    R["auc_change"] = R.auc_after - R.auc_before
    thr_b = thr_a = np.nan
    if null_groups is not None:                                   # AUC null: control guide pools of typical size vs the other controls
        nb, na = [], []
        gid = pd.Series(np.arange(len(Zc))).groupby(null_groups).apply(lambda s: s.values).to_dict(); gl = list(gid)
        for _ in range(n_null):
            RNG.shuffle(gl); ix = []
            for gg in gl:
                ix.extend(gid[gg].tolist())
                if len(ix) >= MAXC: break
            ix = np.array(ix[:MAXC]); mm = np.ones(len(Zc), bool); mm[ix] = False
            nb.append(separation(Zc[ix], Zc[mm])); na.append(separation(Zc_c[ix], Zc_c[mm]))
        thr_b, thr_a = np.percentile(nb, 95), np.percentile(na, 95)
    S = dict(modality=name, n_groups=len(R), cycle_variance_removed_pct_controls=round(100 * (1 - np.trace(np.cov(Zc_c.T)) / np.trace(np.cov(Zc.T))), 2),
             shuffled_variance_removed_pct=round(100 * (1 - np.trace(np.cov(Zc_s.T)) / np.trace(np.cov(Zc.T))), 3),
             control_spread_change_pct=round(100 * (spread(Zc_c) / spread(Zc) - 1), 2),
             median_spread_change_pct=round(R.spread_change_pct.median(), 2), median_spread_change_shuffled_pct=round(R.spread_change_shuffled_pct.median(), 3),
             median_auc_before=round(R.auc_before.median(), 3), median_auc_after=round(R.auc_after.median(), 3), median_auc_change=round(R.auc_change.median(), 4),
             n_separating_before=int((R.auc_before > thr_b).sum()) if np.isfinite(thr_b) else np.nan,
             n_separating_after=int((R.auc_after > thr_a).sum()) if np.isfinite(thr_a) else np.nan, cycle_used=label_ref)
    return R, S


Q1, H_rows, H_sum = [], [], []

# ====================================================================== OPS0077
L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
L = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")]
L = L[~L.index.duplicated()]
ko_phase = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r"); kobs = ko_phase.obs.copy(); kobs["gene"] = kobs["gene"].astype(str).str.upper()
kobs["row"] = np.arange(len(kobs)); kobs = kobs[~kobs.cell_key.duplicated()]          # row = position in the file, kept before de-duplication
pick = kobs.groupby("gene", group_keys=False).apply(lambda d: d.sample(min(len(d), MAXC), random_state=0))
ko_keys = pick.cell_key.astype(str).values; ko_gene = pick.gene.values
log(f"OPS0077: {len(ko_keys):,} knockout cells sampled ({MAXC} max per gene, {len(np.unique(ko_gene))} genes)")

# --- phase
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str); ph = ph[~ph.obs_names.duplicated()]
ntc_keys = ph.obs_names.values; ntc_tile = ph.obs["tile_pheno"].astype(str).values; ntc_guide = ph.obs["sgRNA"].astype(str).values
lab = np.isin(ntc_keys, L.index)
pres = np.isfinite(np.asarray(ph.X)[lab]).mean(0) >= .85
MOD = {"phase contrast": (np.asarray(ph.X, np.float32)[:, pres], None)}
f = h5py.File("OPS0077_allko_phase.h5ad", "r"); Xk = read_rows(f["X"], pick.row.values, cols=pres); f.close()
MOD["phase contrast"] = (MOD["phase contrast"][0], Xk); del ph
log("phase loaded")

# --- 4i protein panel: pRb, Rb, p21 files (same rows), all features, NTC + sampled knockout cells
blocks_n, blocks_k = [], []
for mk in ["prb", "rb", "p21"]:
    fh = h5py.File(f"raw/all_cells_fluor_{mk}.h5ad", "r"); o = ad.read_h5ad(f"raw/all_cells_fluor_{mk}.h5ad", backed="r").obs
    key = (o["well"].astype(str) + "|" + o["segmentation_id"].astype("int64").astype(str)).values
    kp = pd.Series(np.arange(len(key)), index=key); kp = kp[~kp.index.duplicated()]
    want_n, want_k = kp.reindex(ntc_keys).values, kp.reindex(ko_keys).values
    names = np.array(ad.read_h5ad(f"raw/all_cells_fluor_{mk}.h5ad", backed="r").var_names)
    if mk != "prb": keepc = ~np.char.startswith(names.astype(str), "op_cell_")       # shared cell-shape columns only once
    else: keepc = np.ones(len(names), bool)
    blocks_n.append(read_rows(fh["X"], want_n, cols=keepc)); blocks_k.append(read_rows(fh["X"], want_k, cols=keepc)); fh.close()
    log(f"4i {mk}: NTC matched {np.isfinite(want_n).mean():.3f}, knockout matched {np.isfinite(want_k).mean():.3f}, features {keepc.sum()}")
Xn4, Xk4 = np.hstack(blocks_n), np.hstack(blocks_k); p4 = np.isfinite(Xn4[lab]).mean(0) >= .85
MOD["4i protein panel (pRb, Rb, p21)"] = (Xn4[:, p4], Xk4[:, p4]); del blocks_n, blocks_k, Xn4, Xk4

# --- DAPI (NTC only)
dp = ad.read_h5ad(find("OPS0077_NTC_DAPI_cpmeasure.h5ad")); dp = dp[~dp.obs_names.duplicated()]
Xd = pd.DataFrame(np.asarray(dp.X, np.float32), index=dp.obs_names.astype(str)).reindex(ntc_keys).values; pd_ = np.isfinite(Xd[lab]).mean(0) >= .85
dna_rel = L["dna_rel_R234"].reindex(ntc_keys).astype(float).values; state_o = np.where(np.isfinite(dna_rel), dna_state(dna_rel, .25, .65), "NA")
th_final = 2 * np.pi * L["cc_phase"].reindex(ntc_keys).astype(float).values

# (1) scores: measured clocks + each modality's held-out prediction of the final clock
scores = {"4i final cc_phase (measured; uses DNA)": th_final, "4i DNA clock (measured; uses DNA)": 2 * np.pi * L["cc_phase_dna"].reindex(ntc_keys).astype(float).values,
          "4i markers clock (measured; no DNA)": 2 * np.pi * L["cc_phase_markers"].reindex(ntc_keys).astype(float).values}
OOF = {}
for name, Xm in [("phase contrast", MOD["phase contrast"][0]), ("4i protein panel (pRb, Rb, p21)", MOD["4i protein panel (pRb, Rb, p21)"][0]), ("DAPI features (uses DNA)", Xd[:, pd_])]:
    oof = np.full(len(ntc_keys), np.nan); li = np.where(lab)[0]
    for a, b in GroupKFold(5).split(Xm, groups=ntc_tile):
        tr = a[lab[a]]; T = fit_transform(Xm[tr]); P = fit_angle(T(Xm[tr]), th_final[tr]); oof[b] = P(T(Xm[b]))
    OOF[name] = oof; scores[f"{name}: predicted clock"] = oof
    log(f"OPS0077 {name}: held-out median error vs final clock {np.median(np.degrees(np.abs(np.angle(np.exp(1j * (oof[lab] - th_final[lab])))))):.1f} deg")
scores["random angle (null)"] = RNG.uniform(0, 2 * np.pi, len(ntc_keys))
for s_, th in scores.items():
    m = lab & np.isfinite(th) & (state_o != "NA")
    bacc, auc = score_quality(th[m], state_o[m])
    med = np.median(np.degrees(np.abs(np.angle(np.exp(1j * (th[m] - th_final[m])))))) if "measured" not in s_ or "final" in s_ else np.nan
    Q1.append(dict(dataset="OPS0077", score=s_, n=int(m.sum()), DNA_state_balanced_acc=bacc, AUC_4N_vs_2N=auc, median_err_vs_final_clock_deg=round(float(med), 1) if np.isfinite(med) else np.nan))
# variance of each modality explained by the measured clock (labelled NTC)
for name in ["phase contrast", "4i protein panel (pRb, Rb, p21)"]:
    Xm = MOD[name][0]; T = fit_transform(Xm[lab]); Z = T(Xm[lab])
    Q1.append(dict(dataset="OPS0077", score=f"{name}: % variance (PCA-50) explained by the measured clock", n=int(lab.sum()),
                   variance_explained_by_measured_clock_pct=round(var_explained(Z, th_final[lab]), 2), variance_explained_by_shuffled_pct=round(var_explained(Z, RNG.permutation(th_final[lab])), 3)))
pd.DataFrame(Q1).to_csv(f"{OUT}/q1_score_quality.csv", index=False); log("\n" + pd.DataFrame(Q1).to_string(index=False))

# (2) heterogeneity per OPS0077 modality, cycle = the modality's own predicted clock
for name in ["phase contrast", "4i protein panel (pRb, Rb, p21)"]:
    Xn_, Xk_ = MOD[name]; T = fit_transform(Xn_[lab]); P = fit_angle(T(Xn_[lab]), th_final[lab])
    Zn, Zk = T(Xn_), T(Xk_)
    R, S = heterogeneity(f"OPS0077 {name}", Zn, OOF[name], ko_gene, Zk, P(Zk), "modality's own predicted clock", null_groups=ntc_guide)
    H_rows.append(R); H_sum.append(S); log(f"heterogeneity {name}: {S}")
    pd.concat(H_rows).to_csv(f"{OUT}/q2_heterogeneity_by_group.csv", index=False); pd.DataFrame(H_sum).to_csv(f"{OUT}/q2_heterogeneity_summary.csv", index=False)
del MOD, Xd

# ====================================================================== Element
h = h5py.File("/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu", "r")
obs = read_elem(h["obs"]); names = np.array(read_elem(h["mod"]["imaging"]["var"]).index.astype(str))
raw = h["mod"]["imaging"]["layers"]["raw"][:].astype(np.float32); raw[~np.isfinite(raw)] = np.nan
g = lambda n: raw[:, list(names).index(n)].astype(float)
wells = obs["Well"].astype(str).values
dE = np.log2(np.clip((g("Nucleus_Nuclear") - g("Bkg_Nucleus_Nuclear")) * g("NuclearArea"), 1, None)); dE = dE - pd.Series(dE).groupby(wells).transform("median").values
drop = np.array([any(s in n for s in ("Location", "Center", "ObjectNumber", "Bkg_", "Parent", "Children", "Number_", "Metadata")) for n in names])
keep = (~drop) & (np.isfinite(raw).mean(0) >= .85); Xi = raw[:, keep]; fn = names[keep]; del raw
nodna = np.array(["Nucleus" not in n.split("_") and n not in ("Nucleus_Nuclear",) for n in fn])
th = np.radians(obs["cycle_position_deg"].astype(float).values); cond = obs["condition"].astype(str).values; gene = obs["target_gene"].astype(str).values
ctl = np.isin(gene, ["NTC", "OR1I1"]); guide = obs["guide"].astype(str).values
hh, ed = np.histogram(dE[ctl], bins=120, range=(-1.5, 2)); cen = (ed[:-1] + ed[1:]) / 2
m2 = cen[np.argmax(np.where(np.abs(cen) < .4, hh, 0))]; m4 = cen[np.argmax(np.where(cen > .6, hh, 0))]
state_e = dna_state(dE, m2 + .25 * (m4 - m2), m2 + .75 * (m4 - m2))
RNA = {"RNA (DRVI embedding)": np.asarray(h["mod"]["rna"]["obsm"]["X_drvi"][:], np.float32),
       "RNA (cycleVI cycle-free latent; model-based removal)": np.asarray(h["mod"]["rna"]["obsm"]["X_cyclevi_other"][:], np.float32)}
log(f"Element: {len(Xi):,} cells, imaging features {len(fn)} ({nodna.sum()} without the DNA stain)")

pred = {}
for name, cols in [("imaging (all features; uses DNA stain)", np.ones(len(fn), bool)), ("imaging without the DNA stain", nodna)]:
    p_ = np.full(len(Xi), np.nan)
    for w in np.unique(wells):
        tr = ctl & (wells != w); T = fit_transform(Xi[tr][:, cols]); P = fit_angle(T(Xi[tr][:, cols]), th[tr]); te = wells == w; p_[te] = P(T(Xi[te][:, cols]))
    pred[name] = p_; log(f"Element {name}: leave-one-well-out median error vs RNA clock {np.median(np.degrees(np.abs(np.angle(np.exp(1j * (p_[ctl] - th[ctl])))))):.1f} deg")
for name in RNA:
    if "DRVI" in name:
        p_ = np.full(len(Xi), np.nan)
        for w in np.unique(wells):
            tr = ctl & (wells != w); P = fit_angle(RNA[name][tr], th[tr]); te = wells == w; p_[te] = P(RNA[name][te])
        pred[name] = p_
SE = {"RNA clock (cycleVI, measured)": th, **{f"{k}: predicted clock": v for k, v in pred.items()}, "random angle (null)": RNG.uniform(0, 2 * np.pi, len(th))}
for s_, t_ in SE.items():
    m = ctl & np.isfinite(t_); bacc, auc = score_quality(t_[m], state_e[m])
    med = np.median(np.degrees(np.abs(np.angle(np.exp(1j * (t_[m] - th[m])))))) if "measured" not in s_ else np.nan
    Q1.append(dict(dataset="Element", score=s_, n=int(m.sum()), DNA_state_balanced_acc=bacc, AUC_4N_vs_2N=auc, median_err_vs_final_clock_deg=round(float(med), 1) if np.isfinite(med) else np.nan))
ph_s = obs["phase_scanpy"].astype(str).values
m = ctl; Fp = pd.get_dummies(ph_s[m]).values.astype(float)
pr = cross_val_predict(LogisticRegression(max_iter=2000), Fp, state_e[m], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
cls = np.array(sorted(set(state_e[m]))); kk = np.isin(state_e[m], ["2N", "4N"])
Q1.append(dict(dataset="Element", score="RNA scanpy phase call (G1/S/G2M)", n=int(m.sum()), DNA_state_balanced_acc=round(balanced_accuracy_score(state_e[m], cls[pr.argmax(1)]), 3),
               AUC_4N_vs_2N=round(roc_auc_score(state_e[m][kk] == "4N", pr[kk][:, list(cls).index("4N")]), 3)))
Ti = fit_transform(Xi[ctl]); Zi = Ti(Xi)
for name, Z in [("imaging (all features)", Zi), *RNA.items()]:
    Q1.append(dict(dataset="Element", score=f"{name}: % variance explained by the measured RNA clock", n=int(ctl.sum()),
                   variance_explained_by_measured_clock_pct=round(var_explained(Z[ctl], th[ctl]), 2), variance_explained_by_shuffled_pct=round(var_explained(Z[ctl], RNG.permutation(th[ctl])), 3)))
pd.DataFrame(Q1).to_csv(f"{OUT}/q1_score_quality.csv", index=False)

# (2) Element heterogeneity: perturbed populations per condition vs that condition's controls
for name, Z, cyc, cyc_lab in [("imaging (all features)", Zi, th, "measured RNA clock"), ("imaging (all features)", Zi, pred["imaging (all features; uses DNA stain)"], "imaging's own predicted clock"),
                              ("RNA (DRVI embedding)", RNA["RNA (DRVI embedding)"], th, "measured RNA clock")]:
    rows_c, sums = [], []
    for c_ in ["baseline", "TNFa"]:
        mc = ctl & (cond == c_); mp = (~ctl) & (cond == c_) & (gene != "Undetermined")
        R, S = heterogeneity(f"Element {name} [{c_}]", Z[mc], cyc[mc], gene[mp], Z[mp], cyc[mp], cyc_lab, n_null=20, null_groups=guide[mc])
        R["condition"] = c_; rows_c.append(R)
    mb, mt = ctl & (cond == "baseline"), ctl & (cond == "TNFa")
    Rt, _ = heterogeneity(f"Element {name} [TNFa vs baseline]", Z[mb], cyc[mb], np.array(["TNFa (controls)"] * mt.sum()), Z[mt], cyc[mt], cyc_lab)
    Rt["condition"] = "TNFa vs baseline"; R = pd.concat(rows_c + [Rt]); R["cycle_used"] = cyc_lab; H_rows.append(R)
    Zc = Z[ctl & (cond == "baseline")]; tcb = cyc[ctl & (cond == "baseline")]; B = coef(Zc, tcb)
    H_sum.append(dict(modality=f"Element {name}", cycle_used=cyc_lab, n_groups=len(R),
                      cycle_variance_removed_pct_controls=round(100 * (1 - np.trace(np.cov((Zc - harm(tcb) @ B).T)) / np.trace(np.cov(Zc.T))), 2),
                      median_spread_change_pct=round(R.spread_change_pct.median(), 2), median_spread_change_shuffled_pct=round(R.spread_change_shuffled_pct.median(), 3),
                      median_auc_before=round(R.auc_before.median(), 3), median_auc_after=round(R.auc_after.median(), 3), median_auc_change=round(R.auc_change.median(), 4)))
    log(f"heterogeneity Element {name} / {cyc_lab}:\n" + R[["group", "condition", "n", "spread_change_pct", "spread_change_shuffled_pct", "auc_before", "auc_after"]].round(3).to_string(index=False))
    pd.concat(H_rows).to_csv(f"{OUT}/q2_heterogeneity_by_group.csv", index=False); pd.DataFrame(H_sum).to_csv(f"{OUT}/q2_heterogeneity_summary.csv", index=False)
# RNA: compare with cycleVI's own cycle-free latent (spread / separation of perturbed populations, no regression)
for c_ in ["baseline", "TNFa"]:
    mc = ctl & (cond == c_); mp = (~ctl) & (cond == c_) & (gene != "Undetermined")
    for name in RNA:
        Zr = RNA[name]
        for g_ in pd.unique(gene[mp]):
            mm = mp & (gene == g_)
            H_rows.append(pd.DataFrame([dict(modality=f"Element {name} [{c_}]", group=g_, condition=c_, n=int(mm.sum()), auc_before=separation(Zr[mm], Zr[mc]),
                                             spread_before=spread(Zr[mm]), cycle_used="none (embedding as is)")]))
pd.concat(H_rows).to_csv(f"{OUT}/q2_heterogeneity_by_group.csv", index=False)
log("\nSUMMARY (1) score quality:\n" + pd.DataFrame(Q1).to_string(index=False))
log("\nSUMMARY (2) heterogeneity:\n" + pd.DataFrame(H_sum).to_string(index=False))
log("DONE")
