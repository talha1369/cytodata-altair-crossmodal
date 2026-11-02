#!/usr/bin/env python
"""Knockout cell-cycle shifts from phase-contrast morphology (building on Brian's CDK7/CCNB1 figures), and whether
regressing the cell cycle out reduces the heterogeneity of perturbed populations (the project goal).

Model: phase-contrast morphology -> circular cell-cycle phase (team `cc_phase`), ridge on PCA-100 of robust-scaled features,
trained on NTC cells only.
Part 1  Fair knockout comparison. Knockout cells are never in training, so the NTC reference must also be cells the model
        did not train on: out-of-fold NTC predictions (folds grouped by tile). In-sample NTC is reported only to show the
        artifact it creates. Null = NTC guides pooled to the knockout's cell count (no perturbation, same pipeline).
Part 2  Validation against measured 4i markers on the same knockout cells (pRb for all genes; DNA for the 3 measured genes).
Part 3  Heterogeneity. Remove from every phase feature the part explained by the predicted cycle (sin/cos harmonics 1-2, fit on
        NTC), then compare within-population spread and knockout-vs-NTC separation before vs after, for arrest genes and
        random control genes, with a shuffled-cycle control.
usage: python ko_cycle_heterogeneity.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "ko_cycle"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
QUART = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", ".", "cc_align"]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


# ------------------------------------------------ data
L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
L = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")]
phall = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); phall.obs_names = phall.obs["cell_key"].astype(str)
phall = phall[~phall.obs_names.duplicated()]
lab = phall.obs_names.isin(L.index)                       # NTC cells that have a measured clock (training labels)
present = np.isfinite(np.asarray(phall.X)[lab]).mean(0) >= 0.85
Xall = np.asarray(phall.X, np.float32)[:, present]
tile_all = phall.obs["tile_pheno"].astype(str).values; guide_all = phall.obs["sgRNA"].astype(str).values
L = L.loc[phall.obs_names[lab]]
Xn = Xall[lab]; th = 2 * np.pi * L["cc_phase"].astype(float).values; tile = tile_all[lab]
print(f"NTC phase cells {len(Xall):,}, of which {int(lab.sum()):,} have a measured clock (train) and {int((~lab).sum()):,} do not "
      f"(no reliable single nucleus); the knockout cells were NOT filtered this way, so ALL NTC cells are the reference", flush=True)

kos = []
for fn, cls in [("OPS0077_genes_phase.h5ad", "arrest"), ("OPS0077_ctrlgenes_phase.h5ad", "control")]:
    k = ad.read_h5ad(find(fn)); kos.append(pd.DataFrame(np.asarray(k.X, np.float32)[:, present]).assign(gene=k.obs["gene"].astype(str).str.upper().values,
                                                                                                           cell_key=k.obs["cell_key"].astype(str).values, cls=cls))
K = pd.concat(kos, ignore_index=True).drop_duplicates("cell_key").reset_index(drop=True)
Xk = K.drop(columns=["gene", "cell_key", "cls"]).values.astype(np.float32)
print(f"NTC train {len(L):,} | knockout cells {len(K):,} | genes {K.gene.nunique()} ({(K.drop_duplicates('gene').cls == 'arrest').sum()} arrest, "
      f"{(K.drop_duplicates('gene').cls == 'control').sum()} control)", flush=True)


def fit_transform(Xtr):
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xtr))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5)
    pca = PCA(100, random_state=0).fit(f(Xtr))
    return lambda Z: pca.transform(f(Z))


def fit_angle(Z, t):
    rs = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.sin(t)); rc = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.cos(t))
    return lambda W: np.mod(np.arctan2(rs.predict(W), rc.predict(W)), 2 * np.pi) / (2 * np.pi)


# out-of-fold NTC predictions + in-sample predictions + knockout predictions from the full NTC model
# out-of-fold predictions for ALL NTC cells: folds over all NTC cells by tile; train on labelled cells outside the fold
oof_all = np.full(len(Xall), np.nan); lab_idx = np.where(lab)[0]; pos = {c: i for i, c in enumerate(lab_idx)}
for a, b in GroupKFold(5).split(Xall, groups=tile_all):
    tr = a[lab[a]]; tri = np.array([pos[i] for i in tr])
    T_ = fit_transform(Xall[tr]); P_ = fit_angle(T_(Xall[tr]), th[tri]); oof_all[b] = P_(T_(Xall[b]))
oof = oof_all[lab]
T = fit_transform(Xn); Zn_lab = T(Xn); P = fit_angle(Zn_lab, th)
insample = P(Zn_lab); Zk = T(Xk); K["pred"] = P(Zk)
Zn = T(Xall)                                          # morphology space for ALL NTC cells (reference for Part 3)
meas = L["cc_phase"].astype(float).values
print("prediction error (deg), NTC: out-of-fold", round(float(np.median(np.abs((oof - meas + .5) % 1 - .5)) * 360), 1),
      "| in-sample", round(float(np.median(np.abs((insample - meas + .5) % 1 - .5)) * 360), 1), flush=True)


def comp(x):
    h = np.histogram(np.mod(x, 1), bins=[0, .25, .5, .75, 1.0001])[0]; return 100 * h / h.sum()


ref = {"measured cc_phase (4i), labelled NTC": comp(meas), "NTC in-sample prediction, labelled": comp(insample),
       "NTC out-of-fold, labelled only (biased)": comp(oof), "NTC out-of-fold, unlabelled only": comp(oof_all[~lab]),
       "NTC out-of-fold prediction (fair)": comp(oof_all)}

# null: NTC guides pooled to n cells, scored with OUT-OF-FOLD predictions
ntc_guides = pd.Series(oof_all).groupby(guide_all)


def null_comp(n, reps=400):
    gl = list(ntc_guides.groups.keys()); out = []
    for _ in range(reps):
        RNG.shuffle(gl); vals = []
        for gname in gl:
            vals.extend(ntc_guides.get_group(gname).values.tolist())
            if len(vals) >= n: break
        out.append(comp(np.array(vals[:n])))
    return np.array(out)


rows = []; nulls = {}
for gname, d in K.groupby("gene"):
    n = len(d); c = comp(d["pred"].values); ref_ = ref["NTC out-of-fold prediction (fair)"]
    key = int(round(n / 50) * 50)
    if key not in nulls: nulls[key] = null_comp(max(key, 50))
    nd = nulls[key] - ref_; delta = c - ref_
    p = [(np.sum(np.abs(nd[:, j]) >= abs(delta[j])) + 1) / (len(nd) + 1) for j in range(4)]
    shift_vs_insample = c - ref["NTC out-of-fold, labelled only (biased)"]
    rows.append(dict(gene=gname, cls=d["cls"].iloc[0], n=n, **{f"d_{QUART[j]}": round(delta[j], 2) for j in range(4)},
                     **{f"p_{QUART[j]}": round(p[j], 4) for j in range(4)},
                     **{f"d_vs_labelled_only_{QUART[j]}": round(shift_vs_insample[j], 2) for j in range(4)}))
R = pd.DataFrame(rows)
for j in range(4):
    pv = R[f"p_{QUART[j]}"].values; o = np.argsort(pv); q = np.empty_like(pv); m = len(pv)
    q[o] = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]; R[f"q_{QUART[j]}"] = np.round(np.minimum(q, 1), 4)
R.to_csv(f"{OUT}/ko_predicted_cycle_composition.csv", index=False)
print("\nreference compositions (% of cells per quarter of cc_phase):")
for k_, v_ in ref.items(): print(f"  {k_:36s}", np.round(v_, 1))
sig = R[[f"q_{QUART[j]}" for j in range(4)]].min(1) < 0.05
print(f"\nknockouts with a significant composition shift vs out-of-fold NTC (FDR 5%, guide-pooled null): {int(sig.sum())} / {len(R)}"
      f"  (arrest {int(sig[R.cls == 'arrest'].sum())}/{int((R.cls == 'arrest').sum())}, control {int(sig[R.cls == 'control'].sum())}/{int((R.cls == 'control').sum())})")
for g_ in ["CDK7", "CCNB1", "MCM3", "CENPC"]:
    r_ = R[R.gene == g_]
    if len(r_): r_ = r_.iloc[0]; print(f"  {g_:6s} n={r_.n:4d} shift vs fair NTC {[r_[f'd_{q}'] for q in QUART]} q {[r_[f'q_{q}'] for q in QUART]} | vs labelled-only NTC {[r_[f'd_vs_labelled_only_{q}'] for q in QUART]}")
print("mean shift per quarter (signed, all 120 genes), vs labelled-only NTC:", np.round(R[[f"d_vs_labelled_only_{q}" for q in QUART]].mean().values, 2),
      "| vs ALL NTC (fair):", np.round(R[[f"d_{q}" for q in QUART]].mean().values, 2))

# ------------------------------------------------ Part 2: measured markers for the same knockout cells
a = ad.read_h5ad("raw/all_cells_fluor_prb.h5ad", backed="r")
j = list(a.var_names).index("cp_single_object_pRb_cell_Intensity_MeanIntensity")
mp = pd.DataFrame({"cell_key": (a.obs["well"].astype(str) + "|" + a.obs["segmentation_id"].astype("int64").astype(str)).values,
                   "gene": a.obs["gene"].astype(str).str.upper().values, "prb": np.asarray(a[:, j].to_memory().X).ravel()}).dropna()
base = np.log2(mp.loc[mp.gene == "NTC", "prb"].median())
meas_prb = np.log2(mp[mp.gene.isin(R.gene)].groupby("gene")["prb"].median()) - base
R["measured_pRb_shift_log2"] = R.gene.map(meas_prb)
from scipy.stats import spearmanr
for q in QUART:
    ok = R["measured_pRb_shift_log2"].notna()
    rho, pv = spearmanr(R.loc[ok, f"d_{q}"], R.loc[ok, "measured_pRb_shift_log2"])
    print(f"  predicted % in quarter {q} vs measured pRb shift across {ok.sum()} knockouts: rho {rho:+.2f} (p={pv:.2g})")
R.to_csv(f"{OUT}/ko_predicted_cycle_composition.csv", index=False)
dna3 = None
if os.path.exists("ko3_dna_percell.csv"):
    d3 = pd.read_csv("ko3_dna_percell.csv", index_col=0); kk = K.set_index("cell_key")
    d3 = d3[d3.index.isin(kk.index)]; d3["pred"] = kk.loc[d3.index, "pred"].values
    ntc_dna = L["dna_rel_R234"].astype(float)
    dna3 = []
    for g_, dd in d3.groupby("gene"):
        m4 = 100 * ((dd.dna_rel_R234 > .65).mean() - (ntc_dna > .65).mean())
        p4 = R.loc[R.gene == g_, ["d_0.5-0.75", "d_0.75-1"]].values.ravel()
        dna3.append(dict(gene=g_, n=len(dd), measured_4N_pct_shift=round(m4, 1), predicted_shift_0p5_0p75=p4[0], predicted_shift_0p75_1=p4[1]))
    dna3 = pd.DataFrame(dna3); print("\nmeasured DNA vs predicted shift (percentage points):\n", dna3.to_string(index=False)); dna3.to_csv(f"{OUT}/ko3_measured_vs_predicted.csv", index=False)

# ------------------------------------------------ Part 3: regressing the cycle out
def harm(p):
    t = 2 * np.pi * p; return np.c_[np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)]


Hn = harm(oof_all)        # NTC cycle covariates from OUT-OF-FOLD predictions, ALL NTC cells (no leakage)
Hk = harm(K["pred"].values)
B = np.linalg.lstsq(np.c_[Hn, np.ones(len(Hn))], Zn, rcond=None)[0]
corr_ = lambda Z, H: Z - H @ B[:4]
Zn_c, Zk_c = corr_(Zn, Hn), corr_(Zk, Hk)
Hs = harm(RNG.permutation(oof_all)); Bs = np.linalg.lstsq(np.c_[Hs, np.ones(len(Hs))], Zn, rcond=None)[0]
Hks = harm(RNG.permutation(K["pred"].values))
Zn_s, Zk_s = Zn - Hs @ Bs[:4], Zk - Hks @ Bs[:4]
tot = lambda Z: float(np.trace(np.cov(Z.T)))
print(f"\nvariance of NTC morphology (PCA-100 space) removed by regressing out the predicted cycle: {100 * (1 - tot(Zn_c) / tot(Zn)):.1f}%"
      f" | by a SHUFFLED cycle: {100 * (1 - tot(Zn_s) / tot(Zn)):.2f}%")


def spread(Z):   # within-population heterogeneity = mean distance to centroid
    return float(np.mean(np.linalg.norm(Z - Z.mean(0), axis=1)))


def separation(Zg, Zref):   # held-out knockout-vs-NTC AUC (balanced, logistic on the same space)
    n = min(len(Zg), 3000); idx = RNG.choice(len(Zref), n, replace=False)
    X = np.r_[Zg[:n], Zref[idx]]; y = np.r_[np.ones(n), np.zeros(n)]; pp = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        pp[b] = LogisticRegression(C=.05, max_iter=2000).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return roc_auc_score(y, pp)


H = []
for gname, d in K.groupby("gene"):
    ix = d.index.values
    H.append(dict(gene=gname, cls=d["cls"].iloc[0], n=len(ix),
                  spread_before=spread(Zk[ix]), spread_after=spread(Zk_c[ix]), spread_shuffled=spread(Zk_s[ix]),
                  auc_before=separation(Zk[ix], Zn), auc_after=separation(Zk_c[ix], Zn_c), auc_shuffled=separation(Zk_s[ix], Zn_s)))
H = pd.DataFrame(H)
H["spread_change_pct"] = 100 * (H.spread_after / H.spread_before - 1); H["spread_change_shuffled_pct"] = 100 * (H.spread_shuffled / H.spread_before - 1)
H["auc_change"] = H.auc_after - H.auc_before; H["auc_change_shuffled"] = H.auc_shuffled - H.auc_before
ntc_spread = (spread(Zn), spread(Zn_c))
H.to_csv(f"{OUT}/ko_heterogeneity_before_after.csv", index=False)
print(f"NTC spread {ntc_spread[0]:.2f} -> {ntc_spread[1]:.2f} ({100 * (ntc_spread[1] / ntc_spread[0] - 1):+.1f}%)")
print("\nper class, median over genes:")
print(H.groupby("cls")[["spread_change_pct", "spread_change_shuffled_pct", "auc_before", "auc_after", "auc_change", "auc_change_shuffled"]].median().round(3).to_string())
print("\ngenes whose separation from NTC drops most after removing the cycle (their phenotype IS largely cell cycle):")
print(H.nsmallest(8, "auc_change")[["gene", "cls", "n", "auc_before", "auc_after", "auc_change"]].round(3).to_string(index=False))
print("\ngenes whose separation rises most:")
print(H.nlargest(6, "auc_change")[["gene", "cls", "n", "auc_before", "auc_after", "auc_change"]].round(3).to_string(index=False))

# ------------------------------------------------ figures
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
x = np.linspace(0, 1, 201)
def kde(v, bw=.03):
    v = np.mod(v, 1); d = np.zeros_like(x)
    for s in (-1, 0, 1): d += np.exp(-.5 * ((x[:, None] - (v[None, :] + s)) / bw) ** 2).sum(1)
    return d / (len(v) * bw * np.sqrt(2 * np.pi))
ax[0].plot(x, kde(meas), "k-", lw=1.5, label="NTC, measured cc_phase")
ax[0].plot(x, kde(oof), color="0.5", lw=1.5, ls="--", label="NTC with a measured clock only (biased subset)")
ax[0].plot(x, kde(oof_all), color="tab:blue", lw=2.2, label="ALL NTC, out-of-fold prediction (fair reference)")
for g_, c_ in [("CDK7", "tab:red"), ("CCNB1", "tab:orange")]:
    if (K.gene == g_).any(): ax[0].plot(x, kde(K.loc[K.gene == g_, "pred"].values), color=c_, lw=2, label=f"{g_} KO (n={int((K.gene == g_).sum())})")
for v in (.25, .5, .75): ax[0].axvline(v, color=".8", ls=":")
ax[0].set_xlabel("cell-cycle phase (cc_phase)"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8); ax[0].set_title("predicted cycle distribution: compare like with like")
lim = max(R[[f"d_{q}" for q in QUART]].abs().max().max(), 1)
for cls_, c_ in [("control", "0.6"), ("arrest", "tab:red")]:
    rr = R[R.cls == cls_]
    ax[1].scatter(rr["d_0.5-0.75"], rr["d_0.75-1"], s=20, c=c_, label=cls_, alpha=.8)
for g_ in ["CDK7", "CCNB1", "MCM3", "CENPC"]:
    rr = R[R.gene == g_]
    if len(rr): ax[1].annotate(g_, (rr["d_0.5-0.75"].values[0], rr["d_0.75-1"].values[0]), fontsize=8)
ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5); ax[1].legend()
ax[1].set_xlabel("shift in % of cells at cc_phase 0.5-0.75 (vs fair NTC)"); ax[1].set_ylabel("shift in % at 0.75-1"); ax[1].set_title("knockout cell-cycle shifts (arrest vs control genes)")
for cls_, c_ in [("control", "0.6"), ("arrest", "tab:red")]:
    hh = H[H.cls == cls_]; ax[2].scatter(hh.auc_before, hh.auc_after, s=20, c=c_, label=cls_, alpha=.8)
ax[2].plot([.45, 1], [.45, 1], "k--", lw=.8); ax[2].set_xlabel("knockout vs NTC separation (AUC), before"); ax[2].set_ylabel("after regressing out the cell cycle")
ax[2].legend(); ax[2].set_title("does removing the cycle change the perturbation signal?")
plt.tight_layout(); plt.savefig(f"{OUT}/ko_cycle_overview.png", dpi=105)
print("DONE")
