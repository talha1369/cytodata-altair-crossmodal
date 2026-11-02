#!/usr/bin/env python
"""Follow-up checks on phase_embedding_push.py (stream C). Writes to phase_push_check/ (new folder, nothing overwritten).

Why: the first run's pooled knockout tightening was 0.0% on every PC and a few knockouts sat at centroid distances of 10^4 to 10^5.
Both come from a handful of knockout cells with extreme feature values (z-scores in the thousands) that dominate any variance or
mean. This script reproduces the same embedding and covariate (Brian's gradient-boosting predicted cycle, coefficients fit on NTC),
then
  1. flags outlier knockout cells (PC1-10 norm above the largest NTC cell norm) and reports how many and where;
  2. recomputes pooled knockout tightening per PC without them (plus a robust version using the median absolute deviation);
  3. builds an NTC null for the per-knockout distance change: 1,000 random sets of 200 NTC cells, same before/after distance;
  4. adds a noise-corrected centroid distance (squared distance minus the sampling noise expected from the cell spread);
  5. compares cycle-shifting (q<0.05) vs other knockouts with a Mann-Whitney test, on outlier-free centroids;
  6. recomputes knockout feature effects robustly (per-knockout median instead of mean) for the tail-overlap question, and
     where along the clock the strongest tail features peak.
usage: python phase_push_ko_check.py"""
import os, time
import numpy as np, pandas as pd, anndata as ad, h5py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import mannwhitneyu, spearmanr

OUT = "phase_push_check"; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0); NB = 48; T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
L = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")]; L = L[~L.index.duplicated()]
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str); ph = ph[~ph.obs_names.duplicated()]
ph = ph[ph.obs_names.isin(L.index)]; present = np.isfinite(np.asarray(ph.X)).mean(0) >= .85
Xn = np.asarray(ph.X, np.float32)[:, present]; tile = ph.obs["tile_pheno"].astype(str).values; Q = L.loc[ph.obs_names]
cc = Q["cc_phase"].astype(float).values
imp = SimpleImputer(strategy="median").fit(Xn); zs = StandardScaler().fit(imp.transform(Xn))
emb = lambda X: np.nan_to_num(zs.transform(imp.transform(X)))
pca = PCA(50, random_state=0).fit(emb(Xn)); Zn = pca.transform(emb(Xn))
log(f"NTC {len(Xn):,} cells, {present.sum()} features")


def onehot_circ(t, nb=NB): b = np.floor(np.mod(t, 1) * nb).astype(int) % nb; return np.eye(nb)[b][:, 1:]
def fitB(A, Z): return np.linalg.lstsq(np.c_[A, np.ones(len(A))], Z, rcond=None)[0][:-1]
def wfv(Z, g): D = pd.DataFrame(Z); return (D - D.groupby(g).transform("mean")).var().values
def wf_mad(Z, g): D = pd.DataFrame(Z); R = D - D.groupby(g).transform("median"); return ((R - R.median()).abs().median() * 1.4826).values ** 2


bf_imp = lambda X: np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0); t = 2 * np.pi * cc
oof = np.full(len(Xn), np.nan)
for k, (a, b) in enumerate(GroupKFold(5).split(Xn, groups=tile)):
    s_ = StandardScaler().fit(bf_imp(Xn[a])); A_, B_ = s_.transform(bf_imp(Xn[a])), s_.transform(bf_imp(Xn[b]))
    ps = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(A_, np.sin(t[a])).predict(B_)
    pc = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(A_, np.cos(t[a])).predict(B_)
    oof[b] = np.mod(np.arctan2(ps, pc), 2 * np.pi) / (2 * np.pi); log(f"fold {k + 1}/5")
An = onehot_circ(oof); Bh = fitB(An, Zn); Zn_r = Zn - An @ Bh
s_full = StandardScaler().fit(bf_imp(Xn))
hs = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(s_full.transform(bf_imp(Xn)), np.sin(t))
hc = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(s_full.transform(bf_imp(Xn)), np.cos(t))

ko = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r"); ob = ko.obs.copy(); ob["row"] = np.arange(len(ob)); ob["gene"] = ob["gene"].astype(str).str.upper()
pick = ob.drop_duplicates("cell_key").groupby("gene", group_keys=False).apply(lambda d: d.sample(min(len(d), 200), random_state=0)).sort_values("row")
f = h5py.File("OPS0077_allko_phase.h5ad", "r"); r_ = pick.row.values
Xk_all = np.vstack([f["X"][r_[i:i + 5000]] for i in range(0, len(r_), 5000)]).astype(np.float32); f.close()
Xk = Xk_all[:, present]; gk = pick.gene.values; tk = pick.tile_pheno.astype(str).values
Ek = emb(Xk); Zk = pca.transform(Ek)
pk = np.mod(np.arctan2(hs.predict(s_full.transform(bf_imp(Xk))), hc.predict(s_full.transform(bf_imp(Xk)))), 2 * np.pi) / (2 * np.pi)
Ak = onehot_circ(pk); Zk_r = Zk - Ak @ Bh
log(f"knockout cells {len(Xk):,} in {len(set(gk))} knockouts")

# 1. outlier cells
nn = np.linalg.norm(Zn[:, :10], axis=1); nk = np.linalg.norm(Zk[:, :10], axis=1); thr = nn.max()
out = nk > thr; maxz = np.abs(Ek).max(1)
log(f"NTC max PC1-10 norm {thr:.1f}; knockout cells above it: {out.sum()} of {len(out):,} in {len(set(gk[out]))} knockouts; "
    f"max |z| of those cells: median {np.median(maxz[out]):.0f}, max {maxz.max():.3g}")
O = pd.DataFrame({"gene": gk[out], "tile_pheno": tk[out], "pc1_10_norm": nk[out], "max_abs_z": maxz[out]}).sort_values("pc1_10_norm", ascending=False)
O.to_csv(f"{OUT}/knockout_outlier_cells.csv", index=False)

# 2. pooled tightening, outliers excluded
keep = ~out
def red(Z0, Z1, g, fn): return 100 * (1 - fn(Z1, g) / fn(Z0, g))
rows = []
rn = red(Zn, Zn_r, tile, wfv); rows.append(dict(cells="NTC (gradient-boosting predicted, out of fold)", measure="variance", n_cells=len(Zn), **{f"PC{k+1}": round(rn[k], 1) for k in range(10)}))
rn = red(Zn, Zn_r, tile, wf_mad); rows.append(dict(cells="NTC (gradient-boosting predicted, out of fold)", measure="robust (MAD^2)", n_cells=len(Zn), **{f"PC{k+1}": round(rn[k], 1) for k in range(10)}))
for lab, m in [("knockouts pooled, all cells", np.ones(len(Zk), bool)), ("knockouts pooled, outlier cells removed", keep)]:
    for meas, fn in [("variance", wfv), ("robust (MAD^2)", wf_mad)]:
        rk = red(Zk[m], Zk_r[m], tk[m], fn); rows.append(dict(cells=lab, measure=meas, n_cells=int(m.sum()), **{f"PC{k+1}": round(rk[k], 1) for k in range(10)}))
# control: knockout predicted cycle permuted across knockout cells, same NTC coefficients (should not tighten; adds noise)
Zk_p = Zk - onehot_circ(RNG.permutation(pk)) @ Bh
rk = red(Zk[keep], Zk_p[keep], tk[keep], wfv); rows.append(dict(cells="knockouts, predicted cycle permuted (control)", measure="variance", n_cells=int(keep.sum()), **{f"PC{k+1}": round(rk[k], 1) for k in range(10)}))
P = pd.DataFrame(rows); P["top3_mean"] = P[["PC1", "PC2", "PC3"]].mean(1).round(1)
P.to_csv(f"{OUT}/pooled_per_pc_reduction_checked.csv", index=False); log("pooled per-PC within-field variance removed (%):\n" + P.to_string(index=False))
# predicted cycle distribution: NTC vs knockouts
hist = lambda p: np.histogram(np.mod(p, 1), bins=8, range=(0, 1))[0] / len(p)
log("predicted cycle occupancy by eighth of the clock, NTC out-of-fold:", np.round(hist(oof), 3), " knockouts:", np.round(hist(pk[keep]), 3))

# 3./4. per-knockout distances, outlier cells removed, with noise correction and NTC null
lib = pd.read_csv("ko_library/ko_library_cycle.csv").set_index("gene")
P10 = slice(0, 10); mu0, mu1 = Zn[:, P10].mean(0), Zn_r[:, P10].mean(0)
def dist_stats(a0, a1, n_ref):
    d0, d1 = np.linalg.norm(a0.mean(0) - mu0), np.linalg.norm(a1.mean(0) - mu1)
    # expected squared distance from sampling noise alone: trace of covariance / n (knockout) ; NTC reference noise negligible (n=35k)
    e0, e1 = np.trace(np.cov(a0.T)) / len(a0), np.trace(np.cov(a1.T)) / len(a1)
    c0, c1 = np.sqrt(max(d0 ** 2 - e0, 0)), np.sqrt(max(d1 ** 2 - e1, 0))
    return d0, d1, c0, c1, np.sqrt(e0), np.sqrt(e1)
rows = []
for g_ in np.unique(gk):
    m = (gk == g_) & keep; d0, d1, c0, c1, e0, e1 = dist_stats(Zk[m][:, P10], Zk_r[m][:, P10], len(Zn))
    rows.append(dict(gene=g_, n=int(m.sum()), n_outlier_cells_removed=int(((gk == g_) & out).sum()), cycle_shift_q=lib.loc[g_, "q_cycle"] if g_ in lib.index else np.nan,
                     dist_before=d0, dist_after=d1, noise_floor_before=e0, noise_floor_after=e1, corrected_dist_before=c0, corrected_dist_after=c1))
G = pd.DataFrame(rows); G["cycle_shifting"] = G.cycle_shift_q < .05
G["dist_change_pct"] = 100 * (G.dist_after / G.dist_before - 1)
G["corrected_dist_change_pct"] = np.where(G.corrected_dist_before > 0, 100 * (G.corrected_dist_after / G.corrected_dist_before.where(G.corrected_dist_before > 0) - 1), np.nan)
G["above_noise_before"] = G.corrected_dist_before > 0
G.round(4).to_csv(f"{OUT}/knockouts_distance_checked.csv", index=False)
nul = []
for i in range(1000):
    j = RNG.choice(len(Zn), 200, replace=False); d0, d1, c0, c1, e0, e1 = dist_stats(Zn[j][:, P10], Zn_r[j][:, P10], len(Zn))
    nul.append(dict(draw=i, dist_before=d0, dist_after=d1, dist_change_pct=100 * (d1 / d0 - 1)))
N = pd.DataFrame(nul); N.round(4).to_csv(f"{OUT}/ntc_null_200cell_draws.csv", index=False)
log(f"NTC null (1,000 random 200-cell NTC sets): distance before median {N.dist_before.median():.2f}, after {N.dist_after.median():.2f}, "
    f"change median {N.dist_change_pct.median():.1f}% (IQR {N.dist_change_pct.quantile(.25):.1f} to {N.dist_change_pct.quantile(.75):.1f})")
S = G.groupby("cycle_shifting").agg(n_knockouts=("gene", "size"), dist_before_median=("dist_before", "median"), dist_change_pct_median=("dist_change_pct", "median"),
                                     n_above_noise=("above_noise_before", "sum"), corrected_change_pct_median=("corrected_dist_change_pct", "median")).round(2)
log("per knockout, outlier cells removed:\n" + S.to_string())
a, b = G[G.cycle_shifting].dist_change_pct, G[~G.cycle_shifting].dist_change_pct
log(f"Mann-Whitney cycle-shifting vs other, raw distance change: p = {mannwhitneyu(a, b).pvalue:.2g}")
a, b = G[G.cycle_shifting].corrected_dist_change_pct.dropna(), G[~G.cycle_shifting].corrected_dist_change_pct.dropna()
log(f"Mann-Whitney, noise-corrected distance change (knockouts above noise): n {len(a)} vs {len(b)}, p = {mannwhitneyu(a, b).pvalue:.2g}")
big = G[G.corrected_dist_before > 2 * G.noise_floor_before]
log(f"knockouts with corrected distance > 2x noise floor: {len(big)}; their median corrected change {big.corrected_dist_change_pct.median():.1f}% "
    f"(cycle-shifting {big[big.cycle_shifting].corrected_dist_change_pct.median():.1f}% n={big.cycle_shifting.sum()}, other {big[~big.cycle_shifting].corrected_dist_change_pct.median():.1f}% n={(~big.cycle_shifting).sum()})")
log("largest pull toward NTC among knockouts > 2x noise floor:\n" + big.nsmallest(12, "corrected_dist_change_pct")[["gene", "cycle_shift_q", "dist_before", "dist_after", "corrected_dist_before", "corrected_dist_after", "corrected_dist_change_pct"]].round(2).to_string(index=False))
log("largest push away from NTC among knockouts > 2x noise floor:\n" + big.nlargest(6, "corrected_dist_change_pct")[["gene", "cycle_shift_q", "corrected_dist_before", "corrected_dist_after", "corrected_dist_change_pct"]].round(2).to_string(index=False))
log(f"Spearman, -log10 q_cycle vs corrected distance change (n {big.shape[0]}): {spearmanr(-np.log10(big.cycle_shift_q), big.corrected_dist_change_pct)[0]:.2f}")

# 6. robust knockout feature effects for the tail question
T = pd.read_csv("phase_push/phase_feature_cycle_tail.csv")
X = pd.DataFrame(np.asarray(ph.X, float), columns=ph.var_names, index=ph.obs_names)[T.feature]; X = X.fillna(X.median())
cols = [list(ko.var_names).index(c) for c in T.feature]
Xk2 = pd.DataFrame(Xk_all[keep][:, cols], columns=T.feature).fillna(X.median()); gk2 = gk[keep]
mu, sd = X.median(), (X - X.median()).abs().median() * 1.4826
sd = sd.where(sd > 0, X.std())
E = ((Xk2.groupby(gk2).median() - mu) / sd).abs()
T["median_abs_KO_effect_robust"] = E.median().values
top = T.nlargest(100, "median_abs_KO_effect_robust")
log(f"robust (per-knockout median, MAD-scaled): of the 100 features that best separate knockouts from NTC, {int(top.in_tail.sum())} are tail features "
    f"(tail = {T.in_tail.sum()} of {len(T)}, {100 * T.in_tail.mean():.0f}%); families of top 100: {top.family.value_counts().head(6).to_dict()}")
log(f"robust Spearman, cycle loss (1 - ratio) vs knockout effect: {spearmanr(1 - T.ratio_cc_phase, T.median_abs_KO_effect_robust)[0]:.2f}; "
    f"partial check within non-tail features only: {spearmanr(1 - T[~T.in_tail].ratio_cc_phase, T[~T.in_tail].median_abs_KO_effect_robust)[0]:.2f}")
# where along the clock do the strongest tail features peak (bin means, 16 bins)
b16 = np.floor(np.mod(cc, 1) * 16).astype(int) % 16
tt = T.nsmallest(30, "ratio_cc_phase").feature
BM = X[tt].groupby(b16).mean(); BMz = (BM - X[tt].mean()) / X[tt].std()
peak = (BMz.abs().idxmax() + .5) / 16
T["clock_peak_bin_center"] = np.nan; T.loc[T.feature.isin(tt), "clock_peak_bin_center"] = T.loc[T.feature.isin(tt), "feature"].map(peak).values
log("clock position of largest deviation, 30 strongest tail features: " + pd.Series(np.round(peak.values, 3)).value_counts().sort_index().to_string().replace("\n", "; "))
BMz.round(3).to_csv(f"{OUT}/tail_top30_clock_profile_z.csv")
size = T[T.feature.str.contains("area|perimeter|axis|diameter|volume|mass", regex=True)]
log(f"size-like features (area/perimeter/axis/diameter/volume/mass): n {len(size)}, median ratio {size.ratio_cc_phase.median():.3f}, in tail {int(size.in_tail.sum())}; "
    f"op_cell_area ratio {T.set_index('feature').loc['op_cell_area', 'ratio_cc_phase']:.3f}")
T.round(4).to_csv(f"{OUT}/phase_feature_cycle_tail_robust.csv", index=False)
log("DONE")
