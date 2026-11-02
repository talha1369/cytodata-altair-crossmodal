#!/usr/bin/env python
"""Diego's embedding analysis (item 7), pushed further on phase contrast, where we have every knockout cell.

Embedding: phase-contrast OrganelleProfiler features, median-imputed, z-scored, PCA-50 fit on NTC cells with a measured clock (Diego's
setting). Cycle removed by 48 clock-bin means (or 48 DNA-content bins), coefficients fit on NTC.
Cycle covariates
  measured cc_phase (4i)           NTC only
  measured DNA content only        NTC only
  predicted, gradient boosting     Brian's model (HistGB on phase features, NaN -> 0, z-scored), out of fold for NTC (folds by tile)
  predicted, ridge on PCA-100      ours, out of fold for NTC
Readouts
  NTC: within-field variance of PC1-PC10 before / after (Diego's bars), and his PC1-PC2 view raw vs cycle removed
  knockouts (all 1,000, up to 200 cells each, NTC axes, predicted cycle): pooled per-PC tightening; per knockout, distance of its
  centroid from NTC in PC1-10 and its spread, before / after; the cycle-shifting knockouts overlaid on NTC before / after
usage: python phase_embedding_push.py [OUT_DIR]"""
import os, sys, time
import numpy as np, pandas as pd, anndata as ad, h5py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "phase_push"
os.makedirs(OUT, exist_ok=True)
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
cc = Q["cc_phase"].astype(float).values; dna = Q["dna_rel_R234"].astype(float).values
imp = SimpleImputer(strategy="median").fit(Xn); zs = StandardScaler().fit(imp.transform(Xn))
emb = lambda X: np.nan_to_num(zs.transform(imp.transform(X)))
pca = PCA(50, random_state=0).fit(emb(Xn)); Zn = pca.transform(emb(Xn))
log(f"NTC {len(Xn):,} cells, {present.sum()} features; PC1-10 explain {100 * pca.explained_variance_ratio_[:10].sum():.1f}% of variance")


def onehot_circ(t, nb=NB): b = np.floor(np.mod(t, 1) * nb).astype(int) % nb; return np.eye(nb)[b][:, 1:]
def onehot_q(v, edges): b = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2); return np.eye(len(edges) - 1)[b][:, 1:]
def fitB(A, Z): return np.linalg.lstsq(np.c_[A, np.ones(len(A))], Z, rcond=None)[0][:-1]
def wfv(Z, g): D = pd.DataFrame(Z); return (D - D.groupby(g).transform("mean")).var().values


# predicted cycle for NTC, out of fold: gradient boosting (Brian) and ridge (ours)
bf_imp = lambda X: np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
oof_hgb, oof_rdg = np.full(len(Xn), np.nan), np.full(len(Xn), np.nan); t = 2 * np.pi * cc
for k, (a, b) in enumerate(GroupKFold(5).split(Xn, groups=tile)):
    s_ = StandardScaler().fit(bf_imp(Xn[a])); A_, B_ = s_.transform(bf_imp(Xn[a])), s_.transform(bf_imp(Xn[b]))
    ps = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(A_, np.sin(t[a])).predict(B_)
    pc = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(A_, np.cos(t[a])).predict(B_)
    oof_hgb[b] = np.mod(np.arctan2(ps, pc), 2 * np.pi) / (2 * np.pi)
    im2 = SimpleImputer(strategy="median").fit(Xn[a]); rs = RobustScaler(quantile_range=(10, 90)).fit(im2.transform(Xn[a]))
    g_ = lambda X: np.clip(rs.transform(im2.transform(X)), -5, 5); p_ = PCA(100, random_state=0).fit(g_(Xn[a]))
    Ra, Rb = p_.transform(g_(Xn[a])), p_.transform(g_(Xn[b]))
    oof_rdg[b] = np.mod(np.arctan2(RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Ra, np.sin(t[a])).predict(Rb),
                                   RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Ra, np.cos(t[a])).predict(Rb)), 2 * np.pi) / (2 * np.pi)
    log(f"fold {k + 1}/5")
err = lambda p: float(np.median(np.abs((p - cc + .5) % 1 - .5)) * 360)
log(f"held-out median error vs cc_phase: gradient boosting {err(oof_hgb):.1f} deg | ridge {err(oof_rdg):.1f} deg")

dq = np.nanquantile(dna, np.linspace(0, 1, NB + 1))
COV = {"measured cc_phase (4i)": onehot_circ(cc), "measured DNA content only": onehot_q(dna, dq),
       "predicted cycle, gradient boosting (Brian)": onehot_circ(oof_hgb), "predicted cycle, ridge (ours)": onehot_circ(oof_rdg),
       "shuffled cc_phase (control)": onehot_circ(RNG.permutation(cc))}
v0 = wfv(Zn, tile); rows = []; ZR = {}
for name, A in COV.items():
    B = fitB(A, Zn); Zr = Zn - A @ B; ZR[name] = Zr; red = 100 * (1 - wfv(Zr, tile) / v0)
    rows.append(dict(covariate=name, **{f"PC{k + 1}": round(red[k], 1) for k in range(10)}, top3_mean=round(red[:3].mean(), 1), total_50_PCs=round(100 * (1 - wfv(Zr, tile).sum() / v0.sum()), 2)))
N = pd.DataFrame(rows); N.to_csv(f"{OUT}/ntc_per_pc_reduction.csv", index=False); log("NTC, within-field variance removed per PC:\n" + N.to_string(index=False))

# full models for knockout cells
s_full = StandardScaler().fit(bf_imp(Xn)); hs = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(s_full.transform(bf_imp(Xn)), np.sin(t))
hc = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(s_full.transform(bf_imp(Xn)), np.cos(t))
ko = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r"); ob = ko.obs.copy(); ob["row"] = np.arange(len(ob)); ob["gene"] = ob["gene"].astype(str).str.upper()
pick = ob.drop_duplicates("cell_key").groupby("gene", group_keys=False).apply(lambda d: d.sample(min(len(d), 200), random_state=0)).sort_values("row")
f = h5py.File("OPS0077_allko_phase.h5ad", "r"); r_ = pick.row.values
Xk = np.vstack([f["X"][r_[i:i + 5000]] for i in range(0, len(r_), 5000)])[:, present].astype(np.float32); f.close()
gk = pick.gene.values; tk = pick.tile_pheno.astype(str).values
Zk = pca.transform(emb(Xk)); pk = np.mod(np.arctan2(hs.predict(s_full.transform(bf_imp(Xk))), hc.predict(s_full.transform(bf_imp(Xk)))), 2 * np.pi) / (2 * np.pi)
log(f"knockout cells {len(Xk):,} in {len(set(gk))} knockouts")
Bh = fitB(COV["predicted cycle, gradient boosting (Brian)"], Zn)                           # coefficients fit on NTC (out-of-fold predicted cycle)
Ak = onehot_circ(pk); Zk_r = Zk - Ak @ Bh; Zn_r = ZR["predicted cycle, gradient boosting (Brian)"]
Bs = fitB(COV["shuffled cc_phase (control)"], Zn); Zk_s = Zk - onehot_circ(RNG.permutation(pk)) @ Bs
vk0 = wfv(Zk, tk); red_k = 100 * (1 - wfv(Zk_r, tk) / vk0)
K1 = pd.DataFrame([dict(cells="knockouts pooled (gradient-boosting predicted cycle)", **{f"PC{k + 1}": round(red_k[k], 1) for k in range(10)},
                        top3_mean=round(red_k[:3].mean(), 1), total_50_PCs=round(100 * (1 - wfv(Zk_r, tk).sum() / vk0.sum()), 2))])
K1.to_csv(f"{OUT}/knockouts_pooled_per_pc_reduction.csv", index=False); log("knockouts pooled:\n" + K1.to_string(index=False))

lib = pd.read_csv("ko_library/ko_library_cycle.csv").set_index("gene")
P10 = slice(0, 10); mu_n0, mu_n1 = Zn[:, P10].mean(0), Zn_r[:, P10].mean(0)
rows = []
for g_ in np.unique(gk):
    m = gk == g_; a0, a1, as_ = Zk[m][:, P10], Zk_r[m][:, P10], Zk_s[m][:, P10]
    rows.append(dict(gene=g_, n=int(m.sum()), cycle_shift_q=lib.loc[g_, "q_cycle"] if g_ in lib.index else np.nan,
                     dist_to_NTC_before=float(np.linalg.norm(a0.mean(0) - mu_n0)), dist_to_NTC_after=float(np.linalg.norm(a1.mean(0) - mu_n1)),
                     spread_before=float(np.mean(np.linalg.norm(a0 - a0.mean(0), axis=1))), spread_after=float(np.mean(np.linalg.norm(a1 - a1.mean(0), axis=1))),
                     spread_shuffled=float(np.mean(np.linalg.norm(as_ - as_.mean(0), axis=1)))))
G = pd.DataFrame(rows); G["dist_change_pct"] = 100 * (G.dist_to_NTC_after / G.dist_to_NTC_before - 1); G["spread_change_pct"] = 100 * (G.spread_after / G.spread_before - 1)
G["spread_change_shuffled_pct"] = 100 * (G.spread_shuffled / G.spread_before - 1); G["cycle_shifting"] = G.cycle_shift_q < .05
G.round(4).to_csv(f"{OUT}/knockouts_pc1_10_distance_and_spread.csv", index=False)
S = G.groupby("cycle_shifting")[["dist_change_pct", "spread_change_pct", "spread_change_shuffled_pct"]].median().round(2)
log("per knockout, PC1-10, median change (%):\n" + S.to_string())
log("cycle-shifting knockouts pulled closest to NTC:\n" + G[G.cycle_shifting].nsmallest(10, "dist_change_pct")[["gene", "n", "dist_to_NTC_before", "dist_to_NTC_after", "dist_change_pct", "spread_change_pct"]].round(2).to_string(index=False))

# figures: Diego's view for phase contrast, and the overlays
sub = RNG.choice(len(Zn), min(8000, len(Zn)), replace=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True, sharey=True)
for a_, Z_, ttl in [(ax[0], Zn, "phase contrast: PCA, raw"), (ax[1], ZR["measured cc_phase (4i)"], "same axes, cc_phase regressed out (48 bins)")]:
    sc_ = a_.scatter(Z_[sub, 0], Z_[sub, 1], c=cc[sub], cmap="twilight", s=3, vmin=0, vmax=1); a_.set_title(ttl); a_.set_xlabel("PC1"); a_.set_ylabel("PC2")
plt.colorbar(sc_, ax=ax, label="cc_phase"); plt.savefig(f"{OUT}/phase_pca_raw_vs_regressed.png", dpi=110, bbox_inches="tight"); plt.close()
fig, ax = plt.subplots(figsize=(13, 4.8)); x = np.arange(10); w = .16
for k_, (_, r) in enumerate(N.iterrows()):
    ax.bar(x + (k_ - 2) * w, [r[f"PC{i + 1}"] for i in range(10)], w, label=r.covariate, color=["tab:red", "tab:blue", "tab:green", "tab:orange", "0.6"][k_])
ax.bar(x + 3 * w, [K1.iloc[0][f"PC{i + 1}"] for i in range(10)], w, label="knockouts pooled (gradient-boosting predicted)", color="tab:purple")
ax.set_xticks(x); ax.set_xticklabels([f"PC{i + 1}" for i in range(10)]); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("% within-field variance removed")
ax.set_title("phase contrast: within-field variance removed per PC"); ax.legend(fontsize=7.5, ncol=2); plt.tight_layout(); plt.savefig(f"{OUT}/phase_per_pc_by_covariate.png", dpi=110); plt.close()
show = [g for g in ["CDC45", "MTBP", "POLR1B", "RPL27A", "KIF23", "POLA2"] if g in set(gk)][:4]
fig, ax = plt.subplots(2, len(show), figsize=(4.4 * len(show), 8.4), sharex=True, sharey=True)
for j, g_ in enumerate(show):
    m = gk == g_
    for i, (Zn_, Zk_, lab_) in enumerate([(Zn, Zk, "raw"), (Zn_r, Zk_r, "cycle regressed out")]):
        a_ = ax[i, j]; a_.scatter(Zn_[sub, 0], Zn_[sub, 1], s=2, c="0.82"); a_.scatter(Zk_[m, 0], Zk_[m, 1], s=7, c="tab:red", alpha=.7)
        a_.scatter(*Zn_[:, :2].mean(0), marker="x", c="k", s=60); a_.scatter(*Zk_[m, :2].mean(0), marker="x", c="tab:red", s=60)
        r_ = G[G.gene == g_].iloc[0]; d_ = r_.dist_to_NTC_before if i == 0 else r_.dist_to_NTC_after
        a_.set_title(f"{g_} ({lab_})\ncentroid distance to NTC, PC1-10: {d_:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/phase_knockout_overlays.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 5))
for flag, c_, lab_ in [(False, "0.6", "other knockouts"), (True, "tab:red", "cycle-shifting knockouts (q < 0.05)")]:
    d_ = G[G.cycle_shifting == flag]; ax.scatter(d_.dist_to_NTC_before, d_.dist_to_NTC_after, s=10 if not flag else 22, c=c_, label=lab_, alpha=.8)
lim = [0, G[["dist_to_NTC_before", "dist_to_NTC_after"]].max().max() * 1.05]; ax.plot(lim, lim, "k--", lw=.7)
ax.set_xlabel("distance of knockout centroid to NTC (PC1-10), raw"); ax.set_ylabel("after removing the cycle"); ax.legend(fontsize=8)
ax.set_title("does removing the cycle pull knockouts back to NTC?"); plt.tight_layout(); plt.savefig(f"{OUT}/phase_knockout_distance.png", dpi=110)
log("DONE")
