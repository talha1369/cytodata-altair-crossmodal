#!/usr/bin/env python
"""Does normalising / weighting by cell-cycle state
  Q1  reduce field (tile) and well heterogeneity of control-cell morphology,
  Q2  find perturbations that change the cell-cycle DISTRIBUTION of the population, and
  Q3  separate them from perturbations that change the cells THEMSELVES (morphology at a matched cycle state)?

Two normalisations of the cycle:
  regress   remove from every morphology dimension the part explained by the cycle (sin/cos harmonics 1-2, fit on controls)
  reweight  weight reference cells so their cycle distribution matches the target population (12 cycle bins)
Decomposition for a perturbation g (morphology centroids, PCA space):
  total shift D = mean(g) - mean(ctrl) = C + W
  C = mean(ctrl reweighted to g's cycle distribution) - mean(ctrl)   "what the cycle shift alone would do"   (Q2)
  W = mean(g) - mean(ctrl reweighted to g's cycle distribution)       "change at a matched cycle state"        (Q3)
Datasets
  OPS0077  phase-contrast morphology; cycle = morphology-predicted cc_phase (out-of-fold for NTC); measured 4i clock as a check
  Element  imaging features; cycle = MEASURED RNA clock (independent of morphology); TNFa stimulus + 5 knockouts
usage: python cycle_norm_questions.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, h5py
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "cycle_norm"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
NBIN = 12


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", ".", "cc_align"]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


def fit_transform(Xtr, k=100):
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xtr))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5)
    pca = PCA(k, random_state=0).fit(f(Xtr))
    return lambda Z: pca.transform(f(Z)).astype(np.float32)


def fit_angle(Z, t):
    rs = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.sin(t)); rc = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.cos(t))
    return lambda W: np.mod(np.arctan2(rs.predict(W), rc.predict(W)), 2 * np.pi) / (2 * np.pi)


def harm(p):
    t = 2 * np.pi * p; return np.c_[np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)]


def regress_coef(Z, p):
    return np.linalg.lstsq(np.c_[harm(p), np.ones(len(p))], Z, rcond=None)[0][:4]


def bins(p): return np.floor(np.mod(p, 1) * NBIN).astype(int) % NBIN


def hist(p, w=None):
    h = np.bincount(bins(p), weights=w, minlength=NBIN).astype(float); return h / h.sum()


def rw_weights(p_src, p_tgt):
    """weights on the source cells so that their cycle histogram equals the target's"""
    hs, ht = hist(p_src), hist(p_tgt); r = np.where(hs > 0, ht / np.maximum(hs, 1e-12), 0)
    return r[bins(p_src)]


def group_hist(codes, p, G):
    H = np.bincount(codes * NBIN + bins(p), minlength=G * NBIN).reshape(G, NBIN).astype(float)
    return H / H.sum(1, keepdims=True)


def between_frac(Z, g, p=None):
    """share of total variance that lies between group centroids; with p, each group's centroid is computed after
    reweighting its cells to the pooled cycle distribution"""
    codes, _ = pd.factorize(g); G = codes.max() + 1; n = np.bincount(codes, minlength=G)
    M = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(G, len(codes)))
    if p is None: w = np.ones(len(codes))
    else:
        Hs = group_hist(codes, p, G); r = np.where(Hs > 0, hist(p)[None, :] / np.maximum(Hs, 1e-12), 0); w = r[codes, bins(p)]
    ws = np.asarray(M @ w).ravel(); C = np.asarray(M @ (Z * w[:, None])) / np.maximum(ws, 1e-12)[:, None]
    mu = Z.mean(0); tot = float(np.trace(np.cov(Z.T)))
    return float(np.sum(n * np.sum((C - mu) ** 2, 1))) / len(Z) / tot


def late_share_sd(g, p):
    codes, _ = pd.factorize(g); return float(np.std(group_hist(codes, p, codes.max() + 1)[:, NBIN // 2:].sum(1)))


def heterogeneity_table(name, Z, p, groups, label, p_shuf, coef, coef_s, n_perm=10):
    """Q1: between-group (well / tile) variance share before and after each cycle normalisation, minus the share
    expected from sampling noise alone (same procedure on permuted group labels)"""
    arms = {"no cycle normalisation": (Z, None), "regress out cycle": (Z - harm(p) @ coef, None),
            "reweight to common cycle distribution": (Z, p), "regress out SHUFFLED cycle (control)": (Z - harm(p_shuf) @ coef_s, None)}
    rows = []
    for arm, (Za, pa) in arms.items():
        obs = between_frac(Za, groups, pa)
        floor = np.mean([between_frac(Za, RNG.permutation(groups), pa) for _ in range(n_perm)])
        rows.append(dict(dataset=name, grouping=label, n_groups=len(np.unique(groups)), arm=arm,
                         between_share_pct=round(100 * obs, 3), noise_floor_pct=round(100 * floor, 3), excess_pct=round(100 * (obs - floor), 3)))
    # does cycle composition itself differ between groups beyond sampling?
    obs_sd = late_share_sd(groups, p)
    perm_sd = np.mean([late_share_sd(RNG.permutation(groups), p) for _ in range(n_perm)])
    for r in rows: r["sd_across_groups_of_late_cycle_share_pct"] = round(100 * obs_sd, 2); r["same_if_random_pct"] = round(100 * perm_sd, 2)
    return rows


def decompose(Zg, pg, Zr, pr):
    D = Zg.mean(0) - Zr.mean(0); w = rw_weights(pr, pg); mrw = np.average(Zr, axis=0, weights=w)
    C = mrw - Zr.mean(0); W = Zg.mean(0) - mrw
    tv = 0.5 * np.abs(hist(pg) - hist(pr)).sum()
    return dict(cycle_TV=tv, total_shift=float(np.linalg.norm(D)), cycle_part=float(np.linalg.norm(C)), within_cycle_part=float(np.linalg.norm(W)),
                cycle_share_of_shift=float(C @ D / max(D @ D, 1e-12)))


def bh(pv):
    pv = np.asarray(pv, float); o = np.argsort(pv); m = len(pv); q = np.empty(m)
    q[o] = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]; return np.minimum(q, 1)


HET = []

# ====================================================================== OPS0077
L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
L = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")]
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str)
ph = ph[~ph.obs_names.duplicated()]
lab = ph.obs_names.isin(L.index)
present = np.isfinite(np.asarray(ph.X)[lab]).mean(0) >= 0.85
Xall = np.asarray(ph.X, np.float32)[:, present]
tile = ph.obs["tile_pheno"].astype(str).values; guide = ph.obs["sgRNA"].astype(str).values
well = np.array([k.split("|")[0] for k in ph.obs_names])
L = L.loc[ph.obs_names[lab]]; th = 2 * np.pi * L["cc_phase"].astype(float).values

oof = np.full(len(Xall), np.nan); li = np.where(lab)[0]; pos = {c: i for i, c in enumerate(li)}
for a, b in GroupKFold(5).split(Xall, groups=tile):
    tr = a[lab[a]]; T_ = fit_transform(Xall[tr]); P_ = fit_angle(T_(Xall[tr]), th[[pos[i] for i in tr]]); oof[b] = P_(T_(Xall[b]))
T = fit_transform(Xall[lab]); P = fit_angle(T(Xall[lab]), th); Zn = T(Xall)
print(f"OPS0077 NTC cells {len(Zn):,} | wells {len(np.unique(well))} | tiles {len(np.unique(tile))}", flush=True)

# Q1 OPS0077, predicted cycle on all NTC
p_sh = RNG.permutation(oof); cf = regress_coef(Zn, oof); cfs = regress_coef(Zn, p_sh)
tile_ok = pd.Series(tile).map(pd.Series(tile).value_counts()).values >= 30
HET += heterogeneity_table("OPS0077 (predicted cycle, all NTC)", Zn, oof, well, "well", p_sh, cf, cfs)
HET += heterogeneity_table("OPS0077 (predicted cycle, all NTC)", Zn[tile_ok], oof[tile_ok], tile[tile_ok], "field (tile, >=30 cells)",
                           p_sh[tile_ok], cf, cfs)
# same with the MEASURED 4i clock (labelled cells only), so the cycle is not derived from the morphology itself
Zl, pm = Zn[lab], L["cc_phase"].astype(float).values; pms = RNG.permutation(pm)
cfm, cfms = regress_coef(Zl, pm), regress_coef(Zl, pms)
tl = tile[lab]; tl_ok = pd.Series(tl).map(pd.Series(tl).value_counts()).values >= 30
HET += heterogeneity_table("OPS0077 (MEASURED 4i clock, labelled NTC)", Zl, pm, well[lab], "well", pms, cfm, cfms)
HET += heterogeneity_table("OPS0077 (MEASURED 4i clock, labelled NTC)", Zl[tl_ok], pm[tl_ok], tl[tl_ok], "field (tile, >=30 cells)", pms[tl_ok], cfm, cfms)
pd.DataFrame(HET).to_csv(f"{OUT}/q1_field_well_heterogeneity.csv", index=False)
print(pd.DataFrame(HET).to_string(index=False), flush=True)

# Q2/Q3 OPS0077 knockouts
kos = []
for fn, cls in [("OPS0077_genes_phase.h5ad", "arrest"), ("OPS0077_ctrlgenes_phase.h5ad", "control")]:
    k = ad.read_h5ad(find(fn)); kos.append(pd.DataFrame(np.asarray(k.X, np.float32)[:, present]).assign(
        gene=k.obs["gene"].astype(str).str.upper().values, cell_key=k.obs["cell_key"].astype(str).values, cls=cls))
K = pd.concat(kos, ignore_index=True).drop_duplicates("cell_key").reset_index(drop=True)
Zk = T(K.drop(columns=["gene", "cell_key", "cls"]).values.astype(np.float32)); pk = P(Zk)
gidx = pd.Series(np.arange(len(Zn))).groupby(guide).apply(lambda s: s.values).to_dict()
gl = list(gidx)


def null_stats(n, reps=int(os.environ.get("NULL_REPS", 200))):
    out = []
    for _ in range(reps):
        RNG.shuffle(gl); ix = []
        for g_ in gl:
            ix.extend(gidx[g_].tolist())
            if len(ix) >= n: break
        ix = np.array(ix[:n]); m = np.ones(len(Zn), bool); m[ix] = False
        out.append(decompose(Zn[ix], oof[ix], Zn[m], oof[m]))
    return pd.DataFrame(out)


rows, nulls = [], {}
for g_, d in K.groupby("gene"):
    ix = d.index.values; key = max(int(round(len(ix) / 100) * 100), 100)
    if key not in nulls: nulls[key] = null_stats(key)
    r = decompose(Zk[ix], pk[ix], Zn, oof); nd = nulls[key]
    r.update(gene=g_, cls=d.cls.iloc[0], n=len(ix),
             p_cycle=(np.sum(nd.cycle_TV >= r["cycle_TV"]) + 1) / (len(nd) + 1),
             p_within=(np.sum(nd.within_cycle_part >= r["within_cycle_part"]) + 1) / (len(nd) + 1),
             within_vs_null_median=r["within_cycle_part"] / nd.within_cycle_part.median(),
             cycle_TV_vs_null_median=r["cycle_TV"] / nd.cycle_TV.median())
    rows.append(r)
R = pd.DataFrame(rows); R["q_cycle"] = bh(R.p_cycle); R["q_within"] = bh(R.p_within)
R["class"] = np.select([(R.q_cycle < .05) & (R.q_within >= .05), (R.q_cycle >= .05) & (R.q_within < .05), (R.q_cycle < .05) & (R.q_within < .05)],
                       ["cycle distribution only", "cells change, cycle distribution unchanged", "both"], "neither")
R.round(4).to_csv(f"{OUT}/q23_ops0077_knockout_decomposition.csv", index=False)
print("\nOPS0077 knockouts (FDR 5%, null = NTC guides pooled to the same cell count):")
print(pd.crosstab(R["class"], R.cls, margins=True).to_string())
print("\nmedian share of the morphology shift explained by the cycle shift, by class:")
print(R.groupby("class")["cycle_share_of_shift"].median().round(3).to_string())
cols = ["gene", "cls", "n", "cycle_TV", "q_cycle", "within_vs_null_median", "q_within", "cycle_share_of_shift", "class"]
print("\nlargest cycle-distribution changers:\n", R.nsmallest(10, "p_cycle").sort_values("cycle_TV", ascending=False)[cols].round(3).to_string(index=False))
print("\nlargest within-cycle changers:\n", R.sort_values("within_vs_null_median", ascending=False).head(10)[cols].round(3).to_string(index=False), flush=True)

# ====================================================================== Element (measured RNA clock)
h = h5py.File("/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu", "r")
obs = read_elem(h["obs"]); names = np.array(read_elem(h["mod"]["imaging"]["var"]).index.astype(str))
drop = np.array([any(s in n for s in ("Location", "Center", "ObjectNumber", "Bkg_", "Parent", "Children", "Number_", "Metadata")) for n in names])
raw = h["mod"]["imaging"]["layers"]["raw"][:, :][:, ~drop].astype(np.float32); raw[~np.isfinite(raw)] = np.nan
ok = (obs["qc_fail"].astype(str) != "True").values & np.isfinite(obs["cycle_position_deg"].astype(float).values)
raw = raw[ok]; o = obs[ok].reset_index(drop=True)
keep = np.isfinite(raw).mean(0) >= 0.85; raw = raw[:, keep]
pe = o["cycle_position_deg"].astype(float).values / 360
ctl = o["target_gene"].astype(str).isin(["NTC", "OR1I1"]).values; cond = o["condition"].astype(str).values
Te = fit_transform(raw[ctl & (cond == "baseline")], k=50); Ze = Te(raw); del raw
ewell = o["Well"].astype(str).values; etile = (o["Well"].astype(str) + "|" + o["tile_id"].astype(str)).values
print(f"\nElement cells {len(Ze):,} (QC pass, clock present) | features {int(keep.sum())} | wells x condition:\n",
      pd.crosstab(o["Well"], o["condition"]).to_string(), flush=True)

for c_ in ["baseline", "TNFa"]:
    m = ctl & (cond == c_); Zc, pc = Ze[m], pe[m]; ps = RNG.permutation(pc)
    cf_, cfs_ = regress_coef(Zc, pc), regress_coef(Zc, ps)
    HET += heterogeneity_table(f"Element {c_} controls (MEASURED RNA clock)", Zc, pc, ewell[m], "well", ps, cf_, cfs_)
    t_ok = pd.Series(etile[m]).map(pd.Series(etile[m]).value_counts()).values >= 30
    HET += heterogeneity_table(f"Element {c_} controls (MEASURED RNA clock)", Zc[t_ok], pc[t_ok], etile[m][t_ok], "field (tile, >=30 cells)", ps[t_ok], cf_, cfs_)
Hdf = pd.DataFrame(HET); Hdf.to_csv(f"{OUT}/q1_field_well_heterogeneity.csv", index=False)
print(Hdf[Hdf.dataset.str.startswith("Element")].to_string(index=False), flush=True)

# Q2/Q3 Element: TNFa vs baseline (controls), null = splitting baseline wells into two halves
erows = []
mb, mt = ctl & (cond == "baseline"), ctl & (cond == "TNFa")
r = decompose(Ze[mt], pe[mt], Ze[mb], pe[mb]); r.update(contrast="TNFa vs baseline (control guides)", n=int(mt.sum()))
bw = np.unique(ewell[mb]); import itertools
nl = []
for half in itertools.combinations(bw, len(bw) // 2):
    a_ = mb & np.isin(ewell, half); b_ = mb & ~np.isin(ewell, half); nl.append(decompose(Ze[a_], pe[a_], Ze[b_], pe[b_]))
nl = pd.DataFrame(nl); r.update(null="baseline wells split in halves", null_max_cycle_TV=nl.cycle_TV.max(), null_max_within=nl.within_cycle_part.max())
erows.append(r)
# knockouts within each condition, null = each control guide vs the other control guides
for c_ in ["baseline", "TNFa"]:
    mc = ctl & (cond == c_); cg = o["guide"].astype(str).values
    nl = pd.DataFrame([decompose(Ze[mc & (cg == g_)], pe[mc & (cg == g_)], Ze[mc & (cg != g_)], pe[mc & (cg != g_)]) for g_ in np.unique(cg[mc])])
    for g_ in sorted(set(o.target_gene.astype(str)) - {"NTC", "OR1I1"}):
        mg = (o.target_gene.astype(str).values == g_) & (cond == c_)
        r = decompose(Ze[mg], pe[mg], Ze[mc], pe[mc]); r.update(contrast=f"{g_} KO vs controls, {c_}", n=int(mg.sum()), null="single control guide vs other control guides",
                                                              null_max_cycle_TV=nl.cycle_TV.max(), null_max_within=nl.within_cycle_part.max())
        erows.append(r)
E = pd.DataFrame(erows)
E["cycle_beyond_null"] = E.cycle_TV > E.null_max_cycle_TV; E["within_beyond_null"] = E.within_cycle_part > E.null_max_within
E.round(4).to_csv(f"{OUT}/q23_element_decomposition.csv", index=False)
print("\nElement decomposition (measured RNA clock):\n", E[["contrast", "n", "cycle_TV", "null_max_cycle_TV", "cycle_beyond_null", "total_shift", "cycle_part",
                                                           "within_cycle_part", "null_max_within", "within_beyond_null", "cycle_share_of_shift"]].round(3).to_string(index=False))
print("DONE")
