#!/usr/bin/env python
"""Building on Diego's and Brian's analyses (hackathon doc items 7 and 6).

Diego (item 7): regressing cc_phase out of NTC single-cell embeddings (PCA of features) tightens the top PCs (within-field
variance of PC1-PC3: -28/-26/-33% in 4i, -9/-19/-9% in phase contrast). Here: the same readout, with the cycle covariate varied.
  cc_phase (Diego's)   = Louise's final clock: principal curve through 4i pRb / p21 / phospho-Rb, DNA, and per-channel PCs of the
                         pRb, Rb and p21 features (per_cell_pca_embeddings.csv), so it is partly built from the 4i features
  cc_phase_dna         = principal curve through 4i markers + DNA (no PC layer)
  cc_phase_markers     = principal curve through 4i markers only
  DNA content only     = measured DAPI (rounds 2-4), cubic; independent of both the 4i and the phase features
  shuffled cc_phase    = control
  Readout per embedding (phase contrast; 4i panel = the NTC pRb/Rb/p21 OrganelleProfiler file): within-field variance of PC1-PC10
  before / after, and the total over all PCs. Then the knockouts (which Diego did not have): same axes, each modality's own
  predicted cycle, pooled knockout cells vs NTC.
Brian (item 6, the 4i part): where measured-2N NTC cells land on each 4i clock, split into his Early G1 (pRb low) and Late G1
  (pRb high); and how often a 2N cell lands in the S or G2/M half of the clock.
usage: python build_on_team.py [OUT_DIR]"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, h5py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "build_on_team"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0); NPC = 50


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
L = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")]; L = L[~L.index.duplicated()]


def load_ntc(fn):
    a = ad.read_h5ad(find(fn))
    if "cell_key" in a.obs: key = a.obs["cell_key"].astype(str).values
    elif {"well", "segmentation_id"} <= set(a.obs.columns): key = (a.obs["well"].astype(str) + "|" + a.obs["segmentation_id"].astype("int64").astype(str)).values
    else: key = np.array(["|".join(s.split("|")[:2]) for s in a.obs_names.astype(str)])
    a.obs_names = key; a = a[~a.obs_names.duplicated()]; a = a[a.obs_names.isin(L.index)]
    X = np.asarray(a.X, np.float32); keep = np.isfinite(X).mean(0) >= .85
    print(f"{fn}: {a.n_obs:,} NTC cells with a clock, {keep.sum()} features", flush=True)
    return pd.DataFrame(X[:, keep], index=a.obs_names, columns=np.where(keep)[0]), a.obs["tile_pheno"].astype(str).values if "tile_pheno" in a.obs else None


EMB = {"phase contrast": load_ntc("OPS0077_NTC_phase_OrganelleProfiler.h5ad"), "4i panel (pRb, Rb, p21)": load_ntc("OPS0077_NTC_OrganelleProfiler_pRb_Rb_p21.h5ad")}
common = EMB["phase contrast"][0].index.intersection(EMB["4i panel (pRb, Rb, p21)"][0].index)
tile = pd.Series(EMB["phase contrast"][1], index=EMB["phase contrast"][0].index).loc[common].values
Q = L.loc[common]; print(f"NTC cells with both embeddings and a clock: {len(common):,} | fields {len(set(tile)):,}", flush=True)
dna = Q["dna_rel_R234"].astype(float).values
COV = {"cc_phase (Diego; built from 4i markers, DNA and 4i PCs)": ("angle", 2 * np.pi * Q["cc_phase"].astype(float).values),
       "cc_phase_dna (4i markers + DNA)": ("angle", 2 * np.pi * Q["cc_phase_dna"].astype(float).values),
       "cc_phase_markers (4i markers only)": ("angle", 2 * np.pi * Q["cc_phase_markers"].astype(float).values),
       "DNA content only (independent of both embeddings)": ("dna", dna),
       "shuffled cc_phase (control)": ("angle", RNG.permutation(2 * np.pi * Q["cc_phase"].astype(float).values))}


def design(kind, v):
    if kind == "angle": return np.c_[np.sin(v), np.cos(v), np.sin(2 * v), np.cos(2 * v), np.ones(len(v))]
    z = (v - np.nanmean(v)) / np.nanstd(v); z = np.nan_to_num(z); return np.c_[z, z ** 2, z ** 3, np.ones(len(v))]


def within_field_var(Z, groups):
    D = pd.DataFrame(Z); return (D - D.groupby(groups).transform("mean")).var().values


rows, fits = [], {}
for name, (Xdf, _) in EMB.items():
    X = Xdf.loc[common].values; imp = SimpleImputer(strategy="median").fit(X); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(X))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5); pca = PCA(NPC, random_state=0).fit(f(X)); Z = pca.transform(f(X))
    fits[name] = (imp, sc, pca)
    v0 = within_field_var(Z, tile)
    for cname, (kind, v) in COV.items():
        A = design(kind, v); B = np.linalg.lstsq(A, Z, rcond=None)[0]; Zr = Z - A[:, :-1] @ B[:-1]
        v1 = within_field_var(Zr, tile); red = 100 * (1 - v1 / v0)
        rows.append(dict(embedding=name, cycle_covariate=cname, **{f"PC{k + 1}": round(red[k], 1) for k in range(10)},
                         top3_mean=round(red[:3].mean(), 1), total_all_50_PCs=round(100 * (1 - v1.sum() / v0.sum()), 2),
                         total_within_field_variance_share_of_PC1_3=round(100 * v0[:3].sum() / v0.sum(), 1)))
    print(name, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/diego_per_pc_reduction.csv", index=False)
print("\n% reduction of within-field variance per PC (NTC):\n", D.to_string(index=False), flush=True)

# knockouts (phase contrast only here; 4i knockouts are in notebook 19): same NTC axes, phase-predicted cycle, pooled knockout cells
ph_name = "phase contrast"; imp, sc, pca = fits[ph_name]
Xn = EMB[ph_name][0].loc[common].values; Zn = pca.transform(np.clip(sc.transform(imp.transform(Xn)), -5, 5))
th = COV["cc_phase (Diego; built from 4i markers, DNA and 4i PCs)"][1]
rs = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Zn, np.sin(th)); rc = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Zn, np.cos(th))
ko = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r"); ob = ko.obs.copy(); ob["row"] = np.arange(len(ob))
pick = ob.drop_duplicates("cell_key").groupby("gene", group_keys=False).apply(lambda d: d.sample(min(len(d), 100), random_state=0))
phase_vars = np.array(ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad"), backed="r").var_names)
ntc_cols = EMB[ph_name][0].columns.values                                    # positions of the kept columns in the full var list
a_ntc = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad"), backed="r"); keepmask = np.zeros(len(phase_vars), bool); keepmask[ntc_cols] = True
f_ = h5py.File("OPS0077_allko_phase.h5ad", "r"); rows_ = np.sort(pick.row.values)
Xk = np.vstack([f_["X"][rows_[i:i + 5000]] for i in range(0, len(rows_), 5000)])[:, keepmask]; f_.close()
tk = ob.set_index("row").loc[rows_, "tile_pheno"].astype(str).values
Zk = pca.transform(np.clip(sc.transform(imp.transform(Xk)), -5, 5)); pk = np.arctan2(rs.predict(Zk), rc.predict(Zk)); pn = np.arctan2(rs.predict(Zn), rc.predict(Zn))
krow = []
for lab_, Zx, tx, px in [("NTC (cycle = phase-predicted)", Zn, tile, pn), ("knockouts, pooled (cycle = phase-predicted)", Zk, tk, pk)]:
    A = design("angle", px); An = design("angle", pn); B = np.linalg.lstsq(An, Zn, rcond=None)[0]
    v0 = within_field_var(Zx, tx); v1 = within_field_var(Zx - A[:, :-1] @ B[:-1], tx); red = 100 * (1 - v1 / v0)
    krow.append(dict(cells=lab_, n=len(Zx), **{f"PC{k + 1}": round(red[k], 1) for k in range(10)}, top3_mean=round(red[:3].mean(), 1), total_all_50_PCs=round(100 * (1 - v1.sum() / v0.sum()), 2)))
K = pd.DataFrame(krow); K.to_csv(f"{OUT}/diego_knockouts_phase.csv", index=False); print("\nphase contrast, NTC vs knockouts (NTC axes):\n", K.to_string(index=False), flush=True)

# Brian item 6, the 4i part: where do measured-2N cells land on each clock?
prb = Q["z_pRb"].astype(float).values; two = dna < .25; thr = np.nanpercentile(prb[two], 35)
grp = np.where(two & (prb <= thr), "2N, pRb low (Early G1)", np.where(two, "2N, pRb high (Late G1)", np.where(dna > .65, "4N (G2/M)", "S (2N-4N)")))
B6 = []
for cname in ["cc_phase (Diego; built from 4i markers, DNA and 4i PCs)", "cc_phase_dna (4i markers + DNA)", "cc_phase_markers (4i markers only)"]:
    q = np.minimum((np.mod(COV[cname][1] / (2 * np.pi), 1) * 4).astype(int), 3)
    for g_ in ["2N, pRb low (Early G1)", "2N, pRb high (Late G1)", "S (2N-4N)", "4N (G2/M)"]:
        m = grp == g_; h = np.bincount(q[m], minlength=4) / m.sum() * 100
        B6.append(dict(clock=cname, measured_state=g_, n=int(m.sum()), **{f"% in quarter {k + 1} ({['early G1', 'late G1', 'S', 'G2/M'][k]})": round(h[k], 1) for k in range(4)}))
B6 = pd.DataFrame(B6); B6.to_csv(f"{OUT}/brian_2N_cells_on_4i_clocks.csv", index=False)
print("\nmeasured state vs quarter of each 4i clock (row %):\n", B6.to_string(index=False), flush=True)

# figure: Diego's bars for each covariate
fig, ax = plt.subplots(1, 2, figsize=(17, 5.2), sharey=True)
cols_ = ["tab:red", "tab:orange", "tab:purple", "tab:blue", "0.6"]
for a_, name in zip(ax, EMB):
    d = D[D.embedding == name]; x = np.arange(10); w = .16
    for k_, (_, r) in enumerate(d.iterrows()):
        a_.bar(x + (k_ - 2) * w, [r[f"PC{i + 1}"] for i in range(10)], w, color=cols_[k_], label=r.cycle_covariate)
    a_.set_xticks(x); a_.set_xticklabels([f"PC{i + 1}" for i in range(10)]); a_.axhline(0, color="k", lw=.5)
    a_.set_title(f"{name}: within-field variance removed per PC (NTC)"); a_.set_ylabel("% reduction")
ax[0].legend(fontsize=7, loc="upper right"); plt.tight_layout(); plt.savefig(f"{OUT}/diego_per_pc_by_covariate.png", dpi=110)
print("DONE")
