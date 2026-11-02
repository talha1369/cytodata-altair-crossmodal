#!/usr/bin/env python
"""Brian's G1 concern: are G1-arrested knockout cells that keep growing (2N DNA, but the size / dry mass of a 4N cell) called
G2/M by the phase-contrast model?  Tested against MEASURED DNA content of knockout cells (ko_dna_measure.py), with NTC as reference.

Models (phase-contrast OrganelleProfiler features -> PCA-100 -> logistic regression, 4N vs not, trained on NTC with measured DNA):
  standard      as in ko_cycle_library.py
  size-balanced training weights make 2N and 4N cells equally common within each size bin (20 quantile bins of log phase
                cell area), so cell size alone cannot separate them
  size-removed  as standard, after regressing log cell area (and its square) out of every principal component (fit on NTC)
  DNA floor     (Brian) call a cell 4N only if the standard 4N probability > 0.5 AND a ridge-predicted DNA content (same PCA-100
                front end, target dna_rel_R234, RidgeCV) >= 0.32 (= 2.5N). Score for AUC: standard probability, halved for
                cells below the floor (so the >0.5 call equals the rule and the ranking within each group is kept)
Per knockout (cells with measured DNA): measured % 4N vs each model's predicted % 4N; among measured-2N cells, the mean predicted
4N probability and the share called 4N (false 4N), relative to NTC 2N cells; and how large the measured-2N cells are.
usage: python g1_hypertrophy_check.py [OUT_DIR]"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, h5py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "g1_hypertrophy"
os.makedirs(OUT, exist_ok=True)


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key"); L = L[(L["dna_reliable"] == True)]; L = L[~L.index.duplicated()]
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str); ph = ph[~ph.obs_names.duplicated()]
ph = ph[ph.obs_names.isin(L.index)]; var = np.array(ph.var_names)
present = np.isfinite(np.asarray(ph.X)).mean(0) >= .85
Xn = np.asarray(ph.X, np.float32)[:, present]; tile = ph.obs["tile_pheno"].astype(str).values
area_col = list(var[present]).index("op_cell_area")
dn = L.loc[ph.obs_names, "dna_rel_R234"].astype(float).values; y = (dn > .65).astype(int); two = dn < .25

K = pd.read_csv("ko_dna/ko_measured_dna_percell.csv", index_col=0)
ko = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r"); ob = ko.obs.copy(); ob["row"] = np.arange(len(ob)); ob = ob.drop_duplicates("cell_key").set_index("cell_key")
K = K[K.index.isin(ob.index)]; rows = ob.loc[K.index, "row"].values; o = np.argsort(rows)
f = h5py.File("OPS0077_allko_phase.h5ad", "r"); Xk = np.empty((len(rows), present.sum()), np.float32)
Xk[o] = np.vstack([f["X"][np.sort(rows)[i:i + 5000]] for i in range(0, len(rows), 5000)])[:, present]; f.close()
print(f"NTC {len(Xn):,} cells with measured DNA | knockout cells with measured DNA {len(K):,} in {K.gene.nunique()} knockouts", flush=True)


def front(Xtr):
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xtr))
    g = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5); p = PCA(100, random_state=0).fit(g(Xtr)); return lambda Z: p.transform(g(Z))


def logarea(X): return np.log2(np.clip(np.nan_to_num(X[:, area_col], nan=np.nanmedian(X[:, area_col])), 1, None))


def size_weights(la, yy, nb=20):
    b = np.digitize(la, np.quantile(la, np.linspace(0, 1, nb + 1)[1:-1])); w = np.ones(len(yy))
    for k in range(nb):
        m = b == k; n1, n0 = (yy[m] == 1).sum(), (yy[m] == 0).sum()
        if n1 and n0: w[m & (yy == 1)] = .5 * m.sum() / n1; w[m & (yy == 0)] = .5 * m.sum() / n0
        else: w[m] = 0
    return w


DNA_FLOOR = .32   # 2.5N in log2 units relative to the 2N peak


def fit(model, Xtr, ytr, dtr=None):
    T = front(Xtr); Z = T(Xtr); la = logarea(Xtr); w = None; R = None; rd = None
    if model == "size-removed":
        A = np.c_[la, la ** 2, np.ones(len(la))]; R = np.linalg.lstsq(A, Z, rcond=None)[0]; Z = Z - A @ R
    if model == "size-balanced": w = size_weights(la, ytr)
    if model == "DNA floor": rd = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, dtr)
    m = LogisticRegression(C=.1, max_iter=3000).fit(Z, ytr, sample_weight=w)
    def predict(X, return_dna=False):
        Zx = T(X)
        if R is not None: lx = logarea(X); Zx = Zx - np.c_[lx, lx ** 2, np.ones(len(lx))] @ R
        p = m.predict_proba(Zx)[:, 1]
        if rd is None: return p
        dp = rd.predict(Zx); s = np.where(dp >= DNA_FLOOR, p, .5 * p)
        return (s, dp) if return_dna else s
    return predict


rows_s, P_ntc, P_ko = [], {}, {}
dna_oof = np.full(len(Xn), np.nan)
for model in ["standard", "size-balanced", "size-removed", "DNA floor"]:
    oof = np.zeros(len(Xn))
    for a, b in GroupKFold(5).split(Xn, groups=tile):
        if model == "DNA floor": oof[b], dna_oof[b] = fit(model, Xn[a], y[a], dn[a])(Xn[b], return_dna=True)
        else: oof[b] = fit(model, Xn[a], y[a], dn[a])(Xn[b])
    P_ntc[model] = oof
    if model == "DNA floor": P_ko[model], dna_ko = fit(model, Xn, y, dn)(Xk, return_dna=True)
    else: P_ko[model] = fit(model, Xn, y, dn)(Xk)
    big2 = two & (logarea(Xn) > np.quantile(logarea(Xn)[two], .9))
    rows_s.append(dict(model=model, NTC_AUC_4N=round(roc_auc_score(y, oof), 3), NTC_pred_pct_4N=round(100 * (oof > .5).mean(), 1), NTC_measured_pct_4N=round(100 * y.mean(), 1),
                       NTC_2N_false_4N_pct=round(100 * (oof[two] > .5).mean(), 2), NTC_largest_2N_false_4N_pct=round(100 * (oof[big2] > .5).mean(), 2),
                       AUC_of_cell_size_alone=round(roc_auc_score(y, logarea(Xn)), 3),
                       NTC_measured_4N_called_4N_pct=round(100 * (oof[y == 1] > .5).mean(), 1)))
    print(rows_s[-1], flush=True)
ok_ = np.isfinite(dna_oof)
print(f"ridge DNA from phase (NTC, held out by tile, n={ok_.sum():,}): Spearman with measured {spearmanr(dna_oof[ok_], dn[ok_])[0]:.3f}, "
      f"Pearson {np.corrcoef(dna_oof[ok_], dn[ok_])[0, 1]:.3f}; % of NTC cells predicted >= 2.5N: {100 * (dna_oof >= DNA_FLOOR).mean():.1f}; "
      f"% of measured-4N NTC cells predicted >= 2.5N: {100 * (dna_oof[y == 1] >= DNA_FLOOR).mean():.1f}; "
      f"% of measured-2N NTC cells predicted >= 2.5N: {100 * (dna_oof[two] >= DNA_FLOOR).mean():.1f}", flush=True)
mk = np.isfinite(K.dna.values)
print(f"ridge DNA from phase (knockout cells, n={mk.sum():,}): Spearman with measured {spearmanr(dna_ko[mk], K.dna.values[mk])[0]:.3f}, "
      f"Pearson {np.corrcoef(dna_ko[mk], K.dna.values[mk])[0, 1]:.3f}", flush=True)
S = pd.DataFrame(rows_s); S.to_csv(f"{OUT}/model_summary_ntc.csv", index=False)

# knockouts
ex = pd.read_csv("genes_cc.csv"); exp = dict(zip(ex.Gene_Symbol.str.upper(), ex.Primary_Arrest_Phase))
la_k = logarea(Xk); ntc_2n_size = np.median(logarea(Xn)[two])
G = []
for g_, d in K.assign(i=np.arange(len(K))).groupby("gene"):
    i = d.i.values; m2 = d.dna.values < .25; meas4 = 100 * (d.dna.values > .65).mean() - 100 * y.mean()
    r = dict(gene=g_, expected_arrest_phase=exp.get(g_, ""), n=len(i), n_2N=int(m2.sum()), measured_d_pct_4N=meas4,
             size_2N_vs_NTC_2N_log2=float(np.median(la_k[i][m2]) - ntc_2n_size) if m2.sum() >= 10 else np.nan)
    for model in P_ko:
        p = P_ko[model][i]; r[f"{model}: predicted d%4N"] = 100 * (p > .5).mean() - 100 * (P_ntc[model] > .5).mean()
        r[f"{model}: false 4N among measured 2N (%)"] = 100 * (p[m2] > .5).mean() if m2.sum() >= 10 else np.nan
    G.append(r)
G = pd.DataFrame(G)
for model in P_ko: G[f"{model}: false 4N excess vs NTC (points)"] = G[f"{model}: false 4N among measured 2N (%)"] - float(S.loc[S.model == model, "NTC_2N_false_4N_pct"].iloc[0])
G.round(3).to_csv(f"{OUT}/knockouts_measured_vs_models.csv", index=False)
rows_c = []
for model in P_ko:
    ok = G[f"{model}: predicted d%4N"].notna()
    rows_c.append(dict(model=model, spearman_pred_vs_measured_d4N=round(spearmanr(G.loc[ok, f"{model}: predicted d%4N"], G.loc[ok, "measured_d_pct_4N"])[0], 3),
                       pearson=round(np.corrcoef(G.loc[ok, f"{model}: predicted d%4N"], G.loc[ok, "measured_d_pct_4N"])[0, 1], 3),
                       median_false_4N_excess_points=round(G[f"{model}: false 4N excess vs NTC (points)"].median(), 2),
                       G1_list_median_false_4N_excess=round(G.loc[G.expected_arrest_phase == "G1 Phase", f"{model}: false 4N excess vs NTC (points)"].median(), 2)))
C = pd.DataFrame(rows_c); C.to_csv(f"{OUT}/model_vs_measured_across_knockouts.csv", index=False)
print("\nacross knockouts, predicted vs measured change in % 4N:\n", C.to_string(index=False))
print("\ndo knockouts whose 2N cells are larger get more false 4N calls (standard model)? Spearman",
      round(spearmanr(G.size_2N_vs_NTC_2N_log2, G["standard: false 4N excess vs NTC (points)"], nan_policy="omit")[0], 2))
cols = ["gene", "expected_arrest_phase", "n", "measured_d_pct_4N", "standard: predicted d%4N", "size-balanced: predicted d%4N", "DNA floor: predicted d%4N", "size_2N_vs_NTC_2N_log2",
        "standard: false 4N excess vs NTC (points)", "size-balanced: false 4N excess vs NTC (points)", "DNA floor: false 4N excess vs NTC (points)"]
for model in P_ko:
    print(f"size of measured-2N cells vs false-4N excess, {model}: Spearman",
          round(spearmanr(G.size_2N_vs_NTC_2N_log2, G[f"{model}: false 4N excess vs NTC (points)"], nan_policy="omit")[0], 2), "| knockouts with >=10 measured-2N cells:", int(G.size_2N_vs_NTC_2N_log2.notna().sum()))
print("\nknockouts with the largest false-4N excess (standard model):\n", G.nlargest(12, "standard: false 4N excess vs NTC (points)")[cols].round(2).to_string(index=False))
print("\nG1-list knockouts:\n", G[G.expected_arrest_phase == "G1 Phase"][cols].round(2).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
for k_, (model, c_) in enumerate(zip(P_ko, ["tab:blue", "tab:green", "tab:orange", "tab:purple"])):
    ax[0].scatter(G[f"{model}: predicted d%4N"], G.measured_d_pct_4N, s=14, c=c_, alpha=.7, label=f"{model} (rho {C.spearman_pred_vs_measured_d4N.iloc[k_]:.2f})")
ax[0].axline((0, 0), slope=1, color="k", lw=.6); ax[0].set_xlabel("predicted change in % 4N (phase contrast)"); ax[0].set_ylabel("measured change in % 4N (DAPI)"); ax[0].legend(fontsize=8)
ax[0].set_title("which model tracks measured DNA across knockouts")
ax[1].scatter(G.size_2N_vs_NTC_2N_log2, G["standard: false 4N excess vs NTC (points)"], s=16, c=np.where(G.expected_arrest_phase == "G1 Phase", "tab:red", "0.5"))
for _, r in G.nlargest(8, "standard: false 4N excess vs NTC (points)").iterrows(): ax[1].annotate(r.gene, (r.size_2N_vs_NTC_2N_log2, r["standard: false 4N excess vs NTC (points)"]), fontsize=7)
ax[1].axhline(0, color="k", lw=.5); ax[1].axvline(0, color="k", lw=.5); ax[1].set_xlabel("size of measured-2N cells vs NTC 2N cells (log2 cell area)")
ax[1].set_ylabel("false 4N calls among measured-2N cells,\nexcess over NTC (points)"); ax[1].set_title("Brian's hypothesis: big 2N cells called G2/M? (red = G1 list)")
x_ = np.arange(len(S)); ax[2].bar(x_ - .2, S.NTC_2N_false_4N_pct, .4, label="all 2N NTC cells"); ax[2].bar(x_ + .2, S.NTC_largest_2N_false_4N_pct, .4, label="largest 10% of 2N NTC cells")
ax[2].set_xticks(x_); ax[2].set_xticklabels(S.model); ax[2].set_ylabel("% of measured-2N cells called 4N"); ax[2].legend(fontsize=8); ax[2].set_title("false 4N calls in NTC, by model")
plt.tight_layout(); plt.savefig(f"{OUT}/g1_hypertrophy_check.png", dpi=110)
print("DONE")
