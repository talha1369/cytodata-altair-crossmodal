#!/usr/bin/env python
"""Map the transcriptional cell-cycle representation of the Element dataset (measured RNA clock, cycleVI `cycle_position_deg`)
to morphological phenotypes in the same cells, and find their correlates in OPS0077.

A  prediction   imaging features -> RNA clock (circular ridge on PCA of robust-scaled features), leave-one-well-out over control
                cells (NTC + OR1I1, both conditions); feature-family ablations; error by clock arc; shuffled-label null.
B  phenotypes   per-feature profile around the RNA clock (control cells, baseline), amplitude in units of the cell-to-cell SD,
                reproducibility across two halves of the wells.
C  perturbations for TNFa and the 5 knockouts, the cycle shift predicted from morphology (out of fold by well) vs the shift
                measured by the RNA clock: does a morphology readout of the cycle recover real perturbation effects?
D  correlates   in OPS0077 (Altair, A549): for features measured in both datasets (nuclear DNA-stain texture and granularity,
                scale-free cell shape), the change between DNA-content states (S vs 2N, 4N vs 2N, by measured DNA in each
                dataset) is compared across datasets; and the OPS0077 phase-contrast features that mark the same states.
usage: python element_cycle_morphology.py [OUT_DIR]
"""
import os, sys, re
import numpy as np, pandas as pd, h5py, anndata as ad
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "element_cycle"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0); NB = 12


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


def fit_transform(Xtr, k=100):
    imp = SimpleImputer(strategy="median").fit(Xtr); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xtr))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5)
    pca = PCA(min(k, Xtr.shape[1]), random_state=0).fit(f(Xtr))
    return lambda Z: np.vstack([pca.transform(f(Z[i:i + 50000])) for i in range(0, len(Z), 50000)])


def fit_angle(Z, t):
    rs = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.sin(t)); rc = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, np.cos(t))
    return lambda W: np.mod(np.arctan2(rs.predict(W), rc.predict(W)), 2 * np.pi)


def err_deg(a, b): return np.abs((np.degrees(a - b) + 180) % 360 - 180)


def bins(rad): return (np.floor(np.mod(rad, 2 * np.pi) / (2 * np.pi) * NB).astype(int)) % NB


def hist(rad):
    h = np.bincount(bins(rad), minlength=NB).astype(float); return h / h.sum()


# ------------------------------------------------ Element
h = h5py.File("/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu", "r")
obs = read_elem(h["obs"]); names = np.array(read_elem(h["mod"]["imaging"]["var"]).index.astype(str))
raw_all = h["mod"]["imaging"]["layers"]["raw"][:].astype(np.float32); raw_all[~np.isfinite(raw_all)] = np.nan
g = lambda n: raw_all[:, list(names).index(n)].astype(float)
dna = np.log2(np.clip((g("Nucleus_Nuclear") - g("Bkg_Nucleus_Nuclear")) * g("NuclearArea"), 1, None))
wells = obs["Well"].astype(str).values
dna = dna - pd.Series(dna).groupby(wells).transform("median").values
drop = np.array([any(s in n for s in ("Location", "Center", "ObjectNumber", "Bkg_", "Parent", "Children", "Number_", "Metadata")) or n in ("AreaShape_Orientation",) for n in names])
keep = (~drop) & (np.isfinite(raw_all).mean(0) >= .85)
X = raw_all[:, keep]; fn = names[keep]; del raw_all
th = np.radians(obs["cycle_position_deg"].astype(float).values)
cond = obs["condition"].astype(str).values; gene = obs["target_gene"].astype(str).values
ctl = np.isin(gene, ["NTC", "OR1I1"]); base = cond == "baseline"
print(f"Element cells {len(X):,} | features {len(fn)} | control cells {ctl.sum():,} | wells {len(np.unique(wells))}", flush=True)

CHANNELS = ["Nucleus", "Actin", "Cell-Membrane", "Mitochondria", "Phospho-ERK1", "ERK1", "Phospho-HSP27", "Phospho-p38", "Mouse", "Rabbit"]


def family(n):
    t = n.split("_")
    if n in ("Area", "NuclearArea", "AreaUm", "NuclearAreaUm") or n.startswith("AreaShape"): return "shape and size"
    for c in CHANNELS:
        if c in t: return {"Nucleus": "DNA stain (Nucleus)", "Actin": "Actin", "Cell-Membrane": "Cell membrane", "Mitochondria": "Mitochondria",
                           "Mouse": "IgG isotype controls (negative control)", "Rabbit": "IgG isotype controls (negative control)"}.get(c, "antibodies (ERK, pERK, pHSP27, p-p38)")
    return "other"


fam = np.array([family(n) for n in fn])
print(pd.Series(fam).value_counts().to_string(), flush=True)

# ------------------------------------------------ A: prediction, leave-one-well-out over control cells
C = np.where(ctl)[0]; Xc, thc, wc = X[C], th[C], wells[C]
sets = {"all imaging features": np.ones(len(fn), bool), "DNA content + nuclear/cell size only": np.isin(fn, ["Nucleus_Nuclear", "NuclearArea", "Area"]),
        **{f"only {f_}": fam == f_ for f_ in pd.unique(fam) if f_ != "other"},
        "all except DNA stain": fam != "DNA stain (Nucleus)"}
rowsA = []; oof_all = None
for name, m in sets.items():
    oof = np.full(len(C), np.nan)
    for w in np.unique(wc):
        tr, te = wc != w, wc == w
        Xtr = Xc[tr][:, m]; T = fit_transform(Xtr); P = fit_angle(T(Xtr), thc[tr]); oof[te] = P(T(Xc[te][:, m]))
    e = err_deg(oof, thc)
    rowsA.append(dict(features=name, n_features=int(m.sum()), median_error_deg=round(float(np.median(e)), 1),
                      pct_within_45deg=round(100 * float(np.mean(e < 45)), 1), mean_cos=round(float(np.mean(np.cos(np.radians(e)))), 3)))
    print(rowsA[-1], flush=True)
    if name == "all imaging features": oof_all = oof
sh = RNG.permutation(thc); e0 = err_deg(sh, thc)
rowsA.append(dict(features="shuffled clock (null)", n_features=0, median_error_deg=round(float(np.median(e0)), 1),
                  pct_within_45deg=round(100 * float(np.mean(e0 < 45)), 1), mean_cos=round(float(np.mean(np.cos(np.radians(e0)))), 3)))
A = pd.DataFrame(rowsA); A.to_csv(f"{OUT}/A_prediction_by_feature_family.csv", index=False)
arc = pd.DataFrame({"arc_deg": (bins(thc) + .5) * 360 / NB, "err": err_deg(oof_all, thc), "cond": cond[C]})
arcT = arc.groupby(["arc_deg"]).err.agg(["median", "size"]).round(1).reset_index()
ph = obs["phase_scanpy"].astype(str).values[C]
arcT = arcT.merge(pd.DataFrame({"arc_deg": (bins(thc) + .5) * 360 / NB, "ph": ph}).groupby("arc_deg").ph.agg(lambda s: s.value_counts().idxmax()).rename("majority_scanpy_phase").reset_index(), on="arc_deg")
arcT.to_csv(f"{OUT}/A_error_by_clock_arc.csv", index=False)
cond_err = arc.groupby("cond").err.median().round(1).to_dict()
print("median error by condition (leave-one-well-out):", cond_err, "\n", arcT.to_string(index=False), flush=True)

# ------------------------------------------------ B: per-feature profiles around the RNA clock (baseline controls)
mb = ctl & base; Xb = X[mb]; tb = th[mb]; wb = wells[mb]
Zb = pd.DataFrame(Xb, columns=fn); Zb = (Zb - Zb.groupby(wb).transform("median")) / Zb.std()      # within-well centring
bb = bins(tb); half = np.isin(wb, np.unique(wb)[:3])
prof = Zb.groupby(bb).mean(); pA = Zb[half].groupby(bb[half]).mean(); pB = Zb[~half].groupby(bb[~half]).mean()
B = pd.DataFrame({"feature": fn, "family": fam, "amplitude_SD": (prof.max() - prof.min()).values,
                  "peak_deg": ((prof.values.argmax(0) + .5) * 360 / NB), "trough_deg": ((prof.values.argmin(0) + .5) * 360 / NB),
                  "split_half_r": [pearsonr(pA[c], pB[c])[0] if pA[c].std() > 0 and pB[c].std() > 0 else np.nan for c in fn]})
B = B.sort_values("amplitude_SD", ascending=False); B.round(3).to_csv(f"{OUT}/B_feature_cycle_profiles.csv", index=False)
fam_sum = B.groupby("family").agg(n=("feature", "size"), median_amplitude_SD=("amplitude_SD", "median"), max_amplitude_SD=("amplitude_SD", "max"),
                                  n_reproducible_and_large=("amplitude_SD", lambda s: int(((s > .5) & (B.loc[s.index, "split_half_r"] > .9)).sum()))).round(3)
fam_sum.to_csv(f"{OUT}/B_family_summary.csv")
print("\nfeature families around the RNA clock:\n", fam_sum.to_string(), "\n\ntop features:\n", B.head(25).round(3).to_string(index=False), flush=True)
comp = pd.DataFrame({q: [np.mean(obs["phase_scanpy"].astype(str).values[mb][bb == k] == q) * 100 for k in range(NB)] for q in ["G1", "S", "G2M"]})
comp["pct_4N"] = [100 * np.mean(dna[mb][bb == k] > .21) for k in range(NB)]; comp["deg"] = (np.arange(NB) + .5) * 360 / NB

# ------------------------------------------------ C: perturbations, predicted (out of fold by well) vs measured cycle shift
pred = np.full(len(X), np.nan)
for w in np.unique(wells):
    tr = ctl & (wells != w); T = fit_transform(X[tr]); P = fit_angle(T(X[tr]), th[tr]); te = wells == w; pred[te] = P(T(X[te]))
rowsC = []
contrasts = [("TNFa vs baseline (controls)", ctl & ~base, ctl & base)] + \
            [(f"{k} KO vs controls, {c_}", (gene == k) & (cond == c_), ctl & (cond == c_)) for c_ in ["baseline", "TNFa"] for k in ["CTNNB1", "GSK3B", "IL1R1", "SMAD7", "TGFBR2"]]
late = [1, 2, 3, 4]                                                    # 30-150 deg: the G2/M-rich arc of the RNA clock
for name, mt, mr in contrasts:
    hm, hr = hist(th[mt]), hist(th[mr]); pm, pr = hist(pred[mt]), hist(pred[mr])
    rowsC.append(dict(contrast=name, n=int(mt.sum()), measured_TV=.5 * np.abs(hm - hr).sum(), predicted_TV=.5 * np.abs(pm - pr).sum(),
                      measured_d_pct_G2M_arc=100 * (hm[late].sum() - hr[late].sum()), predicted_d_pct_G2M_arc=100 * (pm[late].sum() - pr[late].sum()),
                      bin_shift_corr=np.corrcoef(hm - hr, pm - pr)[0, 1]))
Cdf = pd.DataFrame(rowsC).round(3); Cdf.to_csv(f"{OUT}/C_perturbation_measured_vs_predicted.csv", index=False)
print("\nperturbations: cycle shift measured by RNA vs predicted from morphology (out of fold by well):\n", Cdf.to_string(index=False), flush=True)
print("across the 11 contrasts: measured vs predicted d%G2M-arc rho", round(spearmanr(Cdf.measured_d_pct_G2M_arc, Cdf.predicted_d_pct_G2M_arc)[0], 2), flush=True)

# ------------------------------------------------ D: correlates in OPS0077
def states(d, lo, hi):
    return np.select([d < lo, d > hi], ["2N", "4N"], "S")


def effects(df, st):
    out = {}
    for s_ in ["S", "4N"]:
        a, b = df[st == s_], df[st == "2N"]; sd = np.sqrt((a.var() + b.var()) / 2)
        out[f"{s_}_vs_2N"] = (a.mean() - b.mean()) / sd
    return pd.DataFrame(out)


hh, ed = np.histogram(dna[ctl & base], bins=120, range=(-1.5, 2)); cen = (ed[:-1] + ed[1:]) / 2
m2 = cen[np.argmax(np.where(np.abs(cen) < .4, hh, 0))]; m4 = cen[np.argmax(np.where(cen > .6, hh, 0))]
stE = states(dna[mb], m2 + .25 * (m4 - m2), m2 + .75 * (m4 - m2))
L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key"); L = L[L["dna_reliable"] == True]
dO = L["dna_rel_R234"].astype(float)
dp = ad.read_h5ad(find("OPS0077_NTC_DAPI_cpmeasure.h5ad")); dp.obs_names = dp.obs["cell_key"].astype(str) if "cell_key" in dp.obs else dp.obs_names
dp = dp[~dp.obs_names.duplicated()]; dp = dp[dp.obs_names.isin(dO.index)]
DO = pd.DataFrame(np.asarray(dp.X, float), columns=dp.var_names, index=dp.obs_names); stO = states(dO.loc[DO.index].values, .25, .75)
pairs = []
for n in fn:
    m_ = re.match(r"Texture_(\w+?)_3_(\d\d)_256_Nucleus$", n)
    if m_ and f"{m_.group(1)}_3_{m_.group(2)}_256__DAPI_R2" in DO: pairs.append((n, f"{m_.group(1)}_3_{m_.group(2)}_256__DAPI_R2", "DNA-stain texture"))
    m_ = re.match(r"Granularity_(\d+)_Nucleus$", n)
    if m_:
        cand = [c for c in DO if re.fullmatch(rf"Granularity_{m_.group(1)}_?_?_?DAPI_R2", c) or c == f"Granularity_{m_.group(1)}__DAPI_R2"]
        if cand: pairs.append((n, cand[0], "DNA-stain granularity"))
eE = effects(pd.DataFrame(X[mb], columns=fn)[[p[0] for p in pairs]], stE)
eO = effects(DO[[p[1] for p in pairs]], stO)
# scale-free cell shape (OPS0077 4i cell segmentation, cp_cell_*)
sh_pairs = [("AreaShape_Eccentricity", "cp_cell_Eccentricity"), ("AreaShape_Solidity", "cp_cell_Solidity"), ("AreaShape_Extent", "cp_cell_Extent"),
            ("AreaShape_FormFactor", "cp_cell_FormFactor"), ("AreaShape_Compactness", "cp_cell_Compactness")]
a4 = ad.read_h5ad(find("OPS0077_NTC_OrganelleProfiler_pRb_Rb_p21.h5ad"), backed="r")
SO = pd.DataFrame(np.asarray(a4[:, [p[1] for p in sh_pairs]].to_memory().X, float), columns=[p[1] for p in sh_pairs], index=a4.obs["cell_key"].astype(str).values)
SO = SO[~SO.index.duplicated()].reindex(dO.index)
eE2 = effects(pd.DataFrame(X[mb], columns=fn)[[p[0] for p in sh_pairs]], stE); eO2 = effects(SO, states(dO.values, .25, .75))
rowsD = []
for (en, on, kind) in pairs + [(a_, b_, "cell shape (scale-free)") for a_, b_ in sh_pairs]:
    src_E = eE if kind != "cell shape (scale-free)" else eE2; src_O = eO if kind != "cell shape (scale-free)" else eO2
    rowsD.append(dict(kind=kind, element_feature=en, ops0077_feature=on, element_S_vs_2N=src_E.loc[en, "S_vs_2N"], ops_S_vs_2N=src_O.loc[on, "S_vs_2N"],
                      element_4N_vs_2N=src_E.loc[en, "4N_vs_2N"], ops_4N_vs_2N=src_O.loc[on, "4N_vs_2N"]))
D = pd.DataFrame(rowsD).round(3); D.to_csv(f"{OUT}/D_shared_feature_state_effects.csv", index=False)
r4 = spearmanr(D.element_4N_vs_2N, D.ops_4N_vs_2N)[0]; rS = spearmanr(D.element_S_vs_2N, D.ops_S_vs_2N)[0]
sign4 = float(np.mean(np.sign(D.element_4N_vs_2N) == np.sign(D.ops_4N_vs_2N)))
print(f"\nD: {len(D)} shared features | state effects agree across datasets: 4N vs 2N rho {r4:.2f} (same sign {100 * sign4:.0f}%), S vs 2N rho {rS:.2f}", flush=True)
print(D.groupby("kind").apply(lambda d: pd.Series({"n": len(d), "rho_4N": spearmanr(d.element_4N_vs_2N, d.ops_4N_vs_2N)[0] if len(d) > 3 else np.nan,
                                                   "same_sign_4N": np.mean(np.sign(d.element_4N_vs_2N) == np.sign(d.ops_4N_vs_2N))})).round(2).to_string(), flush=True)
# phase-contrast correlates: OPS0077 phase features that mark the same DNA states
phd = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); phd.obs_names = phd.obs["cell_key"].astype(str); phd = phd[~phd.obs_names.duplicated()]
phd = phd[phd.obs_names.isin(dO.index)]; ok = np.isfinite(np.asarray(phd.X)).mean(0) >= .85
PH = pd.DataFrame(np.asarray(phd.X, float)[:, ok], columns=np.array(phd.var_names)[ok], index=phd.obs_names)
eP = effects(PH, states(dO.loc[PH.index].values, .25, .75)); eP["abs_4N"] = eP["4N_vs_2N"].abs()
eP.sort_values("abs_4N", ascending=False).round(3).to_csv(f"{OUT}/D_ops0077_phase_features_by_dna_state.csv")
print("\nOPS0077 phase-contrast features that best separate 4N from 2N:\n", eP.sort_values("abs_4N", ascending=False).head(15).round(3).to_string(), flush=True)

# ------------------------------------------------ figures
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
a_ = A[A.features != "shuffled clock (null)"].sort_values("median_error_deg")
ax[0].barh(a_.features, a_.median_error_deg, color="tab:blue"); ax[0].axvline(float(A.loc[A.features == "shuffled clock (null)", "median_error_deg"].iloc[0]), color="k", ls="--", label="shuffled clock")
ax[0].axvline(30, color="tab:red", ls=":", label="OPS0077 phase contrast -> 4i clock"); ax[0].invert_yaxis(); ax[0].legend(fontsize=8)
ax[0].set_xlabel("median error predicting the RNA clock (deg), held-out wells"); ax[0].set_title("A: which morphology reads the RNA clock")
top = B[(B.split_half_r > .9)].head(30); Pm = prof[top.feature.values]; Pm = Pm.iloc[:, np.argsort(Pm.values.argmax(0))]
im = ax[1].imshow(Pm.T.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, extent=[0, 360, len(Pm.columns), 0])
ax[1].set_yticks(np.arange(len(Pm.columns)) + .5); ax[1].set_yticklabels([c[:38] for c in Pm.columns], fontsize=6)
ax[1].set_xlabel("Element RNA clock (deg)"); ax[1].set_title("B: largest reproducible cycle phenotypes (SD units)"); plt.colorbar(im, ax=ax[1], fraction=.04)
for k_, c_ in [("G1", "tab:blue"), ("S", "tab:green"), ("G2M", "tab:red")]: ax[1].plot(comp.deg, len(Pm.columns) * (1 - comp[k_] / 100), color=c_, lw=1, alpha=.6)
ax[2].scatter(Cdf.measured_d_pct_G2M_arc, Cdf.predicted_d_pct_G2M_arc, c="tab:red")
for _, r in Cdf.iterrows(): ax[2].annotate(r.contrast.replace(" vs controls", "").replace(" (controls)", ""), (r.measured_d_pct_G2M_arc, r.predicted_d_pct_G2M_arc), fontsize=7)
ax[2].axline((0, 0), slope=1, color="k", lw=.6); ax[2].axhline(0, color=".7", lw=.5); ax[2].axvline(0, color=".7", lw=.5)
ax[2].set_xlabel("change in % cells on the G2/M arc, measured (RNA)"); ax[2].set_ylabel("predicted from morphology (held-out wells)"); ax[2].set_title("C: perturbations, measured vs predicted")
plt.tight_layout(); plt.savefig(f"{OUT}/element_cycle_morphology.png", dpi=110); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k_, c_ in [("DNA-stain texture", "tab:blue"), ("DNA-stain granularity", "tab:green"), ("cell shape (scale-free)", "tab:orange")]:
    d_ = D[D.kind == k_]; ax[0].scatter(d_.element_4N_vs_2N, d_.ops_4N_vs_2N, s=18, c=c_, label=k_); ax[1].scatter(d_.element_S_vs_2N, d_.ops_S_vs_2N, s=18, c=c_)
for a_, t_ in zip(ax, [f"4N vs 2N (rho {r4:.2f})", f"S vs 2N (rho {rS:.2f})"]):
    a_.axhline(0, color=".7", lw=.5); a_.axvline(0, color=".7", lw=.5); a_.axline((0, 0), slope=1, color="k", lw=.5)
    a_.set_xlabel("Element, effect size (d)"); a_.set_ylabel("OPS0077, effect size (d)"); a_.set_title(f"D: shared features, {t_}")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/element_ops_correlates.png", dpi=110)
print("DONE")
