"""Part D of element_cycle_morphology.py on its own (correlates of the Element cycle phenotypes in OPS0077), with duplicate cell keys removed."""

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

mb = ctl & base
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
L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key"); L = L[L["dna_reliable"] == True]; L = L[~L.index.duplicated()]
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

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k_, c_ in [("DNA-stain texture", "tab:blue"), ("DNA-stain granularity", "tab:green"), ("cell shape (scale-free)", "tab:orange")]:
    d_ = D[D.kind == k_]; ax[0].scatter(d_.element_4N_vs_2N, d_.ops_4N_vs_2N, s=18, c=c_, label=k_); ax[1].scatter(d_.element_S_vs_2N, d_.ops_S_vs_2N, s=18, c=c_)
for a_, t_ in zip(ax, [f"4N vs 2N (rho {r4:.2f})", f"S vs 2N (rho {rS:.2f})"]):
    a_.axhline(0, color=".7", lw=.5); a_.axvline(0, color=".7", lw=.5); a_.axline((0, 0), slope=1, color="k", lw=.5)
    a_.set_xlabel("Element, effect size (d)"); a_.set_ylabel("OPS0077, effect size (d)"); a_.set_title(f"D: shared features, {t_}")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/element_ops_correlates.png", dpi=110)
print("DONE")
