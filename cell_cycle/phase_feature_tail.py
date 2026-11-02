#!/usr/bin/env python
"""Diego's per-feature result on phase contrast, followed up: which phase features lose the most variance when the cell cycle is
removed, are they cell size / dry mass, and do they carry the knockout signal?

Per feature (NTC cells with a measured clock): within-field variance after / before removing cc_phase (48 clock-bin means fit within
each well, as Diego did), and the same with measured DNA content only (48 bins). Tail = ratio < 0.9.
Then: correlation of each feature with phase cell area; feature families of the tail; and for the 1,000 knockouts, whether the
features that best separate knockouts from NTC (median |effect size| across knockouts) are tail features.
usage: python phase_feature_tail.py [OUT_DIR]"""
import os, sys, re
import numpy as np, pandas as pd, anndata as ad, h5py

OUT = sys.argv[1] if len(sys.argv) > 1 else "phase_push"
os.makedirs(OUT, exist_ok=True)
NB = 48


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
L = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")]; L = L[~L.index.duplicated()]
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str); ph = ph[~ph.obs_names.duplicated()]
ph = ph[ph.obs_names.isin(L.index)]; X = pd.DataFrame(np.asarray(ph.X, float), columns=ph.var_names, index=ph.obs_names)
X = X.loc[:, X.notna().mean() >= .85]; X = X.loc[:, X.std() > 0]; X = X.fillna(X.median())
well = np.array([k.split("|")[0] for k in X.index]); tile = ph.obs["tile_pheno"].astype(str).values
Q = L.loc[X.index]; cc = Q["cc_phase"].astype(float).values; dna = Q["dna_rel_R234"].astype(float).values
print(f"NTC {len(X):,} cells, {X.shape[1]} phase features", flush=True)


def remove(Xv, b):
    """subtract per-bin means within each well (Diego: cycle term fitted within well)"""
    D = pd.DataFrame(Xv); key = pd.Series(well).astype(str) + "_" + pd.Series(b).astype(str)
    return (D - D.groupby(key.values).transform("mean") + D.groupby(well).transform("mean")).values


def wfv(Xv): D = pd.DataFrame(Xv); return (D - D.groupby(tile).transform("mean")).var().values


bc = np.floor(np.mod(cc, 1) * NB).astype(int) % NB
bd = np.clip(np.digitize(dna, np.nanquantile(dna, np.linspace(0, 1, NB + 1))[1:-1]), 0, NB - 1)
v0 = wfv(X.values); r_cc = wfv(remove(X.values, bc)) / v0; r_dna = wfv(remove(X.values, bd)) / v0
area_col = "op_cell_area" if "op_cell_area" in X else X.columns[0]
size_r = np.array([np.corrcoef(X[c], np.log2(X[area_col].clip(lower=1)))[0, 1] for c in X.columns])


def family(n):
    n = n.replace("op_", "")
    for k in ["focus3d", "cell", "nucleus", "nuclear", "cytoplasm", "texture", "intensity", "granularity", "zernike", "radial"]:
        if k in n: return k
    return n.split("_")[0]


F = pd.DataFrame({"feature": X.columns, "family": [family(c) for c in X.columns], "ratio_cc_phase": r_cc, "ratio_DNA_only": r_dna,
                  "corr_with_log_cell_area": size_r})
F["in_tail"] = F.ratio_cc_phase < .9
print(f"median ratio: cc_phase {np.median(r_cc):.3f} | DNA only {np.median(r_dna):.3f}; tail (< 0.9): {int(F.in_tail.sum())} features", flush=True)
print("tail by family:\n", F[F.in_tail].family.value_counts().head(12).to_string())
print(f"|corr with cell area|: tail median {F[F.in_tail].corr_with_log_cell_area.abs().median():.2f} vs rest {F[~F.in_tail].corr_with_log_cell_area.abs().median():.2f}")
print("strongest 15:\n", F.nsmallest(15, "ratio_cc_phase").round(3).to_string(index=False), flush=True)

# knockouts: effect size of each feature per knockout vs NTC, up to 200 cells per knockout
ko = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r"); ob = ko.obs.copy(); ob["row"] = np.arange(len(ob)); ob["gene"] = ob["gene"].astype(str).str.upper()
pick = ob.drop_duplicates("cell_key").groupby("gene", group_keys=False).apply(lambda d: d.sample(min(len(d), 200), random_state=0)).sort_values("row")
cols = [list(ko.var_names).index(c) for c in X.columns]
f = h5py.File("OPS0077_allko_phase.h5ad", "r"); r_ = pick.row.values
Xk = pd.DataFrame(np.vstack([f["X"][r_[i:i + 5000]] for i in range(0, len(r_), 5000)])[:, cols], columns=X.columns); f.close()
Xk = Xk.fillna(X.median()); mu, sd = X.mean(), X.std()
E = ((Xk.groupby(pick.gene.values).mean() - mu) / sd).abs()
F["median_abs_KO_effect"] = E.median().values; F["n_KO_effect_gt_0.3"] = (E > .3).sum().values
top = F.nlargest(100, "median_abs_KO_effect")
print(f"\nof the 100 features that best separate knockouts from NTC, {int(top.in_tail.sum())} are cycle-tail features "
      f"(tail is {100 * F.in_tail.mean():.0f}% of all features)", flush=True)
from scipy.stats import spearmanr
print("feature-level: cycle loss (1 - ratio) vs knockout effect, Spearman", round(spearmanr(1 - F.ratio_cc_phase, F.median_abs_KO_effect)[0], 2))
F.round(4).to_csv(f"{OUT}/phase_feature_cycle_tail.csv", index=False)
print("DONE")
