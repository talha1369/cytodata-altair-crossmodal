#!/usr/bin/env python
"""Per-bin profiles around the Element RNA clock for a selected set of imaging features (part B of element_cycle_morphology.py
saved only amplitude/peak/trough; this saves the curves). Same cells (baseline controls: NTC + OR1I1), same 12 bins, same
normalisation (within-well median centring, then divided by the per-feature SD over those cells). Reads only the needed columns.
Also saves, per bin: number of cells, scanpy phase composition, median DNA content and % of cells at 4N (DNA-stain integrated
intensity, log2 relative to the well median; 4N threshold 0.21 as in part B).
usage: python element_profiles.py [OUT_DIR]"""
import os, sys
import numpy as np, pandas as pd, h5py
from scipy.stats import pearsonr
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "element_cycle"
NB = 12
B = pd.read_csv(f"{OUT}/B_feature_cycle_profiles.csv")
rep = B[B.split_half_r > .9]
sel = list(rep.head(30).feature)
for f_, d_ in B.groupby("family"):                                  # top 3 per family (any reproducibility, so IgG is shown too)
    sel += list(d_.sort_values("amplitude_SD", ascending=False).head(3).feature)
sel += ["Nucleus_Nuclear", "NuclearArea", "Area", "Granularity_1_Nucleus"]
sel = list(dict.fromkeys(s for s in sel if s in set(B.feature)))

h = h5py.File("/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu", "r")
obs = read_elem(h["obs"]); names = list(read_elem(h["mod"]["imaging"]["var"]).index.astype(str))
need = list(dict.fromkeys(sel + ["Nucleus_Nuclear", "Bkg_Nucleus_Nuclear", "NuclearArea"]))
idx = sorted(names.index(n) for n in need)
M = h["mod"]["imaging"]["layers"]["raw"][:, idx].astype(float); M[~np.isfinite(M)] = np.nan
R = pd.DataFrame(M, columns=[names[i] for i in idx])
wells = obs["Well"].astype(str).values
dna = np.log2(np.clip((R["Nucleus_Nuclear"] - R["Bkg_Nucleus_Nuclear"]).values * R["NuclearArea"].values, 1, None))
dna = dna - pd.Series(dna).groupby(wells).transform("median").values
th = np.radians(obs["cycle_position_deg"].astype(float).values)
cond = obs["condition"].astype(str).values; gene = obs["target_gene"].astype(str).values
mb = np.isin(gene, ["NTC", "OR1I1"]) & (cond == "baseline")
bb = (np.floor(np.mod(th[mb], 2 * np.pi) / (2 * np.pi) * NB).astype(int)) % NB
wb = wells[mb]
Zb = R.loc[mb, sel].reset_index(drop=True)
Zb = (Zb - Zb.groupby(wb).transform("median")) / Zb.std()
half = np.isin(wb, np.unique(wb)[:3])
prof = Zb.groupby(bb).mean(); pA = Zb[half].groupby(bb[half]).mean(); pB = Zb[~half].groupby(bb[~half]).mean()
deg = (np.arange(NB) + .5) * 360 / NB
fam = B.set_index("feature").family
rows = []
for f_ in sel:
    for k in range(NB):
        rows.append(dict(feature=f_, family=fam[f_], bin_deg=deg[k], mean_SD=prof.loc[k, f_], half1_SD=pA.loc[k, f_], half2_SD=pB.loc[k, f_]))
P = pd.DataFrame(rows).round(4); P.to_csv(f"{OUT}/B2_profiles_by_bin.csv", index=False)
chk = pd.DataFrame({"feature": sel, "amplitude_recomputed": (prof.max() - prof.min()).values,
                    "split_half_r_recomputed": [pearsonr(pA[c], pB[c])[0] for c in sel]}).merge(B[["feature", "amplitude_SD", "split_half_r"]], on="feature")
print("max |amplitude difference| vs B table:", float((chk.amplitude_recomputed - chk.amplitude_SD).abs().max()), flush=True)
ph = obs["phase_scanpy"].astype(str).values[mb]; d_ = dna[mb]
comp = pd.DataFrame({"bin_deg": deg, "n_cells": [int((bb == k).sum()) for k in range(NB)],
                     **{f"pct_{q}": [100 * np.mean(ph[bb == k] == q) for k in range(NB)] for q in ["G1", "S", "G2M"]},
                     "median_dna_log2": [float(np.nanmedian(d_[bb == k])) for k in range(NB)],
                     "pct_4N": [100 * np.mean(d_[bb == k] > .21) for k in range(NB)],
                     "n_wells": len(np.unique(wb)), "n_wells_half1": len(np.unique(wb[half]))}).round(3)
comp.to_csv(f"{OUT}/B2_bin_composition.csv", index=False)
print(f"baseline control cells {mb.sum():,} | features profiled {len(sel)}\n", comp.to_string(index=False), flush=True)
print("DONE")
