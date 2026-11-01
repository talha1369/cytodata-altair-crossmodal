#!/usr/bin/env python
"""Merge the per-part cp_measure DAPI outputs into one AnnData keyed to the NTC OrganelleProfiler table, add
background-subtracted DNA content per round, and validate it (2N/4N peaks, DNA x pRb).

usage: python merge_dapi.py DAPI_DIR NTC_H5AD OUT_H5AD FIG_PNG
"""
import glob, sys
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

D, NTC, OUT, FIG = sys.argv[1:5]
SCALE = 65535.0

feat = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{D}/part*/feat_*.parquet"))])
qc = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{D}/part*/qc_*.parquet"))]).set_index("cell_key")
assert feat.index.is_unique and qc.index.is_unique, "duplicate cells across parts"
qc = qc.reindex(feat.index)
print("cells", len(feat), "| features", feat.shape[1])

ntc = ad.read_h5ad(NTC, backed="r")
o = ntc.obs.drop_duplicates("cell_key").set_index("cell_key")
obs = o.reindex(feat.index)[["well", "tile_pheno", "segmentation_id", "gene", "sgRNA", "barcode", "x_cp", "y_cp",
                             "pheno_cell_uid", "fixed_cell_uid", "pheno_to_fixed_distance_px"]].copy()
obs = obs.join(qc)

# Local background needs >100 cell-free pixels in the crop; in dense fields the cell masks fill the crop and it is
# missing (~47% of cells). Background is mostly an optical offset per tile and round, so a missing value takes the
# median of the measured backgrounds in the same tile, else the same well. Flagged per cell in bg_imputed_R<k>.
rel = {}
for k in range(1, 6):
    b = obs[f"R{k}_bg_median"].astype(float)
    tile_med = b.groupby(obs["tile_pheno"].astype(str)).transform("median")
    well_med = b.groupby(obs["well"].astype(str)).transform("median")
    obs[f"bg_imputed_R{k}"] = b.isna()
    obs[f"R{k}_bg_used"] = b.fillna(tile_med).fillna(well_med)
    integ = feat[f"Intensity_IntegratedIntensity__DAPI_R{k}"] * SCALE
    area = feat[f"nucleus_R{k}_Area"]
    dna = integ - obs[f"R{k}_bg_used"] * area
    lg = np.log2(dna.where(dna > 0))
    obs[f"dna_R{k}_log2"] = lg
    rel[k] = lg - lg.median()
obs["dna_rel_R234"] = pd.concat([rel[2], rel[3], rel[4]], axis=1).mean(1, skipna=False)   # 0 = median cell, +1 = 2x
obs["dna_R1_any_saturated"] = obs["R1_nucleus_saturated_frac"] > 0
# Imputed-background cells (dense fields) show a compressed 4N peak (+0.56 vs +0.82 log2): neighbouring-nucleus haze
# raises the true local background, so a borrowed background under-subtracts. Only measured-background cells with a
# single nucleus in R2-R4 are trusted for DNA content until a nuclear-exclusion background is computed for all cells.
obs["dna_reliable"] = (~obs[["bg_imputed_R2", "bg_imputed_R3", "bg_imputed_R4"]].any(axis=1)
                       & (obs["R2_n_nuclei_in_cell"].astype(float) <= 1)
                       & obs["dna_rel_R234"].notna())
obs["nucleus_found_R234"] = (obs[["R2_nucleus_id", "R3_nucleus_id", "R4_nucleus_id"]] >= 0).all(1)

# pRb from the NTC table for staging
prb_col = "cp_single_object_pRb_cell_Intensity_MeanIntensity"
prb = pd.Series(np.asarray(ntc[:, prb_col].X).ravel(), index=ntc.obs["cell_key"].values)
prb = prb[~prb.index.duplicated()]
obs["pRb_mean"] = prb.reindex(feat.index).values

var = pd.DataFrame(index=feat.columns)
var["round"] = [next((f"R{k}" for k in range(1, 6) if f"R{k}" in c), "") for c in feat.columns]
var["group"] = [c.split("_")[0] if not c.startswith("nucleus_") else "SizeShape" for c in feat.columns]
out = ad.AnnData(X=feat.values.astype(np.float32), obs=obs, var=var)
for c in out.obs.columns:
    if out.obs[c].dtype == object:
        out.obs[c] = out.obs[c].astype(str)
out.uns["provenance"] = {
    "source": "Biohub_OPS0077 zarr (public S3), 4i_R1..R5_nuclei_DAPI with 4i_R<k>_nuclear_seg; cell = 4i_cell_seg",
    "tool": "cp_measure featurizer, accelerator=numba", "intensity_scale": "raw/65535",
    "nucleus_rule": "round-k nucleus with largest overlap with the cell, kept if >=50% of it lies in the cell",
    "dna_rel_R234": "mean over R2-R4 of log2(Integrated*65535 - bg_median*Area) minus that round's median; R1 saturates"}
out.write_h5ad(OUT, compression="gzip")
print("wrote", OUT, out.shape)

# ---- validation ----
ok = obs["nucleus_found_R234"] & obs["dna_rel_R234"].notna() & (obs["R2_n_nuclei_in_cell"] <= 1)
x = obs.loc[ok, "dna_rel_R234"].values
x = x[(x > -1.5) & (x < 2.5)]
gm = GaussianMixture(2, random_state=0).fit(x[:, None])
mu = np.sort(gm.means_.ravel()); w = gm.weights_[np.argsort(gm.means_.ravel())]
gm1 = GaussianMixture(1, random_state=0).fit(x[:, None])
print(f"single-nucleus cells used: {ok.sum()}")
print(f"2-peak fit: means {mu.round(3)}  gap {mu[1] - mu[0]:.3f} log2 (expect ~1.0 for 2N->4N)  weights {w.round(3)}")
print(f"BIC 1 vs 2 components: {gm1.bic(x[:, None]):.0f} vs {gm.bic(x[:, None]):.0f} (lower is better)")
imp = obs.loc[ok, ["bg_imputed_R2", "bg_imputed_R3", "bg_imputed_R4"]].any(axis=1)
for lab, m in (("measured bg", ~imp), ("imputed bg", imp)):
    xx = obs.loc[ok, "dna_rel_R234"][m]; xx = xx[(xx > -1.5) & (xx < 2.5)].values
    g = GaussianMixture(2, random_state=0).fit(xx[:, None]); mm = np.sort(g.means_.ravel())
    ww = g.weights_[np.argsort(g.means_.ravel())]
    print(f"  {lab}: n={len(xx):,} peaks {mm.round(3)} weights {ww.round(3)}")
print("Spearman between rounds (log2 DNA):")
print(obs.loc[ok, [f"dna_R{k}_log2" for k in range(1, 6)]].corr("spearman").round(2).to_string())

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].hist(x, 200, color="0.3"); ax[0].set_xlabel("relative DNA content, log2 (R2-R4 consensus)")
for m in mu: ax[0].axvline(m, color="tab:red", lw=1)
ax[0].set_title(f"NTC cells, n={len(x):,}: peaks {mu[0]:.2f} / {mu[1]:.2f} (gap {mu[1]-mu[0]:.2f})")
y = obs.loc[ok, "dna_R1_log2"] - obs.loc[ok, "dna_R1_log2"].median()
ax[1].hist(y[(y > -1.5) & (y < 2.5)], 200, color="0.6"); ax[1].set_xlabel("relative DNA content, log2 (R1 only, saturates)")
p = obs.loc[ok, "pRb_mean"]; okp = p > 0
ax[2].hexbin(obs.loc[ok, "dna_rel_R234"][okp], np.log2(p[okp]), gridsize=90, bins="log", extent=(-1.5, 2.5, 7, 10))
ax[2].set_xlabel("relative DNA content, log2"); ax[2].set_ylabel("log2 pRb (whole-cell mean)")
ax[2].set_title("DNA x pRb staging")
plt.tight_layout(); plt.savefig(FIG, dpi=90)
print("figure", FIG)
