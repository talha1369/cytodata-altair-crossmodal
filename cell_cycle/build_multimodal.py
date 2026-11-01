#!/usr/bin/env python
"""Assemble OPS0077_NTC_Multimodal_4i_Phase_DNA.h5ad in the team's schema (Brian's Multimodal_Description_h5aD.md),
improved: full 4i panel (pRb + Rb + p21 + shape), DNA content, a DNA x pRb cell-cycle stage, and robust-scaled
feature matrices ready to use.

Layout
  .X                      robust-scaled + clipped 4i OrganelleProfiler features (per-feature median/IQR, clip +-5,
                          NaN -> 0 after centring). RAW values stay in the source per-marker tables.
  layers['raw']           the raw 4i values (float32), for anyone who wants to rescale differently
  obsm['X_phase']         robust-scaled + clipped per-cell phase OrganelleProfiler features
  obsm['X_phase_raw']     raw phase values
  obsm['spatial']         fixed 4i centroid (x_cp, y_cp)
  obsm['spatial_pheno']   live phase centroid (x_pheno, y_pheno)
  obs                     cell-cycle readouts (pRb, Rb, p21, pRb/Rb, DNA content), cycle_stage, QC flags, ids
  var                     4i feature name, marker, group
  uns['phase_var_names']  the obsm['X_phase'] column names; uns['provenance'], uns['scaling']

Cells: 4i cells that also have phase (the common set). DNA columns are NaN where dna_reliable is False.
usage: python build_multimodal.py OUT.h5ad
"""
import sys
import numpy as np, pandas as pd, anndata as ad

OUT = sys.argv[1]
op = ad.read_h5ad("OPS0077_NTC_OrganelleProfiler_pRb_Rb_p21.h5ad")
ph = ad.read_h5ad("OPS0077_NTC_phase_OrganelleProfiler.h5ad")
fr = pd.read_parquet("OPS0077_NTC_cellcycle_frame.parquet")   # per cell_key summary + cycle_stage


def dedup(a, key="cell_key"):
    k = a.obs[key].astype(str).values
    keep = ~pd.Index(k).duplicated()
    b = a[keep].copy(); b.obs_names = k[keep]
    return b


op, ph = dedup(op), dedup(ph)
cells = op.obs_names.intersection(ph.obs_names)
cells = fr.index.intersection(cells)                          # keep frame order where possible
op, ph = op[cells], ph[cells]
fr = fr.loc[cells]
print(f"common 4i+phase cells: {len(cells):,}")


def robust(X):
    X = np.asarray(X, dtype=np.float64)
    med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, med)                      # median-impute per feature
    q10, q90 = np.nanpercentile(X, [10, 90], axis=0)
    iqr = np.where(q90 - q10 > 0, q90 - q10, 1.0)
    Z = np.clip((X - med) / iqr, -5, 5)
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)       # all-NaN features (e.g. fluor-contact in phase) -> 0


X4i_raw = np.asarray(op.X, dtype=np.float32)
Xph_raw = np.asarray(ph.X, dtype=np.float32)

obs = pd.DataFrame(index=cells)
obs["cell_id"] = op.obs["pheno_cell_uid"].astype(str).values
obs["well"] = fr["well"].values; obs["tile_pheno"] = fr["tile_pheno"].values
obs["barcode"] = op.obs["barcode"].astype(str).values; obs["sgRNA"] = op.obs["sgRNA"].astype(str).values
obs["perturbation"] = "NTC"; obs["control_type"] = "non-targeting"
for c in ["pRb", "Rb", "p21", "prb_rb_ratio", "log2_pRb", "nuclear_area_4i", "cell_area_4i",
          "dna_content_log2", "dna_reliable", "dense_field", "cycle_stage",
          "pheno_to_fixed_distance_px", "has_4i_protein"]:
    obs[c] = fr[c].values
obs["has_phase"] = True
obs["trusted_match"] = obs["pheno_to_fixed_distance_px"].astype(float) < 25

var = op.var.copy()
var["marker"] = ["pRb" if "pRb" in n or "prb" in n else "Rb" if (("_Rb_" in n) or n.endswith("_rb") or "4i_r3_rb" in n)
                 else "p21" if ("p21" in n) else "shared" for n in var.index]

out = ad.AnnData(X=robust(X4i_raw), obs=obs, var=var)
out.layers["raw"] = X4i_raw
out.obsm["X_phase"] = robust(Xph_raw)
out.obsm["X_phase_raw"] = Xph_raw
coord = op.obs if "x_cp" in op.obs.columns else ph.obs
out.obsm["spatial"] = np.c_[coord["x_cp"].astype(float), coord["y_cp"].astype(float)].astype(np.float32)
out.obsm["spatial_pheno"] = np.c_[coord["x_pheno"].astype(float), coord["y_pheno"].astype(float)].astype(np.float32)
out.uns["phase_var_names"] = list(ph.var_names)
out.uns["scaling"] = "X and X_phase: per-feature median-impute, (x-median)/IQR(10-90), clip +-5. Raw in layers['raw'] / obsm['X_phase_raw']."
out.uns["provenance"] = {"screen": "Biohub_OPS0077 (ops0144_20260406)", "population": "NTC (non-targeting)",
                         "4i_markers": "pRb, Rb, p21 + cell/nucleus shape (OrganelleProfiler)",
                         "phase": "per-cell OrganelleProfiler phase (atlas all_cells_phase.h5ad)",
                         "dna": "cp_measure DAPI R2-R4, per-cell non-nuclear background; dna_reliable flag",
                         "cycle_stage": "DNA x pRb gating: G0/G1-early, G1-late, S, G2/M (reliable-DNA cells only)",
                         "builder": "build_multimodal.py; supersedes OPS0077_NTC_Multimodal_4i_Phase.h5ad (pRb-only, raw)"}
for c in out.obs.columns:
    if out.obs[c].dtype == object:
        out.obs[c] = out.obs[c].astype(str)
out.write_h5ad(OUT, compression="gzip")
import os
print(f"wrote {OUT}  {out.shape}  {os.path.getsize(OUT)/1e6:.0f} MB")
print(" .X 4i:", out.X.shape, "| obsm X_phase:", out.obsm['X_phase'].shape)
print(" markers in .X var:", pd.Series(var['marker']).value_counts().to_dict())
print(" obs cycle_stage:", out.obs['cycle_stage'].value_counts(dropna=False).to_dict())
print(" DNA reliable:", int((out.obs['dna_reliable'].astype(str)=='True').sum()), "| trusted match:", int((out.obs['trusted_match'].astype(str)=='True').sum()))
print(" .X finite:", np.isfinite(out.X).mean().round(4), "| X_phase finite:", np.isfinite(out.obsm['X_phase']).mean().round(4))
