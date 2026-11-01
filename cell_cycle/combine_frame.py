#!/usr/bin/env python
"""Assemble one per-cell cell-cycle frame for the OPS0077 NTC population: the cell-cycle readouts from all three
modalities aligned on cell_key, plus a DNA x pRb cell-cycle stage.

Inputs (all in this dir, same cell_key = well|4i segmentation_id):
  OPS0077_NTC_OrganelleProfiler_pRb_Rb_p21.h5ad  4i protein (p21, Rb, pRb) + cell/nucleus shape
  OPS0077_NTC_DAPI_cpmeasure.h5ad                DNA content (per-cell background) + QC
  OPS0077_NTC_phase_OrganelleProfiler.h5ad       label-free phase (presence flag only here)

Output: OPS0077_NTC_cellcycle_frame.parquet  (one row per NTC cell; the aggregated summary the meeting asked for)
"""
import numpy as np, pandas as pd, anndata as ad

op = ad.read_h5ad("OPS0077_NTC_OrganelleProfiler_pRb_Rb_p21.h5ad")
dn = ad.read_h5ad("OPS0077_NTC_DAPI_cpmeasure.h5ad", backed="r").obs
ph = ad.read_h5ad("OPS0077_NTC_phase_OrganelleProfiler.h5ad", backed="r").obs


def opcol(name):
    return pd.Series(np.asarray(op[:, name].X).ravel(), index=op.obs["cell_key"].astype(str).values)


prot = pd.DataFrame(index=op.obs["cell_key"].astype(str))
prot = prot[~prot.index.duplicated()]
for lab, col in [("pRb", "cp_single_object_pRb_cell_Intensity_MeanIntensity"),
                 ("Rb", "cp_single_object_Rb_cell_Intensity_MeanIntensity"),
                 ("p21", "cp_single_object_p21_cell_Intensity_MeanIntensity"),
                 ("nuclear_area_4i", "op_nuclei_area_mean"), ("cell_area_4i", "op_cell_area")]:
    s = opcol(col); prot[lab] = s[~s.index.duplicated()].reindex(prot.index)

d = dn.set_index(dn["cell_key"].astype(str)) if "cell_key" in dn else dn
d = d[~d.index.duplicated()]
dna = pd.DataFrame(index=d.index)
dna["dna_content_log2"] = d["dna_rel_R234"].astype(float)
dna["dna_reliable"] = d["dna_reliable"].astype(str) == "True"
dna["dense_field"] = d["dense_field"].astype(str) == "True"

phset = set(ph["cell_key"].astype(str)) if "cell_key" in ph else set(ph.index.astype(str))

# union of all NTC cells, plus per-cell match quality
base = op.obs.drop_duplicates("cell_key").set_index(op.obs.drop_duplicates("cell_key")["cell_key"].astype(str))
F = pd.DataFrame(index=base.index)
F["well"] = base["well"].astype(str); F["tile_pheno"] = base["tile_pheno"].astype(str)
F["pheno_to_fixed_distance_px"] = base["pheno_to_fixed_distance_px"].astype(float)
F["has_4i_protein"] = base["has_4i_features"].astype(bool)
F = F.join(prot).join(dna)
F["has_phase"] = F.index.isin(phset)
F["prb_rb_ratio"] = F["pRb"] / F["Rb"]
F["log2_pRb"] = np.log2(F["pRb"].clip(lower=1))

# DNA x pRb cell-cycle stage on reliable cells: 2N/4N by DNA, pRb low/high at the 2N restriction point
rel = F["dna_reliable"].fillna(False)
g1mode = F.loc[rel, "dna_content_log2"].median()          # robust G1 anchor (population is G1-dominant)
twoN = rel & (F["dna_content_log2"] < 0.3)
prb_thr = np.nanmedian(F.loc[twoN, "log2_pRb"])           # restriction-point split within 2N
stage = pd.Series(pd.NA, index=F.index, dtype=object)
stage[rel & (F["dna_content_log2"] < 0.3) & (F["log2_pRb"] < prb_thr)] = "G0/G1-early"
stage[rel & (F["dna_content_log2"] < 0.3) & (F["log2_pRb"] >= prb_thr)] = "G1-late"
stage[rel & (F["dna_content_log2"] >= 0.3) & (F["dna_content_log2"] < 0.7)] = "S"
stage[rel & (F["dna_content_log2"] >= 0.7)] = "G2/M"
F["cycle_stage"] = stage

F.index.name = "cell_key"
F.to_parquet("OPS0077_NTC_cellcycle_frame.parquet")
print(f"frame rows {len(F):,}")
print("modality coverage:", {"4i protein": int(F.has_4i_protein.fillna(False).sum()),
                             "DNA reliable": int(F.dna_reliable.fillna(False).sum()),
                             "phase": int(F.has_phase.sum()),
                             "all three": int((F.has_4i_protein.fillna(False) & F.dna_reliable.fillna(False) & F.has_phase).sum())})
print("\ncycle stage (reliable cells):")
print(F["cycle_stage"].value_counts(dropna=False).to_string())
print("\nmedian readouts by stage:")
print(F.dropna(subset=["cycle_stage"]).groupby("cycle_stage")[["dna_content_log2", "log2_pRb", "prb_rb_ratio", "p21", "nuclear_area_4i"]].median().round(3).to_string())
print("\np21 near-background check: fraction of reliable cells with p21 in [120,125]:",
      round(float(((F.loc[rel, "p21"] >= 120) & (F.loc[rel, "p21"] <= 125)).mean()), 3))
