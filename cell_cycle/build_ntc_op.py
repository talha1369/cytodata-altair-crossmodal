#!/usr/bin/env python
"""Build an NTC single-cell OrganelleProfiler table for Biohub OPS0077 (4i) from the public CZ Biohub atlas.

Sources (anonymous S3, ops-explorer-public, us-west-2):
  leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/all_cells_fluor_{prb,rb,p21}.h5ad
Each file has the same 4i cells (experiment ops0144_20260406) in the same row order; it carries the shared cell and
nucleus features plus the features of its own marker. This script keeps the NTC cells, takes the shared features
once (checked identical across files) and the marker-specific features from every file, and joins the team's NTC list
(OPS0077_NTC_stamped.h5ad) on well + 4i segmentation id.

usage: python build_ntc_op.py RAW_DIR NTC_STAMPED_H5AD OUT_H5AD
"""
import os, sys, subprocess
import anndata as ad, numpy as np, pandas as pd

RAW, STAMPED, OUT = sys.argv[1:4]
MARKERS = ["prb", "rb", "p21"]
S3 = ("s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/"
      "all_cells_fluor_{}.h5ad")

os.makedirs(RAW, exist_ok=True)
for m in MARKERS:
    dst = os.path.join(RAW, f"all_cells_fluor_{m}.h5ad")
    if not os.path.exists(dst):
        print("download", m, flush=True)
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--region", "us-west-2", S3.format(m), dst], check=True)


def key(obs):
    return obs["well"].astype(str) + "|" + obs["segmentation_id"].astype("int64").astype(str)


A = {m: ad.read_h5ad(os.path.join(RAW, f"all_cells_fluor_{m}.h5ad"), backed="r") for m in MARKERS}
ref = A["prb"].obs
for m in MARKERS[1:]:
    assert A[m].n_obs == A["prb"].n_obs, m
    assert (key(A[m].obs).values == key(ref).values).all(), f"row order differs: {m}"
    assert (A[m].obs["gene"].astype(str).values == ref["gene"].astype(str).values).all(), f"gene differs: {m}"
print("same cells, same order in", MARKERS, flush=True)

ntc = np.where(ref["gene"].astype(str).values == "NTC")[0]
print("atlas NTC rows", len(ntc), flush=True)

blocks, names, shared = [], [], None
Xs = {m: A[m][ntc].to_memory().X for m in MARKERS}
V = {m: list(A[m].var_names) for m in MARKERS}
common = [n for n in V["prb"] if all(n in V[m] for m in MARKERS)]
for n in common:                       # shared features must be identical across the files
    a = Xs["prb"][:, V["prb"].index(n)]
    for m in MARKERS[1:]:
        b = Xs[m][:, V[m].index(n)]
        assert np.array_equal(np.nan_to_num(a, nan=-1e30), np.nan_to_num(b, nan=-1e30)), f"shared feature differs: {n} ({m})"
print("shared features", len(common), "identical across files", flush=True)
idx = [V["prb"].index(n) for n in common]
blocks.append(Xs["prb"][:, idx]); names += common
var_src = ["shared"] * len(common)
var_frames = [A["prb"].var.iloc[idx]]
seen = set(common)
for m in MARKERS:
    own = []
    for i, n in enumerate(V[m]):
        if n in seen:
            continue
        holders = [o for o in MARKERS if n in V[o]]       # pairwise colocalisation features sit in both partners' files
        for o in holders[1:]:
            b = Xs[o][:, V[o].index(n)]
            assert np.array_equal(np.nan_to_num(Xs[m][:, i], nan=-1e30), np.nan_to_num(b, nan=-1e30)), f"differs: {n} ({m}/{o})"
        own.append(i); seen.add(n)
    blocks.append(Xs[m][:, own]); names += [V[m][i] for i in own]
    var_src += ["+".join(o for o in MARKERS if V[o].count(V[m][i])) for i in own]
    var_frames.append(A[m].var.iloc[own])
    print(m, "features taken from this file", len(own), flush=True)
X = np.hstack(blocks).astype(np.float32)
var = pd.concat(var_frames); var.index = names; var["source_file"] = var_src
assert var.index.is_unique

obs = ref.iloc[ntc].copy()
obs["cell_key"] = key(obs).values
obs.index = obs["cell_key"].values + "|" + np.arange(len(obs)).astype(str)

st = ad.read_h5ad(STAMPED).obs.copy()
st["cell_key"] = (st["well"].str[0] + "/" + st["well"].str[1:] + "/0|" + st["fixed_segmentation_id"].astype("int64").astype(str))
keep = ["cell_id", "fixed_cell_uid", "pheno_to_fixed_distance_px", "fixed_area_px2", "sgRNA_protospacer", "control_type"]
st = st.drop_duplicates("cell_key")[["cell_key"] + keep].rename(columns={"cell_id": "pheno_cell_uid"})
obs = obs.reset_index().rename(columns={"index": "obs_name"}).merge(st, on="cell_key", how="left").set_index("obs_name")
obs["in_team_ntc_list"] = obs["pheno_cell_uid"].notna()
obs["dup_key"] = obs["cell_key"].duplicated(keep=False)
print("in team NTC list", int(obs["in_team_ntc_list"].sum()), "of", len(obs), "| team list size", len(st),
      "| duplicated cell keys", int(obs["dup_key"].sum()), flush=True)

out = ad.AnnData(X=X, obs=obs, var=var)
out.uns["provenance"] = {"source": S3.format("{prb,rb,p21}"), "experiment": "ops0144_20260406",
                         "dataset": "Biohub_OPS0077", "filter": "gene == 'NTC'",
                         "builder": "build_ntc_op.py"}
for c in out.obs.columns:
    if out.obs[c].dtype == object:
        out.obs[c] = out.obs[c].astype(str)
out.write_h5ad(OUT, compression="gzip")
print("wrote", OUT, out.shape, f"{os.path.getsize(OUT) / 1e6:.0f} MB", flush=True)
for m, f in [("pRb", "cp_single_object_pRb_cell_Intensity_MeanIntensity"),
             ("Rb", "cp_single_object_Rb_cell_Intensity_MeanIntensity"),
             ("p21", "cp_single_object_p21_cell_Intensity_MeanIntensity")]:
    print(m, "present in", f"{np.isfinite(out[:, f].X).mean():.3f}", "of NTC cells")
print("DONE")
