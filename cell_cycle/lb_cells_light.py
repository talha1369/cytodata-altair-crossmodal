#!/usr/bin/env python
"""Low-memory cell list for a few genes from the public atlas obs: reads the experiment and gene codes, then the other
columns only at the matching rows (same output columns as ko_dna_light.py list, plus sgRNA; no cap).
usage: python lb_cells_light.py GENES_CSV OUT_CSV"""
import sys
import numpy as np, pandas as pd, h5py, s3fs

ATLAS = ("ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/"
         "all_cells_phase.h5ad")
G = set(pd.read_csv(sys.argv[1])["gene"].astype(str).str.upper())
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"}, default_block_size=8 * 2**20)
with fs.open(ATLAS, "rb") as f, h5py.File(f, "r") as h:
    o = h["obs"]; cats = o["experiment"]["categories"][:].astype(str); ci = int(np.where(cats == "ops0144_20260406")[0][0])
    exp = o["experiment"]["codes"]; n = exp.shape[0]; rows = []
    for s in range(0, n, 5_000_000):
        rows.append(s + np.where(exp[s:s + 5_000_000] == ci)[0])
    rows = np.concatenate(rows); gcat = np.char.upper(o["gene"]["categories"][:].astype(str))
    want = np.where(np.isin(gcat, list(G)))[0]; gc = o["gene"]["codes"]
    lo, hi = rows[0], rows[-1] + 1; hit = []
    for s in range(lo, hi, 2_000_000):
        e = min(s + 2_000_000, hi); c = gc[s:e]; hit.append(s + np.where(np.isin(c, want))[0])
    hit = np.intersect1d(np.concatenate(hit), rows)
    def col(nm):
        x = o[nm]
        if isinstance(x, h5py.Group):
            return x["categories"][:].astype(str)[x["codes"][hit]]
        return x[hit]
    d = pd.DataFrame({"gene": pd.Series(col("gene")).str.upper().values, "well": col("well"), "tile_pheno": col("tile_pheno"),
                      "segmentation_id": col("segmentation_id"), "x_cp": col("x_cp"), "y_cp": col("y_cp"), "sgRNA": col("sgRNA")})
d["cell_key"] = d.well.astype(str) + "|" + d.segmentation_id.astype("int64").astype(str); d = d.drop_duplicates("cell_key")
d.to_csv(sys.argv[2], index=False); print(d.gene.value_counts().to_string()); print(len(d), "cells")
