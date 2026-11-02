#!/usr/bin/env python
"""Extract per-cell phase-contrast OrganelleProfiler features for OPS0077 cells of a set of target genes (the 59
arrest genes), by byte-range read of the 870 GB atlas all_cells_phase.h5ad. Writes OPS0077_genes_phase.h5ad with
obs: gene, cell_key, well, tile_pheno, segmentation_id, x_cp, y_cp.
usage: python extract_phase_genes.py genes_cc.csv OUT.h5ad
"""
import sys, time
import numpy as np, pandas as pd, anndata as ad, h5py, s3fs

GENES_CSV, OUT = sys.argv[1:3]
G = set(pd.read_csv(GENES_CSV)["Gene_Symbol"].astype(str).str.upper())
P = ("ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/"
     "all_cells_phase.h5ad")
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"}, default_block_size=32 * 2**20)

t = time.time()
with fs.open(P, "rb") as f, h5py.File(f, "r") as h:
    cats = h["obs"]["experiment"]["categories"][:].astype(str); ci = int(np.where(cats == "ops0144_20260406")[0][0])
    codes = h["obs"]["experiment"]["codes"][:]; rows = np.where(codes == ci)[0]
    lo, hi = rows[0], rows[-1] + 1; idx = rows - lo
    gcat = h["obs"]["gene"]["categories"][:].astype(str); gene = pd.Series(gcat[h["obs"]["gene"]["codes"][lo:hi][idx]]).str.upper()
    sel = gene.isin(G).values
    seg = h["obs"]["segmentation_id"][lo:hi][idx][sel]
    wcat = h["obs"]["well"]["categories"][:].astype(str); well = wcat[h["obs"]["well"]["codes"][lo:hi][idx][sel]]
    tcat = h["obs"]["tile_pheno"]["categories"][:].astype(str); tile = tcat[h["obs"]["tile_pheno"]["codes"][lo:hi][idx][sel]]
    xcp = h["obs"]["x_cp"][lo:hi][idx][sel]; ycp = h["obs"]["y_cp"][lo:hi][idx][sel]
    grows = rows[sel]
    var = h["var"]["_index"][:].astype(str)
    X = h["X"]; off, n_feat, dt = X.id.get_offset(), X.shape[1], X.dtype
    assert X.compression is None and X.chunks is None and off is not None
print(f"selected {sel.sum():,} gene-KO cells | {n_feat} feats | header {time.time()-t:.0f}s", flush=True)

order = np.argsort(grows); grows_s = grows[order]
rb = n_feat * np.dtype(dt).itemsize
starts = [int(off + r * rb) for r in grows_s]; ends = [s + rb for s in starts]
t = time.time(); data = []; B = 4000
for i in range(0, len(starts), B):
    data += fs.cat_ranges([P] * len(starts[i:i+B]), starts[i:i+B], ends[i:i+B])
    print(f"  fetched {min(i+B,len(starts)):,}/{len(starts):,} | {time.time()-t:.0f}s", flush=True)
Xg = np.frombuffer(b"".join(data), dtype=dt).reshape(len(grows_s), n_feat)

key = pd.Series(well).str[:] + "|" + pd.Series(seg).astype("int64").astype(str)
obs = pd.DataFrame({"gene": pd.Series(gene[sel].values)[order].values, "cell_key": key.values[order],
                    "well": well[order], "tile_pheno": tile[order], "segmentation_id": seg[order].astype("int64"),
                    "x_cp": xcp[order], "y_cp": ycp[order]})
out = ad.AnnData(X=Xg.astype(np.float32), obs=obs.reset_index(drop=True), var=pd.DataFrame(index=var))
out.obs_names = out.obs["cell_key"].astype(str) + "__" + out.obs.index.astype(str)
for c in out.obs.columns:
    if out.obs[c].dtype == object: out.obs[c] = out.obs[c].astype(str)
out.write_h5ad(OUT, compression="gzip")
print(f"wrote {OUT} {out.shape} | finite {np.isfinite(Xg).mean():.3f} | genes {out.obs.gene.nunique()}", flush=True)
print("DONE")
