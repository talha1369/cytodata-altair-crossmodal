#!/usr/bin/env python
"""Extract per-cell phase-contrast OrganelleProfiler features for ALL OPS0077 knockout cells (every gene except NTC), by
byte-range read of the 870 GB atlas all_cells_phase.h5ad. Same output schema as extract_phase_genes.py; the feature matrix is
preallocated and filled batch by batch so peak memory stays near one copy (~8 GB for the full library).
usage: python extract_phase_all.py OUT.h5ad [MAX_GENES]   (MAX_GENES for a quick test on the first genes alphabetically)
"""
import sys, time
import numpy as np, pandas as pd, anndata as ad, h5py, s3fs

OUT = sys.argv[1]
MAXG = int(sys.argv[2]) if len(sys.argv) > 2 else None
P = ("ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/"
     "all_cells_phase.h5ad")
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"}, default_block_size=32 * 2**20)

t = time.time()
with fs.open(P, "rb") as f, h5py.File(f, "r") as h:
    cats = h["obs"]["experiment"]["categories"][:].astype(str); ci = int(np.where(cats == "ops0144_20260406")[0][0])
    codes = h["obs"]["experiment"]["codes"][:]; rows = np.where(codes == ci)[0]
    lo, hi = rows[0], rows[-1] + 1; idx = rows - lo
    gcat = h["obs"]["gene"]["categories"][:].astype(str); gene = pd.Series(gcat[h["obs"]["gene"]["codes"][lo:hi][idx]]).str.upper()
    G = sorted(set(gene) - {"NTC"})[:MAXG]
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
# the selected rows are nearly all of the experiment's contiguous block: read it in large contiguous chunks and keep the
# selected rows (far faster than one byte range per cell)
t = time.time(); CH = 2000; PAR = 16
Xg = np.empty((len(grows_s), n_feat), dtype=np.float32)
blocks = list(range(int(grows_s[0]), int(grows_s[-1]) + 1, CH)); filled = 0
print(f"block span {grows_s[-1] - grows_s[0] + 1:,} rows for {len(grows_s):,} selected", flush=True)
for i in range(0, len(blocks), PAR):
    bs = blocks[i:i + PAR]; be = [min(b + CH, int(grows_s[-1]) + 1) for b in bs]
    raw = fs.cat_ranges([P] * len(bs), [int(off + b * rb) for b in bs], [int(off + e * rb) for e in be])
    for b, e, r in zip(bs, be, raw):
        A = np.frombuffer(r, dtype=dt).reshape(e - b, n_feat)
        lo_, hi_ = np.searchsorted(grows_s, [b, e]); Xg[lo_:hi_] = A[grows_s[lo_:hi_] - b]; filled += hi_ - lo_
    if (i // PAR) % 5 == 0: print(f"  rows filled {filled:,}/{len(grows_s):,} | {time.time()-t:.0f}s", flush=True)
assert filled == len(grows_s)

key = pd.Series(well).str[:] + "|" + pd.Series(seg).astype("int64").astype(str)
obs = pd.DataFrame({"gene": pd.Series(gene[sel].values)[order].values, "cell_key": key.values[order],
                    "well": well[order], "tile_pheno": tile[order], "segmentation_id": seg[order].astype("int64"),
                    "x_cp": xcp[order], "y_cp": ycp[order]})
out = ad.AnnData(X=Xg, obs=obs.reset_index(drop=True), var=pd.DataFrame(index=var))
out.obs_names = out.obs["cell_key"].astype(str) + "__" + out.obs.index.astype(str)
for c in out.obs.columns:
    if out.obs[c].dtype == object: out.obs[c] = out.obs[c].astype(str)
out.write_h5ad(OUT)
print(f"wrote {OUT} {out.shape} | finite {np.isfinite(Xg).mean():.3f} | genes {out.obs.gene.nunique()}", flush=True)
print("DONE")
