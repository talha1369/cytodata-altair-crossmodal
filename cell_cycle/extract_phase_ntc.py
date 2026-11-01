#!/usr/bin/env python
"""Extract per-cell OrganelleProfiler PHASE features for the OPS0077 NTC cells from the 870 GB atlas file.

all_cells_phase.h5ad (public S3) is 64.6M cells x 3,317 float32 features, stored contiguous and uncompressed, so
row r sits at byte offset X_offset + r * 3317 * 4. The OPS0077 (ops0144_20260406) rows are scattered between other
screens, so this fetches only the NTC rows with parallel byte-range reads (no full download).
Cells are matched to the NTC OrganelleProfiler table on well + 4i segmentation id (the phase file's segmentation_id
is the same 4i id as in the fluorescent per-marker files).

usage: python extract_phase_ntc.py NTC_H5AD OUT_H5AD
"""
import sys, time
import numpy as np, pandas as pd, anndata as ad, h5py, s3fs

NTC, OUT = sys.argv[1:3]
P = ("ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/"
     "all_cells_phase.h5ad")
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"}, default_block_size=32 * 2**20)

t = time.time()
with fs.open(P, "rb") as f, h5py.File(f, "r") as h:
    cats = h["obs"]["experiment"]["categories"][:].astype(str)
    ci = int(np.where(cats == "ops0144_20260406")[0][0])
    codes = h["obs"]["experiment"]["codes"][:]
    rows = np.where(codes == ci)[0]
    lo, hi = rows[0], rows[-1] + 1
    seg = h["obs"]["segmentation_id"][lo:hi][rows - lo]
    wc = h["obs"]["well"]["codes"][lo:hi][rows - lo]
    wcat = h["obs"]["well"]["categories"][:].astype(str)
    gc = h["obs"]["gene"]["codes"][lo:hi][rows - lo]
    gcat = h["obs"]["gene"]["categories"][:].astype(str)
    X = h["X"]
    off, n_feat, dt = X.id.get_offset(), X.shape[1], X.dtype
    var = h["var"]["_index"][:].astype(str)
    assert X.compression is None and X.chunks is None and off is not None, "X must be contiguous, uncompressed"
print(f"ops0144 rows {len(rows):,} | features {n_feat} | header {time.time() - t:.0f}s", flush=True)

ph = pd.DataFrame({"row": rows, "key": pd.Series(wcat[wc]) + "|" + pd.Series(seg).astype("int64").astype(str).values,
                   "gene_phase": gcat[gc]})
ntc = ad.read_h5ad(NTC, backed="r").obs.drop_duplicates("cell_key")
sel = ph[ph["key"].isin(set(ntc["cell_key"]))].drop_duplicates("key").sort_values("row")
print(f"NTC cells in table {len(ntc):,} | matched in phase file {len(sel):,} | "
      f"phase gene label NTC among matched {(sel.gene_phase == 'NTC').mean():.3f}", flush=True)

rb = n_feat * np.dtype(dt).itemsize
starts = [int(off + r * rb) for r in sel["row"]]
ends = [s + rb for s in starts]
t = time.time()
data = []
B = 4000
for i in range(0, len(starts), B):
    chunk = fs.cat_ranges([P] * len(starts[i:i + B]), starts[i:i + B], ends[i:i + B])
    data += chunk
    print(f"  fetched {min(i + B, len(starts)):,}/{len(starts):,} rows | {time.time() - t:.0f}s", flush=True)
Xn = np.frombuffer(b"".join(data), dtype=dt).reshape(len(sel), n_feat)

obs = ntc.set_index("cell_key").loc[sel["key"].values,
                                    ["well", "tile_pheno", "segmentation_id", "gene", "sgRNA", "barcode",
                                     "pheno_cell_uid", "fixed_cell_uid", "pheno_to_fixed_distance_px"]].copy()
obs["phase_row"] = sel["row"].values
obs.index.name = "cell_key"
out = ad.AnnData(X=Xn.astype(np.float32), obs=obs.reset_index().set_index("cell_key", drop=False),
                 var=pd.DataFrame(index=var))
for c in out.obs.columns:
    if out.obs[c].dtype == object:
        out.obs[c] = out.obs[c].astype(str)
out.uns["provenance"] = {"source": "s3://" + P, "experiment": "ops0144_20260406 (Biohub_OPS0077)",
                         "features": "OrganelleProfiler per-cell phase features (not Cell-DINO; Cell-DINO is gene-level only)",
                         "match": "well + 4i segmentation_id == NTC table cell_key"}
out.write_h5ad(OUT, compression="gzip")
fin = np.isfinite(Xn).mean()
print(f"wrote {OUT} {out.shape} | finite fraction {fin:.3f}", flush=True)
print("DONE")
