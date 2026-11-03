#!/usr/bin/env python
"""Laptop version of the knockout DNA measurement (dapi_cpmeasure.py + bg_pass.py + the DNA step of ko_dna_measure.py), for
the knockouts only the gradient-boosting model calls. Reads only what the DNA number needs from the public Biohub OPS0077 zarr
(anonymous S3): DAPI rounds 2-4, the cell mask and the round 2-4 nuclear masks, in the same 256 px crop around (x_cp, y_cp).
Per round k (same rules as the originals):
  nucleus  = round-k nucleus with the largest overlap with the cell, kept only if >= half of it lies inside the cell
  n_nuclei = round-k nuclei with >= half inside the cell
  bg       = median DAPI over crop pixels outside every round-k nucleus dilated by 3 px (bg_pass.py, needs >= 200 px)
  dna_Rk   = log2(sum of DAPI over the nucleus - bg x nucleus area) - NTC median of dna_Rk_log2 (reliable NTC cells)
dna = mean over rounds 2-4; usable = nucleus found in R2-4 and n_nuclei_R2 <= 1 (ko_dna_measure.py). cp_measure's
IntegratedIntensity x 65535 is exactly the pixel sum, and its Area the pixel count, so no featurizer is needed.

usage:
  python ko_dna_light.py list GENES_CSV OUT_CSV [MAX_CELLS=250]   cells of those genes (column gene) from the public atlas obs
  python ko_dna_light.py validate NTC_DAPI_H5AD OUT_DIR [N=300]   same measurement on NTC cells already in the NTC table
  python ko_dna_light.py measure CELLS_CSV NTC_DAPI_H5AD OUT_DIR  knockout cells -> OUT_DIR/part_*.parquet (resumable)
env THREADS (default 8)."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

ROOT = ("s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/datasets/"
        "Biohub_OPS0077/Biohub_OPS0077.zarr")
ATLAS = ("ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/"
         "all_cells_phase.h5ad")
SO = {"anon": True, "client_kwargs": {"region_name": "us-west-2"}}
DAPI_CH = {2: 6, 3: 8, 4: 10}; H, RING = 128, 3
THREADS = int(os.environ.get("THREADS", 8))
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def cell_list(genes_csv, out_csv, maxc):
    import h5py, s3fs
    G = set(pd.read_csv(genes_csv)["gene"].astype(str).str.upper())
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"}, default_block_size=32 * 2**20)
    with fs.open(ATLAS, "rb") as f, h5py.File(f, "r") as h:
        o = h["obs"]; cats = o["experiment"]["categories"][:].astype(str); ci = int(np.where(cats == "ops0144_20260406")[0][0])
        rows = np.where(o["experiment"]["codes"][:] == ci)[0]; lo, hi = rows[0], rows[-1] + 1; idx = rows - lo
        col = lambda n: (o[n]["categories"][:].astype(str)[o[n]["codes"][lo:hi][idx]] if isinstance(o[n], h5py.Group) else o[n][lo:hi][idx])
        d = pd.DataFrame({"gene": pd.Series(col("gene")).str.upper().values, "well": col("well"), "tile_pheno": col("tile_pheno"),
                          "segmentation_id": col("segmentation_id"), "x_cp": col("x_cp"), "y_cp": col("y_cp")})
    d = d[d.gene.isin(G)].copy(); d["cell_key"] = d.well.astype(str) + "|" + d.segmentation_id.astype("int64").astype(str)
    d = d.drop_duplicates("cell_key")
    d = d.groupby("gene", group_keys=False).apply(lambda g: g.sample(min(len(g), maxc), random_state=0)).reset_index(drop=True)
    d.to_csv(out_csv, index=False); log(f"{d.gene.nunique()} of {len(G)} genes | {len(d):,} cells -> {out_csv}")


def open_arrays(wells):
    import zarr
    A = {}
    for w in wells:
        A[w] = {"img": zarr.open_array(f"{ROOT}/A/{w}/0/0", mode="r", storage_options=SO),
                "cell": zarr.open_array(f"{ROOT}/A/{w}/0/labels/4i_cell_seg/0", mode="r", storage_options=SO)}
        for k in DAPI_CH: A[w][k] = zarr.open_array(f"{ROOT}/A/{w}/0/labels/4i_R{k}_nuclear_seg/0", mode="r", storage_options=SO)
    return A


def measure_one(r, A):
    from scipy.ndimage import binary_dilation
    w = str(r.well).split("/")[1]; a = A[w]; Y, X = a["cell"].shape[-2:]
    yc, xc, sid = int(round(float(r.y_cp))), int(round(float(r.x_cp))), int(r.segmentation_id)
    sy, sx = slice(max(0, yc - H), min(Y, yc + H)), slice(max(0, xc - H), min(X, xc + H))
    cell = np.asarray(a["cell"][0, 0, 0, sy, sx]); cm = cell == sid
    img = np.asarray(a["img"][0, [DAPI_CH[k] for k in DAPI_CH], 0, sy, sx], dtype=np.float64)
    rec = {"cell_key": r.cell_key, "cell_px": int(cm.sum())}
    st = np.ones((2 * RING + 1, 2 * RING + 1), bool)
    for j, k in enumerate(DAPI_CH):
        nuc = np.asarray(a[k][0, 0, 0, sy, sx])
        ids, cnt = np.unique(nuc[cm], return_counts=True); keep = ids != 0; ids, cnt = ids[keep], cnt[keep]
        tot = np.array([(nuc == i_).sum() for i_ in ids]) if len(ids) else np.array([]); frac = cnt / np.maximum(tot, 1)
        rec[f"R{k}_n_nuclei_in_cell"] = int((frac >= .5).sum())
        px = img[j][~binary_dilation(nuc > 0, structure=st)]
        rec[f"R{k}_bg"] = float(np.median(px)) if px.size >= 200 else np.nan
        if len(ids) and frac[np.argmax(cnt)] >= .5:
            m = nuc == ids[np.argmax(cnt)]; rec[f"R{k}_found"] = True
            rec[f"R{k}_sum"] = float(img[j][m].sum()); rec[f"R{k}_area"] = int(m.sum())
        else:
            rec[f"R{k}_found"] = False; rec[f"R{k}_sum"] = np.nan; rec[f"R{k}_area"] = np.nan
    return rec


def run(cells, out_dir, batch=200):
    os.makedirs(out_dir, exist_ok=True); A = open_arrays(sorted(cells.well.astype(str).str.split("/").str[1].unique()))
    pool = ThreadPoolExecutor(THREADS); t0 = time.time(); todo = range(0, len(cells), batch)
    for s in todo:
        fo = f"{out_dir}/part_{s:06d}.parquet"
        if os.path.exists(fo): continue
        recs = list(pool.map(lambda i: measure_one(cells.iloc[i], A), range(s, min(s + batch, len(cells)))))
        pd.DataFrame(recs).to_parquet(fo)
        done = min(s + batch, len(cells)); el = (time.time() - t0) / 60
        log(f"{done:,}/{len(cells):,} cells | {el:.1f} min")


def dna_table(raw, ntc_h5ad):
    import anndata as ad
    ntc = ad.read_h5ad(ntc_h5ad, backed="r").obs; ok = ntc["dna_reliable"].astype(str) == "True"
    med = {k: float(ntc.loc[ok, f"dna_R{k}_log2"].astype(float).median()) for k in DAPI_CH}
    d = raw.set_index("cell_key")
    for k in DAPI_CH:
        v = d[f"R{k}_sum"] - d[f"R{k}_bg"] * d[f"R{k}_area"]; d[f"dna_R{k}_log2_abs"] = np.log2(v.where(v > 0)); d[f"dna_R{k}"] = d[f"dna_R{k}_log2_abs"] - med[k]
    d["dna"] = d[[f"dna_R{k}" for k in DAPI_CH]].mean(1, skipna=False)
    d["ok"] = d[[f"R{k}_found" for k in DAPI_CH]].all(1) & (d["R2_n_nuclei_in_cell"] <= 1)
    return d


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "list":
        cell_list(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 250)
    elif mode == "validate":
        import anndata as ad
        h5, out = sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        o = ad.read_h5ad(h5, backed="r").obs
        o = o[o["dna_reliable"].astype(str) == "True"].sample(n, random_state=1).copy(); o["cell_key"] = o.index.astype(str)
        t = time.time(); run(o.reset_index(drop=True), out); log(f"{n} cells in {(time.time() - t) / 60:.1f} min with {THREADS} threads")
        d = dna_table(pd.concat([pd.read_parquet(p) for p in sorted(__import__("glob").glob(f"{out}/part_*.parquet"))]), h5)
        ref = o.set_index("cell_key")
        for k in DAPI_CH:
            a_, b_ = d[f"dna_R{k}_log2_abs"], ref.loc[d.index, f"dna_R{k}_log2"].astype(float)
            ok = a_.notna() & b_.notna(); print(f"R{k}: n {ok.sum()} max |diff| {np.abs(a_[ok] - b_[ok]).max():.2e}  r {np.corrcoef(a_[ok], b_[ok])[0, 1]:.6f}")
        a_, b_ = d["dna"], ref.loc[d.index, "dna_rel_R234"].astype(float); ok = a_.notna() & b_.notna()
        print(f"dna_rel_R234: n {ok.sum()} median |diff| {np.median(np.abs(a_[ok] - b_[ok])):.2e} max {np.abs(a_[ok] - b_[ok]).max():.2e} | usable {int((d.ok & d.dna.notna()).sum())}/{len(d)}")
        d.to_csv(f"{out}/validate_percell.csv")
    elif mode == "measure":
        cells = pd.read_csv(sys.argv[2]); run(cells, sys.argv[4])
        import glob
        d = dna_table(pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{sys.argv[4]}/part_*.parquet"))]), sys.argv[3])
        d = d.join(cells.set_index("cell_key")[["gene", "well", "tile_pheno"]]); d.to_csv(f"{sys.argv[4]}/ko_measured_dna_percell.csv")
        log(f"usable {int((d.ok & d.dna.notna()).sum()):,} of {len(d):,}")
