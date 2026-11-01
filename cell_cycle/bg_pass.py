#!/usr/bin/env python
"""Per-cell local DAPI background from non-nuclear pixels, for every NTC cell in the DAPI table.

Same crops as dapi_cpmeasure.py (256 px around x_cp, y_cp in the public Biohub OPS0077 zarr). For each round k the
background is taken over crop pixels outside every round-k nucleus, after dilating the nucleus masks by RING px so
blur at nucleus edges is excluded. Unlike the cell-free background (which needs gaps between cells and is missing in
dense fields), this exists for nearly every cell and reflects the local haze from neighbouring nuclei.

usage: python bg_pass.py DAPI_H5AD OUT_DIR [--n N] [--part i --nparts N] [--threads 48]
"""
import argparse, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, anndata as ad, zarr
from scipy.ndimage import binary_dilation

ROOT = ("s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/datasets/"
        "Biohub_OPS0077/Biohub_OPS0077.zarr")
SO = {"anon": True, "client_kwargs": {"region_name": "us-west-2"}}
DAPI_CH = {1: 4, 2: 6, 3: 8, 4: 10, 5: 13}
H, RING = 128, 3
STRUCT = np.ones((2 * RING + 1, 2 * RING + 1), bool)

ap = argparse.ArgumentParser()
ap.add_argument("dapi"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
ap.add_argument("--threads", type=int, default=48); ap.add_argument("--batch", type=int, default=512)
a = ap.parse_args()

obs = ad.read_h5ad(a.dapi, backed="r").obs[["well", "x_cp", "y_cp"]].copy()
obs["cell_key"] = obs.index.astype(str)
obs = obs.reset_index(drop=True)
if a.n:
    obs = obs.sample(n=a.n, random_state=0).reset_index(drop=True)
if a.nparts > 1:
    obs = obs.iloc[a.part::a.nparts].reset_index(drop=True)
    a.out = os.path.join(a.out, f"part{a.part:02d}")
os.makedirs(a.out, exist_ok=True)
print(f"{len(obs)} cells | ring {RING}px | threads {a.threads}", flush=True)

ARR = {}
for w in sorted(obs["well"].astype(str).str.split("/").str[1].unique()):
    ARR[w] = {"img": zarr.open_array(f"{ROOT}/A/{w}/0/0", mode="r", storage_options=SO)}
    for k in DAPI_CH:
        ARR[w][k] = zarr.open_array(f"{ROOT}/A/{w}/0/labels/4i_R{k}_nuclear_seg/0", mode="r", storage_options=SO)


def one(i):
    r = obs.iloc[i]; w = str(r.well).split("/")[1]; A = ARR[w]
    yc, xc = int(round(float(r.y_cp))), int(round(float(r.x_cp)))
    Y, X = A[1].shape[-2:]
    sy, sx = slice(max(0, yc - H), min(Y, yc + H)), slice(max(0, xc - H), min(X, xc + H))
    img = np.asarray(A["img"][0, [DAPI_CH[k] for k in DAPI_CH], 0, sy, sx], dtype=np.float32)
    rec = {"cell_key": r.cell_key}
    for j, k in enumerate(DAPI_CH):
        nuc = np.asarray(A[k][0, 0, 0, sy, sx]) > 0
        bgm = ~binary_dilation(nuc, structure=STRUCT)
        px = img[j][bgm]
        rec[f"R{k}_bg_nonnuc_npix"] = int(px.size)
        rec[f"R{k}_bg_nonnuc_median"] = float(np.median(px)) if px.size >= 200 else np.nan
        rec[f"R{k}_bg_nonnuc_p10"] = float(np.percentile(px, 10)) if px.size >= 200 else np.nan
    return rec


pool = ThreadPoolExecutor(a.threads)
t0 = time.time()
for s in range(0, len(obs), a.batch):
    fo = os.path.join(a.out, f"bg_{s:06d}.parquet")
    if os.path.exists(fo):
        continue
    recs = list(pool.map(one, range(s, min(s + a.batch, len(obs)))))
    pd.DataFrame(recs).to_parquet(fo)
    print(f"{min(s + a.batch, len(obs))}/{len(obs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
print("DONE", flush=True)
