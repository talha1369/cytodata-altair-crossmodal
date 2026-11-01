#!/usr/bin/env python
"""Per-cell DAPI features for the OPS0077 NTC cells with cp_measure (numba accelerator).

For every NTC cell in the NTC OrganelleProfiler table (has_4i_features, unique cell key) this reads a crop around the
cell from the public Biohub OPS0077 zarr (anonymous S3):
  image  A/<well>/0/0           channels 4,6,8,10,13 = 4i_R1..R5_nuclei_DAPI   (0.325 um/px)
  labels 4i_cell_seg            the cell (label == segmentation_id)
         4i_R<k>_nuclear_seg    nuclei of round k (ids are per round and unrelated to the cell id)
Per round k the cell's nucleus is the round-k nucleus with the largest overlap with the cell mask, kept only if at
least half of that nucleus lies inside the cell. DAPI_Rk is then featurized inside that nucleus with cp_measure's
featurizer (intensity, size/shape, texture, granularity, radial distribution, Zernike, Feret).
QC columns per round: nucleus id, overlap fraction, number of nuclei inside the cell, crop-border truncation, local
background (median DAPI where neither a cell nor a round-k nucleus is labelled in the crop).

usage: python dapi_cpmeasure.py NTC_H5AD OUT_DIR [--n N] [--half 128] [--batch 256] [--threads 48] [--rounds 1,2,3,4,5]
Output: OUT_DIR/feat_<start>.parquet + OUT_DIR/qc_<start>.parquet per batch (existing batches are skipped -> resumable).
"""
import argparse, os, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, anndata as ad, zarr
import cp_measure
from cp_measure.featurizer import make_featurizer_config, featurize

ROOT = ("s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/datasets/"
        "Biohub_OPS0077/Biohub_OPS0077.zarr")
SO = {"anon": True, "client_kwargs": {"region_name": "us-west-2"}}
DAPI_CH = {1: 4, 2: 6, 3: 8, 4: 10, 5: 13}
SCALE = 65535.0              # omero window of every DAPI channel is 0..65535 (16-bit); features are in units of SCALE
SAT = 65535.0 * 0.999        # round-1 DAPI saturates in bright nuclei; recorded per nucleus as QC
EXPECTED = ["4i_R1_nuclei_DAPI", "4i_R2_nuclei_DAPI", "4i_R3_nuclei_DAPI", "4i_R4_nuclei_DAPI", "4i_R5_nuclei_DAPI"]

ap = argparse.ArgumentParser()
ap.add_argument("ntc"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=0); ap.add_argument("--half", type=int, default=128)
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--threads", type=int, default=48)
ap.add_argument("--rounds", default="1,2,3,4,5")
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
a = ap.parse_args()
ROUNDS = [int(r) for r in a.rounds.split(",")]
H = a.half
os.makedirs(a.out, exist_ok=True)
cp_measure.set_accelerator("numba")
warnings.filterwarnings("ignore")

obs = ad.read_h5ad(a.ntc, backed="r").obs
obs = obs[obs["has_4i_features"].astype(bool)].drop_duplicates("cell_key")
obs = obs[["cell_key", "well", "x_cp", "y_cp", "segmentation_id"]].reset_index(drop=True)
if a.n:
    obs = obs.sample(n=a.n, random_state=0).reset_index(drop=True)
if a.nparts > 1:                       # interleaved split, so every part sees all wells; each part writes its own dir
    obs = obs.iloc[a.part::a.nparts].reset_index(drop=True)
    a.out = os.path.join(a.out, f"part{a.part:02d}")
    os.makedirs(a.out, exist_ok=True)
print(f"{len(obs)} cells, rounds {ROUNDS}, crop {2 * H}px, batch {a.batch}, threads {a.threads}", flush=True)

ARR = {}
for w in sorted(obs["well"].str.split("/").str[1].unique()):
    grp = zarr.open_group(f"{ROOT}/A/{w}/0", mode="r", storage_options=SO)
    labels = [c["label"] for c in grp.attrs["ome"]["omero"]["channels"]]
    assert [labels[DAPI_CH[k]] for k in range(1, 6)] == EXPECTED, (w, labels)
    ARR[w] = {"img": grp["0"], "cell": zarr.open_array(f"{ROOT}/A/{w}/0/labels/4i_cell_seg/0", mode="r", storage_options=SO)}
    for k in ROUNDS:
        ARR[w][k] = zarr.open_array(f"{ROOT}/A/{w}/0/labels/4i_R{k}_nuclear_seg/0", mode="r", storage_options=SO)


def crop(arr, sl_y, sl_x, pad, lead):
    x = arr[lead + (sl_y, sl_x)]
    return np.pad(x, pad) if any(p != (0, 0) for p in pad) else x


def load(i):
    r = obs.iloc[i]; w = r.well.split("/")[1]; A = ARR[w]
    yc, xc, sid = int(round(r.y_cp)), int(round(r.x_cp)), int(r.segmentation_id)
    Y, X = A["cell"].shape[-2:]
    y0, y1, x0, x1 = yc - H, yc + H, xc - H, xc + H
    pad = ((max(0, -y0), max(0, y1 - Y)), (max(0, -x0), max(0, x1 - X)))
    sy, sx = slice(max(0, y0), min(Y, y1)), slice(max(0, x0), min(X, x1))
    cell = crop(A["cell"], sy, sx, pad, (0, 0, 0))
    cm = cell == sid
    img = np.stack([crop(A["img"], sy, sx, pad, (0, DAPI_CH[k], 0)) for k in ROUNDS]).astype(np.float32)
    masks = np.zeros((len(ROUNDS), 2 * H, 2 * H), np.uint8)
    qc = {"cell_key": r.cell_key, "cell_px_in_crop": int(cm.sum()),
          "cell_touches_crop_border": bool(cm[0].any() or cm[-1].any() or cm[:, 0].any() or cm[:, -1].any())}
    for j, k in enumerate(ROUNDS):
        nuc = crop(A[k], sy, sx, pad, (0, 0, 0))
        ids, cnt_in = np.unique(nuc[cm], return_counts=True)
        keep = ids != 0; ids, cnt_in = ids[keep], cnt_in[keep]
        tot = np.array([(nuc == i_).sum() for i_ in ids]) if len(ids) else np.array([])
        frac = cnt_in / np.maximum(tot, 1)
        inside = ids[frac >= 0.5]
        qc[f"R{k}_n_nuclei_in_cell"] = int(len(inside))
        bgpix = img[j][(nuc == 0) & (cell == 0)]
        qc[f"R{k}_bg_median"] = float(np.median(bgpix)) if bgpix.size > 100 else np.nan
        if len(ids) and frac[np.argmax(cnt_in)] >= 0.5:
            best = ids[np.argmax(cnt_in)]
            m = nuc == best
            masks[j][m] = 1
            qc[f"R{k}_nucleus_id"] = int(best); qc[f"R{k}_nucleus_frac_in_cell"] = float(frac[np.argmax(cnt_in)])
            qc[f"R{k}_nucleus_touches_crop_border"] = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
            qc[f"R{k}_nucleus_saturated_frac"] = float((img[j][m] >= SAT).mean())
        else:
            qc[f"R{k}_nucleus_id"] = -1; qc[f"R{k}_nucleus_frac_in_cell"] = np.nan
            qc[f"R{k}_nucleus_touches_crop_border"] = False
            qc[f"R{k}_nucleus_saturated_frac"] = np.nan
    # cp_measure (like CellProfiler) wants intensities in [0, 1]: one fixed 16-bit scale for every cell, never per crop
    return np.clip(img / SCALE, 0, 1), masks, qc


CFG = {k: make_featurizer_config([f"DAPI_R{k}"], objects=[f"nucleus_R{k}"]) for k in ROUNDS}
pool = ThreadPoolExecutor(a.threads)
t_all = time.time()
for start in range(0, len(obs), a.batch):
    fo, qo = os.path.join(a.out, f"feat_{start:06d}.parquet"), os.path.join(a.out, f"qc_{start:06d}.parquet")
    if os.path.exists(fo) and os.path.exists(qo):
        continue
    t = time.time()
    idx = list(range(start, min(start + a.batch, len(obs))))
    res = list(pool.map(load, idx))
    t_read = time.time() - t
    keys = [q["cell_key"] for _, _, q in res]
    imgs = np.stack([r[0] for r in res]); msks = np.stack([r[1] for r in res])
    per_round = []
    for j, k in enumerate(ROUNDS):
        data, cols, rows = featurize(imgs[:, [j]], msks[:, [j]].astype(np.int32), CFG[k], image_ids=keys)
        cols = [c if f"R{k}" in c else f"nucleus_R{k}_{c}" for c in cols]   # shape features carry no round tag
        df = pd.DataFrame(data, columns=cols, index=[r_[0] for r_ in rows])
        df = df[~df.index.duplicated()]                       # one object per crop by construction
        per_round.append(df.reindex(keys))
    feat = pd.concat(per_round, axis=1); feat.index.name = "cell_key"
    feat.astype(np.float32).to_parquet(fo)
    pd.DataFrame([q for _, _, q in res]).to_parquet(qo)
    done = min(start + a.batch, len(obs))
    print(f"{done}/{len(obs)} cells | batch read {t_read:.0f}s featurize {time.time() - t - t_read:.0f}s | "
          f"elapsed {(time.time() - t_all) / 60:.1f} min | {feat.shape[1]} features", flush=True)
print("DONE", flush=True)
