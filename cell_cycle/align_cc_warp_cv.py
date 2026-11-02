#!/usr/bin/env python
"""Can the OPS0077 4i clock be mapped onto the Element RNA clock at all?  Leave-one-anchor-out test.

A monotone circular piecewise-linear warp (K knots; K=1 is a rigid rotation) maps OPS0077 angle -> Element angle. It is
fitted on a subset of the three scale-free anchors (DNA content, nuclear area, cell area; each log2 relative to the well
median) and scored on an anchor it never saw. If a warp does not generalise to the held-out anchor, the two clocks are
not related by a monotone re-timing on these anchors, and no per-cell alignment should be trusted.

usage: python align_cc_warp_cv.py [OUT_DIR]
"""
import os, sys, itertools
import numpy as np, pandas as pd, h5py
from scipy.optimize import differential_evolution
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "cc_align"
os.makedirs(OUT, exist_ok=True)
NB, STEP = 72, 5.0
CEN = (np.arange(NB) + .5) * STEP
FEATS = ["dna", "nuc_area", "cell_area"]


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


h = h5py.File("/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu", "r")
obs = read_elem(h["obs"]); names = list(read_elem(h["mod"]["imaging"]["var"]).index)
raw = h["mod"]["imaging"]["layers"]["raw"][:]
ci = {n: names.index(n) for n in ["Nucleus_Nuclear", "Bkg_Nucleus_Nuclear", "NuclearArea", "Area"]}
dna = np.clip((raw[:, ci["Nucleus_Nuclear"]] - raw[:, ci["Bkg_Nucleus_Nuclear"]]) * raw[:, ci["NuclearArea"]], 1, None)
E = pd.DataFrame({"angle": obs["cycle_position_deg"].astype(float).values, "well": obs["Well"].astype(str).values,
                  "cond": obs["condition"].astype(str).values, "gene": obs["target_gene"].astype(str).values,
                  "dna": np.log2(dna), "nuc_area": np.log2(np.clip(raw[:, ci["NuclearArea"]], 1, None)),
                  "cell_area": np.log2(np.clip(raw[:, ci["Area"]], 1, None))}).replace([np.inf, -np.inf], np.nan).dropna()
del raw
for f in FEATS: E[f] -= E.groupby("well")[f].transform("median")
ctrl = E[E.gene.isin(["NTC", "OR1I1"]) & (E.cond == "baseline")]

L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key"); fr = pd.read_parquet(find("OPS0077_NTC_cellcycle_frame.parquet"))
Q = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")].copy()
Q["angle"] = Q["cc_phase"].astype(float) * 360; Q["dna"] = Q["dna_rel_R234"].astype(float)
Q["nuc_area"] = np.log2(Q["nuc_area_px"].astype(float).clip(lower=1))
Q["cell_area"] = np.log2(fr["cell_area_4i"].reindex(Q.index).astype(float).clip(lower=1))
Q = Q.replace([np.inf, -np.inf], np.nan).dropna(subset=["angle"] + FEATS)
for f in FEATS: Q[f] -= Q.groupby("well")[f].transform("median")


def profile(angle, v, min_n=20):
    b = np.floor((np.asarray(angle) % 360) / STEP).astype(int) % NB
    n = np.bincount(b, minlength=NB); p = np.where(n >= min_n, np.bincount(b, weights=v, minlength=NB) / np.maximum(n, 1), np.nan)
    out = np.full(NB, np.nan)
    for i in range(NB):
        w = p[[(i - 1) % NB, i, (i + 1) % NB]]
        if np.isfinite(w).any(): out[i] = np.nanmean(w)
    idx = np.arange(NB); ok = np.isfinite(out)
    out[~ok] = np.interp(idx[~ok], idx[ok], out[ok], period=NB)
    return out


Pr = {f: profile(ctrl["angle"].values, ctrl[f].values) for f in FEATS}
Pq = {f: profile(Q["angle"].values, Q[f].values) for f in FEATS}


def make_warp(params, K):
    phi0 = params[0]; w = np.asarray(params[1:1 + K]); d = 360 * w / w.sum()
    th = 360 * np.arange(K + 1) / K; ph = phi0 + np.r_[0, np.cumsum(d)]
    return th, ph


def warped_query_at_reference(P, params, K, s):
    th, ph = make_warp(params, K)                       # query angle th_k -> element angle ph_k (increasing, total 360)
    q = P if s == 1 else P[::-1]
    # Element angle c -> query angle t:  invert the piecewise-linear map
    c = (CEN - ph[0]) % 360 + ph[0]
    t = np.interp(c, ph, th)
    return np.interp(t, np.r_[CEN - 360, CEN, CEN + 360], np.r_[q, q, q])


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 8 else -1.0


def objective(params, K, s, anchors, lam=0.15):
    w = np.asarray(params[1:1 + K]); d = w / w.sum() * K
    pen = lam * np.mean(np.log(d) ** 2)
    return -np.mean([corr(warped_query_at_reference(Pq[f], params, K, s), Pr[f]) for f in anchors]) + pen


def fit(K, anchors, seed=0):
    best = None
    for s in (1, -1):
        bounds = [(0, 360)] + [(0.15, 1.0)] * K
        r = differential_evolution(objective, bounds, args=(K, s, anchors), seed=seed, maxiter=80, popsize=14, tol=1e-6, polish=False)
        if best is None or r.fun < best[0]: best = (r.fun, r.x, s)
    return best


def score(best, K, anchors):
    _, x, s = best
    return {f: corr(warped_query_at_reference(Pq[f], x, K, s), Pr[f]) for f in anchors}


rows = []
for K in (1, 4, 6, 8):
    for held in FEATS:
        train = [f for f in FEATS if f != held]
        b = fit(K, train)
        tr = np.mean(list(score(b, K, train).values())); te = score(b, K, [held])[held]
        nulls = np.array([corr(warped_query_at_reference(Pq[held], b[1], K, b[2]), np.roll(Pr[held], sh)) for sh in range(NB)])
        rows.append(dict(knots=K, warp="rigid rotation" if K == 1 else f"warp, {K} knots", train_on="+".join(train), held_out=held,
                         direction=b[2], train_corr=round(tr, 3), HELD_OUT_corr=round(te, 3),
                         null_95th=round(float(np.percentile(nulls, 95)), 3), p_vs_random_shift=round(float(np.mean(nulls >= te)), 3)))
        print(rows[-1], flush=True)
    # DNA-only fit, tested on both areas (the question we started with)
    b = fit(K, ["dna"]); sc = score(b, K, ["nuc_area", "cell_area"])
    nulls = np.array([np.mean([corr(warped_query_at_reference(Pq[f], b[1], K, b[2]), np.roll(Pr[f], sh)) for f in ["nuc_area", "cell_area"]]) for sh in range(NB)])
    te2 = float(np.mean(list(sc.values())))
    rows.append(dict(knots=K, warp="rigid rotation" if K == 1 else f"warp, {K} knots", train_on="dna", held_out="nuc_area+cell_area",
                     direction=b[2], train_corr=round(score(b, K, ["dna"])["dna"], 3), HELD_OUT_corr=round(te2, 3),
                     null_95th=round(float(np.percentile(nulls, 95)), 3), p_vs_random_shift=round(float(np.mean(nulls >= te2)), 3)))
    print(rows[-1], flush=True)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/cc_warp_leave_one_anchor_out.csv", index=False)

# how well identified is the DNA-only rotation? resample cells in both datasets
rng = np.random.default_rng(0); bs = []
for _ in range(100):
    e = ctrl.iloc[rng.integers(0, len(ctrl), len(ctrl))]; q = Q.iloc[rng.integers(0, len(Q), len(Q))]
    pe, pq = profile(e["angle"].values, e["dna"].values), profile(q["angle"].values, q["dna"].values)
    c, s_, k_ = max((corr(np.roll(pq, k) if s == 1 else np.roll(pq[::-1], k), pe), s, k) for s in (1, -1) for k in range(NB))
    bs.append((s_, k_ * STEP))
bs = pd.DataFrame(bs, columns=["s", "off"]); modal = int(bs.s.mode()[0]); bo = bs[bs.s == modal].off.values
d_ = (bo - np.median(bo) + 180) % 360 - 180
print(f"\nDNA-only rigid alignment, bootstrap over cells: direction {modal:+d} in {100 * (bs.s == modal).mean():.0f}% of resamples; "
      f"offset median {np.median(bo):.0f} deg, 90% interval [{np.median(bo) + np.percentile(d_, 5):.0f}, {np.median(bo) + np.percentile(d_, 95):.0f}]")

piv = res[res.held_out.isin(FEATS)].pivot_table(index="warp", columns="held_out", values="HELD_OUT_corr", sort=False)
print("\nheld-out correlation (fit on the other two anchors):\n", piv.round(3).to_string())
print("\nversus random circular shifts of the held-out Element profile (null_95th = 95th percentile; p = fraction of shifts as good):")
print(res[["warp", "train_on", "held_out", "HELD_OUT_corr", "null_95th", "p_vs_random_shift"]].to_string(index=False))
fig, ax = plt.subplots(figsize=(7.5, 4.2)); piv.plot.bar(ax=ax); ax.axhline(0, color="k", lw=.6)

ax.set_ylabel("correlation with Element on the HELD-OUT anchor"); ax.set_xlabel(""); plt.xticks(rotation=20, ha="right")
ax.set_title("does the alignment generalise to an anchor it never saw?"); plt.tight_layout(); plt.savefig(f"{OUT}/cc_align_5_leave_one_out.png", dpi=110)
print("DONE")
