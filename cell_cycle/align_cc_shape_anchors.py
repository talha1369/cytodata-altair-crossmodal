#!/usr/bin/env python
"""Does adding shared CELL-SHAPE anchors make the OPS0077 -> Element cell-cycle alignment identifiable?

Element `AreaShape_*` features are cell-object CellProfiler features (EquivalentDiameter == sqrt(4*Area/pi), Solidity ==
Area/ConvexArea exactly), and OPS0077 has the same CellProfiler features as `cp_cell_*`. Pixel sizes differ (cell area
median 15,594 px vs 3,656 px), so only scale-free quantities are used (ratios and bounded shape descriptors), each z-scored
within well.

Anchor FAMILIES (features in one family are correlated, so held-out testing is leave-one-FAMILY-out, never leave-one-feature-out):
  dna    DNA content (log2 rel. to well median)
  area   nuclear area, cell area (log2 rel.)
  shape  cell eccentricity, solidity, extent, form factor, compactness, major/minor axis ratio, max/mean radius ratio

For every split a monotone circular warp (1 knot = rigid rotation, or 4/6 knots) is fit on the training families and scored on the
held-out family against a null of random circular shifts of the held-out Element profiles (p = fraction of shifts as good).
usage: python align_cc_shape_anchors.py [OUT_DIR]
"""
import os, sys, itertools
import numpy as np, pandas as pd, h5py, anndata as ad
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
SHAPE = ["eccentricity", "solidity", "extent", "formfactor", "compactness", "axis_ratio", "radius_ratio"]
FAMILY = {"dna": ["dna"], "area": ["nuc_area", "cell_area"], "shape": SHAPE}
ALL = [f for fs in FAMILY.values() for f in fs]


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


# ---------------- Element ----------------
h = h5py.File("/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu", "r")
obs = read_elem(h["obs"]); names = list(read_elem(h["mod"]["imaging"]["var"]).index)
raw = h["mod"]["imaging"]["layers"]["raw"][:]
g = lambda n: raw[:, names.index(n)].astype(float)
dna = np.clip((g("Nucleus_Nuclear") - g("Bkg_Nucleus_Nuclear")) * g("NuclearArea"), 1, None)
E = pd.DataFrame({"angle": obs["cycle_position_deg"].astype(float).values, "well": obs["Well"].astype(str).values,
                  "cond": obs["condition"].astype(str).values, "gene": obs["target_gene"].astype(str).values,
                  "dna": np.log2(dna), "nuc_area": np.log2(np.clip(g("NuclearArea"), 1, None)), "cell_area": np.log2(np.clip(g("Area"), 1, None)),
                  "eccentricity": g("AreaShape_Eccentricity"), "solidity": g("AreaShape_Solidity"), "extent": g("AreaShape_Extent"),
                  "formfactor": g("AreaShape_FormFactor"), "compactness": g("AreaShape_Compactness"),
                  "axis_ratio": np.log2(g("AreaShape_MajorAxisLength") / g("AreaShape_MinorAxisLength")),
                  "radius_ratio": np.log2(g("AreaShape_MaximumRadius") / g("AreaShape_MeanRadius"))})
del raw
E = E.replace([np.inf, -np.inf], np.nan).dropna()
for f in ALL: E[f] = E[f] - E.groupby("well")[f].transform("median")
ctrl = E[E.gene.isin(["NTC", "OR1I1"]) & (E.cond == "baseline")]

# ---------------- OPS0077 ----------------
L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
a = ad.read_h5ad(find("OPS0077_NTC_OrganelleProfiler_pRb_Rb_p21.h5ad"), backed="r")
v = list(a.var_names); need = ["Eccentricity", "Solidity", "Extent", "FormFactor", "Compactness", "MajorAxisLength", "MinorAxisLength",
                                "MaximumRadius", "MeanRadius", "Area"]
M = pd.DataFrame(np.asarray(a[:, [f"cp_cell_{n}" for n in need]].to_memory().X), columns=need, index=a.obs["cell_key"].astype(str).values)
M = M[~M.index.duplicated()]
Q = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")].copy()
M = M.reindex(Q.index)
Q["angle"] = Q["cc_phase"].astype(float) * 360; Q["dna"] = Q["dna_rel_R234"].astype(float)
Q["nuc_area"] = np.log2(Q["nuc_area_px"].astype(float).clip(lower=1)); Q["cell_area"] = np.log2(M["Area"].astype(float).clip(lower=1))
Q["eccentricity"], Q["solidity"], Q["extent"] = M["Eccentricity"], M["Solidity"], M["Extent"]
Q["formfactor"], Q["compactness"] = M["FormFactor"], M["Compactness"]
Q["axis_ratio"] = np.log2(M["MajorAxisLength"] / M["MinorAxisLength"]); Q["radius_ratio"] = np.log2(M["MaximumRadius"] / M["MeanRadius"])
Q = Q.replace([np.inf, -np.inf], np.nan).dropna(subset=["angle"] + ALL)
for f in ALL: Q[f] = Q[f] - Q.groupby("well")[f].transform("median")
print(f"Element controls {len(ctrl):,} | OPS0077 {len(Q):,}")


def profile(angle, v, min_n=20):
    b = np.floor((np.asarray(angle) % 360) / STEP).astype(int) % NB
    n = np.bincount(b, minlength=NB); p = np.where(n >= min_n, np.bincount(b, weights=v, minlength=NB) / np.maximum(n, 1), np.nan)
    out = np.full(NB, np.nan)
    for i in range(NB):
        w = p[[(i - 1) % NB, i, (i + 1) % NB]]
        if np.isfinite(w).any(): out[i] = np.nanmean(w)
    idx = np.arange(NB); ok = np.isfinite(out); out[~ok] = np.interp(idx[~ok], idx[ok], out[ok], period=NB)
    return out


Pr = {f: profile(ctrl["angle"].values, ctrl[f].values) for f in ALL}
Pq = {f: profile(Q["angle"].values, Q[f].values) for f in ALL}


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 8 else -1.0


def make_warp(params, K):
    d = 360 * np.asarray(params[1:1 + K]) / np.sum(params[1:1 + K])
    return 360 * np.arange(K + 1) / K, params[0] + np.r_[0, np.cumsum(d)]


def warped(P, params, K, s):
    th, ph = make_warp(params, K); q = P if s == 1 else P[::-1]
    c = (CEN - ph[0]) % 360 + ph[0]; t = np.interp(c, ph, th)
    return np.interp(t, np.r_[CEN - 360, CEN, CEN + 360], np.r_[q, q, q])


def obj(params, K, s, feats, lam=0.15):
    w = np.asarray(params[1:1 + K]); d = w / w.sum() * K
    return -np.mean([corr(warped(Pq[f], params, K, s), Pr[f]) for f in feats]) + lam * np.mean(np.log(d) ** 2)


def fit(K, feats):
    best = None
    for s in (1, -1):
        r = differential_evolution(obj, [(0, 360)] + [(0.15, 1.0)] * K, args=(K, s, feats), seed=0, maxiter=70, popsize=12, tol=1e-6, polish=False)
        if best is None or r.fun < best[0]: best = (r.fun, r.x, s)
    return best


# per-anchor quality: how much does each profile vary around the clock in each dataset? (flat profiles carry no alignment information)
amp = pd.DataFrame({"Element range": {f: float(np.ptp(Pr[f])) for f in ALL}, "OPS0077 range": {f: float(np.ptp(Pq[f])) for f in ALL}})
amp["Element range / SD of cells"] = [float(np.ptp(Pr[f]) / ctrl[f].std()) for f in ALL]
amp["OPS0077 range / SD of cells"] = [float(np.ptp(Pq[f]) / Q[f].std()) for f in ALL]
print("\nhow much each anchor moves around the clock (profile range, and range in units of the cell-to-cell SD):")
print(amp.round(3).to_string()); amp.round(3).to_csv(f"{OUT}/cc_shape_anchor_amplitude.csv")

rows = []
fams = list(FAMILY)
for K in (1, 4, 6):
    for held in fams:
        train = [f for fam in fams if fam != held for f in FAMILY[fam]]
        b = fit(K, train); _, x, s = b
        tr = np.mean([corr(warped(Pq[f], x, K, s), Pr[f]) for f in train])
        te = np.mean([corr(warped(Pq[f], x, K, s), Pr[f]) for f in FAMILY[held]])
        null = np.array([np.mean([corr(warped(Pq[f], x, K, s), np.roll(Pr[f], sh)) for f in FAMILY[held]]) for sh in range(NB)])
        rows.append(dict(warp="rigid rotation" if K == 1 else f"warp, {K} knots", train_on="+".join(f for f in fams if f != held), held_out=held,
                         direction=s, train_corr=round(float(tr), 3), HELD_OUT_corr=round(float(te), 3),
                         null_95th=round(float(np.percentile(null, 95)), 3), p_vs_random_shift=round(float(np.mean(null >= te)), 3)))
        print(rows[-1], flush=True)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/cc_shape_anchor_leave_family_out.csv", index=False)

# shape family alone: is it identified, and does it agree with DNA?
print("\nrigid rotation fitted on each FAMILY alone (direction, offset, correlation on the anchors it was fit on):")
fam_rows = []
for fam in fams:
    best_ = max(((np.mean([corr(np.roll(Pq[f] if s == 1 else Pq[f][::-1], k), Pr[f]) for f in FAMILY[fam]]), s, k) for s in (1, -1) for k in range(NB)))
    fam_rows.append(dict(anchor_family=fam, direction=best_[1], offset_deg=best_[2] * STEP, profile_corr=round(float(best_[0]), 3)))
    print(f"  {fam:6s}: direction {best_[1]:+d}, offset {best_[2] * STEP:5.0f} deg, correlation {best_[0]:.3f}")
pd.DataFrame(fam_rows).to_csv(f"{OUT}/cc_shape_anchor_family_rigid.csv", index=False)

# figure: profiles of the shape anchors in each dataset under the DNA-only rigid rotation
sh_best = max(((corr(np.roll(Pq["dna"] if s == 1 else Pq["dna"][::-1], k), Pr["dna"]), s, k) for s in (1, -1) for k in range(NB)))
_, s_d, k_d = sh_best
fig, ax = plt.subplots(2, 4, figsize=(18, 7))
for a_, f in zip(ax.ravel(), SHAPE + ["cell_area"]):
    a_.plot(CEN, (Pr[f] - Pr[f].mean()) / Pr[f].std(), "k", lw=2.2, label="Element (RNA clock)")
    q = np.roll(Pq[f] if s_d == 1 else Pq[f][::-1], k_d); a_.plot(CEN, (q - q.mean()) / q.std(), color="tab:red", lw=1.8, label=f"OPS0077, DNA-only rotation ({k_d * STEP:.0f} deg)")
    a_.set_title(f); a_.set_xlabel("Element clock angle (deg)")
ax[0, 0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/cc_align_8_shape_anchors.png", dpi=105)
print("DONE")
