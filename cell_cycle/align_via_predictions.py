#!/usr/bin/env python
"""Aligning cell-cycle clocks THROUGH predictions.

Part A (within OPS0077, same cells): the out-of-fold phase-contrast predictions of the 4i DNA and 4i markers regimes.
  Do the predictions line up once the MEASURED relation between the regimes is applied, and does combining them help?
Part B (across datasets): train a model on Element (RNA clock <- features that BOTH datasets measure), apply it to OPS0077 cells,
  and compare the predicted Element-clock angle with Louise's 4i clock ON THE SAME CELLS. Shared predictors are scale-free
  (DNA content, nuclear area, cell area, 7 cell-shape descriptors), robust z-scored within well in each dataset.
  Predictor sets WITHOUT DNA are the fair test, because Louise's clock is built partly from DNA.
usage: python align_via_predictions.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, h5py, anndata as ad
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "cc_align"
os.makedirs(OUT, exist_ok=True)
SHAPE = ["eccentricity", "solidity", "extent", "formfactor", "compactness", "axis_ratio", "radius_ratio"]
SETS = {"DNA + area + shape": ["dna", "nuc_area", "cell_area"] + SHAPE, "area + shape (no DNA)": ["nuc_area", "cell_area"] + SHAPE,
        "shape only (no DNA, no area)": SHAPE, "DNA only": ["dna"]}
ALL = ["dna", "nuc_area", "cell_area"] + SHAPE


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", OUT, "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


def cd(x, y): return (x - y + .5) % 1 - .5


def align_stats(x, y):
    """best rotation (and optional reflection) of angle x (fractions) onto y, chosen by the rotation-SENSITIVE mean cos"""
    sh = np.linspace(0, 1, 180, endpoint=False)
    sc, f, o = max(((np.mean(np.cos(2 * np.pi * (f_ * x + s_ - y))), f_, s_) for f_ in (1, -1) for s_ in sh))
    e = np.abs(cd(f * x + o, y)) * 360
    return dict(mean_cos=round(float(sc), 3), direction=int(f), rotation_deg=round(o * 360 % 360), median_err_deg=round(float(np.median(e)), 1),
                within_45deg=round(float((e < 45).mean()), 3))


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")

# ======================= Part A: within OPS0077 =======================
pa = pd.read_csv(find("phase_predicted_angles.csv"), index_col=0)
Lm = L.loc[pa.index]
meas = {"4i DNA": Lm["cc_phase_dna"].values, "4i markers": Lm["cc_phase_markers"].values, "4i DNA+PCs (final)": Lm["cc_phase"].values}
pred = {k: pa[k].values for k in pa.columns}
rowsA = []
m = np.isfinite(meas["4i markers"]) & np.isfinite(meas["4i DNA"])
rel = align_stats(meas["4i markers"][m], meas["4i DNA"][m])                                  # MEASURED markers -> DNA relation
print("measured relation, markers -> DNA clock:", rel)
dir_, shift = rel["direction"], rel["rotation_deg"] / 360
pm_al = (dir_ * pred["4i markers"] + shift) % 1                                              # predicted markers angle mapped into the DNA frame
canon = meas["4i DNA+PCs (final)"]
for name, a_, b_ in [("predicted DNA vs measured DNA", pred["4i DNA"], meas["4i DNA"]),
                     ("predicted markers (mapped by the measured relation) vs measured DNA", pm_al, meas["4i DNA"]),
                     ("predicted markers (mapped by the measured relation) vs predicted DNA", pm_al, pred["4i DNA"]),
                     ("predicted markers vs predicted DNA (best rotation of the predictions themselves)", pred["4i markers"], pred["4i DNA"])]:
    ok = np.isfinite(a_) & np.isfinite(b_)
    if "mapped" in name:   # no further alignment: the mapping is fixed by the MEASURED relation
        e = np.abs(cd(a_[ok], b_[ok])) * 360; r = dict(mean_cos=round(float(np.mean(np.cos(2 * np.pi * cd(a_[ok], b_[ok])))), 3), direction=dir_,
                                                        rotation_deg=rel["rotation_deg"], median_err_deg=round(float(np.median(e)), 1), within_45deg=round(float((e < 45).mean()), 3))
    else:
        r = align_stats(a_[ok], b_[ok])
    rowsA.append(dict(comparison=name, **r)); print(rowsA[-1])
# does combining the two predictions help?  vs the canonical (final) measured angle
ens = np.angle(0.5 * np.exp(2j * np.pi * pred["4i DNA+PCs (final)"]) + 0.5 * np.exp(2j * np.pi * pm_al)) / (2 * np.pi) % 1
for name, a_ in [("final-trained prediction alone", pred["4i DNA+PCs (final)"]), ("markers-trained prediction mapped to the DNA frame", pm_al),
                 ("circular average of the two", ens)]:
    ok = np.isfinite(a_) & np.isfinite(canon); e = np.abs(cd(a_[ok], canon[ok])) * 360
    arcs = [float(np.median(e[np.floor((canon[ok] * 360) / 30).astype(int) == k])) for k in range(12)]
    rowsA.append(dict(comparison=f"vs measured final cc_phase: {name}", mean_cos=round(float(np.mean(np.cos(2 * np.pi * cd(a_[ok], canon[ok])))), 3),
                      direction=1, rotation_deg=0, median_err_deg=round(float(np.median(e)), 1), within_45deg=round(float((e < 45).mean()), 3),
                      worst_arc_median_err_deg=round(max(arcs), 1)))
    print(rowsA[-1])
pd.DataFrame(rowsA).to_csv(f"{OUT}/cc_predictions_within_ops0077.csv", index=False)

# ======================= Part B: across datasets =======================
hh = h5py.File("/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu", "r")
obs = read_elem(hh["obs"]); names = list(read_elem(hh["mod"]["imaging"]["var"]).index)
raw = hh["mod"]["imaging"]["layers"]["raw"][:]
g = lambda n: raw[:, names.index(n)].astype(float)
dna = np.clip((g("Nucleus_Nuclear") - g("Bkg_Nucleus_Nuclear")) * g("NuclearArea"), 1, None)
E = pd.DataFrame({"angle": obs["cycle_position_deg"].astype(float).values, "well": obs["Well"].astype(str).values,
                  "dna": np.log2(dna), "nuc_area": np.log2(np.clip(g("NuclearArea"), 1, None)), "cell_area": np.log2(np.clip(g("Area"), 1, None)),
                  "eccentricity": g("AreaShape_Eccentricity"), "solidity": g("AreaShape_Solidity"), "extent": g("AreaShape_Extent"),
                  "formfactor": g("AreaShape_FormFactor"), "compactness": g("AreaShape_Compactness"),
                  "axis_ratio": np.log2(g("AreaShape_MajorAxisLength") / g("AreaShape_MinorAxisLength")),
                  "radius_ratio": np.log2(g("AreaShape_MaximumRadius") / g("AreaShape_MeanRadius"))})
del raw
E = E.replace([np.inf, -np.inf], np.nan).dropna()

a = ad.read_h5ad(find("OPS0077_NTC_OrganelleProfiler_pRb_Rb_p21.h5ad"), backed="r")
need = ["Eccentricity", "Solidity", "Extent", "FormFactor", "Compactness", "MajorAxisLength", "MinorAxisLength", "MaximumRadius", "MeanRadius", "Area"]
M = pd.DataFrame(np.asarray(a[:, [f"cp_cell_{n}" for n in need]].to_memory().X), columns=need, index=a.obs["cell_key"].astype(str).values)
M = M[~M.index.duplicated()]
Q = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")].copy(); M = M.reindex(Q.index)
Q["well"] = Q["well"].astype(str); Q["angle_L"] = Q["cc_phase"].astype(float)
Q["dna"] = Q["dna_rel_R234"].astype(float); Q["nuc_area"] = np.log2(Q["nuc_area_px"].astype(float).clip(lower=1)); Q["cell_area"] = np.log2(M["Area"].astype(float).clip(lower=1))
Q["eccentricity"], Q["solidity"], Q["extent"], Q["formfactor"], Q["compactness"] = M["Eccentricity"], M["Solidity"], M["Extent"], M["FormFactor"], M["Compactness"]
Q["axis_ratio"] = np.log2(M["MajorAxisLength"] / M["MinorAxisLength"]); Q["radius_ratio"] = np.log2(M["MaximumRadius"] / M["MeanRadius"])
Q = Q.replace([np.inf, -np.inf], np.nan).dropna(subset=["angle_L"] + ALL)


def robust_z(df):
    out = df.copy()
    for f in ALL:
        grp = out.groupby("well")[f]; med = grp.transform("median"); iqr = (grp.transform(lambda s: s.quantile(.75) - s.quantile(.25)) / 1.349).replace(0, np.nan)
        out[f] = ((out[f] - med) / iqr).clip(-5, 5)
    return out


E, Q = robust_z(E).dropna(subset=ALL), robust_z(Q).dropna(subset=ALL)
ang = np.deg2rad(E["angle"].values); sector = np.floor(E["angle"].values / 30).astype(int)
w = 1.0 / pd.Series(sector).map(pd.Series(sector).value_counts()).values; w = w / w.mean()          # every 30-degree sector counts equally
print(f"\nElement {len(E):,} cells | OPS0077 {len(Q):,} cells | train weights balanced over 12 sectors")


def fit_predict(Xtr, ytr_s, ytr_c, wtr, Xte):
    kw = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=60, random_state=0)
    ps = HistGradientBoostingRegressor(**kw).fit(Xtr, ytr_s, sample_weight=wtr).predict(Xte)
    pc = HistGradientBoostingRegressor(**kw).fit(Xtr, ytr_c, sample_weight=wtr).predict(Xte)
    return np.arctan2(ps, pc)


def balanced_err(true_deg, pred_deg):
    e = np.abs((pred_deg - true_deg + 180) % 360 - 180); sec = np.floor(true_deg / 30).astype(int)
    return float(np.median(e)), float(np.mean([np.median(e[sec == k]) for k in range(12) if (sec == k).sum() > 20]))


Pr = None
rowsB = []; keep = {}
for sname, feats in SETS.items():
    X = E[feats].values; Xq = Q[feats].values; wells = E["well"].values
    oof = np.zeros(len(E))
    for tr, te in GroupKFold(4).split(X, groups=wells):
        oof[te] = np.degrees(fit_predict(X[tr], np.sin(ang[tr]), np.cos(ang[tr]), w[tr], X[te])) % 360
    med, bal = balanced_err(E["angle"].values, oof)
    th_q = (np.degrees(fit_predict(X, np.sin(ang), np.cos(ang), w, Xq)) % 360) / 360          # predicted Element-clock angle (fraction) for OPS0077 cells
    tr_ = align_stats(th_q, Q["angle_L"].values)
    # independent check where DNA is NOT a predictor: does the OPS0077 DNA profile vs predicted angle reproduce Element's?
    dprof = np.nan
    if "dna" not in feats:
        nb = 36; be = np.floor(E["angle"].values / 360 * nb).astype(int) % nb; bq = np.floor(th_q * nb).astype(int) % nb
        pe = pd.Series(E["dna"].values).groupby(be).mean().reindex(range(nb)).values; pq = pd.Series(Q["dna"].values).groupby(bq).mean().reindex(range(nb)).values
        okk = np.isfinite(pe) & np.isfinite(pq); dprof = float(np.corrcoef(pe[okk], pq[okk])[0, 1])
    rowsB.append(dict(predictors=sname, element_heldout_median_err_deg=round(med, 1), element_heldout_balanced_err_deg=round(bal, 1),
                      **{f"transfer_{k}": v for k, v in tr_.items()}, dna_profile_corr_no_dna_predictor=round(dprof, 3) if np.isfinite(dprof) else ""))
    keep[sname] = (oof, th_q); print(rowsB[-1], flush=True)
tab = pd.DataFrame(rowsB); tab.to_csv(f"{OUT}/cc_predictions_across_datasets.csv", index=False)

# ======================= figure =======================
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
for r_, sname in enumerate(["area + shape (no DNA)", "DNA + area + shape"]):
    oof, th_q = keep[sname]; row = tab[tab.predictors == sname].iloc[0]
    ax[r_, 0].hexbin(E["angle"].values, oof, gridsize=60, bins="log", extent=(0, 360, 0, 360)); ax[r_, 0].plot([0, 360], [0, 360], "w--", lw=.8)
    ax[r_, 0].set_xlabel("Element RNA clock (deg)"); ax[r_, 0].set_ylabel("predicted from shared features (held-out wells)")
    ax[r_, 0].set_title(f"{sname}: Element held-out, median error {row.element_heldout_median_err_deg:.0f} deg")
    sh = row.transfer_rotation_deg / 360; d_ = row.transfer_direction; al = (d_ * th_q + sh) % 1
    ax[r_, 1].hexbin(Q["angle_L"].values * 360, al * 360, gridsize=60, bins="log", extent=(0, 360, 0, 360)); ax[r_, 1].plot([0, 360], [0, 360], "w--", lw=.8)
    ax[r_, 1].set_xlabel("Louise's 4i clock (deg)"); ax[r_, 1].set_ylabel("Element-clock prediction, rotated onto it")
    ax[r_, 1].set_title(f"OPS0077 cells: direction {d_:+d}, rotation {row.transfer_rotation_deg:.0f} deg, median error {row.transfer_median_err_deg:.0f} deg")
    nb = 36
    be = np.floor(E["angle"].values / 360 * nb).astype(int) % nb; bq = np.floor(th_q * nb).astype(int) % nb
    ax[r_, 2].plot((np.arange(nb) + .5) * 10, pd.Series(E["dna"].values).groupby(be).mean().reindex(range(nb)).values, "k-", lw=2.2, label="Element: DNA vs RNA clock")
    ax[r_, 2].plot((np.arange(nb) + .5) * 10, pd.Series(Q["dna"].values).groupby(bq).mean().reindex(range(nb)).values, "r-", lw=1.8, label="OPS0077: DNA vs predicted Element clock")
    ax[r_, 2].set_xlabel("Element clock angle (deg)"); ax[r_, 2].set_ylabel("DNA content (robust z)"); ax[r_, 2].legend(fontsize=8)
    ax[r_, 2].set_title("DNA is " + ("NOT a predictor here (independent check)" if "no DNA" in sname else "a predictor"))
plt.tight_layout(); plt.savefig(f"{OUT}/cc_align_9_alignment_through_predictions.png", dpi=105)
print("DONE")
