#!/usr/bin/env python
"""Align cell-cycle angular scores across scoring regimes.

Regimes
  Element RNA clock   Tim's cycleVI position `cycle_position_deg` (0..360, 0 = peak of the S score, runs toward G2/M)
  OPS0077 4i clock    Louise's `cc_phase` (0..1, starts at G1) from 4i markers + DNA
  OPS0077 morphology  phase-contrast morphology -> cc_phase (held-out), on Louise's convention

The two datasets share no cells, so the alignment goes through scale-free anchors that are measured the same way in both:
  DNA content  log2 of background-subtracted integrated nuclear DAPI, relative to the well median (Tim's definition)
  nuclear area, cell area  log2 relative to the well median
The transform is FIT on DNA only and VALIDATED on nuclear and cell area, which the fit never sees.

Two transforms are compared: a rigid rotation (direction + offset) and a landmark warp (piecewise-linear through the
DNA-rise, DNA-peak and DNA-fall landmarks, which absorbs differences in how fast each clock moves through each phase).

usage: python align_cc_regimes.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, h5py, anndata as ad
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "cc_align"
os.makedirs(OUT, exist_ok=True)
H5MU = "/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu"
NB = 72                                    # 5-degree bins
STEP = 360 / NB
CEN = (np.arange(NB) + 0.5) * STEP
FEATS = ["dna", "nuc_area", "cell_area"]
LABEL = {"dna": "DNA content (log2 rel. to well median)", "nuc_area": "nuclear area (log2 rel.)", "cell_area": "cell area (log2 rel.)"}


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", "."]:
        if os.path.exists(f"{r}/{n}"):
            return f"{r}/{n}"
    raise FileNotFoundError(n)


# ============================ Element (reference regime) ============================
h = h5py.File(H5MU, "r")
obs = read_elem(h["obs"])
names = list(read_elem(h["mod"]["imaging"]["var"]).index)
raw = h["mod"]["imaging"]["layers"]["raw"][:]
ci = {n: names.index(n) for n in ["Nucleus_Nuclear", "Bkg_Nucleus_Nuclear", "NuclearArea", "Area"]}
dna = np.clip((raw[:, ci["Nucleus_Nuclear"]] - raw[:, ci["Bkg_Nucleus_Nuclear"]]) * raw[:, ci["NuclearArea"]], 1, None)
E = pd.DataFrame({"angle": obs["cycle_position_deg"].astype(float).values, "radius": obs["cycle_radius"].astype(float).values,
                  "well": obs["Well"].astype(str).values, "cond": obs["condition"].astype(str).values,
                  "gene": obs["target_gene"].astype(str).values, "dna": np.log2(dna),
                  "nuc_area": np.log2(np.clip(raw[:, ci["NuclearArea"]], 1, None)),
                  "cell_area": np.log2(np.clip(raw[:, ci["Area"]], 1, None))})
del raw
E = E.replace([np.inf, -np.inf], np.nan).dropna()
for f in FEATS:
    E[f] -= E.groupby("well")[f].transform("median")
ctrl = E[E.gene.isin(["NTC", "OR1I1"]) & (E.cond == "baseline")]
print(f"Element: {len(E):,} cells | controls at baseline {len(ctrl):,}")

# ============================ OPS0077 (query regime) ============================
L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
fr = pd.read_parquet(find("OPS0077_NTC_cellcycle_frame.parquet"))
Q = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")].copy()
Q["angle"] = Q["cc_phase"].astype(float) * 360
Q["dna"] = Q["dna_rel_R234"].astype(float)
Q["nuc_area"] = np.log2(Q["nuc_area_px"].astype(float).clip(lower=1))
Q["cell_area"] = np.log2(fr["cell_area_4i"].reindex(Q.index).astype(float).clip(lower=1))
Q = Q.replace([np.inf, -np.inf], np.nan).dropna(subset=["angle"] + FEATS)
for f in FEATS:
    Q[f] -= Q.groupby("well")[f].transform("median")
print(f"OPS0077: {len(Q):,} cells with a 4i clock")


# ============================ profiles ============================
def profile(angle, values, min_n=20):
    b = np.floor((np.asarray(angle) % 360) / STEP).astype(int) % NB
    n = np.bincount(b, minlength=NB).astype(float)
    p = np.where(n >= min_n, np.bincount(b, weights=values, minlength=NB) / np.maximum(n, 1), np.nan)
    ps = np.array([np.nanmean(p[[(i - 1) % NB, i, (i + 1) % NB]]) if np.isfinite(p[[(i - 1) % NB, i, (i + 1) % NB]]).any() else np.nan
                   for i in range(NB)])
    return ps, n


def profiles(df, angle=None):
    a = df["angle"].values if angle is None else angle
    return {f: profile(a, df[f].values)[0] for f in FEATS}


def shifted(P, s, k):
    """profile after the map phi = s*theta + k*STEP (see derivation in comments of the script header)."""
    return np.roll(P, k) if s == 1 else np.roll(P[::-1], k)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 8 else np.nan


def scan(Pq, Pr, feats):
    res = np.full((2, NB), np.nan)
    for si, s in enumerate((1, -1)):
        for k in range(NB):
            res[si, k] = np.nanmean([corr(shifted(Pq[f], s, k), Pr[f]) for f in feats])
    return res


def best(res):
    si, k = np.unravel_index(np.nanargmax(res), res.shape)
    return (1, -1)[si], int(k), float(res[si, k])


Pr = profiles(ctrl); Pq = profiles(Q)
res = {"DNA only": scan(Pq, Pr, ["dna"]), "area only (nuc + cell)": scan(Pq, Pr, ["nuc_area", "cell_area"]),
       "all three": scan(Pq, Pr, FEATS)}
print("\nrigid alignment OPS0077 -> Element (phi = s*theta + offset):")
rigid = {}
for name, r in res.items():
    s, k, c = best(r); rigid[name] = (s, k)
    print(f"  {name:24s}: direction {s:+d}, offset {k * STEP:5.0f} deg, profile correlation {c:.3f}")

# bootstrap the direction and offset (resample cells in both datasets), for each anchor set
def boot(feats, n=100):
    rng = np.random.default_rng(0); out = []
    for _ in range(n):
        e = ctrl.iloc[rng.integers(0, len(ctrl), len(ctrl))]; q = Q.iloc[rng.integers(0, len(Q), len(Q))]
        s_, k_, _ = best(scan(profiles(q), profiles(e), feats)); out.append((s_, k_ * STEP))
    bs = pd.DataFrame(out, columns=["s", "off"]); modal = int(bs.s.mode()[0]); bo = bs[bs.s == modal].off.values
    d = (bo - np.median(bo) + 180) % 360 - 180
    return modal, 100 * float((bs.s == modal).mean()), float(np.median(bo)), float(np.median(bo) + np.percentile(d, 5)), float(np.median(bo) + np.percentile(d, 95))


summary_rows = []
for name, feats in [("DNA only", ["dna"]), ("area only (nuc + cell)", ["nuc_area", "cell_area"]), ("all three", FEATS)]:
    modal, pct, med, lo, hi = boot(feats)
    s_best, k_best, c_best = best(res[name])
    summary_rows.append(dict(anchors=name, best_direction=s_best, best_offset_deg=k_best * STEP, best_profile_corr=round(c_best, 3),
                             bootstrap_direction=modal, bootstrap_pct_same_direction=round(pct), bootstrap_offset_median_deg=round(med),
                             bootstrap_offset_90pct_interval=f"[{lo:.0f}, {hi:.0f}]"))
    print(f"  bootstrap ({name}): direction {modal:+d} in {pct:.0f}% of resamples; offset median {med:.0f} deg, 90% interval [{lo:.0f}, {hi:.0f}]")
pd.DataFrame(summary_rows).to_csv(f"{OUT}/cc_alignment_summary.csv", index=False)

# ============================ landmark warp (fit on DNA only) ============================
def landmarks(P):
    p = P.copy(); idx = np.arange(NB); ok = np.isfinite(p)
    p[~ok] = np.interp(idx[~ok], idx[ok], p[ok], period=NB)
    lo, hi = np.percentile(p, 10), np.percentile(p, 90)
    mid, top = lo + .5 * (hi - lo), lo + .8 * (hi - lo)
    sel = p >= top
    ang = np.deg2rad(CEN[sel])
    peak = (np.rad2deg(np.angle(np.mean(np.exp(1j * ang)))) % 360)
    j = int(np.argmin(np.abs(CEN - peak)))
    r = j
    while p[r % NB] >= mid and (r - j) < NB: r += 1                      # forward from the peak until DNA falls below mid
    down = CEN[(r - 1) % NB] + STEP / 2 * 0
    l = j
    while p[l % NB] >= mid and (j - l) < NB: l -= 1                      # backward from the peak until DNA falls below mid
    up = CEN[(l + 1) % NB]
    return up % 360, peak % 360, down % 360


def unwrap_triplet(t):
    u, p, d = t
    while p < u: p += 360
    while d < p: d += 360
    return u, p, d


def warp_map(lq, lr):
    uq, pq, dq = unwrap_triplet(lq); ur, pr, dr = unwrap_triplet(lr)
    kq = np.array([0, pq - uq, dq - uq, 360.0]); kr = np.array([0, pr - ur, dr - ur, 360.0])
    def f(theta):
        t = (np.asarray(theta) - uq) % 360
        return (ur + np.interp(t, kq, kr)) % 360
    return f


s_dna = rigid["DNA only"][0]
LQ = landmarks(shifted(Pq["dna"], 1, 0) if s_dna == 1 else Pq["dna"][::-1] * 1.0)
LR = landmarks(Pr["dna"])
print(f"\nDNA landmarks (rise-mid, peak, fall-mid), degrees:  Element {tuple(round(x) for x in LR)}   OPS0077 {tuple(round(x) for x in LQ)}")
sgn = s_dna


def apply_rigid(angle, name="all three"):
    s, k = rigid[name]
    return (s * np.asarray(angle) + k * STEP) % 360


warp = warp_map(LQ, LR)


def apply_warp(angle):
    a = np.asarray(angle) % 360
    if sgn == -1:
        a = (-a) % 360
    return warp(a)


# ============================ independent validation: area profiles ============================
variants = {"unaligned": lambda a: np.asarray(a) % 360,
            "rigid (DNA only)": lambda a: apply_rigid(a, "DNA only"),
            "rigid (all three)": lambda a: apply_rigid(a, "all three"),
            "landmark warp (DNA only)": apply_warp}
rows = []
for vn, fn in variants.items():
    Pa = profiles(Q, fn(Q["angle"].values))
    row = {"alignment": vn}
    for f in FEATS:
        row[f"{f} r"] = corr(Pa[f], Pr[f])
    row["area r (mean of nuc, cell)"] = np.nanmean([row["nuc_area r"], row["cell_area r"]])
    rows.append(row)
val = pd.DataFrame(rows).set_index("alignment")
print("\nprofile correlation with Element after alignment (DNA is fitted; nuclear/cell area are INDEPENDENT checks):")
print(val.round(3).to_string())
val.to_csv(f"{OUT}/cc_alignment_validation.csv")

# ============================ within-OPS0077 regime agreement ============================
Lf = L.loc[Q.index]
regimes = {"4i DNA": Lf["cc_phase_dna"], "4i markers": Lf["cc_phase_markers"], "4i DNA+PCs": Lf["cc_phase_dna_pcs"], "4i final": Lf["cc_phase"]}

# ours: held-out morphology prediction of the 4i clock (circular regression, folds grouped by tile)
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str)
com = ph.obs_names[ph.obs_names.isin(Q.index)]
ph = ph[com]; Qc = Q.loc[com]
present = np.isfinite(np.asarray(ph.X)).mean(0) >= 0.85
Xc = np.asarray(ph.X, np.float32)[:, present]
th = np.deg2rad(Qc["angle"].values); gr = ph.obs["tile_pheno"].astype(str).values
ps_, pc_ = np.zeros(len(th)), np.zeros(len(th))
for a, b in GroupKFold(5).split(Xc, th, gr):
    imp = SimpleImputer(strategy="median").fit(Xc[a]); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xc[a]))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5)
    pca = PCA(100, random_state=0).fit(f(Xc[a])); A, B = pca.transform(f(Xc[a])), pca.transform(f(Xc[b]))
    ps_[b] = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(A, np.sin(th[a])).predict(B)
    pc_[b] = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(A, np.cos(th[a])).predict(B)
morph = pd.Series(np.mod(np.arctan2(ps_, pc_), 2 * np.pi) / (2 * np.pi), index=com)
regimes["morphology (phase contrast)"] = morph.reindex(Q.index)


def circ_r(a, b):
    a, b = 2 * np.pi * a, 2 * np.pi * b
    a0, b0 = a - np.angle(np.mean(np.exp(1j * a))), b - np.angle(np.mean(np.exp(1j * b)))
    return float(np.sum(np.sin(a0) * np.sin(b0)) / np.sqrt(np.sum(np.sin(a0) ** 2) * np.sum(np.sin(b0) ** 2)))


def aligned_err(a, b):
    """Best rotation (+ optional reflection) mapping regime a onto regime b, chosen by the ROTATION-SENSITIVE mean cos of the
    difference (not by the sign of a circular correlation, which misleads when the clocks move at different speeds).
    returns (mean cos after alignment, offset deg, median angular error deg, flip)"""
    sh = np.linspace(0, 1, 180, endpoint=False)
    sc, f, o = max(((np.mean(np.cos(2 * np.pi * (f_ * a + s_ - b))), f_, s_) for f_ in (1, -1) for s_ in sh))
    e = np.abs((f * a + o - b + .5) % 1 - .5) * 360
    return float(sc), float(o * 360 % 360), float(np.median(e)), int(f)


keys = list(regimes); n = len(keys)
R = np.full((n, n), np.nan); OFF = np.full((n, n), np.nan); ERR = np.full((n, n), np.nan); FLIP = np.zeros((n, n), int)
for i, ki in enumerate(keys):
    for j, kj in enumerate(keys):
        m = regimes[ki].notna() & regimes[kj].notna()
        if m.sum() > 100:
            R[i, j], OFF[i, j], ERR[i, j], FLIP[i, j] = aligned_err(regimes[ki][m].values, regimes[kj][m].values)
print("\nagreement among OPS0077 scoring regimes (same cells), after the best rotation: mean cos of difference | rotation deg | direction | median angular error")
for i in range(n):
    for j in range(i + 1, n):
        print(f"  {keys[i]:24s} -> {keys[j]:28s}: mean cos {R[i, j]:.2f} | rotate {OFF[i, j]:5.0f} | direction {FLIP[i, j]:+d} | error {ERR[i, j]:4.0f} deg")

# ============================ per-cell output in the common (Element) frame ============================
out = pd.DataFrame({"cc_phase_4i": Q["cc_phase"]})
out["cc_deg_rigid"] = apply_rigid(Q["angle"].values, "all three")
out["cc_deg_warp"] = apply_warp(Q["angle"].values)
mp = regimes["morphology (phase contrast)"].reindex(Q.index)
out["morph_cc_phase"] = mp
out["morph_deg_warp"] = np.where(mp.notna(), apply_warp(np.nan_to_num(mp.values) * 360), np.nan)
out["morph_deg_rigid"] = np.where(mp.notna(), apply_rigid(np.nan_to_num(mp.values) * 360, "all three"), np.nan)
out.index.name = "cell_key"; out.to_csv(f"{OUT}/cc_aligned_ops0077.csv")

# ============================ figures ============================
col = {"el": "k", "q": "tab:blue", "w": "tab:red", "m": "tab:orange"}
# A: native conventions
fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex="col")
for i, f in enumerate(FEATS):
    ax[i, 0].plot(CEN, Pr[f], color=col["el"], lw=2); ax[i, 0].set_ylabel(LABEL[f].split("(")[0].strip())
    ax[i, 1].plot(CEN, Pq[f], color=col["q"], lw=2)
ax[0, 0].set_title("Element RNA clock (cycleVI), degrees"); ax[0, 1].set_title("OPS0077 4i clock (cc_phase x 360)")
ax[2, 0].set_xlabel("native angle (deg)"); ax[2, 1].set_xlabel("native angle (deg)")
for k, (nm, lm) in enumerate([("rise", 0), ("peak", 1), ("fall", 2)]):
    ax[0, 0].axvline(LR[lm], color="0.6", lw=.8, ls=":"); ax[0, 1].axvline(LQ[lm], color="0.6", lw=.8, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/cc_align_1_native.png", dpi=110); plt.close()

# B: offset scan
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for si, s in enumerate((1, -1)):
    for (nm, r), c in zip(res.items(), ["tab:red", "tab:green", "k"]):
        ax[si].plot(np.arange(NB) * STEP, r[si], color=c, lw=2 if nm == "all three" else 1.3, label=nm)
    ax[si].axhline(0, color="0.8", lw=.6); ax[si].set_xlabel("offset applied to OPS0077 angle (deg)")
    ax[si].set_title(f"direction {s:+d}" + ("  (same sense of rotation)" if s == 1 else "  (reversed)"))
ax[0].set_ylabel("profile correlation with Element"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/cc_align_2_offset_scan.png", dpi=110); plt.close()

# C: after alignment
fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
for i, f in enumerate(FEATS):
    ax[i].plot(CEN, Pr[f], color=col["el"], lw=2.4, label="Element (RNA clock)")
    ax[i].plot(CEN, profiles(Q, apply_rigid(Q["angle"].values, "all three"))[f], color=col["q"], lw=1.4, ls="--", label="OPS0077 4i, rigid")
    ax[i].plot(CEN, profiles(Q, apply_warp(Q["angle"].values))[f], color=col["w"], lw=2, label="OPS0077 4i, landmark warp")
    mm = Q.loc[mp.notna().values] if mp.notna().any() else None
    if mm is not None:
        ax[i].plot(CEN, profiles(mm, apply_warp(mp.dropna().values * 360))[f], color=col["m"], lw=1.6, label="OPS0077 morphology, warp")
    ax[i].set_title(LABEL[f] + ("\n(FITTED)" if f == "dna" else "\n(independent check)")); ax[i].set_xlabel("common angle (Element degrees)")
ax[0].legend(fontsize=7)
axp = plt.subplot(1, 4, 4, projection="polar")
for df_, a_, c_, lab in [(ctrl, ctrl["angle"].values, "k", "Element"), (Q, apply_warp(Q["angle"].values), "tab:red", "OPS0077 (warp)")]:
    cnt, ed = np.histogram(np.deg2rad(a_), bins=36, range=(0, 2 * np.pi), density=True)
    axp.plot(np.r_[ed[:-1], ed[0]] + np.pi / 36, np.r_[cnt, cnt[0]], color=c_, lw=2, label=lab)
axp.set_theta_zero_location("N"); axp.set_theta_direction(-1); axp.set_title("cell density around the common clock"); axp.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/cc_align_3_aligned.png", dpi=110); plt.close()

# D: regime agreement
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a_, M, ttl, fmt, cm in [(ax[0], R, "agreement after the best rotation\n(mean cos of the difference; 1 = identical)", "{:.2f}", "viridis"), (ax[1], ERR, "median angular error after the\nbest rotation (deg; random = 90)", "{:.0f}", "viridis_r")]:
    im = a_.imshow(M, cmap=cm); a_.set_xticks(range(n)); a_.set_xticklabels(keys, rotation=35, ha="right"); a_.set_yticks(range(n)); a_.set_yticklabels(keys)
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]): a_.text(j, i, fmt.format(M[i, j]), ha="center", va="center", color="w", fontsize=9)
    a_.set_title(ttl); plt.colorbar(im, ax=a_, fraction=.046)
plt.tight_layout(); plt.savefig(f"{OUT}/cc_align_4_regime_agreement.png", dpi=110); plt.close()

print("\nDONE; outputs in", OUT)
