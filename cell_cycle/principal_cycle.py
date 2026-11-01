#!/usr/bin/env python
"""Step 2: continuous cell-cycle pseudotime for OPS0077 NTC cells.

Fit a principal curve through the interpretable 4i cell-cycle vector (DNA content, pRb, pRb/Rb ratio, nuclear area;
the cell_cycle_markers.md Y) on the reliable-DNA cells, project every cell to arc length -> cycle_pseudotime in [0,1],
oriented so DNA content increases (G0/G1 -> S -> G2/M). Validate the marker ordering, check phase predicts it, and
write cycle_pseudotime back into the frame and the multimodal h5ad.

usage: python principal_cycle.py
"""
import os
import numpy as np, pandas as pd, anndata as ad
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"):
            return f"{r}/{n}"
    raise FileNotFoundError(n)


fr = pd.read_parquet(find("OPS0077_NTC_cellcycle_frame.parquet"))
rel = (fr["dna_reliable"].astype(str) == "True").values
F = fr.loc[rel]
Y = np.c_[F["dna_content_log2"].astype(float), F["log2_pRb"].astype(float),
          F["prb_rb_ratio"].astype(float), np.log2(F["nuclear_area_4i"].astype(float).clip(lower=1))]
Y = np.where(np.isfinite(Y), Y, np.nanmedian(Y, 0))
Z = (Y - Y.mean(0)) / Y.std(0)


def smooth_sorted(vals, order, win):
    out = np.empty_like(vals)
    c = np.cumsum(np.insert(vals[order], 0, 0.0))
    n = len(vals); h = win // 2
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        out[order[i]] = (c[b] - c[a]) / (b - a)
    return out


def principal_curve(Z, iters=8, win=None):
    win = win or max(50, len(Z) // 50)
    t = PCA(2, random_state=0).fit_transform(Z)[:, 0]        # init on PC1
    for _ in range(iters):
        order = np.argsort(t)
        curve = np.column_stack([smooth_sorted(Z[:, j], order, win) for j in range(Z.shape[1])])
        cs = curve[order]
        seg = np.r_[0, np.linalg.norm(np.diff(cs, axis=0), axis=1).cumsum()]
        arc = np.empty(len(t)); arc[order] = seg
        tree = cKDTree(curve)
        _, idx = tree.query(Z, k=1)
        t = arc[idx]
    return t


t = principal_curve(Z)
if np.corrcoef(t, Y[:, 0])[0, 1] < 0:                        # orient by DNA content
    t = -t
t = (t - t.min()) / (t.max() - t.min())
F = F.assign(cycle_pseudotime=t)

print(f"reliable cells {len(F):,}")
print("Spearman of pseudotime with markers:",
      {k: round(float(pd.Series(t).corr(F[k].astype(float).reset_index(drop=True), method="spearman")), 3)
       for k in ["dna_content_log2", "log2_pRb", "prb_rb_ratio", "nuclear_area_4i"]})
print("\nmean pseudotime by DNA x pRb stage (should increase G0/G1-early -> G2/M):")
print(F.groupby("cycle_stage")["cycle_pseudotime"].agg(["mean", "size"]).round(3).to_string())

# does phase predict the continuous pseudotime? (held-out, tile-grouped)
try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad"))
    ph.obs_names = ph.obs["cell_key"].astype(str)
    Xp = np.asarray(ph[F.index].X, np.float32)
    present = np.isfinite(Xp).mean(0) >= 0.85
    Xp = Xp[:, present]
    gr = F["tile_pheno"].astype(str).values
    pred = np.zeros(len(t))
    for a, b in GroupKFold(5).split(Xp, t, gr):
        imp = SimpleImputer(strategy="median").fit(Xp[a]); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xp[a]))
        f = lambda Q: np.clip(sc.transform(imp.transform(Q)), -5, 5)
        p = PCA(100, random_state=0).fit(f(Xp[a]))
        pred[b] = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(p.transform(f(Xp[a])), t[a]).predict(p.transform(f(Xp[b])))
    r2 = r2_score(t, pred)
    print(f"\nphase -> cycle_pseudotime held-out R2 = {r2:.3f} (n={len(t):,})")
except Exception as e:
    r2, pred = None, None
    print("phase prediction skipped:", e)

# write back
full = pd.Series(np.nan, index=fr.index); full.loc[F.index] = t
fr["cycle_pseudotime"] = full.values
fr.to_parquet(find("OPS0077_NTC_cellcycle_frame.parquet"))
mm = find("OPS0077_NTC_Multimodal_4i_Phase_DNA.h5ad")
A = ad.read_h5ad(mm)
A.obs["cycle_pseudotime"] = full.reindex(A.obs_names).values
A.write_h5ad(mm, compression="gzip")
print("wrote cycle_pseudotime into frame and", mm)

# figure
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
o = np.argsort(t)
for name, lab in [("dna_content_log2", "DNA content"), ("log2_pRb", "pRb"), ("prb_rb_ratio", "pRb/Rb ratio"), ("nuclear_area_4i", "nuclear area")]:
    v = F[name].astype(float).values; v = np.where(np.isfinite(v), v, np.nanmedian(v))   # a few cells lack nuclear area
    v = (v - v.mean()) / v.std()                                                          # z-score so all markers share one axis
    ax[0].plot(np.sort(t), smooth_sorted(v, o, max(50, len(t) // 50))[o], label=lab)
ax[0].set_xlabel("cycle pseudotime"); ax[0].set_ylabel("marker (z-scored, smoothed)"); ax[0].legend(); ax[0].set_title("markers along pseudotime")
sc = ax[1].scatter(F["dna_content_log2"].astype(float), F["log2_pRb"].astype(float), c=t, s=3, cmap="viridis")
ax[1].set_xlabel("DNA content (log2)"); ax[1].set_ylabel("log2 pRb"); ax[1].set_title("pseudotime in DNA x pRb"); plt.colorbar(sc, ax=ax[1])
if pred is not None:
    ax[2].hexbin(t, pred, gridsize=70, bins="log", extent=(0, 1, 0, 1)); ax[2].plot([0, 1], [0, 1], "r--", lw=1)
    ax[2].set_xlabel("4i pseudotime"); ax[2].set_ylabel("phase-predicted"); ax[2].set_title(f"phase -> pseudotime (R2={r2:.2f})")
plt.tight_layout(); plt.savefig("cycle_pseudotime.png", dpi=95)
print("figure cycle_pseudotime.png")
