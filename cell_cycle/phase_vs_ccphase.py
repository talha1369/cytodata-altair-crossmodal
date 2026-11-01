#!/usr/bin/env python
"""Overlay label-free phase on the team's cell-cycle clock (cell_cycle_scores.csv `cc_phase`, a circular 0..1 phase).

Predict the clock from per-cell phase morphology with held-out, tile-grouped circular regression (predict sin and cos
of the angle, recover theta). Report circular correlation and angular error vs a size-only baseline and a shuffled null.
usage: python phase_vs_ccphase.py
"""
import os
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "."]:
        if os.path.exists(f"{r}/{n}"):
            return f"{r}/{n}"
    raise FileNotFoundError(n)


cc = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
cc = cc[(cc["dna_reliable"] == True) & (cc["cc_phase_source"] == "dna+pcs")]       # the cells with a defined clock
theta = (cc["cc_phase"].astype(float).values % 1.0) * 2 * np.pi

ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad"))
ph.obs_names = ph.obs["cell_key"].astype(str)
common = ph.obs_names[ph.obs_names.isin(cc.index)]
ph = ph[common]; cc2 = cc.loc[common]; theta = (cc2["cc_phase"].astype(float).values % 1.0) * 2 * np.pi
present = np.isfinite(np.asarray(ph.X)).mean(0) >= 0.85
Xp = np.asarray(ph.X, np.float32)[:, present]
gr = ph.obs["tile_pheno"].astype(str).values
area = np.c_[cc2["nuc_area_px"].astype(float), cc2["dapi_log_area"].astype(float)]
area = np.where(np.isfinite(area), area, np.nanmedian(area, 0))
print(f"cells {len(common):,} | phase feats {int(present.sum())} | tiles {len(set(gr)):,}")


def prep(a, b, pca):
    imp = SimpleImputer(strategy="median").fit(a); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(a))
    f = lambda Q: np.clip(sc.transform(imp.transform(Q)), -5, 5); A, B = f(a), f(b)
    if pca and A.shape[1] > 100:
        p = PCA(100, random_state=0).fit(A); A, B = p.transform(A), p.transform(B)
    return A, B


def predict_angle(X, th, pca=True, shuffle=False):
    if shuffle:
        th = np.random.default_rng(0).permutation(th)
    s, co = np.sin(th), np.cos(th); ps = np.zeros(len(th)); pc = np.zeros(len(th))
    for a, b in GroupKFold(5).split(X, th, gr):
        A, B = prep(X[a], X[b], pca)
        ps[b] = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(A, s[a]).predict(B)
        pc[b] = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(A, co[a]).predict(B)
    return np.arctan2(ps, pc)


def circ_corr(a, b):
    a0 = a - np.angle(np.mean(np.exp(1j * a))); b0 = b - np.angle(np.mean(np.exp(1j * b)))
    return np.sum(np.sin(a0) * np.sin(b0)) / np.sqrt(np.sum(np.sin(a0) ** 2) * np.sum(np.sin(b0) ** 2))


def report(th, pred, name):
    err = np.abs(np.angle(np.exp(1j * (th - pred)))) * 180 / np.pi
    print(f"  {name:24s}: circular r = {circ_corr(th, pred):.3f} | median angular error {np.median(err):.1f} deg | "
          f"within 45 deg {np.mean(err < 45):.2f}")
    return err


pred = predict_angle(Xp, theta)
print("phase -> team cc_phase (held-out, tile-grouped):")
err = report(theta, pred, "phase morphology")
report(theta, predict_angle(area, theta, pca=False), "size-only baseline")
report(theta, predict_angle(Xp, theta, shuffle=True), "shuffled null")

fig, ax = plt.subplots(1, 3, figsize=(18, 5.3))
ax[0].hexbin(theta, np.mod(pred, 2 * np.pi), gridsize=70, bins="log", extent=(0, 2 * np.pi, 0, 2 * np.pi))
ax[0].plot([0, 2 * np.pi], [0, 2 * np.pi], "r--", lw=1); ax[0].set_xlabel("team cc_phase angle"); ax[0].set_ylabel("phase-predicted angle")
ax[0].set_title(f"phase recovers the clock (circ r={circ_corr(theta, pred):.2f})")
sc = ax[1].scatter(cc2["z_dna"], cc2["z_pRb"], c=np.mod(pred, 2 * np.pi), s=3, cmap="twilight")
ax[1].set_xlabel("z DNA"); ax[1].set_ylabel("z pRb"); ax[1].set_title("DNA x pRb coloured by phase-PREDICTED clock"); plt.colorbar(sc, ax=ax[1])
sc2 = ax[2].scatter(cc2["x_cp"], cc2["y_cp"], c=err, s=3, cmap="viridis", vmax=90)
ax[2].set_xlabel("x"); ax[2].set_ylabel("y"); ax[2].set_title("angular error across the well"); plt.colorbar(sc2, ax=ax[2], label="deg")
plt.tight_layout(); plt.savefig("phase_vs_ccphase.png", dpi=95); print("figure phase_vs_ccphase.png")
