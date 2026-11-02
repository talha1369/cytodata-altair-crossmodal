#!/usr/bin/env python
"""How well does phase-contrast morphology predict each cell-cycle scoring regime, and where around the cycle?

Targets (OPS0077 NTC cells, all measured on the same cells as the morphology features)
  regime angles   4i DNA | 4i markers | 4i DNA+PCs (the final cc_phase)        -> circular regression (sin/cos)
  components      DNA content, pRb, p21, phospho-Rb fraction (z-scores)        -> ridge regression
Held-out: 5-fold, folds grouped by imaging tile. Preprocessing (median-impute, robust-scale, clip, PCA 100) fit on train only.
Baselines: cell/nuclear size only, and a shuffled-label null.
Also: do the PREDICTED regimes keep the disagreement the MEASURED regimes have (markers vs DNA)?

usage: python phase_predicts_regimes.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "cc_align"
os.makedirs(OUT, exist_ok=True)


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


L = pd.read_csv(find("cell_cycle_scores.csv")).set_index("cell_key")
fr = pd.read_parquet(find("OPS0077_NTC_cellcycle_frame.parquet"))
Q = L[(L["dna_reliable"] == True) & (L["cc_phase_source"] == "dna+pcs")].copy()
ph = ad.read_h5ad(find("OPS0077_NTC_phase_OrganelleProfiler.h5ad")); ph.obs_names = ph.obs["cell_key"].astype(str)
com = ph.obs_names[ph.obs_names.isin(Q.index)]; ph = ph[com]; Q = Q.loc[com]
present = np.isfinite(np.asarray(ph.X)).mean(0) >= 0.85
X = np.asarray(ph.X, np.float32)[:, present]
groups = ph.obs["tile_pheno"].astype(str).values
n = len(Q); print(f"{n:,} cells | {int(present.sum())} morphology features | {len(set(groups)):,} tiles")

ANG = {"4i DNA": Q["cc_phase_dna"], "4i markers": Q["cc_phase_markers"], "4i DNA+PCs (final)": Q["cc_phase"]}
ANG = {k: 2 * np.pi * v.astype(float).values for k, v in ANG.items()}
COMP = {"DNA content": Q["z_dna"], "pRb": Q["z_pRb"], "p21": Q["z_p21"], "phospho-Rb fraction": Q["z_phos"]}
COMP = {k: v.astype(float).values for k, v in COMP.items()}
size = np.c_[np.log2(Q["nuc_area_px"].astype(float).clip(lower=1)), np.log2(fr["cell_area_4i"].reindex(Q.index).astype(float).clip(lower=1))]
size = np.where(np.isfinite(size), size, np.nanmedian(size, 0))
null_theta = np.random.default_rng(0).permutation(ANG["4i DNA+PCs (final)"])

PS = {k: np.full(n, np.nan) for k in ANG}; PC = {k: np.full(n, np.nan) for k in ANG}
SS = {k: np.full(n, np.nan) for k in ANG}; SC = {k: np.full(n, np.nan) for k in ANG}
NS, NC = np.full(n, np.nan), np.full(n, np.nan)
PR = {k: np.full(n, np.nan) for k in COMP}; SR = {k: np.full(n, np.nan) for k in COMP}
RA = np.logspace(-2, 5, 15)

for fold, (a, b) in enumerate(GroupKFold(5).split(X, groups=groups)):
    imp = SimpleImputer(strategy="median").fit(X[a]); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(X[a]))
    f = lambda Z: np.clip(sc.transform(imp.transform(Z)), -5, 5)
    pca = PCA(100, random_state=0).fit(f(X[a])); A, B = pca.transform(f(X[a])), pca.transform(f(X[b]))
    ss = StandardScaler().fit(size[a]); Aa, Bb = ss.transform(size[a]), ss.transform(size[b])
    for k, th in ANG.items():
        ok = np.isfinite(th[a])
        PS[k][b] = RidgeCV(alphas=RA).fit(A[ok], np.sin(th[a][ok])).predict(B); PC[k][b] = RidgeCV(alphas=RA).fit(A[ok], np.cos(th[a][ok])).predict(B)
        SS[k][b] = RidgeCV(alphas=RA).fit(Aa[ok], np.sin(th[a][ok])).predict(Bb); SC[k][b] = RidgeCV(alphas=RA).fit(Aa[ok], np.cos(th[a][ok])).predict(Bb)
    NS[b] = RidgeCV(alphas=RA).fit(A, np.sin(null_theta[a])).predict(B); NC[b] = RidgeCV(alphas=RA).fit(A, np.cos(null_theta[a])).predict(B)
    for k, v in COMP.items():
        ok = np.isfinite(v[a])
        PR[k][b] = RidgeCV(alphas=RA).fit(A[ok], v[a][ok]).predict(B); SR[k][b] = RidgeCV(alphas=RA).fit(Aa[ok], v[a][ok]).predict(Bb)
    print(f"fold {fold + 1}/5 done", flush=True)


def circ_r(a, b):
    a0, b0 = a - np.angle(np.mean(np.exp(1j * a))), b - np.angle(np.mean(np.exp(1j * b)))
    return float(np.sum(np.sin(a0) * np.sin(b0)) / np.sqrt(np.sum(np.sin(a0) ** 2) * np.sum(np.sin(b0) ** 2)))


def metrics(th, ps, pc):
    m = np.isfinite(th) & np.isfinite(ps)
    p = np.arctan2(ps[m], pc[m]); t = th[m]
    err = np.abs(np.angle(np.exp(1j * (t - p)))) * 180 / np.pi
    sector = np.floor((np.degrees(t) % 360) / 30).astype(int)
    per = [float(np.median(err[sector == s])) for s in range(12) if (sector == s).sum() > 20]
    return dict(circ_r=round(circ_r(t, p), 3), median_err_deg=round(float(np.median(err)), 1),
                balanced_err_deg=round(float(np.mean(per)), 1), within_45deg=round(float((err < 45).mean()), 3), n=int(m.sum())), err, m


rows = []
for k, th in ANG.items():
    m_, _, _ = metrics(th, PS[k], PC[k]); s_, _, _ = metrics(th, SS[k], SC[k])
    rows.append({"target": k, "predictor": "phase-contrast morphology", **m_}); rows.append({"target": k, "predictor": "cell/nuclear size only", **s_})
nl, _, _ = metrics(ANG["4i DNA+PCs (final)"], NS, NC); rows.append({"target": "4i DNA+PCs (final)", "predictor": "shuffled-label null", **nl})
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/phase_predicts_regimes_angles.csv", index=False)
print("\nangular regimes predicted from morphology (held-out; a random guess has median error 90 deg):")
print(tab.to_string(index=False))

crow = []
for k, v in COMP.items():
    ok = np.isfinite(v)
    crow.append({"target": k, "R2 morphology": round(r2_score(v[ok], PR[k][ok]), 3), "R2 size only": round(r2_score(v[ok], SR[k][ok]), 3)})
comp = pd.DataFrame(crow); comp.to_csv(f"{OUT}/phase_predicts_regimes_components.csv", index=False)
print("\ncell-cycle components predicted from morphology (held-out R2):"); print(comp.to_string(index=False))

# do the PREDICTED regimes agree with each other as much as the MEASURED ones do?  (rotation-sensitive alignment)
def align_stats(x, y):
    sh = np.linspace(0, 1, 180, endpoint=False)
    sc, f, o = max(((np.mean(np.cos(2 * np.pi * (f_ * x + s_ - y))), f_, s_) for f_ in (1, -1) for s_ in sh))
    e = np.abs((f * x + o - y + .5) % 1 - .5) * 360
    return sc, f, o * 360 % 360, float(np.median(e))


keys = list(ANG); pred = {k: (np.arctan2(PS[k], PC[k]) % (2 * np.pi)) / (2 * np.pi) for k in keys}; meas = {k: ANG[k] / (2 * np.pi) for k in keys}
print("\nDNA clock vs markers clock, after the best rotation (random error = 90 deg):")
for nm, src in [("MEASURED", meas), ("PREDICTED from morphology", pred)]:
    sc, f, o, e = align_stats(src["4i markers"], src["4i DNA"])
    print(f"  {nm:26s}: mean cos {sc:.2f} | direction {f:+d} | rotation {o:.0f} deg | median error {e:.0f} deg")
print("\nmedian angular error after the best rotation: rows = MEASURED regime, columns = PREDICTED regime (from morphology)")
Mx = pd.DataFrame({kp: {km: align_stats(pred[kp][np.isfinite(meas[km])], meas[km][np.isfinite(meas[km])])[3] for km in keys} for kp in keys})
print(Mx.round(0).to_string())
Mx.round(1).to_csv(f"{OUT}/phase_predicts_regimes_cross.csv")
rowsm = []
for nm, src in [("measured", meas), ("predicted from morphology", pred)]:
    sc_, f_, o_, e_ = align_stats(src["4i markers"], src["4i DNA"]); rowsm.append(dict(angles=nm, mean_cos=round(sc_, 2), direction=f_, rotation_deg=round(o_), median_err_deg=round(e_)))
pd.DataFrame(rowsm).to_csv(f"{OUT}/phase_predicts_regimes_dna_vs_markers.csv", index=False)

# figures
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
w = .26; xs = np.arange(len(keys))
for j, (pr, c) in enumerate([("phase-contrast morphology", "tab:red"), ("cell/nuclear size only", "tab:gray")]):
    ax[0].bar(xs + (j - .5) * w, [tab[(tab.target == k) & (tab.predictor == pr)].circ_r.values[0] for k in keys], w, color=c, label=pr)
ax[0].bar(xs[-1] + 1.5 * w, nl["circ_r"], w, color="k", alpha=.4, label="shuffled null")
ax[0].set_xticks(xs); ax[0].set_xticklabels([k.replace(" (final)", "\n(final)") for k in keys]); ax[0].set_ylabel("circular correlation (held-out)"); ax[0].legend(fontsize=8)
ax[0].set_title("predicting each scoring regime")
canon = np.degrees(ANG["4i DNA+PCs (final)"]) % 360
arc_rows = []
for k, c in zip(keys, ["tab:blue", "tab:green", "tab:red"]):
    m_, err, mk = metrics(ANG[k], PS[k], PC[k]); sec = np.floor(canon[mk] / 30).astype(int)
    med = [float(np.median(err[sec == s_])) if (sec == s_).sum() > 20 else np.nan for s_ in range(12)]
    ax[1].plot(np.arange(12) * 30 + 15, med, "o-", color=c, label=k)
    arc_rows += [dict(regime=k, canonical_arc_deg=f"{s_ * 30}-{s_ * 30 + 30}", median_err_deg=round(med[s_], 1), n=int((sec == s_).sum())) for s_ in range(12)]
pd.DataFrame(arc_rows).to_csv(f"{OUT}/phase_predicts_regimes_by_canonical_arc.csv", index=False)
ax[1].axhline(90, color="k", ls=":", lw=1, label="random guess"); ax[1].set_xlabel("position on the canonical clock (final cc_phase, deg), same cells for every regime"); ax[1].set_ylabel("median angular error (deg)")
ax[1].legend(fontsize=8); ax[1].set_title("where around the cycle morphology is accurate\n(each regime scored against its own measured angle)")
xs2 = np.arange(len(comp)); ax[2].bar(xs2 - .2, comp["R2 morphology"], .4, color="tab:red", label="phase-contrast morphology"); ax[2].bar(xs2 + .2, comp["R2 size only"], .4, color="tab:gray", label="size only")
ax[2].set_xticks(xs2); ax[2].set_xticklabels(comp["target"], rotation=15); ax[2].set_ylabel("held-out R2"); ax[2].legend(fontsize=8); ax[2].set_title("predicting the cell-cycle readouts themselves")
plt.tight_layout(); plt.savefig(f"{OUT}/cc_align_6_phase_predicts_regimes.png", dpi=110)
print("DONE")
