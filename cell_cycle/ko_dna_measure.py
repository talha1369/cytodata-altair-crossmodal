#!/usr/bin/env python
"""Measured DNA content for the cycle-shifting knockouts (and calibration knockouts), compared with the morphology prediction.

Inputs: dapi_cpmeasure.py + bg_pass.py runs on OPS0077_KOcyc_input.h5ad (ko_dna_build_input.py), the NTC DAPI table, and
ko_library/ko_library_cycle.csv (predicted change in % 4N per knockout). DNA content per cell = mean over DAPI rounds 2-4 of
log2(integrated DAPI - local non-nuclear background x nuclear area), minus the NTC median per round (same as the NTC table).
Cells need a nucleus in rounds 2-4 and a single nucleus in the cell. States: 2N < 0.25 <= S <= 0.65 < 4N (as for NTC).
Null for each knockout: NTC guides pooled to the knockout's cell count (2,000 draws), BH FDR.
usage: python ko_dna_measure.py [OUT_DIR]"""
import os, sys, glob
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "ko_dna"
os.makedirs(OUT, exist_ok=True)
SCALE = 65535.0; RNG = np.random.default_rng(0); REPS = 2000


def load(d, pat):
    fs = sorted(glob.glob(f"{d}/part*/{pat}")) or sorted(glob.glob(f"{d}/{pat}"))
    return pd.concat([pd.read_parquet(p) for p in fs])


USE = os.environ.get("KO_DNA_BATCHES", "kocyc" + (",koarrest" if os.path.exists("koarrest_done.txt") else "")).split(",")   # only finished batches
BATCHES = [(d, b, i) for name, (d, b, i) in [("kocyc", ("dapi_kocyc", "bg_kocyc", "OPS0077_KOcyc_input.h5ad")),
                                              ("koarrest", ("dapi_koarrest", "bg_koarrest", "OPS0077_KOarrest_input.h5ad"))]
           if name in USE and glob.glob(f"{d}/*") and glob.glob(f"{b}/*")]
print("batches:", [b[0] for b in BATCHES], flush=True)
feat = pd.concat([load(d, "feat_*.parquet") for d, _, _ in BATCHES]); qc = pd.concat([load(d, "qc_*.parquet") for d, _, _ in BATCHES])
for t_ in (feat, qc):
    if "cell_key" in t_.columns: t_.set_index("cell_key", inplace=True)
feat = feat[~feat.index.duplicated()]; qc = qc[~qc.index.duplicated()].reindex(feat.index)
bg = pd.concat([load(b, "bg_*.parquet") for _, b, _ in BATCHES])
if "cell_key" in bg.columns: bg = bg.set_index("cell_key")
bg = bg[~bg.index.duplicated()].reindex(feat.index)
inp = pd.concat([ad.read_h5ad(i, backed="r").obs for _, _, i in BATCHES]).drop_duplicates("cell_key").set_index("cell_key")
ntc = ad.read_h5ad("OPS0077_NTC_DAPI_cpmeasure.h5ad", backed="r").obs
rel_ok = ntc["dna_reliable"].astype(str) == "True"
med = {k: float(ntc.loc[rel_ok, f"dna_R{k}_log2"].astype(float).median()) for k in (2, 3, 4)}
rel = {}
for k in (2, 3, 4):
    dna = feat[f"Intensity_IntegratedIntensity__DAPI_R{k}"] * SCALE - bg[f"R{k}_bg_nonnuc_median"] * feat[f"nucleus_R{k}_Area"]
    rel[k] = np.log2(dna.where(dna > 0)) - med[k]
K = pd.DataFrame({"gene": inp.loc[feat.index, "gene"].astype(str).values, "set": inp.loc[feat.index, "set"].astype(str).values,
                  "dna": pd.concat(rel, axis=1).mean(1, skipna=False).values}, index=feat.index)
K["ok"] = (qc[["R2_nucleus_id", "R3_nucleus_id", "R4_nucleus_id"]] >= 0).all(1).values & (qc["R2_n_nuclei_in_cell"].astype(float) <= 1).values
print(f"knockout cells measured {len(K):,} | usable (nucleus in R2-4, single nucleus) {int((K.ok & K.dna.notna()).sum()):,}", flush=True)
K = K[K.ok & K.dna.notna()]
K["nuc_area_R2"] = feat.loc[K.index, "nucleus_R2_Area"].values
K.to_csv(f"{OUT}/ko_measured_dna_percell.csv")                       # per cell, for g1_hypertrophy_check.py

N = pd.DataFrame({"dna": ntc.loc[rel_ok, "dna_rel_R234"].astype(float).values, "guide": ntc.loc[rel_ok, "sgRNA"].astype(str).values})
state = lambda d: (100 * np.mean(d < .25), 100 * np.mean((d >= .25) & (d <= .65)), 100 * np.mean(d > .65))
ref = np.array(state(N.dna.values)); print(f"NTC n={len(N):,}: %2N {ref[0]:.1f}  %S {ref[1]:.1f}  %4N {ref[2]:.1f}", flush=True)
gidx = N.groupby("guide").indices; gl = list(gidx)


def null(n):
    out = []
    for _ in range(REPS):
        RNG.shuffle(gl); ix = []
        for g_ in gl:
            ix.extend(gidx[g_].tolist())
            if len(ix) >= n: break
        out.append(state(N.dna.values[ix[:n]]))
    return np.array(out) - ref


P = pd.read_csv("ko_library/ko_library_cycle.csv").set_index("gene")
rows, nulls = [], {}
for g_, d in K.groupby("gene"):
    n = len(d); key = max(int(round(n / 25) * 25), 25)
    if key not in nulls: nulls[key] = null(key)
    s = np.array(state(d.dna.values)) - ref; nd = nulls[key]
    p4 = (np.sum(np.abs(nd[:, 2]) >= abs(s[2])) + 1) / (REPS + 1); p2 = (np.sum(np.abs(nd[:, 0]) >= abs(s[0])) + 1) / (REPS + 1)
    rows.append(dict(gene=g_, set=d.set.iloc[0], n_measured=n, measured_d_pct_2N=s[0], measured_d_pct_S=s[1], measured_d_pct_4N=s[2],
                     p_4N=p4, p_2N=p2, predicted_d_pct_4N=P.loc[g_, "d_pct_4N_pred"], predicted_cycle_TV=P.loc[g_, "cycle_TV"],
                     q_cycle_pred=P.loc[g_, "q_cycle"], measured_pRb_shift_log2=P.loc[g_, "measured_pRb_shift_log2"]))
R = pd.DataFrame(rows)
if os.path.exists("genes_cc.csv"):                                   # expected arrest phase for the 60-gene arrest list
    ex = pd.read_csv("genes_cc.csv"); R["expected_arrest_phase"] = R.gene.map(dict(zip(ex.Gene_Symbol.str.upper(), ex.Primary_Arrest_Phase)))
for c_ in ["4N", "2N"]:
    pv = R[f"p_{c_}"].values; o = np.argsort(pv); m = len(pv); q = np.empty(m)
    q[o] = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]; R[f"q_{c_}"] = np.minimum(q, 1)
R["measured_significant"] = (R.q_4N < .05) | (R.q_2N < .05)
R.round(4).to_csv(f"{OUT}/ko_measured_vs_predicted_dna.csv", index=False)

rho, p = spearmanr(R.predicted_d_pct_4N, R.measured_d_pct_4N); r_, _ = pearsonr(R.predicted_d_pct_4N, R.measured_d_pct_4N)
cs = R[R.set == "cycle_shift"]; cal = R[R.set == "calibration"]
sig = cs[cs.measured_significant]
print(f"\npredicted vs measured change in % 4N across {len(R)} knockouts: Spearman {rho:.2f} (p {p:.1g}), Pearson {r_:.2f}")
print(f"cycle-shift knockouts confirmed by measured DNA (FDR 5% on %4N or %2N): {len(sig)} / {len(cs)} | calibration knockouts significant: {int(cal.measured_significant.sum())} / {len(cal)}")
print(f"sign agreement (predicted vs measured %4N) among confirmed: {100 * np.mean(np.sign(sig.predicted_d_pct_4N) == np.sign(sig.measured_d_pct_4N)):.0f}%")
print(f"slope measured on predicted (least squares): {np.polyfit(R.predicted_d_pct_4N, R.measured_d_pct_4N, 1)[0]:.2f}")
if "expected_arrest_phase" in R:
    print("\nby expected arrest phase (arrest list): measured change in % 2N / % 4N, median and fraction significant")
    print(R.dropna(subset=["expected_arrest_phase"]).groupby("expected_arrest_phase").agg(n=("gene", "size"), median_d_pct_2N=("measured_d_pct_2N", "median"),
          median_d_pct_4N=("measured_d_pct_4N", "median"), n_significant=("measured_significant", "sum")).round(2).to_string())
cols = ["gene", "set", "n_measured", "measured_d_pct_2N", "measured_d_pct_S", "measured_d_pct_4N", "q_4N", "predicted_d_pct_4N", "measured_pRb_shift_log2"]
print("\nlargest measured 4N gains:\n", R.nlargest(12, "measured_d_pct_4N")[cols].round(2).to_string(index=False))
print("\nlargest measured 4N losses:\n", R.nsmallest(12, "measured_d_pct_4N")[cols].round(2).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
LAB = {"calibration": ("0.6", "calibration (no predicted shift)"), "cycle_shift": ("tab:red", "predicted cycle shift"), "arrest_list": ("tab:blue", "rest of the arrest list")}
for s_, (c_, lab_) in LAB.items():
    d_ = R[R.set == s_]
    if len(d_): ax[0].scatter(d_.predicted_d_pct_4N, d_.measured_d_pct_4N, s=np.where(d_.measured_significant, 40, 14), c=c_, label=f"{lab_} (n={len(d_)})")
for _, r in R[R.measured_significant].iterrows(): ax[0].annotate(r.gene, (r.predicted_d_pct_4N, r.measured_d_pct_4N), fontsize=7)
ax[0].axline((0, 0), slope=1, color="k", lw=.6); ax[0].axhline(0, color=".7", lw=.5); ax[0].axvline(0, color=".7", lw=.5)
ax[0].set_xlabel("change in % 4N cells predicted from phase morphology"); ax[0].set_ylabel("change in % 4N cells, measured DAPI")
ax[0].set_title(f"predicted vs measured DNA content (rho {rho:.2f})"); ax[0].legend(fontsize=8)
bins_ = np.linspace(-1, 1.8, 70); ax[1].hist(N.dna.clip(-1, 1.8), bins_, density=True, histtype="step", color="k", lw=2, label="NTC")
for g_, c_ in zip(R.reindex(R.measured_d_pct_4N.abs().sort_values(ascending=False).index).gene.head(4), ["tab:red", "tab:orange", "tab:green", "tab:blue"]):
    ax[1].hist(K.loc[K.gene == g_, "dna"].clip(-1, 1.8), bins_, density=True, histtype="step", lw=1.6, color=c_, label=g_)
ax[1].axvline(0, color=".7", lw=.6); ax[1].axvline(1, color=".7", lw=.6); ax[1].legend(fontsize=8)
ax[1].set_xlabel("relative DNA content, log2 (0 = 2N, 1 = 4N)"); ax[1].set_title("measured DNA content, largest shifts")
plt.tight_layout(); plt.savefig(f"{OUT}/ko_measured_vs_predicted_dna.png", dpi=110)
print("DONE")
