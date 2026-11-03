#!/usr/bin/env python
"""Measured DNA for the 111 knockouts only the gradient-boosting (HistGB) library run calls, with the statistics of
ko_dna_measure.py (NTC guides pooled to the knockout's cell count, 2,000 draws, two-sided, BH FDR on %4N and %2N; confirmed
if either q < 0.05), then combined with the 158 knockouts measured before.
FDR is reported two ways: within this batch, and pooled over all 269 measured knockouts (used for the comparison by
which model calls the knockout, so every category is judged on the same scale).
usage: python ko_dna_hgb_only_stats.py PERCELL_CSV NTC_DAPI_H5AD REPORT_DIR OUT_DIR
REPORT_DIR holds ko_measured_vs_predicted_dna_all.csv, compare_ridge_vs_hgb.csv, ko_library_cycle_hgb.csv."""
import os, sys
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PC, NTC, REP, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0); REPS = 2000

K = pd.read_csv(PC, index_col=0); K = K[(K.ok.astype(str) == "True") & K.dna.notna()]
ntc = ad.read_h5ad(NTC, backed="r").obs; rel = ntc["dna_reliable"].astype(str) == "True"
N = pd.DataFrame({"dna": ntc.loc[rel, "dna_rel_R234"].astype(float).values, "guide": ntc.loc[rel, "sgRNA"].astype(str).values})
state = lambda d: (100 * np.mean(d < .25), 100 * np.mean((d >= .25) & (d <= .65)), 100 * np.mean(d > .65))
ref = np.array(state(N.dna.values)); gidx = N.groupby("guide").indices; gl = list(gidx)
print(f"usable knockout cells {len(K):,} in {K.gene.nunique()} knockouts | NTC n={len(N):,}: %2N {ref[0]:.1f} %S {ref[1]:.1f} %4N {ref[2]:.1f}")


def null(n):
    out = []
    for _ in range(REPS):
        RNG.shuffle(gl); ix = []
        for g_ in gl:
            ix.extend(gidx[g_].tolist())
            if len(ix) >= n: break
        out.append(state(N.dna.values[ix[:n]]))
    return np.array(out) - ref


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; return np.minimum(q, 1)


Hg = pd.read_csv(f"{REP}/ko_library_cycle_hgb.csv").set_index("gene")
C = pd.read_csv(f"{REP}/compare_ridge_vs_hgb.csv").set_index("gene")
rows, nulls = [], {}
for g_, d in K.groupby("gene"):
    n = len(d); key = max(int(round(n / 25) * 25), 25)
    if key not in nulls: nulls[key] = null(key)
    s = np.array(state(d.dna.values)) - ref; nd = nulls[key]
    rows.append(dict(gene=g_, set="hgb_only", n_measured=n, measured_d_pct_2N=s[0], measured_d_pct_S=s[1], measured_d_pct_4N=s[2],
                     p_4N=(np.sum(np.abs(nd[:, 2]) >= abs(s[2])) + 1) / (REPS + 1), p_2N=(np.sum(np.abs(nd[:, 0]) >= abs(s[0])) + 1) / (REPS + 1)))
R = pd.DataFrame(rows)
for c_ in ["4N", "2N"]: R[f"q_{c_}_batch"] = bh(R[f"p_{c_}"])
R["significant_batch"] = (R.q_4N_batch < .05) | (R.q_2N_batch < .05)

A = pd.read_csv(f"{REP}/ko_measured_vs_predicted_dna_all.csv")
cols = ["gene", "set", "n_measured", "measured_d_pct_2N", "measured_d_pct_S", "measured_d_pct_4N", "p_4N", "p_2N"]
T = pd.concat([A[cols].assign(batch="earlier 158"), R[cols].assign(batch="HistGB-only 111")], ignore_index=True)
for c_ in ["4N", "2N"]: T[f"q_{c_}_pooled"] = bh(T[f"p_{c_}"])
T["confirmed"] = (T.q_4N_pooled < .05) | (T.q_2N_pooled < .05)
T = T.join(C[["d_pct_4N_pred_ridge", "d_pct_4N_pred_hgb", "sig_ridge", "sig_hgb"]], on="gene")
T["called_by"] = np.select([T.sig_ridge & T.sig_hgb, T.sig_hgb & ~T.sig_ridge, T.sig_ridge & ~T.sig_hgb], ["both", "HistGB only", "ridge only"], "neither")
T["sign_agrees_hgb"] = np.sign(T.d_pct_4N_pred_hgb) == np.sign(T.measured_d_pct_4N)
R = R.merge(T[["gene", "q_4N_pooled", "q_2N_pooled", "confirmed", "d_pct_4N_pred_hgb", "d_pct_4N_pred_ridge"]], on="gene")
R.round(4).to_csv(f"{OUT}/hgb_only_measured_dna.csv", index=False); T.round(4).to_csv(f"{OUT}/all_measured_knockouts.csv", index=False)

print(f"\nthis batch ({len(R)} knockouts): confirmed within-batch FDR {int(R.significant_batch.sum())} | pooled FDR over {len(T)} {int(R.confirmed.sum())}")
cf = R[R.confirmed]
print(f"sign agreement (HistGB predicted vs measured % 4N): all {100 * np.mean(np.sign(R.d_pct_4N_pred_hgb) == np.sign(R.measured_d_pct_4N)):.0f}% | confirmed {100 * np.mean(np.sign(cf.d_pct_4N_pred_hgb) == np.sign(cf.measured_d_pct_4N)):.0f}%")
print(f"within this batch: Spearman HistGB predicted vs measured d%4N {spearmanr(R.d_pct_4N_pred_hgb, R.measured_d_pct_4N)[0]:.2f}, ridge {spearmanr(R.d_pct_4N_pred_ridge, R.measured_d_pct_4N)[0]:.2f}")
print("\nconfirmation by which library run calls the knockout (pooled FDR over all measured):")
S = T.groupby("called_by").agg(n=("gene", "size"), confirmed=("confirmed", "sum"), sign_agrees_hgb=("sign_agrees_hgb", "mean"),
                               median_abs_measured_d4N=("measured_d_pct_4N", lambda x: np.median(np.abs(x))))
S["confirmed_pct"] = 100 * S.confirmed / S.n; print(S.round(2).to_string()); S.round(3).to_csv(f"{OUT}/confirmation_by_model_call.csv")
for sub, lab in [(T, f"all {len(T)}"), (T[T.set.isin(["arrest_list", "calibration"])], "unselected (arrest list + calibration)")]:
    print(f"{lab}: Spearman vs measured d%4N  HistGB {spearmanr(sub.d_pct_4N_pred_hgb, sub.measured_d_pct_4N)[0]:.2f}  ridge {spearmanr(sub.d_pct_4N_pred_ridge, sub.measured_d_pct_4N)[0]:.2f}  n {len(sub)}")
c2 = ["gene", "n_measured", "measured_d_pct_2N", "measured_d_pct_4N", "d_pct_4N_pred_hgb", "q_4N_pooled", "q_2N_pooled"]
print("\nconfirmed in this batch, largest |measured change in % 4N|:\n", cf.reindex(cf.measured_d_pct_4N.abs().sort_values(ascending=False).index)[c2].head(20).round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
for c_, col in [("neither", "0.7"), ("ridge only", "tab:orange"), ("both", "tab:red"), ("HistGB only", "tab:blue")]:
    d_ = T[T.called_by == c_]
    ax[0].scatter(d_.d_pct_4N_pred_hgb, d_.measured_d_pct_4N, s=np.where(d_.confirmed, 34, 10), c=col, label=f"{c_} (n={len(d_)}, {int(d_.confirmed.sum())} confirmed)")
ax[0].axline((0, 0), slope=1, color="k", lw=.6); ax[0].axhline(0, color=".8", lw=.5); ax[0].axvline(0, color=".8", lw=.5)
ax[0].set_xlabel("change in % 4N predicted (gradient boosting)"); ax[0].set_ylabel("change in % 4N, measured DAPI")
ax[0].set_title(f"{len(T)} measured knockouts, by which model calls them"); ax[0].legend(fontsize=8)
COL = {"both": "tab:red", "HistGB only": "tab:blue", "ridge only": "tab:orange", "neither": "0.7"}
Sp = S.reindex([k for k in COL if k in S.index])
ax[1].bar(Sp.index, Sp.confirmed_pct, color=[COL[k] for k in Sp.index])
for i, (c_, r) in enumerate(Sp.iterrows()): ax[1].text(i, r.confirmed_pct + 1, f"{int(r.confirmed)}/{int(r.n)}", ha="center", fontsize=9)
ax[1].set_ylabel("% confirmed by measured DNA (FDR 5%)"); ax[1].set_title("confirmation rate by model call")
plt.tight_layout(); plt.savefig(f"{OUT}/hgb_only_measured_dna.png", dpi=110); print("DONE")
