#!/usr/bin/env python
"""Figure fixes for stream C (reads saved CSVs only, light). Writes new files in phase_push_check/; the originals are left as they are.
  phase_per_pc_by_covariate_v2.png : NTC bars per covariate + knockouts pooled with outlier cells removed (the original purple
                                     bar was 0 because a few extreme knockout cells dominated the variance)
  phase_knockout_distance_v2.png   : (a) noise-corrected centroid distance before vs after, log axes, outlier cells removed;
                                     (b) % change of centroid distance: NTC 200-cell null vs other vs cycle-shifting knockouts
usage: python phase_push_figs_v2.py"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "phase_push_check"
N = pd.read_csv("phase_push/ntc_per_pc_reduction.csv"); P = pd.read_csv(f"{D}/pooled_per_pc_reduction_checked.csv")
G = pd.read_csv(f"{D}/knockouts_distance_checked.csv"); Z = pd.read_csv(f"{D}/ntc_null_200cell_draws.csv")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. per-PC bars
kp = P[(P.cells == "knockouts pooled, outlier cells removed") & (P.measure == "variance")].iloc[0]
series = [(r.covariate + " [NTC]", [r[f"PC{i+1}"] for i in range(10)]) for _, r in N.iterrows()]
series.append((f"knockouts pooled, Brian's predicted cycle, NTC coefficients (n={int(kp.n_cells):,} cells)", [kp[f"PC{i+1}"] for i in range(10)]))
cols = ["#c0392b", "#2c7fb8", "#2ca25f", "#e6852b", "#9e9e9e", "#7b4fa0"]
fig, ax = plt.subplots(figsize=(14, 5.4)); x = np.arange(10); w = .14
for k, ((lab, v), c) in enumerate(zip(series, cols)):
    ax.bar(x + (k - 2.5) * w, v, w * .9, label=lab, color=c)
ax.set_xticks(x); ax.set_xticklabels([f"PC{i+1}" for i in range(10)]); ax.axhline(0, color="k", lw=.6)
ax.set_ylabel("% of within-field variance removed")
ax.set_title(f"Phase contrast: how much of each top PC is cell cycle (NTC n={35633:,} cells; PCA fit on NTC)")
ax.legend(fontsize=8.5, ncol=2, frameon=False, loc="upper left"); ax.grid(axis="y", lw=.4, alpha=.5)
plt.tight_layout(); plt.savefig(f"{D}/phase_per_pc_by_covariate_v2.png", dpi=130); plt.close()

# 2. distance figure
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.1]})
a = ax[0]; H = G[G.corrected_dist_before > 0]
for flag, c, lab in [(False, "#9e9e9e", "other knockouts"), (True, "#c0392b", "cycle-shifting knockouts (q_cycle < 0.05)")]:
    d = H[H.cycle_shifting == flag]
    a.scatter(d.corrected_dist_before, d.corrected_dist_after.clip(lower=.05), s=26 if flag else 12, c=c, alpha=.85, label=f"{lab}, n={len(d)}",
              edgecolors="white", linewidths=.4)
lim = [.05, H[["corrected_dist_before", "corrected_dist_after"]].max().max() * 1.3]
a.plot(lim, lim, "k--", lw=.8); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(.3, lim[1]); a.set_ylim(.05, lim[1])
offs = [(-10, -26), (8, -14), (16, -30), (30, -6), (24, -22)]
for (_, r), o in zip(H[H.corrected_dist_before > 2 * H.noise_floor_before].nsmallest(5, "corrected_dist_change_pct").iterrows(), offs):
    a.annotate(r.gene, (r.corrected_dist_before, r.corrected_dist_after), fontsize=8, xytext=o, textcoords="offset points",
               arrowprops=dict(arrowstyle="-", lw=.5, color="0.3"))
n_clip = int((H.corrected_dist_after < .05).sum())
if n_clip: a.text(.98, .02, f"{n_clip} knockouts with after-distance below 0.05 drawn at 0.05", transform=a.transAxes, ha="right", fontsize=8, color="0.3")
a.set_xlabel("knockout centroid distance from NTC (PC1-10), raw\n(noise corrected; outlier cells removed)")
a.set_ylabel("same distance after removing the cycle"); a.legend(fontsize=8.5, frameon=False, loc="upper left")
a.set_title(f"(a) below the dashed line = pulled toward NTC\n({len(H)} of {len(G)} knockouts sit above the sampling-noise floor)", fontsize=10.5)
b = ax[1]
groups = [("NTC null:\nrandom 200-cell\nNTC sets", Z.dist_change_pct, "#bdbdbd"),
          ("other\nknockouts", G[~G.cycle_shifting].dist_change_pct, "#9e9e9e"),
          ("cycle-shifting\nknockouts", G[G.cycle_shifting].dist_change_pct, "#c0392b")]
bp = b.boxplot([g[1] for g in groups], widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.5))
for patch, g in zip(bp["boxes"], groups): patch.set_facecolor(g[2]); patch.set_alpha(.75)
for i, g in enumerate(groups):
    jit = np.random.default_rng(0).normal(0, .06, len(g[1])) if len(g[1]) < 1500 else None
    if jit is not None: b.scatter(i + 1 + jit, g[1], s=5, c="k", alpha=.25)
    b.text(i + 1, b.get_ylim()[1] if False else np.percentile(g[1], 99) + 2, f"n={len(g[1])}\nmedian {np.median(g[1]):.1f}%", ha="center", fontsize=8.5)
b.set_xticks([1, 2, 3]); b.set_xticklabels([g[0] for g in groups]); b.axhline(0, color="k", lw=.6)
b.set_ylabel("% change in centroid distance to NTC (PC1-10)\nafter removing the cycle (negative = closer)")
b.set_title("(b) raw distance change, compared with the shrinkage expected\nfrom less noise alone (NTC null)", fontsize=10.5)
b.grid(axis="y", lw=.4, alpha=.5)
plt.tight_layout(); plt.savefig(f"{D}/phase_knockout_distance_v2.png", dpi=130); plt.close()
print("DONE")
