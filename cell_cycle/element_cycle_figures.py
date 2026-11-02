#!/usr/bin/env python
"""Figures for notebook 20 (Element: the RNA cell-cycle clock mapped to morphology). Light: reads only the CSVs written by
element_cycle_morphology.py (A-C), element_part_d.py (D) and element_profiles.py (B2 per-bin curves, optional).
usage: python element_cycle_figures.py [IN_DIR] [OUT_DIR]"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import seaborn as sns; sns.set_context("poster"); sns.set_style("ticks")
except Exception:
    pass
def save(path):
    w = plt.gcf().get_size_inches()[0]; plt.savefig(path, dpi=min(150, 1950 / w)); plt.close()


plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})

IN = sys.argv[1] if len(sys.argv) > 1 else "element_cycle"
OUT = sys.argv[2] if len(sys.argv) > 2 else IN
os.makedirs(OUT, exist_ok=True)
rd = lambda n: pd.read_csv(f"{IN}/{n}")
INK, MUTED = "#0b0b0b", "#8a8984"
# fixed family -> colour (reference categorical order); the IgG negative control is always grey
FAM = {"DNA stain (Nucleus)": "#2a78d6", "antibodies (ERK, pERK, pHSP27, p-p38)": "#eb6834", "shape and size": "#1baf7a",
       "Cell membrane": "#eda100", "Mitochondria": "#e87ba4", "Actin": "#6250d6", "IgG isotype controls (negative control)": "#9a9994"}
PHASE = {"G1": "#2a78d6", "S": "#1baf7a", "G2M": "#e34948"}
summary = {}

# ---------------------------------------------------------------- (i) error predicting the RNA clock, by feature family
A = rd("A_prediction_by_feature_family.csv")
null = float(A.loc[A.features == "shuffled clock (null)", "median_error_deg"].iloc[0])
a = A[A.features != "shuffled clock (null)"].sort_values("median_error_deg").reset_index(drop=True)
def a_col(f):
    for k, c in FAM.items():
        if f == f"only {k}": return c
    return "#52514e"
fig, ax = plt.subplots(figsize=(11, 6.2))
y = np.arange(len(a))
ax.barh(y, a.median_error_deg, color=[a_col(f) for f in a.features], height=.7)
for i, r in a.iterrows():
    ax.text(r.median_error_deg + .8, i, f"{r.median_error_deg:.1f}  (n={r.n_features})", va="center", fontsize=11, color=INK)
ax.set_yticks(y); ax.set_yticklabels([f.replace("only ", "only: ") for f in a.features], fontsize=12); ax.invert_yaxis()
ax.axvline(null, color=INK, ls="--", lw=1.5); ax.text(null + 1, 1, f"shuffled\nclock\n{null:.1f} deg", ha="left", va="center", fontsize=11)
ax.axvline(30, color="#e34948", ls=":", lw=2); ax.text(30.6, -.95, "OPS0077 phase contrast reads the 4i clock to 30 deg", ha="left", va="center", fontsize=10, color="#b02a29")
igg = float(a.loc[a.features.str.contains("IgG"), "median_error_deg"].iloc[0])
ax.axvspan(igg - 1.5, igg + 1.5, color="#9a9994", alpha=.15, lw=0)
ax.set_xlim(0, 92); ax.set_ylim(len(a) - .4, -1.4); ax.set_xlabel("median error predicting the RNA clock (degrees), held-out wells")
ax.set_title("Which morphology reads the Element RNA cell-cycle clock\n(95,868 control cells, leave-one-well-out over 12 wells; n = features used)", fontsize=14)
plt.tight_layout(); save(f"{OUT}/fig_A_error_by_family.png")

# ---------------------------------------------------------------- (ii) error by arc of the RNA clock (+ bin composition)
arc = rd("A_error_by_clock_arc.csv")
comp = pd.read_csv(f"{IN}/B2_bin_composition.csv") if os.path.exists(f"{IN}/B2_bin_composition.csv") else None
nrow = 2 if comp is not None else 1
fig, axs = plt.subplots(nrow, 1, figsize=(11, 4.6 * nrow), sharex=True, squeeze=False); axs = axs[:, 0]
ax = axs[0]
ax.bar(arc.arc_deg, arc["median"], width=26, color=[PHASE.get(p, MUTED) for p in arc.majority_scanpy_phase])
for _, r in arc.iterrows():
    ax.text(r.arc_deg, r["median"] + 2, f"n={int(r['size']):,}", ha="center", fontsize=9, color="#52514e", rotation=90, va="bottom")
ax.axhline(null, color=INK, ls="--", lw=1.5); ax.text(358, null + 2, "shuffled", ha="right", fontsize=10)
ax.axhline(30, color="#e34948", ls=":", lw=2)
ax.set_ylabel("median error (deg)"); ax.set_ylim(0, 185)
ax.set_title("Where on the RNA clock imaging fails (all imaging features, held-out wells)\nbar colour = majority scanpy phase of the arc", fontsize=14)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=k) for k, c in PHASE.items()], title="scanpy phase", loc="upper left", frameon=False)
if comp is not None:
    ax = axs[1]
    for q, c in PHASE.items(): ax.plot(comp.bin_deg, comp[f"pct_{q}"], color=c, marker="o", ms=6, label=f"% scanpy {q}")
    ax.plot(comp.bin_deg, comp.pct_4N, color=INK, marker="s", ms=6, ls="-", label="% above 4N threshold (DNA stain)")
    ax.set_ylabel("% of cells in the arc"); ax.set_ylim(0, 100); ax.legend(frameon=False, ncol=2, fontsize=10, loc="upper right")
    ax.set_title(f"What the arcs contain (baseline controls, n={comp.n_cells.sum():,} cells)", fontsize=14)
axs[-1].set_xlabel("Element RNA clock, cycle_position_deg (12 arcs of 30 deg)"); axs[-1].set_xticks(np.arange(0, 361, 30))
plt.tight_layout(); save(f"{OUT}/fig_A_error_by_arc.png")

# ---------------------------------------------------------------- (iii) cycle phenotypes by family, and curves around the clock
B = rd("B_feature_cycle_profiles.csv"); Bf = rd("B_family_summary.csv")
order = list(Bf.sort_values("median_amplitude_SD", ascending=False).family)
fig, axs = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
rng = np.random.default_rng(0)
for i, f in enumerate(order):
    d = B[B.family == f]; rep = (d.amplitude_SD > .5) & (d.split_half_r > .9)
    ax.scatter(d.amplitude_SD, i + rng.uniform(-.28, .28, len(d)), s=14, color=np.where(rep, FAM[f], "#d8d7d2"), lw=0)
    ax.plot([d.amplitude_SD.median()] * 2, [i - .35, i + .35], color=INK, lw=2.5)
    ax.text(1.42, i, f"{int(rep.sum())}/{len(d)}", va="center", fontsize=11)
ax.text(1.42, -.85, "large and\nreproducible", fontsize=10, va="center")
ax.axvline(.5, color=MUTED, ls=":", lw=1.2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([f.replace(" (negative control)", "\n(negative control)") for f in order], fontsize=11)
ax.set_ylim(len(order) - .5, -1.3); ax.set_xlim(0, 1.6)
ax.set_xlabel("amplitude around the RNA clock (SD units)")
ax.set_title("Size of the cycle phenotype per feature\n(coloured = amplitude > 0.5 SD and split-half r > 0.9; bar = median)", fontsize=13)
ax = axs[1]
top = B[(B.amplitude_SD > .5) & (B.split_half_r > .9)]
for f in order:
    d = top[top.family == f]
    if len(d): ax.scatter(d.peak_deg + rng.uniform(-9, 9, len(d)), d.amplitude_SD, s=22, color=FAM[f], lw=0, label=f"{f.split(' (')[0]} ({len(d)})", alpha=.85)
ax.set_xticks(np.arange(0, 361, 60)); ax.set_xlim(0, 360)
ax.set_xlabel("RNA-clock arc where the feature peaks (deg)"); ax.set_ylabel("amplitude (SD units)")
ax.set_title(f"Where the {len(top)} large, reproducible\nfeatures peak", fontsize=13)
ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(1.01, 1), title="family (n features)")
plt.tight_layout(); save(f"{OUT}/fig_B_family_amplitude.png")

if os.path.exists(f"{IN}/B2_profiles_by_bin.csv"):
    P = rd("B2_profiles_by_bin.csv")
    W = P.pivot(index="feature", columns="bin_deg", values="mean_SD")
    t30 = [f for f in B[B.split_half_r > .9].head(30).feature if f in W.index]
    H = W.loc[t30]; H = H.iloc[np.lexsort((-H.max(1).values, H.values.argmax(1)))]
    fig, axs = plt.subplots(1, 2, figsize=(18, 8.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    im = ax.imshow(H.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, extent=[0, 360, len(H), 0])
    fam_of = P.drop_duplicates("feature").set_index("feature").family
    ax.set_yticks(np.arange(len(H)) + .5); ax.set_yticklabels([s[:46] for s in H.index], fontsize=8.5)
    for t, f in zip(ax.get_yticklabels(), H.index): t.set_color(FAM.get(fam_of[f], INK))
    ax.set_xticks(np.arange(0, 361, 60)); ax.set_xlabel("Element RNA clock (deg)")
    ax.set_title("Top 30 reproducible cycle features (mean, SD units)\nlabel colour = family", fontsize=13)
    cb = plt.colorbar(im, ax=ax, fraction=.035); cb.set_label("SD from well median")
    ax = axs[1]
    reps = []
    for f in order:                               # strongest feature per family (IgG included as the negative control)
        d = B[B.family == f].sort_values("amplitude_SD", ascending=False)
        d = d[d.feature.isin(W.index)]
        if len(d): reps.append((f, d.feature.iloc[0]))
    for f, ft in reps:
        r = P[P.feature == ft].sort_values("bin_deg")
        ax.plot(r.bin_deg, r.mean_SD, color=FAM[f], marker="o", ms=5, label=f"{f.split(' (')[0]}: {ft[:38]}")
        ax.fill_between(r.bin_deg, r[["half1_SD", "half2_SD"]].min(1), r[["half1_SD", "half2_SD"]].max(1), color=FAM[f], alpha=.15, lw=0)
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(np.arange(0, 361, 60)); ax.set_xlim(0, 360)
    if comp is not None:
        lo = comp.loc[comp.n_cells < comp.n_cells.median() * .5, "bin_deg"]
        for b in lo: ax.axvspan(b - 15, b + 15, color="#d8d7d2", alpha=.35, lw=0)
    ax.set_xlabel("Element RNA clock (deg)"); ax.set_ylabel("mean (SD units)")
    ax.set_title("Strongest feature of each family\n(band = the two halves of the wells; grey arcs = sparse arcs)", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -.13), ncol=1)
    plt.tight_layout(); save(f"{OUT}/fig_B_profiles.png")

# ---------------------------------------------------------------- (iv) perturbations, measured vs predicted
C = rd("C_perturbation_measured_vs_predicted.csv")
C["cond"] = np.where(C.contrast.str.startswith("TNFa vs"), "TNFa vs baseline", np.where(C.contrast.str.endswith("TNFa"), "KO, TNFa", "KO, baseline"))
CC = {"KO, baseline": "#2a78d6", "KO, TNFa": "#eb6834", "TNFa vs baseline": "#1baf7a"}
rho_all, p_all = spearmanr(C.measured_d_pct_G2M_arc, C.predicted_d_pct_G2M_arc)
rho_tv, p_tv = spearmanr(C.measured_TV, C.predicted_TV)
ko = C[C.cond != "TNFa vs baseline"]
sign = float(np.mean(np.sign(C.measured_d_pct_G2M_arc) == np.sign(C.predicted_d_pct_G2M_arc)))
big = C[C.measured_d_pct_G2M_arc.abs() >= 1]
sign_big = float(np.mean(np.sign(big.measured_d_pct_G2M_arc) == np.sign(big.predicted_d_pct_G2M_arc)))
summary.update(C_p_dG2M=round(p_all, 3), C_p_TV=round(p_tv, 3), C_n_contrasts=len(C), C_rho_dG2M=round(rho_all, 2), C_rho_dG2M_KO_only=round(spearmanr(ko.measured_d_pct_G2M_arc, ko.predicted_d_pct_G2M_arc)[0], 2),
               C_rho_TV=round(rho_tv, 2), C_same_sign=round(sign, 2), C_n_measured_ge1pt=len(big), C_same_sign_measured_ge1pt=round(sign_big, 2),
               C_median_bin_shift_corr=round(float(C.bin_shift_corr.median()), 2))
fig, axs = plt.subplots(1, 2, figsize=(16, 6.8))
lab = lambda s: s.replace(" vs controls", "").replace(" (controls)", "").replace("KO, ", "KO ")
for ax, xm, ym, ttl in [(axs[0], "measured_d_pct_G2M_arc", "predicted_d_pct_G2M_arc", f"Change in % cells on the G2/M-rich arc (30-150 deg)\nSpearman rho {rho_all:.2f} over {len(C)} contrasts"),
                        (axs[1], "measured_TV", "predicted_TV", f"Total change of the clock distribution (TV distance)\nSpearman rho {rho_tv:.2f} over {len(C)} contrasts")]:
    for k, c in CC.items():
        d = C[C.cond == k]; ax.scatter(d[xm], d[ym], s=40 + d.n / 300, color=c, label=k, edgecolor="white", lw=1.5, zorder=3)
    for _, r in C.iterrows(): ax.annotate(lab(r.contrast), (r[xm], r[ym]), fontsize=9, xytext=(5, 3), textcoords="offset points", color="#52514e")
    lim = [min(C[xm].min(), C[ym].min()) * 1.15 - .005, max(C[xm].max(), C[ym].max()) * 1.15 + .005]
    ax.plot(lim, lim, color=INK, lw=.8, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    if xm.endswith("arc"): ax.axhline(0, color=MUTED, lw=.6); ax.axvline(0, color=MUTED, lw=.6)
    ax.set_xlabel("measured with the RNA clock"); ax.set_ylabel("predicted from morphology (held-out well)"); ax.set_title(ttl, fontsize=13)
axs[0].set_xlabel("measured with the RNA clock (percentage points)"); axs[0].legend(frameon=False, loc="upper left", fontsize=10)
plt.tight_layout(); save(f"{OUT}/fig_C_perturbations.png")

# ---------------------------------------------------------------- (v) OPS0077 correlates
D = rd("D_shared_feature_state_effects.csv")
KC = {"DNA-stain texture": "#2a78d6", "DNA-stain granularity": "#1baf7a", "cell shape (scale-free)": "#eb6834"}
rows = []
for k, d in [("all shared", D)] + list(D.groupby("kind")):
    for s in ["S", "4N"]:
        e, o = d[f"element_{s}_vs_2N"], d[f"ops_{s}_vs_2N"]
        rows.append(dict(kind=k, contrast=f"{s} vs 2N", n_features=len(d), spearman_rho=round(spearmanr(e, o)[0], 2) if len(d) > 3 else np.nan,
                         spearman_p=float(f"{spearmanr(e, o)[1]:.2g}") if len(d) > 3 else np.nan,
                         pct_same_sign=round(100 * float(np.mean(np.sign(e) == np.sign(o))), 0),
                         median_abs_d_element=round(float(e.abs().median()), 2), median_abs_d_ops=round(float(o.abs().median()), 2)))
Ds = pd.DataFrame(rows); Ds.to_csv(f"{OUT}/D_summary_by_kind.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(15, 6.4))
for ax, s in zip(axs, ["4N", "S"]):
    for k, c in KC.items():
        d = D[D.kind == k]; ax.scatter(d[f"element_{s}_vs_2N"], d[f"ops_{s}_vs_2N"], s=50, color=c, edgecolor="white", lw=1.2, label=f"{k} (n={len(d)})", zorder=3)
    r_ = Ds[(Ds.kind == "all shared") & (Ds.contrast == f"{s} vs 2N")].iloc[0]
    ax.axhline(0, color=MUTED, lw=.6); ax.axvline(0, color=MUTED, lw=.6); ax.plot([-2, 2], [-2, 2], color=INK, lw=.8, ls="--")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal")
    ax.set_xlabel("Element: effect size d (DNA-stain imaging)"); ax.set_ylabel("OPS0077: effect size d (DAPI / 4i segmentation)")
    ax.set_title(f"{s} vs 2N: Spearman rho {r_.spearman_rho:.2f}, same sign {r_.pct_same_sign:.0f}%\n({r_.n_features} shared features)", fontsize=13)
axs[0].legend(frameon=False, loc="upper left", fontsize=10)
plt.tight_layout(); save(f"{OUT}/fig_D_shared_features.png")

Pp = pd.read_csv(f"{IN}/D_ops0077_phase_features_by_dna_state.csv", index_col=0)
grp = Pp.index.str.replace(r"^op_", "", regex=True)
def pgroup(n):
    t = n.split("_")
    if t[0] in ("phase2d", "focus3d"): return f"{t[1]} objects ({t[0]})" if t[1] in ("tubular", "vesicular") else f"{t[0]} {t[1]}"
    if t[0] in ("nucleoli", "network"): return f"{t[0]} ({t[1]})"
    if t[0] == "nuclei": return "nucleus shape/size" if t[1] not in ("intensity",) else "nucleus phase intensity"
    if t[0] == "cell": return "cell shape/size"
    return t[0]
Pp["group"] = [pgroup(n) for n in grp]
Pg = Pp.groupby("group").agg(n_features=("abs_4N", "size"), median_abs_d_4N=("abs_4N", "median"), max_abs_d_4N=("abs_4N", "max"),
                             n_abs_d_4N_ge_0_8=("abs_4N", lambda s: int((s >= .8).sum()))).sort_values("max_abs_d_4N", ascending=False).round(3)
Pg.to_csv(f"{OUT}/D_phase_family_summary.csv")
cell_area = Pp.loc[[i for i in Pp.index if i in ("op_cell_area", "op_nuclei_area_mean", "op_nuclei_area_max")], ["S_vs_2N", "4N_vs_2N"]]
summary.update(D_phase_n_features=len(Pp), D_phase_n_abs_d4N_ge_0_8=int((Pp.abs_4N >= .8).sum()), D_phase_n_abs_d4N_ge_0_5=int((Pp.abs_4N >= .5).sum()),
               D_phase_top_feature=Pp.abs_4N.idxmax(), D_phase_top_d4N=float(Pp.loc[Pp.abs_4N.idxmax(), "4N_vs_2N"]), D_phase_n_nan=int(Pp.abs_4N.isna().sum()),
               **{f"D_phase_{k}_d4N": float(v) for k, v in cell_area["4N_vs_2N"].items()})
fig, axs = plt.subplots(1, 2, figsize=(17, 7.4), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]
t20 = Pp.sort_values("abs_4N", ascending=False).head(20)
extra = Pp.loc[[i for i in ["op_cell_area", "op_nuclei_area_mean"] if i in Pp.index and i not in t20.index]]
t = pd.concat([t20, extra]); yy = np.arange(len(t))
ax.barh(yy + .2, t["4N_vs_2N"], height=.4, color="#e34948", label="4N vs 2N")
ax.barh(yy - .2, t["S_vs_2N"], height=.4, color="#1baf7a", label="S vs 2N")
ax.set_yticks(yy); ax.set_yticklabels([s.replace("op_", "")[:44] for s in t.index], fontsize=9); ax.invert_yaxis()
if len(extra): ax.axhline(len(t20) - .5, color=MUTED, lw=.8, ls=":"); ax.text(1.72, len(t20) - .55, "below: size, for reference", fontsize=9, ha="right", va="bottom")
ax.set_xlabel("effect size d vs 2N (OPS0077 NTC cells)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.11), ncol=2)
ax.set_title("OPS0077 phase-contrast features that mark the DNA states\n(top 20 by |d| for 4N vs 2N)", fontsize=13)
ax = axs[1]
g = Pg.head(14).iloc[::-1]
ax.barh(np.arange(len(g)), g.max_abs_d_4N, color="#2a78d6", height=.65, label="max |d| in group")
ax.scatter(g.median_abs_d_4N, np.arange(len(g)), color=INK, zorder=3, s=30, label="median |d| in group")
ax.set_yticks(np.arange(len(g))); ax.set_yticklabels([f"{i} (n={int(r.n_features)})" for i, r in g.iterrows()], fontsize=10)
ax.set_xlabel("|d|, 4N vs 2N"); ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_title(f"By phase-feature group ({len(Pp):,} features)", fontsize=13)
plt.tight_layout(); save(f"{OUT}/fig_D_phase_markers.png")

pd.Series(summary).to_csv(f"{OUT}/figures_summary.csv", header=["value"])
print(pd.Series(summary).to_string()); print(Ds.to_string(index=False)); print(Pg.to_string())
print("DONE")
