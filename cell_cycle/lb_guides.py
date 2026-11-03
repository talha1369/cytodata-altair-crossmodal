#!/usr/bin/env python
"""Building on Louise's per-guide result and Brian's panels with measured DNA (all atlas cells of 8 genes).
A. Gene level: change in % 2N, % S, % 4N against NTC, null = NTC guides pooled to the same cell count (ko_dna_measure.py).
B. Per guide: % 2N / S / 4N, change against NTC, 95% bootstrap interval.
C. Do a gene's guides differ? chi-square of the guide x DNA-state table, two nulls:
   (i) cells shuffled among the gene's own guides (the usual permutation test; it assumes guides differ only by chance), and
   (ii) sets of NTC guides with the same cell counts (keeps the guide-to-guide spread that control guides also show).
   Calibration: test (i) applied to sets of NTC guides gives its false-positive rate when no gene is targeted.
D. Louise's per-guide stain-score shifts (G1, S, G2/M) against measured d%2N, d%S, d%4N across her 12 guides (her stain score
   uses DNA as an input, so this checks consistency, not independence).
usage: python lb_guides.py PERCELL_CSV CELLS_CSV NTC_DAPI_H5AD LOUISE_GUIDE_SECTOR_CSV OUT_DIR"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import spearmanr

PC, CELLS, NTC, LG, OUT = sys.argv[1:6]; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0); REPS = 2000
st = lambda x: np.digitize(x, [.25, .65 + 1e-12])            # 0 = 2N (< .25), 1 = S, 2 = 4N (> .65)
pct = lambda s: 100 * np.bincount(s, minlength=3) / max(len(s), 1)

K = pd.read_csv(PC, index_col=0); K = K[(K.ok.astype(str) == "True") & K.dna.notna()]
C = pd.read_csv(CELLS).set_index("cell_key"); K = K.join(C[["sgRNA"]], how="inner"); K["s"] = st(K.dna.values)
o = ad.read_h5ad(NTC, backed="r").obs; rel = o["dna_reliable"].astype(str) == "True"
N = pd.DataFrame({"s": st(o.loc[rel, "dna_rel_R234"].astype(float).values), "guide": o.loc[rel, "sgRNA"].astype(str).values})
gidx = N.groupby("guide").indices; gl = np.array(list(gidx)); gsz = np.array([len(gidx[g]) for g in gl]); ref = pct(N.s.values)
print(f"knockout cells usable {len(K):,} | NTC {len(N):,} cells, {len(gl)} guides | NTC %2N {ref[0]:.1f} %S {ref[1]:.1f} %4N {ref[2]:.1f}")


def chi2(tab):
    tab = tab[tab.sum(1) > 0]; tab = tab[:, tab.sum(0) > 0]; e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    return float(((tab - e) ** 2 / e).sum())


def perm_p(states, labels, reps=REPS):
    labs = np.unique(labels); obs = chi2(np.array([np.bincount(states[labels == l], minlength=3) for l in labs])); cnt = 0
    for _ in range(reps):
        p = RNG.permutation(labels); cnt += chi2(np.array([np.bincount(states[p == l], minlength=3) for l in labs])) >= obs
    return obs, (cnt + 1) / (reps + 1)


def ntc_sets(sizes, reps=REPS):
    ok = np.where(gsz >= max(sizes))[0]; out = []
    for _ in range(reps):
        pick = RNG.choice(ok, len(sizes), replace=False)
        out.append([RNG.choice(gidx[gl[p]], n, replace=False) for p, n in zip(pick, sizes)])
    return out


# A. gene level
A = []
for g_, d in K.groupby("gene"):
    n = len(d); s = pct(d.s.values) - ref; nul = []
    for _ in range(REPS):
        RNG.shuffle(gl_ := gl.copy()); ix = []
        for x in gl_:
            ix.extend(gidx[x].tolist())
            if len(ix) >= n: break
        nul.append(pct(N.s.values[ix[:n]]) - ref)
    nul = np.array(nul); p = [(np.sum(np.abs(nul[:, k]) >= abs(s[k])) + 1) / (REPS + 1) for k in range(3)]
    A.append(dict(gene=g_, n=n, d_pct_2N=s[0], d_pct_S=s[1], d_pct_4N=s[2], p_2N=p[0], p_S=p[1], p_4N=p[2]))
A = pd.DataFrame(A); A.round(4).to_csv(f"{OUT}/gene_level_all_cells.csv", index=False)
print("\nA. gene level, all atlas cells:\n", A.round(3).to_string(index=False))

# B. per guide
B = []
for (g_, sg), d in K.groupby(["gene", "sgRNA"]):
    v = d.s.values; s = pct(v) - ref; bs = np.array([pct(RNG.choice(v, len(v))) - ref for _ in range(1000)])
    B.append(dict(gene=g_, sgRNA=sg, short=sg[:8], n=len(v), d_pct_2N=s[0], d_pct_S=s[1], d_pct_4N=s[2],
                  ci4_lo=np.percentile(bs[:, 2], 2.5), ci4_hi=np.percentile(bs[:, 2], 97.5), ci2_lo=np.percentile(bs[:, 0], 2.5), ci2_hi=np.percentile(bs[:, 0], 97.5)))
B = pd.DataFrame(B); B.round(3).to_csv(f"{OUT}/per_guide_all_cells.csv", index=False)
print("\nB. per guide:\n", B[["gene", "short", "n", "d_pct_2N", "d_pct_S", "d_pct_4N", "ci4_lo", "ci4_hi"]].round(1).to_string(index=False))

# C. guide heterogeneity, two nulls, plus calibration of the permutation test on NTC guide sets
Cr = []
for g_, d in K.groupby("gene"):
    cnt = d.sgRNA.value_counts(); keep = cnt[cnt >= 15].index; d = d[d.sgRNA.isin(keep)]
    obs, p_perm = perm_p(d.s.values, d.sgRNA.values)
    sizes = d.groupby("sgRNA").size().values; nul = [chi2(np.array([np.bincount(N.s.values[ix], minlength=3) for ix in S_])) for S_ in ntc_sets(sizes)]
    Cr.append(dict(gene=g_, n_guides=len(keep), n=len(d), chi2=obs, p_cell_permutation=p_perm, p_vs_ntc_guides=(np.sum(np.array(nul) >= obs) + 1) / (REPS + 1),
                   chi2_ntc_guides_median=float(np.median(nul))))
Cr = pd.DataFrame(Cr); Cr.round(4).to_csv(f"{OUT}/guide_heterogeneity_two_nulls.csv", index=False)
print("\nC. do a gene's guides differ in measured DNA?\n", Cr.round(3).to_string(index=False))
cal = []
for S_ in ntc_sets([110, 110, 110, 110], reps=200):
    lab = np.concatenate([[i] * len(ix) for i, ix in enumerate(S_)]); sv = np.concatenate([N.s.values[ix] for ix in S_])
    cal.append(perm_p(sv, lab, reps=500)[1])
cal = np.array(cal); pd.Series(cal).to_csv(f"{OUT}/permutation_test_on_ntc_guide_sets.csv", index=False)
print(f"calibration: cell-permutation test on 200 sets of 4 NTC guides (110 cells each): p < 0.05 in {100 * np.mean(cal < .05):.0f}% (5% if calibrated)")

# D. Louise's per-guide stain-score shifts against measured DNA
L = pd.read_csv(LG); L = L.pivot_table(index="guide", columns="sector", values="delta")
L.index = L.index.str.split("|").str[1]; M = B.set_index("sgRNA").join(L, how="inner")
for a, b in [("G1", "d_pct_2N"), ("S", "d_pct_S"), ("G2/M", "d_pct_4N")]:
    if a in M:
        r, p = spearmanr(M[a], M[b]); print(f"D. Louise {a} shift vs measured {b} across {len(M)} guides: Spearman {r:.2f} (p {p:.3f})")
M.round(3).to_csv(f"{OUT}/louise_guides_vs_measured_dna.csv")
print("DONE")
