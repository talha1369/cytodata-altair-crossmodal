#!/usr/bin/env python
"""Do the guides of one knockout differ in measured DNA content? (Louise's guide-level observation, tested on measured DNA.)
Per knockout: cells per guide (20-nt sgRNA from the public atlas obs) in 2N / S / 4N (same cut-offs as ko_dna_measure.py),
statistic = chi-square of the guide x state table. Null: the same number of NTC guides with the same cell counts, drawn from
the NTC table (2,000 draws), so ordinary guide-to-guide and well variation among controls is part of the null. BH over knockouts.
Also: for knockouts confirmed by measured DNA, how many of their guides move % 4N in the knockout's direction.
usage: python ko_guide_dna.py PERCELL_CSV NTC_DAPI_H5AD ALL_MEASURED_CSV OUT_DIR"""
import os, sys
import numpy as np, pandas as pd, anndata as ad

PC, NTC, ALLM, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
ATLAS = ("ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/"
         "all_cells_phase.h5ad")
RNG = np.random.default_rng(0); REPS = 2000; MIN_CELLS = 15


def atlas_guides(genes):
    import h5py, s3fs
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"}, default_block_size=32 * 2**20)
    with fs.open(ATLAS, "rb") as f, h5py.File(f, "r") as h:
        o = h["obs"]; cats = o["experiment"]["categories"][:].astype(str); ci = int(np.where(cats == "ops0144_20260406")[0][0])
        rows = np.where(o["experiment"]["codes"][:] == ci)[0]; lo, hi = rows[0], rows[-1] + 1; idx = rows - lo
        col = lambda n: (o[n]["categories"][:].astype(str)[o[n]["codes"][lo:hi][idx]] if isinstance(o[n], h5py.Group) else o[n][lo:hi][idx])
        d = pd.DataFrame({"gene": pd.Series(col("gene")).str.upper().values, "well": col("well"),
                          "segmentation_id": col("segmentation_id"), "sgRNA": col("sgRNA")})
    d = d[d.gene.isin(genes)]; d["cell_key"] = d.well.astype(str) + "|" + d.segmentation_id.astype("int64").astype(str)
    return d.drop_duplicates("cell_key").set_index("cell_key")


state = lambda x: np.digitize(x, [.25, .65 + 1e-12])  # 0 = 2N (< .25), 1 = S, 2 = 4N (> .65)


def chi2(tab):
    tab = tab[tab.sum(1) > 0]; e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    ok = e > 0; return float((((tab - e) ** 2)[ok] / e[ok]).sum())


K = pd.read_csv(PC, index_col=0); K = K[(K.ok.astype(str) == "True") & K.dna.notna()]
G = atlas_guides(set(K.gene.astype(str).str.upper())); K = K.join(G[["sgRNA"]], how="inner")
print(f"knockout cells with a guide: {len(K):,} in {K.gene.nunique()} knockouts, {K.sgRNA.nunique()} guides")
ntc = ad.read_h5ad(NTC, backed="r").obs; rel = ntc["dna_reliable"].astype(str) == "True"
N = pd.DataFrame({"s": state(ntc.loc[rel, "dna_rel_R234"].astype(float).values), "guide": ntc.loc[rel, "sgRNA"].astype(str).values})
gidx = {g: v for g, v in N.groupby("guide").indices.items()}; gl = np.array(list(gidx)); gsize = np.array([len(gidx[g]) for g in gl])
print(f"NTC: {len(N):,} cells in {len(gl)} guides (median {int(np.median(gsize))} cells per guide)")
M = pd.read_csv(ALLM).set_index("gene")

rows, per_guide = [], []
for g_, d in K.groupby("gene"):
    cnt = d.sgRNA.value_counts(); keep = cnt[cnt >= MIN_CELLS].index; d = d[d.sgRNA.isin(keep)]
    if len(keep) < 2: continue
    tab = np.array([[np.sum(state(x.dna.values) == k) for k in range(3)] for _, x in d.groupby("sgRNA")]); obs = chi2(tab)
    sizes = d.groupby("sgRNA").size().values; nul = np.empty(REPS)
    for r in range(REPS):
        pick = RNG.choice(np.where(gsize >= sizes.max())[0], len(sizes), replace=False)
        nul[r] = chi2(np.array([np.bincount(N.s.values[RNG.choice(gidx[gl[p]], n, replace=False)], minlength=3) for p, n in zip(pick, sizes)]))
    d4 = M.loc[g_, "measured_d_pct_4N"] if g_ in M.index else np.nan
    for s_, x in d.groupby("sgRNA"):
        per_guide.append(dict(gene=g_, sgRNA=s_, n=len(x), pct_2N=100 * np.mean(state(x.dna.values) == 0), pct_4N=100 * np.mean(state(x.dna.values) == 2)))
    rows.append(dict(gene=g_, n_guides=len(keep), n_cells=len(d), min_guide_n=int(sizes.min()), chi2=obs,
                     chi2_ntc_median=float(np.median(nul)), p_guides_differ=(np.sum(nul >= obs) + 1) / (REPS + 1),
                     measured_d_pct_4N=d4, confirmed=bool(M.loc[g_, "confirmed"]) if g_ in M.index else np.nan))
R = pd.DataFrame(rows); P = pd.DataFrame(per_guide)
o = np.argsort(R.p_guides_differ.values); m = len(R); q = np.empty(m)
q[o] = np.minimum.accumulate((R.p_guides_differ.values[o] * m / np.arange(1, m + 1))[::-1])[::-1]; R["q_guides_differ"] = np.minimum(q, 1)
ref4 = 100 * np.mean(N.s.values == 2); P["d_pct_4N"] = P.pct_4N - ref4
P = P.merge(R[["gene", "measured_d_pct_4N", "confirmed"]], on="gene")
P["same_direction"] = np.sign(P.d_pct_4N) == np.sign(P.measured_d_pct_4N)
R.round(4).to_csv(f"{OUT}/guide_heterogeneity_measured_dna.csv", index=False); P.round(3).to_csv(f"{OUT}/per_guide_measured_dna.csv", index=False)

print(f"\n{len(R)} knockouts with >= 2 guides of >= {MIN_CELLS} cells (median {int(R.n_guides.median())} guides, {int(R.min_guide_n.median())} cells in the smallest)")
print(f"guides differ more than NTC guides of the same size: p < 0.05 in {int((R.p_guides_differ < .05).sum())} ({100 * np.mean(R.p_guides_differ < .05):.0f}%, 5% expected), q < 0.05 in {int((R.q_guides_differ < .05).sum())}")
print(f"median chi2 knockout {R.chi2.median():.2f} vs matched NTC {R.chi2_ntc_median.median():.2f}")
for c_, lab in [(True, "confirmed"), (False, "not confirmed")]:
    s = R[R.confirmed == c_]; print(f"  {lab}: n {len(s)}, p < 0.05 in {int((s.p_guides_differ < .05).sum())}")
cf = P[P.confirmed == True]; byg = cf.groupby("gene").same_direction.agg(["sum", "size"])
print(f"confirmed knockouts: {int(cf.same_direction.sum())} of {len(cf)} guides ({100 * cf.same_direction.mean():.0f}%) move % 4N in the knockout's direction; "
      f"all guides agree in {int((byg['sum'] == byg['size']).sum())} of {len(byg)} knockouts")
print("\nlargest guide differences:\n", R.sort_values("p_guides_differ").head(10)[["gene", "n_guides", "n_cells", "chi2", "chi2_ntc_median", "p_guides_differ", "q_guides_differ", "measured_d_pct_4N", "confirmed"]].round(3).to_string(index=False))
