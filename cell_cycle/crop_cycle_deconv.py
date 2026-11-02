#!/usr/bin/env python
"""A cell-cycle representation for every CROP-seq knockout (Altair, perturbation-level RNA), learned from single cells.

The Altair CROP-seq data exist only per knockout (DESeq2 KO-vs-control statistics, 1,000 KOs x 9,999 genes). A pseudobulk
fold change mixes two things: a change in how cells are distributed around the cycle, and changes in the cells. Element
(A549, single-cell RNA with a measured RNA clock) tells us how each gene's expression varies around the cycle, so the cycle
part can be deconvolved:

  mu_gb   mean CP10k of gene g in cycle bin b (Element baseline control cells, 12 bins of cycle_position_deg)
  pi_b    fraction of control cells in bin b
  KO distribution  pi'_b = pi_b * (1 + sum_k c_k (h_k(theta_b) - hbar_k)),  h = cos, sin, cos2, sin2 (a smooth tilt)
  linearised:      LFC_g ~ a + sum_k c_k D_gk,   D_gk = sum_b pi_b (h_k(theta_b) - hbar_k) mu_gb / m_g / ln 2

fitted by weighted least squares over strongly cycling genes (weights 1 / (lfcSE^2 + tau^2)), excluding the KO's own target
gene. Representation per KO: c (4 numbers), the implied distribution pi', and readable summaries from Element's per-bin
composition: change in % G1 / S / G2M (scanpy phase calls) and in % 4N DNA (Element imaging).
Null: the cycle profiles are permuted across the cycling genes (breaks the gene-profile link, keeps the fold changes).
Reliability: c fitted on two disjoint halves of the cycling genes must agree across KOs.
usage: python crop_cycle_deconv.py [OUT_DIR]
"""
import os, sys
import numpy as np, pandas as pd, h5py, s3fs
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem

OUT = sys.argv[1] if len(sys.argv) > 1 else "crop_cycle"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
NB = 12; TH = 2 * np.pi * (np.arange(NB) + .5) / NB
H = np.c_[np.cos(TH), np.sin(TH), np.cos(2 * TH), np.sin(2 * TH)]

# ------------------------------------------------ CROP-seq (perturbation level)
P = "ops-explorer-public/leonetti_ops/ops_data_portal_submission/CROPseq/2026_05_28_CropSeq_June2025_perturbation_lvl_ops_8-24update.h5ad"
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
with fs.open(P, "rb") as f, h5py.File(f, "r") as c:
    cv = read_elem(c["var"]); co = read_elem(c["obs"])
    stat = c["layers"]["stat"][:].astype(float); se = c["layers"]["lfcSE"][:].astype(float); base = c["layers"]["baseMean"][:].astype(float)
lfc = stat * se                                         # DESeq2 Wald stat = LFC / SE
ens = np.array([e.split(".")[0] for e in cv.index.astype(str)])
ko_gene = co["gene_symbol"].astype(str).str.upper().values; role = co["role"].astype(str).values
print(f"CROP-seq: {lfc.shape[0]} rows ({(role == 'targeting').sum()} targeting) x {lfc.shape[1]} genes", flush=True)

# ------------------------------------------------ Element single cells: per-bin expression of the CROP-seq genes
h = h5py.File("/srv/team/data/multiomics/cytos_multiomics_cycle_v1.h5mu", "r")
obs = read_elem(h["obs"]); rv = read_elem(h["mod"]["rna"]["var"])
r_ens = np.array([e.split(".")[0] for e in rv["gene_id"].astype(str)])
col_of = {e: i for i, e in enumerate(r_ens)}
shared = np.array([e in col_of for e in ens]); gcols = np.array([col_of[e] for e in ens[shared]])
ctl = (obs["target_gene"].astype(str).isin(["NTC", "OR1I1"]) & (obs["condition"].astype(str) == "baseline")).values
ang = obs["cycle_position_deg"].astype(float).values; b_all = (np.floor(np.mod(ang, 360) / (360 / NB)).astype(int)) % NB
wells = obs["Well"].astype(str).values; half = np.isin(wells, sorted(np.unique(wells[ctl]))[:3])   # split-half by well
X = h["mod"]["rna"]["X"]; ip = X["indptr"]; N = len(ang); ncol = len(rv)
S = {k: np.zeros((NB, shared.sum())) for k in ("all", "A", "B")}; n_b = {k: np.zeros(NB) for k in S}
for s0 in range(0, N, 20000):
    s1 = min(s0 + 20000, N); p = ip[s0:s1 + 1]; d = X["data"][p[0]:p[-1]].astype(np.float32); ix = X["indices"][p[0]:p[-1]]
    M = sparse.csr_matrix((d, ix, p - p[0]), shape=(s1 - s0, ncol))
    tot = np.asarray(M.sum(1)).ravel(); M = sparse.diags(1e4 / np.maximum(tot, 1)) @ M[:, gcols]
    for k, m in (("all", ctl[s0:s1]), ("A", ctl[s0:s1] & half[s0:s1]), ("B", ctl[s0:s1] & ~half[s0:s1])):
        rows = np.where(m)[0]
        if not len(rows): continue
        B = sparse.csr_matrix((np.ones(len(rows)), (b_all[s0:s1][rows], rows)), shape=(NB, s1 - s0))
        S[k] += np.asarray((B @ M).todense()); n_b[k] += np.bincount(b_all[s0:s1][rows], minlength=NB)
mu = {k: S[k] / n_b[k][:, None] for k in S}            # NB x genes
pi = n_b["all"] / n_b["all"].sum()
print(f"Element baseline control cells {int(ctl.sum()):,} | genes shared with CROP-seq {shared.sum():,} | cells per bin {n_b['all'].astype(int).tolist()}", flush=True)

# per-bin composition, for readable summaries
ph = obs["phase_scanpy"].astype(str).values
comp = pd.DataFrame({f"pct_{q}": [100 * np.mean(ph[ctl & (b_all == b)] == q) for b in range(NB)] for q in ["G1", "S", "G2M"]})
raw = h["mod"]["imaging"]["layers"]["raw"][:]; inames = list(read_elem(h["mod"]["imaging"]["var"]).index)
gi = lambda n: raw[:, inames.index(n)].astype(float)
dna = np.log2(np.clip((gi("Nucleus_Nuclear") - gi("Bkg_Nucleus_Nuclear")) * gi("NuclearArea"), 1, None))
dna = dna - pd.Series(dna).groupby(wells).transform("median").values
hh, ed = np.histogram(dna[ctl], bins=120, range=(-1.5, 2)); cen = (ed[:-1] + ed[1:]) / 2
m2 = cen[np.argmax(np.where(np.abs(cen) < .4, hh, 0))]; m4 = cen[np.argmax(np.where(cen > .6, hh, 0))]
thr = (m2 + m4) / 2
comp["pct_4N"] = [100 * np.mean(dna[ctl & (b_all == b)] > thr) for b in range(NB)]
comp["bin_deg"] = (np.arange(NB) + .5) * 360 / NB; comp["pi"] = pi
print(f"Element DNA modes 2N {m2:+.2f}, 4N {m4:+.2f} (log2 rel. to well median), threshold {thr:+.2f}")
print(comp.round(2).to_string(index=False), flush=True)

# ------------------------------------------------ cycling genes
lg = {k: np.log2(mu[k] + .05) for k in mu}; meanexp = (pi[:, None] * mu["all"]).sum(0)
amp = np.ptp(lg["all"], 0)
rel = np.array([np.corrcoef(lg["A"][:, j], lg["B"][:, j])[0, 1] if np.ptp(lg["A"][:, j]) > 0 and np.ptp(lg["B"][:, j]) > 0 else 0 for j in range(lg["A"].shape[1])])
cyc = (meanexp >= .3) & (amp >= .8) & (rel >= .8)
print(f"cycling genes: {cyc.sum()} (mean CP10k >= 0.3, peak-to-trough >= 0.8 log2, split-half profile r >= 0.8)", flush=True)
Hc = H - (pi[:, None] * H).sum(0)                                   # centred harmonics (tilt keeps sum(pi') = 1)
D_all = np.zeros((shared.sum(), 4))
for k in range(4): D_all[:, k] = (pi[:, None] * Hc[:, [k]] * mu["all"]).sum(0) / np.maximum(meanexp, 1e-9) / np.log(2)
cidx = np.where(shared)[0][cyc]                                     # CROP-seq columns of the cycling genes
D = D_all[cyc]; cyc_sym = cv["gene_symbol"].astype(str).str.upper().values[cidx]
pd.DataFrame(mu["all"][:, cyc].T, index=cyc_sym, columns=[f"bin{b}" for b in range(NB)]).assign(amp_log2=amp[cyc], split_half_r=rel[cyc]).to_csv(f"{OUT}/element_cycling_gene_profiles.csv")


def fit(y, s, Dm):
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0); y, s, Dm = y[ok], s[ok], Dm[ok]
    if len(y) < 30: return np.full(4, np.nan), np.nan             # e.g. the control row (no KO-vs-control statistics)
    A = np.c_[Dm, np.ones(len(y))]; b0 = np.linalg.lstsq(A, y, rcond=None)[0]; r = y - A @ b0
    tau2 = max(float(np.var(r) - np.mean(s ** 2)), 1e-4); w = 1 / (s ** 2 + tau2); Aw = A * np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(Aw, y * np.sqrt(w), rcond=None)
    rr = y - A @ beta; r2 = 1 - np.sum(w * rr ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return beta[:4], float(r2)


def implied(cc):
    if not np.all(np.isfinite(cc)): return pi
    q = pi * (1 + Hc @ cc); q = np.clip(q, 0, None); return q / q.sum()


def summaries(cc):
    q = implied(cc); d = q - pi
    return {f"d_pct_{x}": float(d @ comp[f"pct_{x}"].values) for x in ["G1", "S", "G2M", "4N"]} | {
        "cycle_TV": float(.5 * np.abs(d).sum()), "peak_enrichment_deg": float(comp.bin_deg.values[np.argmax(q / pi)])}


rows = []; half_idx = RNG.permutation(len(cidx)); hA, hB = half_idx[: len(cidx) // 2], half_idx[len(cidx) // 2:]
for i in range(lfc.shape[0]):
    keep = cyc_sym != ko_gene[i]                                  # never use the KO's own target gene
    y, s = lfc[i, cidx], se[i, cidx]
    cc, r2 = fit(y[keep], s[keep], D[keep])
    if not np.all(np.isfinite(cc)): print(f"  row {i} ({ko_gene[i]}, {role[i]}): no usable statistics, skipped"); continue
    null = np.array([np.linalg.norm(fit(y[keep], s[keep], D[keep][RNG.permutation(keep.sum())])[0]) for _ in range(200)])
    cA, _ = fit(y[hA][keep[hA]], s[hA][keep[hA]], D[hA][keep[hA]]); cB, _ = fit(y[hB][keep[hB]], s[hB][keep[hB]], D[hB][keep[hB]])
    rows.append(dict(gene=ko_gene[i], role=role[i], n_cells=int(co["n_cells"].values[i]), **{f"c{k + 1}": cc[k] for k in range(4)},
                     shift_size=float(np.linalg.norm(cc)), r2_cycling_genes=r2, p=(np.sum(null >= np.linalg.norm(cc)) + 1) / 201,
                     **summaries(cc), d_pct_4N_halfA=summaries(cA)["d_pct_4N"], d_pct_4N_halfB=summaries(cB)["d_pct_4N"],
                     d_pct_G2M_halfA=summaries(cA)["d_pct_G2M"], d_pct_G2M_halfB=summaries(cB)["d_pct_G2M"]))
R = pd.DataFrame(rows)
tg = R.role == "targeting"; pv = R.loc[tg, "p"].values; o = np.argsort(pv); m = len(pv); q = np.empty(m)
q[o] = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]; R.loc[tg, "q"] = np.minimum(q, 1)
crude = pd.read_csv("cc_ko_scores.csv") if os.path.exists("cc_ko_scores.csv") else None
if crude is not None: R = R.merge(crude[["gene", "S_score", "G2M_score", "p21_stat"]].assign(gene=lambda d: d.gene.str.upper()), on="gene", how="left")
R.to_csv(f"{OUT}/crop_cycle_representation.csv", index=False)

rA = spearmanr(R.d_pct_4N_halfA, R.d_pct_4N_halfB)[0]; rG = spearmanr(R.d_pct_G2M_halfA, R.d_pct_G2M_halfB)[0]
print(f"\nreliability across KOs, two disjoint halves of the cycling genes: d%4N rho {rA:.2f} | d%G2M rho {rG:.2f}")
print(f"KOs with a cycle-composition signal beyond the permuted-profile null (FDR 5%): {int((R.q < .05).sum())} / {int(tg.sum())}")
print("median R2 of cycling-gene fold changes explained by the cycle shift:", round(float(R.r2_cycling_genes.median()), 3))
if crude is not None:
    print("agreement with the crude marker score: d%G2M vs G2M_score rho", round(spearmanr(R.d_pct_G2M, R.G2M_score, nan_policy="omit")[0], 2),
          "| d%S vs S_score rho", round(spearmanr(R.d_pct_S, R.S_score, nan_policy="omit")[0], 2))
cols = ["gene", "n_cells", "d_pct_G1", "d_pct_S", "d_pct_G2M", "d_pct_4N", "cycle_TV", "r2_cycling_genes", "q"]
for lab_, k_, asc in [("largest G2M gain", "d_pct_G2M", False), ("largest G2M loss", "d_pct_G2M", True), ("largest S gain", "d_pct_S", False),
                      ("largest G1 gain", "d_pct_G1", False)]:
    print(f"\n{lab_}:\n", R[R.q < .05].sort_values(k_, ascending=asc).head(8)[cols].round(3).to_string(index=False))
for g_ in ["CCNB1", "CDK7", "AURKB", "POLA1", "CDC45", "RRM1", "MCM7", "PLK1", "CDK1", "WEE1"]:
    r_ = R[R.gene == g_]
    if len(r_): print(f"  {g_:6s}", r_[cols[2:]].round(3).to_dict("records")[0])

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
order = np.argsort(np.argmax(lg["all"][:, cyc], 0)); Zp = lg["all"][:, cyc][:, order]; Zp = (Zp - Zp.mean(0)) / Zp.std(0)
ax[0].imshow(Zp.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, extent=[0, 360, len(order), 0])
ax[0].set_xlabel("Element RNA clock (deg)"); ax[0].set_ylabel(f"{cyc.sum()} cycling genes (sorted by peak)"); ax[0].set_title("Element: cycling genes, per-bin z")
for x_, c_ in [("pct_G1", "tab:blue"), ("pct_S", "tab:green"), ("pct_G2M", "tab:red"), ("pct_4N", "k")]:
    ax[1].plot(comp.bin_deg, comp[x_], "-o", color=c_, label=x_.replace("pct_", "% "))
ax[1].set_xlabel("Element RNA clock (deg)"); ax[1].legend(); ax[1].set_title("what each clock bin means (control cells)")
ax[2].scatter(R.d_pct_4N_halfA, R.d_pct_4N_halfB, s=8, c="0.5"); ax[2].axline((0, 0), slope=1, color="k", lw=.7)
ax[2].set_xlabel("KO change in % 4N, genes half A"); ax[2].set_ylabel("genes half B"); ax[2].set_title(f"reliability across 1,000 KOs (rho {rA:.2f})")
plt.tight_layout(); plt.savefig(f"{OUT}/crop_cycle_overview.png", dpi=105)
print("DONE")
