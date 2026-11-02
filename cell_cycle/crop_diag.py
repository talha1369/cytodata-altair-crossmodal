"""Model-free check of the CROP-seq cycle signal: per KO, mean fold change of genes peaking in G2/M minus genes peaking in G1
(Element peak bin), with split-half reliability; compared with the deconvolution."""
import numpy as np, pandas as pd, h5py, s3fs
from scipy.stats import spearmanr
try:
    from anndata.io import read_elem
except Exception:
    from anndata.experimental import read_elem
prof = pd.read_csv("crop_cycle/element_cycling_gene_profiles.csv", index_col=0)
b = prof[[f"bin{i}" for i in range(12)]].values; peak = np.argmax(np.log2(b + .05), 1)
P = "ops-explorer-public/leonetti_ops/ops_data_portal_submission/CROPseq/2026_05_28_CropSeq_June2025_perturbation_lvl_ops_8-24update.h5ad"
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
with fs.open(P, "rb") as f, h5py.File(f, "r") as c:
    cv = read_elem(c["var"]); co = read_elem(c["obs"]); st = c["layers"]["stat"][:].astype(float); se = c["layers"]["lfcSE"][:].astype(float)
    bm = c["layers"]["baseMean"][:].astype(float)
sym = cv["gene_symbol"].astype(str).str.upper().values; col = {s: i for i, s in enumerate(sym)}
idx = np.array([col.get(s, -1) for s in prof.index]); ok = idx >= 0
lfc = st * se
groups = {"G2M-peak (45-135 deg)": np.isin(peak, [1, 2, 3, 4]), "G1-peak (195-255 deg)": np.isin(peak, [6, 7, 8]), "S-peak (285-15 deg)": np.isin(peak, [9, 10, 11, 0])}
for k, m in groups.items(): print(k, int((m & ok).sum()), "genes")
print("baseMean of cycling genes in CROP-seq: median", np.nanmedian(bm[:, idx[ok]]).round(1), "| all genes", np.nanmedian(bm).round(1))
print("lfcSE of cycling genes: median", np.nanmedian(se[:, idx[ok]]).round(3), "| |LFC| median", np.nanmedian(np.abs(lfc[:, idx[ok]])).round(3))
rng = np.random.default_rng(0)
def contrast(rows_mask):
    g2 = idx[ok & groups["G2M-peak (45-135 deg)"] & rows_mask]; g1 = idx[ok & groups["G1-peak (195-255 deg)"] & rows_mask]
    return np.nanmean(st[:, g2], 1) - np.nanmean(st[:, g1], 1)
half = rng.random(len(idx)) < .5
cA, cB = contrast(half), contrast(~half); call = contrast(np.ones(len(idx), bool))
print("G2M-minus-G1 contrast (mean Wald stat): split-half rho across KOs", round(spearmanr(cA, cB, nan_policy="omit")[0], 3))
# within-group coherence: do G2M-peak genes move together within a KO beyond random gene sets?
g2 = idx[ok & groups["G2M-peak (45-135 deg)"]]; allc = np.where(np.nanmean(bm, 0) > np.nanpercentile(np.nanmean(bm[:, g2], 0), 10))[0]
def coh(cols): return np.nanmean(st[:, cols], 1)
null = np.array([coh(rng.choice(allc, len(g2), replace=False)) for _ in range(200)])
z = (coh(g2) - null.mean(0)) / null.std(0)
print("KOs with |z|>4 for the G2M-peak gene set vs random expression-matched sets:", int((np.abs(z) > 4).sum()), "of", len(z))
d = pd.read_csv("crop_cycle/crop_cycle_representation.csv").set_index("gene")
G = co["gene_symbol"].astype(str).str.upper().values
T = pd.DataFrame({"gene": G, "G2M_minus_G1": call, "G2M_set_z": z}).set_index("gene").join(d[["d_pct_G2M", "d_pct_4N", "cycle_TV"]])
print("deconvolution d%G2M vs model-free contrast rho", round(spearmanr(T.d_pct_G2M, T.G2M_minus_G1, nan_policy="omit")[0], 3))
print(T.sort_values("G2M_set_z").head(8).round(2).to_string()); print(T.sort_values("G2M_set_z").tail(8).round(2).to_string())
for g in ["AURKB", "CCNB1", "CDK7", "CDC45", "RRM1", "POLA1", "MCM7", "PLK1", "CDK1", "RPS29", "EIF6"]:
    if g in T.index: print(g, T.loc[g].round(2).to_dict())
