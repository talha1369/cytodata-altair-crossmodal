"""Per-knockout RNA cell-cycle score from the Altair CROP-seq DE matrix (positive control check)."""
import s3fs, h5py, numpy as np, pandas as pd

fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
p = ("ops-explorer-public/leonetti_ops/ops_data_portal_submission/CROPseq/"
     "2026_05_28_CropSeq_June2025_perturbation_lvl_ops_8-24update.h5ad")

S = {"PCNA": "ENSG00000132646", "TYMS": "ENSG00000176890", "RRM2": "ENSG00000171848", "GMNN": "ENSG00000112312"}
G2M = {"MKI67": "ENSG00000148773", "CCNB1": "ENSG00000134057", "CCNA2": "ENSG00000145386", "CDK1": "ENSG00000170312",
       "TOP2A": "ENSG00000131747", "CDC20": "ENSG00000117399", "UBE2C": "ENSG00000175063", "AURKB": "ENSG00000178999",
       "PLK1": "ENSG00000166851", "BIRC5": "ENSG00000089685"}
P21 = "ENSG00000124762"


def cat(g, k):
    gg = g[k]
    if isinstance(gg, h5py.Group) and "categories" in gg:
        return pd.Categorical.from_codes(gg["codes"][:], gg["categories"][:].astype(str)).astype(str)
    return gg[:].astype(str)


with fs.open(p, "rb") as f, h5py.File(f, "r") as h:
    var = h["var"]["_index"][:].astype(str)
    sym = cat(h["obs"], "gene_symbol")
    role = cat(h["obs"], "role")
    ncell = h["obs"]["n_cells"][:]
    stat = h["layers"]["stat"][:]          # DESeq2 Wald stat per KO x gene (KO vs control)
    lfc = h["X"][:]

col = {g: i for i, g in enumerate(var)}
si = [col[e] for e in S.values() if e in col]
gi = [col[e] for e in G2M.values() if e in col]
pi = col.get(P21)
df = pd.DataFrame({"gene": sym, "role": role, "n_cells": ncell,
                   "S_score": np.nanmean(stat[:, si], 1), "G2M_score": np.nanmean(stat[:, gi], 1),
                   "p21_stat": stat[:, pi] if pi is not None else np.nan})
df["cycle_score"] = df[["S_score", "G2M_score"]].mean(1)
print(f"S genes used {len(si)}/{len(S)}, G2M genes used {len(gi)}/{len(G2M)}, KOs {len(df)}")
print("roles:", df.role.value_counts().to_dict())
print("\nDistribution of per-KO cycle_score:", df.cycle_score.describe().round(2).to_dict())
for g in ["CCNB1", "AURKB"]:
    r = df[df.gene == g]
    if len(r):
        q = (df.cycle_score < r.cycle_score.values[0]).mean()
        print(f"positive control {g}: S={r.S_score.values[0]:.2f} G2M={r.G2M_score.values[0]:.2f} "
              f"p21={r.p21_stat.values[0]:.2f}  (cycle_score percentile {q:.3f})")
ctrl = df[df.role.str.contains("control", case=False)]
print(f"\ncontrol rows ({len(ctrl)}): cycle_score mean {ctrl.cycle_score.mean():.2f}")
print("\nStrongest G2M-DOWN KOs (proliferation loss):")
print(df.nsmallest(8, "G2M_score")[["gene", "S_score", "G2M_score", "p21_stat", "n_cells"]].round(2).to_string(index=False))
print("\nStrongest p21-UP KOs (arrest):")
print(df.nlargest(8, "p21_stat")[["gene", "S_score", "G2M_score", "p21_stat", "n_cells"]].round(2).to_string(index=False))
df.to_csv("cc_ko_scores.csv", index=False)
