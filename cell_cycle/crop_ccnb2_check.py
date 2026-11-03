#!/usr/bin/env python
"""Brian's question on the CCNB1 knockout (no clear arrest): is CCNB1 actually knocked down in the Altair CROP-seq, and does
CCNB2 (or another mitotic cyclin / CDK) go up, as compensation would predict? Also the knockdown of the other panel genes.
Knockout-level DE table (log2 fold change vs NTC in X, Wald stat and adjusted p in layers). A change is judged against the
same gene's change across all other targeting knockouts (z = (value - median) / (1.4826 MAD)), so a gene that moves in many
knockouts is not over-read.
usage: python crop_ccnb2_check.py OUT_DIR"""
import os, sys
import numpy as np, pandas as pd, h5py, s3fs
from anndata.io import read_elem

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
P = "ops-explorer-public/leonetti_ops/ops_data_portal_submission/CROPseq/2026_05_28_CropSeq_June2025_perturbation_lvl_ops_8-24update.h5ad"
KOS = ["CCNB1", "MCM3", "CENPC", "AURKB", "POLA1", "POLA2", "BUB1B", "CDK7"]
READOUT = ["CCNB1", "CCNB2", "CCNB3", "CCNA2", "CDK1", "CDC20", "PLK1", "CCNE1", "CCNE2", "CDKN1A"]
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
with fs.open(P, "rb") as f, h5py.File(f, "r") as c:
    cv = read_elem(c["var"]); co = read_elem(c["obs"])
    X = c["X"][:].astype(float); st = c["layers"]["stat"][:].astype(float); q = c["layers"]["adj_p_value"][:].astype(float)
sym = cv["gene_symbol"].astype(str).str.upper().values; col = {s: i for i, s in enumerate(sym)}
ko = co["gene_symbol"].astype(str).str.upper().values; role = co["role"].astype(str).values
tg = np.where(role == "targeting")[0]; row = {k: i for i, k in enumerate(ko) if role[i] == "targeting"}
print(f"{len(tg)} targeting knockouts x {len(sym)} genes; n_cells per KO median {int(np.median(co['n_cells'].values[tg]))}")


def z_of(i, j):
    v = X[tg, j]; v = v[np.isfinite(v)]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
    return (X[i, j] - med) / mad if mad > 0 else np.nan


rows = []
for k in KOS:
    if k not in row: print("missing KO", k); continue
    i = row[k]; n = int(co["n_cells"].values[i])
    for g in [k] + [r for r in READOUT if r != k]:
        if g not in col: continue
        j = col[g]; rows.append(dict(knockout=k, n_cells=n, gene=g, self=g == k, log2fc=X[i, j], stat=st[i, j], q=q[i, j], z_vs_other_kos=z_of(i, j)))
R = pd.DataFrame(rows); R.round(4).to_csv(f"{OUT}/crop_panel_knockdown_and_cyclins.csv", index=False)
print("\nknockdown of each gene by its own guides:")
print(R[R.self][["knockout", "n_cells", "log2fc", "stat", "q", "z_vs_other_kos"]].round(3).to_string(index=False))
print("\nCCNB1 knockout: mitotic cyclins and partners")
print(R[(R.knockout == "CCNB1") & ~R.self][["gene", "log2fc", "stat", "q", "z_vs_other_kos"]].round(3).to_string(index=False))
if "CCNB1" in row and "CCNB2" in col:
    i = row["CCNB1"]; v = X[tg, col["CCNB2"]]; rk = int((v > X[i, col["CCNB2"]]).sum()) + 1
    print(f"\nCCNB2 change in the CCNB1 knockout ranks {rk} of {np.isfinite(v).sum()} knockouts (1 = largest increase)")
    top = pd.Series(X[i], index=sym).dropna().sort_values()
    print("CCNB1 knockout, largest increases:", ", ".join(f"{g} {v:+.2f}" for g, v in top.tail(8)[::-1].items()))
