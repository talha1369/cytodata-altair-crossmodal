"""Build a dapi_cpmeasure / bg_pass input table for knockout cells: every knockout with a significant predicted cycle shift
(ko_library_cycle.csv, q_cycle < 0.05) plus N_CAL random knockouts without one (calibration across the range), at most
MAX_CELLS cells per knockout, from the full-library phase extraction.
usage: python ko_dna_build_input.py OUT.h5ad [MAX_CELLS=250] [N_CAL=30] [GENES_CSV]
With GENES_CSV (column Gene_Symbol), those genes are taken instead (set = "arrest_list"), minus any already in OPS0077_KOcyc_input.h5ad."""
import sys, numpy as np, pandas as pd, anndata as ad
out = sys.argv[1]; maxc = int(sys.argv[2]) if len(sys.argv) > 2 else 250; ncal = int(sys.argv[3]) if len(sys.argv) > 3 else 30
R = pd.read_csv("ko_library/ko_library_cycle.csv")
if len(sys.argv) > 4:
    done = set(ad.read_h5ad("OPS0077_KOcyc_input.h5ad", backed="r").obs["gene"].astype(str).str.upper())
    shift = sorted(set(pd.read_csv(sys.argv[4])["Gene_Symbol"].astype(str).str.upper()) - done); cal = []
else:
    shift = R.loc[R.q_cycle < .05, "gene"].tolist()
    cal = R.loc[R.q_cycle >= .05, "gene"].sample(ncal, random_state=0).tolist()
o = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r").obs.copy(); o["gene"] = o["gene"].astype(str).str.upper()
o = o[o.gene.isin(shift + cal)].drop_duplicates("cell_key")
o = o.groupby("gene", group_keys=False).apply(lambda d: d.sample(min(len(d), maxc), random_state=0))
o["has_4i_features"] = True; o["set"] = np.where(o.gene.isin(shift), "arrest_list" if len(sys.argv) > 4 else "cycle_shift", "calibration")
o = o[["cell_key", "well", "tile_pheno", "segmentation_id", "x_cp", "y_cp", "gene", "set", "has_4i_features"]].reset_index(drop=True)
a = ad.AnnData(X=np.zeros((len(o), 1), np.float32), obs=o); a.obs_names = o["cell_key"].astype(str).values
for c in a.obs.columns:
    if a.obs[c].dtype == object: a.obs[c] = a.obs[c].astype(str)
a.write_h5ad(out)
print(f"{len(shift)} cycle-shift + {len(cal)} calibration knockouts | cells {len(o):,} | per set {o.set.value_counts().to_dict()}")
