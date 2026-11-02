#!/usr/bin/env python
"""Multimodal 4i + phase-contrast + DNA AnnData for knockout cells, in the same schema as OPS0077_NTC_Multimodal_4i_Phase_DNA.h5ad,
so the two files can be concatenated directly.

  .X                     4i OrganelleProfiler features (same 2,250 columns and order as the NTC file), robust-scaled with the
                         NTC medians / IQR(10-90), clip +-5; layers['raw'] = raw values
  obsm['X_phase']        phase-contrast OrganelleProfiler features (same 3,317 columns as the NTC file), scaled with NTC stats;
                         obsm['X_phase_raw'] = raw values
  obsm['spatial'], obsm['spatial_pheno']   4i and phase centroids
  obs                    gene, sgRNA, barcode, well, tile; pRb, Rb, p21, areas (same definitions as the NTC frame);
                         measured DNA (cp_measure DAPI rounds 2-4, per-cell background; dna_reliable) and the DNA x pRb
                         cycle_stage with the NTC pRb threshold; phase-predicted readouts (models trained on NTC):
                         phase_predicted_cc_phase_ridge / _hgb, phase_predicted_dna, phase_predicted_4N_prob, and
                         phase_predicted_stage (stage labels of CELL_CYCLE_CLOCK_GUIDE.md applied to the gradient-boosting clock)
usage: python build_ko_multimodal.py GENES(comma) DNA_PERCELL_CSV OUT.h5ad"""
import sys
import numpy as np, pandas as pd, anndata as ad, h5py
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

GENES = [g.strip().upper() for g in sys.argv[1].split(",")]; DNA_CSV, OUT = sys.argv[2:4]
NTCD = "/srv/team/data/crossmodal/ops0077_ntc"
ntc = ad.read_h5ad(f"{NTCD}/OPS0077_NTC_Multimodal_4i_Phase_DNA.h5ad")
var4, varph = list(ntc.var_names), list(ntc.uns["phase_var_names"])
print(f"NTC reference {ntc.n_obs:,} cells | 4i features {len(var4)} | phase features {len(varph)}", flush=True)

# ---- knockout cells: phase from the full-library extraction
ko = ad.read_h5ad("OPS0077_allko_phase.h5ad", backed="r"); ob = ko.obs.copy(); ob["row"] = np.arange(len(ob)); ob["gene"] = ob["gene"].astype(str).str.upper()
ob = ob[ob.gene.isin(GENES)].drop_duplicates("cell_key").sort_values("row")
assert list(ko.var_names) == varph, "phase feature order differs from the NTC file"
f = h5py.File("OPS0077_allko_phase.h5ad", "r"); Xph_raw = np.vstack([f["X"][ob.row.values[i:i + 5000]] for i in range(0, len(ob), 5000)]).astype(np.float32); f.close()
keys = ob.cell_key.astype(str).values
print("knockout cells with phase:", ob.gene.value_counts().to_dict(), flush=True)

# ---- 4i: same columns as the NTC file, from the atlas files (identical rows across files)
A = {m: ad.read_h5ad(f"raw/all_cells_fluor_{m}.h5ad", backed="r") for m in ["prb", "rb", "p21"]}
o4 = A["prb"].obs; k4 = (o4["well"].astype(str) + "|" + o4["segmentation_id"].astype("int64").astype(str)).values
pos = pd.Series(np.arange(len(k4)), index=k4); pos = pos[~pos.index.duplicated()]
rows4 = pos.reindex(keys); has4 = rows4.notna().values; r4 = rows4.values[has4].astype(int); o_ = np.argsort(r4)
X4_raw = np.full((len(keys), len(var4)), np.nan, np.float32)
H = {m: h5py.File(f"raw/all_cells_fluor_{m}.h5ad", "r") for m in A}; V = {m: list(A[m].var_names) for m in A}
blk = {m: np.empty((len(r4), len(V[m])), np.float32) for m in A}
for m in A: blk[m][o_] = H[m]["X"][np.sort(r4)]
for j, n in enumerate(var4):
    m = next(mm for mm in ["prb", "rb", "p21"] if n in V[mm]); X4_raw[has4, j] = blk[m][:, V[m].index(n)]
print(f"4i matched {has4.mean():.3f} of knockout cells", flush=True)
o4k = o4.iloc[pos.reindex(keys).fillna(0).astype(int).values]

# ---- scaling with NTC statistics
def ntc_stats(R):
    R = np.asarray(R, np.float64); med = np.nanmedian(R, 0); q10, q90 = np.nanpercentile(R, [10, 90], axis=0)
    return med, np.where(q90 - q10 > 0, q90 - q10, 1.0)
def apply(X, med, iqr):
    X = np.where(np.isfinite(X), X, med); return np.nan_to_num(np.clip((X - med) / iqr, -5, 5), nan=0.0).astype(np.float32)
m4, i4 = ntc_stats(ntc.layers["raw"]); mp, ip = ntc_stats(ntc.obsm["X_phase_raw"])

# ---- obs
obs = pd.DataFrame(index=keys)
obs["gene"] = ob.gene.values; obs["perturbation"] = ob.gene.values; obs["control_type"] = "targeting"
for c in ["well", "tile_pheno"]: obs[c] = ob[c].astype(str).values
for c in ["sgRNA", "barcode"]: obs[c] = o4k[c].astype(str).values if c in o4k else ""
col = lambda n: X4_raw[:, var4.index(n)]
obs["pRb"] = col("cp_single_object_pRb_cell_Intensity_MeanIntensity"); obs["Rb"] = col("cp_single_object_Rb_cell_Intensity_MeanIntensity")
obs["p21"] = col("cp_single_object_p21_cell_Intensity_MeanIntensity"); obs["nuclear_area_4i"] = col("op_nuclei_area_mean"); obs["cell_area_4i"] = col("op_cell_area")
obs["prb_rb_ratio"] = obs.pRb / obs.Rb; obs["log2_pRb"] = np.log2(obs.pRb.clip(lower=1)); obs["has_4i_protein"] = has4; obs["has_phase"] = True
D = pd.read_csv(DNA_CSV, index_col=0); D = D[~D.index.duplicated()].reindex(keys)
obs["dna_content_log2"] = D["dna_rel_R234"].astype(float).values if "dna_rel_R234" in D else D["dna"].astype(float).values
ok_col = "nuc_ok" if "nuc_ok" in D else "ok"
obs["dna_reliable"] = (D[ok_col].astype(str) == "True").values & np.isfinite(obs.dna_content_log2.values) if ok_col in D else np.isfinite(obs.dna_content_log2.values)
n_obs = ntc.obs; rel_n = n_obs.dna_reliable.astype(str) == "True"
prb_thr = float(np.nanmedian(n_obs.loc[rel_n & (n_obs.dna_content_log2.astype(float) < .3), "log2_pRb"].astype(float)))
d_ = obs.dna_content_log2.values; rel = obs.dna_reliable.values; st = np.full(len(obs), None, object)
st[rel & (d_ < .3) & (obs.log2_pRb.values < prb_thr)] = "G0/G1-early"; st[rel & (d_ < .3) & (obs.log2_pRb.values >= prb_thr)] = "G1-late"
st[rel & (d_ >= .3) & (d_ < .7)] = "S"; st[rel & (d_ >= .7)] = "G2/M"; obs["cycle_stage"] = st
print(f"DNA measured for {int(rel.sum())} of {len(obs)} cells | NTC pRb threshold (log2) {prb_thr:.3f}", flush=True)

# ---- phase-predicted readouts, models trained on NTC cells with a measured clock / reliable DNA
L = pd.read_csv("cell_cycle_scores.csv").set_index("cell_key")
L = L[~L.index.duplicated()]; lab = ntc.obs_names.isin(L.index[(L.dna_reliable == True) & (L.cc_phase_source == "dna+pcs")])
Xn = ntc.obsm["X_phase_raw"][lab]; Ln = L.loc[ntc.obs_names[lab]]; t = 2 * np.pi * Ln.cc_phase.astype(float).values; dn = Ln.dna_rel_R234.astype(float).values
pres = np.isfinite(Xn).mean(0) >= .85
imp = SimpleImputer(strategy="median").fit(Xn[:, pres]); rs = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(Xn[:, pres]))
g = lambda X: np.clip(rs.transform(imp.transform(X[:, pres])), -5, 5); pca = PCA(100, random_state=0).fit(g(Xn)); Zn, Zk = pca.transform(g(Xn)), pca.transform(g(Xph_raw))
ang = lambda s, c: np.mod(np.arctan2(s, c), 2 * np.pi) / (2 * np.pi)
obs["phase_predicted_cc_phase_ridge"] = ang(RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Zn, np.sin(t)).predict(Zk), RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Zn, np.cos(t)).predict(Zk))
obs["phase_predicted_dna"] = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Zn, dn).predict(Zk)
obs["phase_predicted_4N_prob"] = LogisticRegression(C=.1, max_iter=3000).fit(Zn, (dn > .65).astype(int)).predict_proba(Zk)[:, 1]
z0 = lambda X: np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0); ss = StandardScaler().fit(z0(Xn))
obs["phase_predicted_cc_phase_hgb"] = ang(HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(ss.transform(z0(Xn)), np.sin(t)).predict(ss.transform(z0(Xph_raw))),
                                          HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(ss.transform(z0(Xn)), np.cos(t)).predict(ss.transform(z0(Xph_raw))))
p = obs.phase_predicted_cc_phase_hgb.values
obs["phase_predicted_stage"] = np.select([p < .25, p < .375, p < .5, p < .625, p < .75], ["G1", "G1/S", "S", "G2", "M/division"], "G1 (after division)")

out = ad.AnnData(X=apply(X4_raw, m4, i4), obs=obs, var=ntc.var.copy())
out.layers["raw"] = X4_raw; out.obsm["X_phase"] = apply(Xph_raw, mp, ip); out.obsm["X_phase_raw"] = Xph_raw
out.obsm["spatial"] = np.c_[ob.x_cp.astype(float), ob.y_cp.astype(float)].astype(np.float32)
xp = o4k["x_pheno"].astype(float).values if "x_pheno" in o4k else np.full(len(ob), np.nan); yp = o4k["y_pheno"].astype(float).values if "y_pheno" in o4k else np.full(len(ob), np.nan)
out.obsm["spatial_pheno"] = np.c_[xp, yp].astype(np.float32)
out.uns["phase_var_names"] = varph
out.uns["scaling"] = "X and X_phase scaled with the NTC file's per-feature median and IQR(10-90), clip +-5, so values are comparable to OPS0077_NTC_Multimodal_4i_Phase_DNA.h5ad. Raw in layers['raw'] / obsm['X_phase_raw']."
out.uns["provenance"] = {"screen": "Biohub_OPS0077 (ops0144_20260406)", "population": f"knockouts: {', '.join(GENES)}",
                         "4i": "atlas all_cells_fluor_{prb,rb,p21}.h5ad, same columns as the NTC file", "phase": "atlas all_cells_phase.h5ad (extract_phase_all.py)",
                         "dna": "cp_measure DAPI R2-R4 with per-cell non-nuclear background (dapi_cpmeasure.py, bg_pass.py)",
                         "cycle_stage": f"DNA x pRb gating with the NTC pRb threshold (log2 {prb_thr:.3f})",
                         "phase_predicted": "models trained on NTC cells with a measured clock; stage labels from CELL_CYCLE_CLOCK_GUIDE.md",
                         "builder": "build_ko_multimodal.py"}
for c in out.obs.columns:
    if out.obs[c].dtype == object: out.obs[c] = out.obs[c].astype(str)
out.write_h5ad(OUT, compression="gzip")
print(f"wrote {OUT} {out.shape} | X_phase {out.obsm['X_phase'].shape}")
print(out.obs.groupby("gene")[["dna_reliable"]].sum().join(out.obs.groupby("gene").size().rename("cells")).to_string())
print(pd.crosstab(out.obs.gene, out.obs.cycle_stage.replace("None", "no DNA")).to_string())
print("DONE")
