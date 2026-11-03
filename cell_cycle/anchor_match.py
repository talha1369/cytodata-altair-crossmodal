#!/usr/bin/env python
"""Daniel's open item: match Element and OPS0077 DNA-stain features through anchors instead of names, then transfer an
Element RNA-clock model to OPS0077 cells.
Anchors: DNA content and nuclear area, each as a rank within well (the public Element features are standardized and show no
separate 4N peak, so DNA is binned into within-well octiles: integrated DNA intensity in Element, measured DNA in OPS0077); both are physical quantities defined the same way in the two datasets. Every feature is turned into within-well
normal scores; its fingerprint is its mean normal score in each cell of an 8 (DNA) x 4 (nuclear area) grid. Features are
matched across datasets by fingerprint correlation, without using names.
1. Benchmark: for the pairs that share a CellProfiler name (texture, intensity, granularity, radial distribution, nuclear area),
   how often is the true partner the best fingerprint match (exact, or same measurement in another texture direction)?
2. Transfer: random forest on Element cells (twelve 30-degree sectors of the RNA clock, cycle_position_deg), applied to OPS0077
   control cells and compared with the 4i clock (cc_phase_4i). The rotation (and direction) between the clocks is fitted on one
   random half of the OPS0077 cells and scored on the other half. Feature sets: name-matched; name-matched pairs whose
   fingerprints agree; anchor-matched (best fingerprint partner, names unused).
usage: python anchor_match.py ELEMENT_H5MU OPS_DAPI_H5AD OPS_CLOCK_CSV OUT_DIR"""
import os, re, sys
import numpy as np, pandas as pd, h5py, anndata as ad
from anndata.io import read_elem
from scipy.stats import norm, rankdata
from sklearn.ensemble import RandomForestClassifier

EH, OH, OC, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0); PER_WELL = 3000; FP_MIN = 0.8
DNA_EDGES = [-np.inf, -.15, .1, .3, .5, .7, .9, 1.15, np.inf]


def well_normal(X, wells):
    Z = np.full(X.shape, np.nan)
    for w in np.unique(wells):
        m = wells == w
        for j in range(X.shape[1]):
            v = X[m, j]; ok = np.isfinite(v)
            if ok.sum() > 20: z = np.full(v.shape, np.nan); z[ok] = norm.ppf((rankdata(v[ok]) - .5) / ok.sum()); Z[np.where(m)[0], j] = z
    return Z


def rel_2n(dna, wells):
    out = np.full(dna.shape, np.nan)
    for w in np.unique(wells):
        m = (wells == w) & np.isfinite(dna); h, e = np.histogram(dna[m], bins=120, range=tuple(np.percentile(dna[m], [1, 99]))); pk = e[np.argmax(h[: len(h) * 3 // 5])] + (e[1] - e[0]) / 2
        out[wells == w] = dna[wells == w] - pk
    return out


def fingerprint(Z, dna_rel, area_z):
    db = np.digitize(dna_rel, norm.ppf(np.arange(1, 8) / 8)); ab = np.digitize(area_z, [-.674, 0, .674]); g = db * 4 + ab   # DNA octile x area quartile, within well
    F = np.full((Z.shape[1], 32), np.nan); n = np.bincount(g, minlength=32)
    for k in range(32):
        if n[k] >= 30: F[:, k] = np.nanmean(Z[g == k], 0)
    return F, n


# ---- Element: QC-passing cells with an RNA clock, up to PER_WELL per well, nucleus-channel features + nuclear area.
# ELEMENT may be the h5mu or the subset parquet written from it (same selection, columns well, condition, cycle_position_deg, features).
if EH.endswith(".parquet"):
    D = pd.read_parquet(EH); ew = D.well.astype(str).values; eclock = D.cycle_position_deg.values.astype(float)
    enames = list(D.columns[3:]); EX = D[enames].values.astype(float)
else:
    with h5py.File(EH, "r") as h:
        O = read_elem(h["obs"]); V = read_elem(h["mod/imaging/var"])
        keep = np.where((~O.qc_fail.astype(bool)).values & O.cycle_position_deg.notna().values)[0]
        wells = O.Metadata_Well.astype(str).values
        pick = np.sort(np.concatenate([RNG.choice(keep[wells[keep] == w], min(PER_WELL, (wells[keep] == w).sum()), replace=False) for w in np.unique(wells[keep])]))
        ecol = [i for i, (n, c) in enumerate(zip(V.index, V.channel.astype(str))) if (c == "Nucleus" and not n.startswith("Bkg_")) or n == "NuclearArea"]
        ecol += [list(V.index).index("Bkg_Nucleus_Nuclear")]
        EX = h["mod/imaging/X"][pick][:, ecol].astype(float)
    enames = [V.index[i] for i in ecol]; ew = wells[pick]; eclock = O.cycle_position_deg.values[pick].astype(float)
ei = {n: i for i, n in enumerate(enames)}
_v = (EX[:, ei["Nucleus_Nuclear"]] - EX[:, ei["Bkg_Nucleus_Nuclear"]]) * EX[:, ei["NuclearArea"]]; e_dna = np.log2(np.where(_v > 0, _v, np.nan))
EX = EX[:, :-1]; enames = enames[:-1]
print(f"Element: {len(ew):,} cells in {len(np.unique(ew))} wells, {len(enames)} features")

# ---- OPS0077: reliable control cells with a 4i clock, round-2 DAPI features
A = ad.read_h5ad(OH, backed="r"); clk = pd.read_csv(OC, index_col=0)["cc_phase_4i"]
ob = A.obs; ok = (ob["dna_reliable"].astype(str) == "True").values & ob.index.isin(clk.index)
ocol = [i for i, v in enumerate(A.var_names) if v.endswith("__DAPI_R2") and not v.startswith("Location_")] + \
       [i for i, v in enumerate(A.var_names) if v.startswith("nucleus_R2_") and not re.search(r"Center|BoundingBox(Min|Max)|Orientation", v)]
OX = np.asarray(A.X[np.where(ok)[0]][:, ocol] if not hasattr(A.X, "toarray") else A.X[np.where(ok)[0]][:, ocol].toarray(), dtype=float)
onames = [A.var_names[i] for i in ocol]; ow = ob["well"].astype(str).values[ok]; oclock = clk.loc[ob.index[ok]].values * 360
o_dna = ob["dna_rel_R234"].astype(float).values[ok]
print(f"OPS0077: {ok.sum():,} cells, {len(onames)} features")

EZ, OZ = well_normal(EX, ew), well_normal(OX, ow)
EF, en_ = fingerprint(EZ, EZ[:, enames.index("Intensity_IntegratedIntensity_Nucleus")], EZ[:, enames.index("NuclearArea")])
OF, on_ = fingerprint(OZ, well_normal(o_dna[:, None], ow)[:, 0], OZ[:, onames.index("nucleus_R2_Area")])
print("cells per grid cell (DNA bin x area quartile), Element:", en_.reshape(8, 4).tolist(), "OPS0077:", on_.reshape(8, 4).tolist())
grid = (en_ >= 30) & (on_ >= 30); EF, OF = EF[:, grid], OF[:, grid]
EF[~np.isfinite(EF).all(1)] = np.nan; OF[~np.isfinite(OF).all(1)] = np.nan   # a feature missing in any used grid cell is not matched
print(f"fingerprint grid cells with >= 30 cells in both datasets: {grid.sum()} of 32; features with a full fingerprint: Element "
      f"{int(np.isfinite(EF).all(1).sum())} of {len(EF)}, OPS0077 {int(np.isfinite(OF).all(1).sum())} of {len(OF)}")
EFc = (EF - EF.mean(1, keepdims=True)) / (EF.std(1, keepdims=True) + 1e-9); OFc = (OF - OF.mean(1, keepdims=True)) / (OF.std(1, keepdims=True) + 1e-9)
Cfp = EFc @ OFc.T / EF.shape[1]                                   # Element features x OPS features fingerprint correlation
Cfp[~np.isfinite(Cfp)] = -2.0                                     # features without a full fingerprint never match


def ops_name(e):
    if e == "NuclearArea": return "nucleus_R2_Area"
    m = re.match(r"Texture_(.+)_Nucleus$", e)
    if m: return f"{m.group(1)}__DAPI_R2"
    m = re.match(r"(Intensity_.+|Granularity_\d+|RadialDistribution_.+)_Nucleus$", e)
    if m: return f"{m.group(1)}__DAPI_R2"
    return None


fam = lambda n: re.sub(r"_3_0\d_256", "", re.sub(r"(_Nucleus|__DAPI_R2)$", "", n.replace("nucleus_R2_", "")).replace("Texture_", ""))
pairs = [(i, onames.index(ops_name(e))) for i, e in enumerate(enames) if ops_name(e) in onames]
n_named = len(pairs); pairs_fp = [(i, j) for i, j in pairs if Cfp[i, j] > -2]
print(f"name-matched pairs: {n_named}; with full fingerprints on both sides: {len(pairs_fp)}")
rows = []
for i, j in pairs_fp:
    order = np.argsort(-Cfp[i]); rank = int(np.where(order == j)[0][0]) + 1
    rows.append(dict(element=enames[i], ops=onames[j], fp_corr=Cfp[i, j], rank_of_true=rank, best=onames[order[0]], best_corr=Cfp[i, order[0]],
                     family_hit=fam(onames[order[0]]) == fam(onames[j])))
B = pd.DataFrame(rows); B.round(3).to_csv(f"{OUT}/benchmark_name_pairs.csv", index=False)
print(f"\n1. {len(B)} name-matched pairs among {len(onames)} OPS features: true partner is the best fingerprint match in {int((B.rank_of_true == 1).sum())} "
      f"({100 * np.mean(B.rank_of_true == 1):.0f}%), in the top 5 in {100 * np.mean(B.rank_of_true <= 5):.0f}%, same measurement family {100 * B.family_hit.mean():.0f}% "
      f"(chance top-1 {100 / len(onames):.1f}%); median fingerprint corr of true pairs {B.fp_corr.median():.2f}")
print(B.groupby(B.element.str.extract(r"^(Texture_[A-Za-z0-9]+|Intensity|Granularity|RadialDistribution|NuclearArea)")[0]).agg(
    n=("rank_of_true", "size"), top1=("rank_of_true", lambda r: np.mean(r == 1)), family=("family_hit", "mean"), fp=("fp_corr", "median")).round(2).to_string())

# ---- transfer
sect = (eclock % 360 // 30).astype(int)
half = RNG.random(len(oclock)) < .5


def evaluate(e_idx, o_idx, label):
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=4, random_state=0, class_weight="balanced")
    Xe = np.nan_to_num(EZ[:, e_idx]); rf.fit(Xe, sect)
    P = rf.predict_proba(np.nan_to_num(OZ[:, o_idx])); ang = np.deg2rad(rf.classes_ * 30 + 15)
    pred = np.rad2deg(np.arctan2(P @ np.sin(ang), P @ np.cos(ang))) % 360
    best = None
    for sgn in (1, -1):
        for off in range(0, 360, 5):
            a = (sgn * pred[half] + off) % 360; s = np.mean(a // 90 == oclock[half] // 90)
            if best is None or s > best[0]: best = (s, sgn, off)
    _, sgn, off = best; a = (sgn * pred[~half] + off) % 360; d = np.abs((a - oclock[~half] + 180) % 360 - 180)
    r = dict(feature_set=label, n_features=len(e_idx), same_quarter_pct=100 * np.mean(a // 90 == oclock[~half] // 90),
             same_30deg_pct=100 * np.mean(a // 30 == oclock[~half] // 30), median_disagreement_deg=float(np.median(d)), direction=sgn, offset_deg=off)
    r["_hit"] = (a // 90 == oclock[~half] // 90); r["_d"] = d
    print(f"  {label}: {len(e_idx)} features, same quarter {r['same_quarter_pct']:.1f}%, same 30-degree bin {r['same_30deg_pct']:.1f}%, median disagreement {r['median_disagreement_deg']:.1f} deg")
    return r


print("\n2. Element RNA-clock model on OPS0077 cells, scored on the held-out half against the 4i clock (chance: 25% quarter, 8.3% bin, 90 deg):")
R = []
ei_, oi_ = [p[0] for p in pairs], [p[1] for p in pairs]; R.append(evaluate(ei_, oi_, "name-matched"))
good = [(i, j) for i, j in pairs if Cfp[i, j] >= FP_MIN]; R.append(evaluate([p[0] for p in good], [p[1] for p in good], f"name-matched, fingerprint corr >= {FP_MIN}"))
am = [(i, int(np.argmax(Cfp[i]))) for i in range(len(enames)) if Cfp[i].max() >= FP_MIN]
R.append(evaluate([p[0] for p in am], [p[1] for p in am], f"anchor-matched (names unused), corr >= {FP_MIN}"))
R.append(evaluate([enames.index("NuclearArea")], [onames.index("nucleus_R2_Area")], "nuclear area only"))
bs = np.random.default_rng(1)
for k in range(1, len(R)):
    dq, dd = [], []
    for _ in range(1000):
        ix = bs.integers(0, len(R[0]["_hit"]), len(R[0]["_hit"]))
        dq.append(100 * (R[k]["_hit"][ix].mean() - R[0]["_hit"][ix].mean())); dd.append(np.median(R[k]["_d"][ix]) - np.median(R[0]["_d"][ix]))
    R[k]["d_same_quarter_vs_name_ci"] = f"{np.mean(dq):+.1f} [{np.percentile(dq, 2.5):+.1f}, {np.percentile(dq, 97.5):+.1f}]"
    R[k]["d_median_deg_vs_name_ci"] = f"{np.mean(dd):+.1f} [{np.percentile(dd, 2.5):+.1f}, {np.percentile(dd, 97.5):+.1f}]"
    print(f"  {R[k]['feature_set']} minus name-matched: same quarter {R[k]['d_same_quarter_vs_name_ci']} points, median disagreement {R[k]['d_median_deg_vs_name_ci']} deg (95% bootstrap over held-out cells)")
for r in R: r.pop("_hit"); r.pop("_d")
pd.DataFrame(R).round(3).to_csv(f"{OUT}/transfer_to_ops0077.csv", index=False)
pd.DataFrame([(enames[i], onames[j], Cfp[i, j], ops_name(enames[i]) == onames[j]) for i, j in am], columns=["element", "ops", "fp_corr", "same_name"]).round(3).to_csv(f"{OUT}/anchor_matches.csv", index=False)
print("DONE")
