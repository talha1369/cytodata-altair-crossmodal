# Cell-cycle clock guide (OPS0077): what `cc_phase` means and how to use it on knockouts

Read this before turning `cc_phase` (or a phase-contrast prediction of it) into cell-cycle stages, and before comparing knockouts
with NTC. Every number below can be reproduced from files in this repository or on the team VM. Sources are listed at the end.

## 1. Quarters of `cc_phase` are not Early G1 / Late G1 / S / G2/M

`cc_phase` runs from 0 to 1 and wraps (0 = start of G1). Measured DNA content (background-corrected DAPI, rounds 2 to 4) shows what
each part of the clock contains. 35,633 NTC cells with reliable DNA; 2N < 0.25 <= S <= 0.65 < 4N in log2 DNA relative to the 2N peak.

| `cc_phase` range | 2N | S | 4N | median pRb (z) | what it is |
|---|---|---|---|---|---|
| 0.000 to 0.125 | 99.9% | 0.1% | 0% | +0.63 | G1 |
| 0.125 to 0.250 | 100% | 0% | 0% | -0.42 | G1 |
| 0.250 to 0.375 | 90.0% | 10.0% | 0% | -0.67 | late G1, entering S |
| 0.375 to 0.500 | 14.8% | 83.8% | 1.4% | -0.21 | S |
| 0.500 to 0.625 | 0% | 12.6% | 87.4% | +0.45 | G2 |
| 0.625 to 0.750 | 2.6% | 69.8% | 27.6% | +0.91 | mitosis and division (intermediate DNA, high pRb) |
| 0.750 to 0.875 | 63.8% | 36.2% | 0% | +0.69 | G1 after division |
| 0.875 to 1.000 | 100% | 0% | 0% | -0.60 | G1 after division |

By quarter: **0.5 to 0.75 holds the 4N cells (76% 4N)** and **0.75 to 1 is 96% 2N**. Of all measured 4N cells, 99.7% sit at 0.5 to 0.75
and none at 0.75 to 1. `cc_phase_dna` has the same layout.

**Consequence.** Labelling the quarters "Early G1 / Late G1 / S / G2/M" shifts every label by about one quarter. A knockout that
accumulates cells in G1 adds cells at 0.75 to 1, which that labelling reports as a "G2/M" gain. This is why the G1/S restriction-point
knockouts appeared to accumulate in G2 (for example CDK7: +11 points in the 0.75 to 1 quarter = more 2N cells after division, a G1
accumulation, as expected for CDK7).

Reproduce:

```python
import numpy as np, pandas as pd
L = pd.read_csv("cell_cycle_scores.csv")
L = L[(L.dna_reliable == True) & (L.cc_phase_source == "dna+pcs")].drop_duplicates("cell_key")
d = L.dna_rel_R234
state = np.select([d < .25, d > .65], ["2N", "4N"], "S")
eighth = np.floor(L.cc_phase * 8) / 8
print(pd.crosstab(eighth, state, normalize="index").round(3))
```

### Recommended stage labels for `cc_phase` (or a prediction of it)

```python
def cc_stage(p):
    """Stage from cc_phase in [0, 1), boundaries from measured DNA content in NTC cells."""
    p = np.mod(np.asarray(p, float), 1)
    return np.select(
        [p < .25, p < .375, p < .5, p < .625, p < .75],
        ["G1", "G1/S", "S", "G2", "M/division"],
        "G1 (after division)")
```

For knockout comparisons it is cleaner still to predict DNA state directly from phase contrast (4N vs 2N, held-out AUC 0.92) than to
read stages off the clock.

## 2. The 4i principal-curve explanation for misplaced G1 cells is not supported

The suggested mechanism was that the 4i curve projects pRb-high 2N cells into late G1 or S. In NTC cells, using the DNA x pRb staging
(2N with pRb at or below its 35th percentile = Early G1, above = Late G1):

| measured state | n | 0 to 0.25 | 0.25 to 0.5 | 0.5 to 0.75 | 0.75 to 1 |
|---|---|---|---|---|---|
| 2N, pRb low (Early G1) | 8,671 | 51.0% | 15.5% | 0.0% | 33.5% |
| 2N, pRb high (Late G1) | 16,103 | 76.3% | 6.7% | 0.3% | 16.8% |
| S | 3,928 | 0.3% | 39.3% | 53.9% | 6.5% |
| 4N (G2/M) | 6,931 | 0.0% | 0.3% | 99.7% | 0.0% |

pRb-high 2N cells are almost never placed in the 4N part of the clock. The final clock includes DNA as an axis, so it separates DNA
states well (4N vs 2N AUC 1.00). Note also that knockout cells have **no** 4i clock: Louise's scores exist for NTC cells only, so
knockout plots use a phase-contrast prediction of the clock, not the 4i model.

## 3. The hypertrophy hypothesis (large 2N cells called G2/M by phase contrast)

**Not supported, tested on measured DNA** (notebook 18, section 2d). Background-corrected DAPI was measured for 111 knockouts (81
with a predicted cycle shift, 30 calibration; 22,152 usable cells), and four phase-contrast 4N models were scored against it:

| model | AUC 4N vs not, NTC (held out) | % of NTC 2N cells called 4N (largest 10% of 2N cells) | agreement with measured change in % 4N across knockouts (Spearman) |
|---|---|---|---|
| standard | 0.919 | 3.4% (15.6%) | 0.86 |
| DNA floor (no 4N call below 2.5N of predicted DNA) | 0.920 | 3.4% (15.6%) | 0.86 |
| size-balanced | 0.902 | 13.8% (21.3%) | 0.75 |
| size-removed (cell area regressed out) | 0.886 | 3.9% (5.8%) | 0.69 |

* Within NTC, the largest 2N cells are over-called as 4N. **Across knockouts this does not matter:** measured-2N cells are larger
  than NTC 2N cells in 102 of 111 knockouts (median about 17% more area), but the excess of false 4N calls over NTC has median
  -1.0 point and does not rise with 2N size (Spearman 0.08, p 0.41).
* The G1-list knockouts in this batch (CUL1, POLE, PSMB6, RPTOR, RRM1) have false-4N excesses of -1.1 to +0.7 points.
* The DNA floor changes no knockout by more than 0.63 points. Size corrections lower agreement with measured DNA.
* **The predictions themselves hold:** predicted vs measured change in % 4N, Spearman 0.76, slope 1.01 (n = 111); 42 of 81 predicted
  shifts confirmed at FDR 5%, 41 with the predicted sign. The confirmed 4N gains are replication genes (MTBP, MCM7, CDC45, POLA2).
* The remaining 47 arrest-list genes are being measured; only 5 G1-list genes are in this first batch.

Already measured: CCNB1 (on the G1/S list) gains 4N cells for real (+4.3 points by DAPI). CCNB1 is the mitotic cyclin.

## 4. Comparing knockouts with NTC

* **Score NTC out of fold.** Knockout cells are never in training, so NTC cells must not be scored in-sample. (For the gradient-boosting
  model in `hackathon_4i_inference_notebook_bf.ipynb` this barely changes the NTC composition: 57/12/18/12 vs 57/12/19/13%.)
* **Use all NTC cells,** not only those with a measured clock (a third of NTC cells have no reliable single nucleus; knockout cells
  were never filtered that way).
* **Use a null with the right noise:** NTC guides pooled to the knockout's cell count, the same comparison, then BH FDR across knockouts.
* **Model choice matters; use gradient boosting on the raw phase features** (the model in that notebook). On the same held-out
  folds (5 folds by tile, 35,633 NTC cells) it predicts the DNA clock to 21.7 to 21.8 degrees vs 29.3 for ridge on PCA-100. NaN
  handling does not matter (NaN to 0 and native NaN tie). Most of the gain needs the raw features: on PCA-100 the same trees reach
  28.3 and a single-target MLP 26.8. Rerunning the 1,000-knockout library with it gives 194 cycle-shifting knockouts instead of 81
  (73 shared), with unchanged agreement with measured pRb (rho 0.38 vs 0.37) and an unchanged heterogeneity result (spread -2.75%,
  separation from NTC unchanged). The extra hits are a more sensitive call under the same null, not independently confirmed
  (`reports/ko_cell_cycle/ko_library_cycle_hgb.csv`, `compare_ridge_vs_hgb.csv`). In that run, with all NTC cells as the reference
  and the guide-pooled null, CDK7 is borderline (q 0.057, +9.7 points in 0.75 to 1) and CCNB1 is not significant (q 0.19, every
  region within 4 points); POLA1, CDC45 and MCM7 gain 4N cells (+10.3, +13.5, +9.8 points).

## 5. The arrest list (`Genes expected to affect cell cycle.csv`)

Several genes sit in a group that does not match their known function: CCNB1 (mitotic cyclin), RRM1, POLE and RPA2 (replication) and
KIF14 (cytokinesis) are listed under G1/S; CDC45 (replication helicase) under G2/M. The `Predicted_Mechanism` text is identical within
each group. Against the library, the S-phase group gains 4N cells as expected (8 of 9, p = 3e-5); the G1 and G2/M groups do not
differ from the other knockouts.

## 6. Regressing the cell cycle out of embeddings (item 7)

* The per-PC tightening reproduces with z-scored features and a flexible cycle fit (48 clock bins): 4i PC1/PC2 25/32%, phase contrast
  11/14%. Over all 50 PCs it is 7.4% (4i) and 4.5% (phase contrast).
* `cc_phase` is built from the 4i markers and from PCs of the pRb, Rb and p21 features (`cell_cycle_4i_DNA/per_cell_pca_embeddings.csv`),
  so regressing it out of 4i embeddings is partly circular. Measured DNA content alone, which is independent of the features, gives
  10% on the top 4i PCs instead of 19%. For phase contrast the clock is independent of the features.
* In every modality the cycle is 2 to 9% of single-cell variance; removing it shrinks perturbed populations by about 1 to 5% and does
  not change their separation from NTC (notebook 19).
* **Phase contrast, extended to knockout cells** (notebook 19, section 3). With the measured clock, phase PC1/PC2/PC5/PC9 lose 11.5,
  14.4, 19.4 and 13.2% of within-field variance. Pooled knockout cells (up to 200 per knockout, gradient-boosting predicted clock)
  tighten like NTC on PC1 to PC3 (16/22/6% vs 20/22/6%) and less on PC5, PC7 and PC9. Removing the cycle moves knockout centroids
  only about 5% closer to NTC (noise-corrected, PC1 to PC10), nearly the same for cycle-shifting knockouts (6.9%) and the rest
  (4.9%); replication knockouts keep 76 to 84% of their distance. 33 knockout cells with feature z-scores up to 2e10 must be
  excluded (`reports/cell_cycle_by_modality/knockout_outlier_cells.csv`), or the pooled numbers collapse to 0%.

### Features and embeddings for perturbed cells

They exist on the team VM:

| what | path | cells |
|---|---|---|
| phase-contrast OrganelleProfiler features, all 1,000 knockouts | `/srv/team/data/crossmodal/ops0077_ko/OPS0077_allko_phase.h5ad` | 591,136 |
| 4i pRb, Rb, p21 features (atlas files; NTC and all knockouts) | `/srv/team/data/crossmodal/ops0077_ko/all_cells_fluor_{prb,rb,p21}.h5ad` | 643,356 |
| CCNB1, MCM3, CENPC knockouts: 4i panel + phase contrast + measured DNA, same schema as the NTC multimodal file (scaled with NTC statistics, so the two concatenate) | `/srv/team/data/crossmodal/ops0077_ko/OPS0077_KO3_CCNB1_MCM3_CENPC_Multimodal_4i_Phase_DNA.h5ad` | 1,603 (1,353 with measured DNA) |
| all 111 measured-DNA knockouts (81 cycle-shifting + 30 calibration): every phase-imaged cell, 4i panel, phase contrast in `obsm["X_phase"]`, measured DNA where measured, `cycle_stage`; same schema and NTC scaling | `/srv/team/data/crossmodal/ops0077_ko/OPS0077_KOcyc_Multimodal_4i_Phase_DNA.h5ad` | 53,471 (22,152 with measured DNA) |
| the same 4i atlas files, public | `s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/cell_features/all_cells_fluor_{prb,rb,p21}.h5ad` (us-west-2, anonymous) | |

Join key: `cell_key = well + "|" + segmentation_id` (the phase file has it in `obs["cell_key"]`; for the atlas files build it from
`obs["well"]` and `obs["segmentation_id"]`). Knockout gene: `obs["gene"]`. Imaging field: `obs["tile_pheno"]`.

## Sources

* Notebooks (branch `talha-cycle-modalities`): `16_phase_to_cell_cycle.ipynb`, `17_align_cell_cycle_regimes.ipynb`,
  `18_ko_cell_cycle_heterogeneity.ipynb`, `19_cell_cycle_by_modality.ipynb`, `20_element_cycle_morphology.ipynb`
* Tables: `reports/ko_cell_cycle/`, `reports/cell_cycle_by_modality/`, `reports/cell_cycle_alignment/`, `reports/element_cycle_morphology/`
* Clock scores: `cell_cycle_scores.csv`; scoring script: `cell_cycle_4i_DNA/ops0077_cell_cycle_score.py`
