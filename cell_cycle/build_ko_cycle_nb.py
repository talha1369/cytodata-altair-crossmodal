#!/usr/bin/env python
"""Emit 18_ko_cell_cycle_heterogeneity.ipynb: knockout cell-cycle shifts from phase-contrast morphology, regressing the cell
cycle out of perturbation profiles, and the cell-cycle representation of the CROP-seq knockouts. Display-only: reads the outputs of
ko_cycle_heterogeneity.py, ko_cycle_library.py, cycle_norm_questions.py, crop_cycle_deconv.py, crop_cycle_sets.py,
map_crop_to_phase.py and ko_cycle_figures.py.   usage: python build_ko_cycle_nb.py OUT.ipynb"""
import sys, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

c = []
def md(s): c.append(new_markdown_cell(s))
def code(s): c.append(new_code_cell(s))

md("""# Knockouts, the cell cycle and heterogeneity

The project goal has two halves: read out cell-cycle state from morphology, then **normalise or regress the cell cycle out of
perturbation profiles and test whether that reduces the apparent heterogeneity of perturbed populations**. Notebooks 16 and 17
built and aligned the cell-cycle readout. This notebook uses it on the knockouts:

1. **Cells of perturbations likely to affect the cycle** (building on Brian's CDK7 / CCNB1 figures): which knockouts shift the
   cell-cycle distribution, compared fairly against non-targeting controls, and checked against measured 4i markers.
2. **The whole library** (1,000 OPS0077 knockouts, 591,136 cells): cycle shifts, and heterogeneity before and after regressing the cycle out.
3. **Three questions about cycle normalisation.** Does it (1) reduce field and well heterogeneity, (2) find perturbations that
   change the cell-cycle distribution, and (3) separate those from perturbations that change the cells themselves?
4. **A cell-cycle representation of the CROP-seq knockouts**, and how it maps to phase-contrast morphology.

> **"phase" is used for two things.** *Cell-cycle phase* (G1/S/G2/M, or the angle `cc_phase`) is what we read out.
> *Phase contrast* (label-free) is the imaging modality. Phase-contrast **morphology** is always the predictor here.

**Model.** Phase-contrast OrganelleProfiler features -> PCA-100 -> ridge to the sine and cosine of the 4i `cc_phase` (circular
regression), and logistic regression to the DNA-content class (4N vs 2N, measured DAPI). Both are trained on NTC cells only, with folds
grouped by imaging tile. Out of fold, the angle is predicted with a median error of 30 degrees and 4N with AUC 0.92.""")

code("""import os, numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 180); pd.set_option('display.max_columns', 30)

def find(n):
    for r in ['reports/ko_cell_cycle', 'ko18_reports', '.']:
        if os.path.exists(f'{r}/{n}'): return f'{r}/{n}'
    raise FileNotFoundError(n)""")

md("""## 1. Building on Brian's figures: 60 likely-arrest knockouts and 60 random controls

**What.** For the 60 knockouts on the likely-arrest list, plus 60 random library genes as controls (other library genes with at
least 300 cells; the library is enriched for essential genes, so these are not neutral), we predict every cell's cycle position
from phase contrast and compare each knockout's distribution with NTC.

**How (and why it differs from a direct comparison).**
* **Same pipeline on both sides.** Knockout cells are never in training, so NTC predictions must be out of fold too.
* **All NTC cells as the reference.** About a third of NTC cells (16,556 of 52,189) have no measured clock (no reliable single
  nucleus), and the knockout cells were never filtered that way. Using only clock-labelled NTC cells as the reference biases every
  comparison (the unlabelled cells are predicted more often in early G1), so the reference is all NTC cells.
* **A null with the right noise.** NTC guides pooled to the knockout's cell count, then the same comparison, 400 draws, BH FDR.

**What the quarters of `cc_phase` contain (measured DNA, section 1b).** 0 to 0.25 is G1 (2N); 0.25 to 0.5 is late G1 into S;
**0.5 to 0.75 is G2/M (mostly 4N)**; **0.75 to 1 is G1 again, after division (96% 2N)**. Labelling the quarters "Early G1 / Late G1 /
S / G2/M" shifts every label by about one quarter, so a G1 accumulation shows up as a gain in the last quarter.

**Result (ridge model on PCA-100).** 8 of 120 knockouts shift the predicted cycle distribution (FDR 5%; 7 arrest genes, 1 control).
* **CDK7 and CCNB1 are not among them** with this model: their last-quarter shifts (0.75 to 1, 2N cells after division) are +3.2 and
  +3.8 percentage points (q 0.12 and 0.13).
* **Brian's model gives larger shifts, and they survive a fair reference** (section 1c). His gradient-boosting model reads the DNA
  clock more accurately (21.5 vs about 30 degrees out of fold). Against out-of-fold NTC, his CDK7 shift in the last quarter is
  +11.5 points, CCNB1 +6.7, POLA1 +10.6 in the S quarter; in-sample and out-of-fold NTC give the same compositions. So the gap
  with our numbers comes from model strength, not from the reference.
* **For CDK7 the last-quarter gain means more 2N cells after division**, a G1 accumulation, which is what its place on the arrest
  list predicts.

**Validation.** Across the 120 knockouts the predicted shifts track measured pRb on the same cells (rho -0.40 for the last
quarter, +0.39 for the second). For the three genes with measured DNA, the predicted gain in the G2/M quarter (0.5 to 0.75) matches
the measured 4N gain for CCNB1 (+4.3 measured vs +4.2 predicted) and MCM3 (+4.2 vs +2.7).""")

md("""### 1b. What each part of the 4i clock contains

Measured DNA state (2N < 0.25 <= S <= 0.65 < 4N) and median pRb in each eighth of `cc_phase`, NTC cells with reliable DNA.""")

code("""L = pd.read_csv(find('cell_cycle_scores.csv'))
L = L[(L.dna_reliable == True) & (L.cc_phase_source == 'dna+pcs')].drop_duplicates('cell_key')
d = L.dna_rel_R234; st = pd.Series(np.select([d < .25, d > .65], ['2N', '4N'], 'S'), index=L.index)
b = (np.floor(L.cc_phase * 8) / 8).rename('cc_phase from')
t = pd.crosstab(b, st, normalize='index').mul(100).round(1)
t['cells'] = b.value_counts().sort_index().values; t['median pRb (z)'] = L.z_pRb.groupby(b).median().round(2).values
t['what it is'] = ['G1', 'G1', 'late G1, entering S', 'S', 'G2 (4N)', 'mitosis / division', 'G1 after division', 'G1 after division']
t""")

md("""### 1c. Brian's model, replicated, against a fair NTC reference

Replication of the model in `hackathon_4i_inference_notebook_bf.ipynb` (gradient boosting on all phase features, missing values set
to 0, target `cc_phase_dna`). The quarter labels are his; see 1b for what each quarter contains.""")

code("""display(pd.read_csv(find('bf_reference_compositions.csv'), index_col=0))
pd.read_csv(find('bf_knockout_shifts.csv'), index_col=0)""")

code("""display(Image(filename=find('ko_cycle_overview.png')))
r = pd.read_csv(find('ko_predicted_cycle_composition.csv'))
q = [c for c in r if c.startswith('q_')]
print('significant at FDR 5%:', int((r[q].min(1) < .05).sum()), 'of', len(r))
r.assign(min_q=r[q].min(1)).sort_values('min_q').head(12)[['gene', 'cls', 'n'] + [c for c in r if c.startswith('d_') and 'labelled' not in c] + ['min_q', 'measured_pRb_shift_log2']].round(3)""")

code("""print(open(find('ko_cycle_heterogeneity_log.txt')).read().split('variance of NTC')[0][-1600:])
pd.read_csv(find('ko3_measured_vs_predicted.csv'))""")

md("""## 2. The whole library: which knockouts change the cell-cycle distribution

**What.** The same model applied to all 1,000 OPS0077 knockouts (591,136 cells).

**How.** For each knockout, the predicted cycle distribution is compared with NTC (total variation over 12 cycle bins), together
with its predicted change in % 4N cells. The null is NTC guides pooled to the knockout's cell count (2,000 draws), with BH FDR over
1,000 knockouts.

**Result.**
* **81 knockouts change the predicted cycle distribution**, and the direction makes biological sense:
  * **DNA replication and mitosis/cytokinesis genes gain 4N cells**, for example CDC45 (+10.5 points), MTBP (+14.3), KIF23 (+11.8),
    POLA1/POLA2/POLE/POLE2, MCM7, RRM1 and AURKB.
  * **Ribosome, translation, RNA polymerase I and chaperonin genes lose 4N cells**, for example POLR1B (-8.5), RPL27A, RPS28, TCP1,
    CCT3/6A/7 and EIF2S1. These cells accumulate before S phase, as expected for nucleolar and translational stress.
* **Validation:** the predicted change in % 4N agrees with measured pRb on the same knockout cells across all 1,000 knockouts (rho +0.37).""")

code("""R = pd.read_csv(find('ko_library_cycle.csv'))
cols = ['gene', 'n', 'cycle_TV', 'q_cycle', 'd_pct_4N_pred', 'measured_pRb_shift_log2', 'within_vs_null_median', 'q_within', 'cycle_share_of_shift', 'auc_before', 'auc_after']
print('knockouts with a significant cycle-distribution change:', int((R.q_cycle < .05).sum()))
R[R.q_cycle < .05].sort_values('cycle_TV', ascending=False)[cols].head(25).round(3)""")

md("""### 2b. Do the likely-arrest knockouts shift the way their expected arrest phase predicts?

**What.** The arrest list annotates each of its 60 genes with an expected arrest phase: G1 (26 genes), S (9) or G2/M (25).
An arrest in G1 should lower the share of 4N cells; an arrest in S or G2/M should raise it.

**How.** For each expected phase, the predicted change in % 4N cells is compared with the other 940 library knockouts (Mann-Whitney).

**Result.**
* **S-phase genes behave as expected.** 8 of 9 gain 4N cells (median +4.0 points; p = 3e-5 vs the rest of the library).
* **The G1 and G2/M groups do not differ from the library** (p 0.10 and 0.13).
* **Part of this is the annotation itself.** Several genes sit in a group that does not match their known function:
  * RRM1, POLE and RPA2 are replication genes listed under G1, and they gain 4N cells like the S group;
  * CCNB1 (the mitotic cyclin) is listed under G1;
  * CDC45 (a replication helicase factor) is listed under G2/M.
* **The mechanism text is identical within each phase group**, so the annotation looks rule-generated, not curated per gene.
* **Most mitotic and ESCRT genes (CHMP*, CENP*, BUB1B) show no predicted 4N gain, and measured DNA agrees** (section 2d, all 60
  genes measured). The G2/M group's median measured change is -2.5 points of 4N cells and +3.2 points of 2N cells, the opposite of
  a G2/M arrest; only CDC45 gains 4N cells, and the two significant G2/M-list shifts (ATRX, ASNS) are 2N gains. These knockouts do
  not arrest in G2/M in this screen, so phase contrast was right not to see an arrest. They are not hidden as binucleate cells
  either: the share of cells with more than one nucleus is 10.0% (median, G2/M list) vs 10.4% in NTC (NTC guides 5 to 18%); only
  KIF23 (31%) is clearly above, with BUB1B and AURKB (17 to 19%) at the edge.""")

code("""display(Image(filename=find('arrest_phase_test.png')))
display(pd.read_csv(find('arrest_phase_test.csv')))
pd.read_csv(find('arrest_phase_by_gene.csv'))""")

md("""### 2c. A better cycle model (gradient boosting), and the library rerun with it

**What.** Brian's notebook uses gradient-boosted trees (HistGB) on all raw phase features instead of our ridge on PCA-100. Is it
better, why, and does it change the library results?

**How.** Five models predict the clock, all scored on held-out tiles (5 folds by tile, 35,633 NTC cells with a 4i clock).
*Median error* is the typical angular distance between predicted and measured clock, in degrees out of 360. The library
(section 2) was then rerun with HistGB as the angle and 4N model; everything else (out-of-fold NTC, guide-pooled null with 2,000
repeats, decomposition, heterogeneity, pRb validation) is unchanged.

**Result: model choice.**
* **HistGB on raw features is clearly better: 21.8 vs 29.3 degrees** on the DNA clock. Brian's NaN-to-0 version and a native-NaN
  version on 1,740 features are tied (21.7 and 21.8), so missing-value handling does not matter.
* **Most of the gain needs the raw features.** On PCA-100, HistGB gains only 1.0 degree over ridge (28.3) and a single-target
  network 2.5 degrees (26.8). No linear model on raw features was fitted, so "raw features" and "nonlinear on raw features"
  are not fully separated.
* **Predicting the DNA x pRb stages directly** reaches balanced accuracy 0.59 (HistGB) vs 0.52 (logistic on PCA-100), chance 0.25.
  S phase stays hard (recall 0.24 and 0.02).

**Result: library rerun.**
* **194 knockouts change the predicted cycle distribution with HistGB, against 81 with ridge** (73 in both). The two runs rank
  genes similarly (Spearman 0.79 for the size of the shift, 0.92 for the predicted change in % 4N).
* **Agreement with measured pRb is unchanged** (rho +0.38 vs +0.37 over 1,000 knockouts; +0.39 among the 121 new hits).
* **The heterogeneity result does not depend on the model.** Removing the cycle shrinks within-knockout spread by 2.75% (ridge
  3.05%), and separation from NTC is unchanged (595 knockouts before, 613 after).
* **Where the hits move.** For 162 of 194 hits the largest gain is in "G1 after division" (0.75 to 1; median +7.4 points) and for
  130 the largest loss is in G2 (0.5 to 0.625; median -5.0). The largest group of new hits (roughly 26 to 34 of 121) is ribosome,
  translation and splicing genes. Replication and mitosis knockouts (CDC45, RRM1, MTBP, KIF11) lose 21 to 28 points of G1
  (0 to 0.25) and gain cells in mitosis/division and in G1 after division.
* **Focus genes.** CDC45, MCM7 and POLA1 are significant in both runs, with larger predicted 4N gains under HistGB (+13.5, +9.8,
  +10.3 points). CDK7 moves from null (q 0.50) to borderline (q 0.057). CCNB1 is not significant in either run (q 0.19 and 0.36).

**Caveats.**
* The extra hits come from a more sensitive model under the same null; the only independent support is the pRb correlation.
* Some new hits are probably not expressed in A549 or belong to multi-copy families (for example OR2T5, TAS2R40, KRTAP2-3,
  SPRR2A, USP17L5, AGAP9). They may be false positives or guide-level effects.
* The 8 ridge-only hits are borderline under HistGB (q 0.06 to 0.14). The smallest attainable q is 0.007 (HistGB) and 0.011
  (ridge), so tiny q differences mean nothing.""")

code("""display(pd.read_csv(find('model_compare_clock.csv')).round(3))
display(pd.read_csv(find('model_compare_stage.csv')).round(3))
print(open(find('compare_ko_ridge_hgb_log.txt')).read()[:1600])""")

code("""H = pd.read_csv(find('ko_library_cycle_hgb.csv'))
print('HistGB library: knockouts with a significant cycle-distribution change:', int((H.q_cycle < .05).sum()))
H[H.q_cycle < .05].sort_values('cycle_TV', ascending=False)[[c_ for c_ in cols if c_ in H]].head(25).round(3)""")

md("""### 2d. Measured DNA for 111 knockouts, and Brian's hypertrophy test

**What.** Are the predicted cycle shifts real? And does Brian's concern hold, that G1-arrested cells keep growing and are then
called G2/M by phase contrast?

**How.**
* **Knockouts:** the 81 with a predicted cycle shift (ridge library, q_cycle < 0.05) and 30 calibration knockouts with none.
* **DNA per cell:** background-corrected DAPI, rounds 2 to 4, log2 relative to the 2N peak. 2N below 0.25, S 0.25 to 0.65, 4N above
  0.65 (the NTC cut-offs). 22,152 of 26,628 measured cells kept (a nucleus in all three rounds, one nucleus per cell).
* **Test:** change in % 2N and % 4N vs NTC guides pooled to the same cell count (2,000 draws), BH FDR separately for each; a
  knockout is "confirmed" if either passes at 5%.
* **Hypertrophy:** four 4N-vs-not phase models trained on NTC and scored on knockout cells with measured DNA. *Standard* (the
  model used so far), *size-balanced* (trained so that size alone cannot separate 2N from 4N), *size-removed* (cell area regressed
  out of the features), and *DNA floor* (Brian's rule: no 4N call below 2.5N of phase-predicted DNA).

**Result: the predictions hold.**
* **Predicted vs measured change in % 4N: Spearman 0.76, Pearson 0.84, slope 1.01** (n = 111; 0.63 within the 30 calibration
  knockouts alone). Predicted shifts are on the right scale.
* **42 of 81 predicted shifters are confirmed** (16 on % 4N, 26 on % 2N only), and 41 of the 42 have the predicted sign (the exception
  is SNRPG). 38 of the 42 lose 4N cells: ribosome-biogenesis, proteasome, chaperone and splicing genes such as HSPA9 (-11.6 points),
  IMP4, PSMD4, SRSF3 and WDR12. The 4 confirmed 4N gains are all DNA replication genes: MTBP +17.0, MCM7 +12.2, CDC45 +8.4,
  POLA2 +7.9.
* **7 of 30 calibration knockouts also shift**, always in the predicted direction and in 6 of 7 more strongly than predicted (MED6:
  predicted -3.3, measured -9.9). The largest is the mitotic kinesin KIF11 (+19.5 points 4N, n = 67). So the library misses some real
  shifts. The 39 unconfirmed shifters may be false positives or may reflect limited power at about 200 cells; this set cannot tell.

**Result: the hypertrophy idea is not supported.**
* **Within NTC there is a size bias:** the standard model calls 3.4% of 2N cells 4N, but 15.6% of the largest tenth of 2N cells.
* **Across knockouts it does not matter.** Measured-2N cells are larger than NTC 2N cells in 102 of 111 knockouts (median about 17%
  more area), yet the excess of false 4N calls over NTC has median -1.0 point and does not rise with 2N cell size (Spearman 0.08,
  p 0.41).
* **The G1-list knockouts in this batch** (CUL1, POLE, PSMB6, RPTOR, RRM1) have false-4N excesses of -1.1 to +0.7 points. RPTOR, the
  only one with a significant measured shift, gains 2N cells (+15.2 points) and phase contrast predicts that direction.
* **Correcting for size makes things worse:** agreement with measured DNA across knockouts is 0.86 (standard), 0.86 (DNA floor), 0.75
  (size-balanced) and 0.69 (size-removed). **The DNA floor changes no knockout by more than 0.63 points**; it is a harmless safety check.

**Caveats.** The 81 shifters were chosen for having predicted shifts, which helps the correlation. With 2,000 draws the smallest p is
about 0.0005, and some large shifts miss significance (GOLGA6L1 +8.8 points, q 0.08). Confirming on either % 2N or % 4N lets the
combined FDR exceed 5% somewhat. The 4N cut (0.65 log2, about 3.1N) counts late-S cells as 4N, which matters for the replication
genes. Only 5 G1-list knockouts are in this batch; the second batch below adds the other 47 arrest-list genes.

**Data.** Every phase-imaged cell of these 111 knockouts (53,471) with 4i, phase contrast and measured DNA where measured:
`/srv/team/data/crossmodal/ops0077_ko/OPS0077_KOcyc_Multimodal_4i_Phase_DNA.h5ad` (same schema and NTC scaling as the NTC file).
pRb, Rb and p21 are missing for 19% of cells, so 3,406 measured 2N-range cells have no `cycle_stage`.""")

code("""display(Image(filename=find('ko_measured_vs_predicted_dna.png')))
M = pd.read_csv(find('ko_measured_vs_predicted_dna.csv'))
M.sort_values('measured_d_pct_4N')[['gene', 'set', 'n_measured', 'measured_d_pct_2N', 'measured_d_pct_4N', 'q_2N', 'q_4N', 'predicted_d_pct_4N', 'measured_significant']].round(3)""")

code("""display(Image(filename=find('g1_hypertrophy_check.png')))
display(pd.read_csv(find('model_summary_ntc.csv')).round(3))
pd.read_csv(find('model_vs_measured_across_knockouts.csv')).round(3)""")

md("""**Second batch: the remaining 47 arrest-list genes (158 knockouts in all, 32,060 cells with measured DNA).**
* **The new knockouts were not selected on a predicted shift, and the prediction still holds:** Spearman 0.64 for the 47 alone,
  0.78 over all 158 (slope 1.02). With 158 tests, 41 of the 81 predicted shifters stay confirmed and 6 of the 30 calibration
  knockouts.
* **By expected arrest phase (all 60 arrest-list genes):**
  * S group (9): median +4.7 points of 4N cells; MCM7 and POLA2 significant.
  * G1 group (26): median +6.1 points of 2N cells and -2.5 of 4N, the direction of a G1 accumulation, but only RPTOR is
    significant on its own. The replication genes on this list (RPA2 +11.5 points 4N, q 0.06; POLE) and CCNB1 gain 4N cells instead.
  * G2/M group (25): median +3.2 points of 2N and -2.5 of 4N, the opposite of a G2/M arrest. Only CDC45 gains 4N cells; ATRX and
    ASNS are significant 2N gains. Mitotic, ESCRT and centromere knockouts are not hidden as binucleate cells (see 2b).
* **Hypertrophy, all 158:** measured-2N cells are larger than NTC 2N cells in 148 knockouts, the excess of false 4N calls has
  median -1.0 point and does not rise with size (Spearman 0.09). The 26 G1-list knockouts range from -3.4 to +3.2 points (median
  -1.0; APC highest). Agreement with measured DNA: standard 0.86, DNA floor 0.86, size-balanced 0.74, size-removed 0.67.""")

code("""display(Image(filename=find('ko_measured_vs_predicted_dna_all.png')))
M2 = pd.read_csv(find('ko_measured_vs_predicted_dna_all.csv'))
display(M2.dropna(subset=['expected_arrest_phase']).groupby('expected_arrest_phase').agg(n=('gene', 'size'),
        median_d_pct_2N=('measured_d_pct_2N', 'median'), median_d_pct_4N=('measured_d_pct_4N', 'median'), n_significant=('measured_significant', 'sum')).round(2))
display(pd.read_csv(find('model_vs_measured_across_knockouts_all.csv')).round(3))
mn = pd.read_csv(find('ko_multinucleate_by_gene.csv'), index_col=0)
mn.sort_values('pct_multi', ascending=False).head(12).round(3)""")

md("""## 3. Regressing the cycle out: does heterogeneity go down?

**What.** The project hypothesis: removing the cell-cycle component from perturbation profiles should reduce the apparent heterogeneity
of perturbed populations.

**How.**
* **What is removed.** From every phase-contrast dimension (PCA-100) we remove the part explained by the predicted cycle (sine and
  cosine, first and second harmonics). The coefficients are fit on NTC cells using out-of-fold predictions.
* **Heterogeneity** is measured two ways: the spread of each knockout population (mean distance to its centroid), and how separable
  it is from NTC (held-out logistic AUC).
* **Controls.** A shuffled cycle is the control. A knockout counts as "separating" when its AUC is above the 95th percentile of NTC
  guide pools of the same size.

**Result: heterogeneity barely changes.**
* **Spread** falls by a median of **3.0%** (shuffled cycle: +0.004%), the same for arrest and non-arrest knockouts.
* **Separation from NTC** is essentially unchanged: median AUC change -0.001. 595 knockouts separate from NTC before and 610 after,
  so removing the cycle neither erases perturbation signal nor sharpens it overall.
* **Per-knockout AUC changes are noise; do not read them gene by gene.** Two independent runs on the same 120 knockouts reproduce
  the AUC itself (r 0.91), but not the before-to-after change (r 0.02; the run-to-run difference, SD 0.036, is as large as the
  changes). Which knockouts' phenotypes are "mostly cell cycle" is answered by the deterministic decomposition in section 4:
  the cycle shift explains at most about a quarter of any knockout's morphology change.

**Why so little:** the cell cycle is a *predictable* but *low-variance* direction of phase morphology. Even the **measured** cycle
explains only 2.9% (harmonics 1 to 2) to 5.7% (all four 4i readouts) of NTC phase-morphology variance, while the first principal
component alone explains 12.3%.""")

code("""print(open(find('cycle_variance_bound.txt')).read())
display(Image(filename=find('library_cycle_decomposition.png')))
R[['spread_change_pct', 'spread_change_shuffled_pct', 'auc_before', 'auc_after', 'auc_change']].describe().round(3)""")

code("""# reproducibility of per-knockout AUC values and AUC changes: the 120-gene run (section 1) vs the library run, same knockouts
a = pd.read_csv(find('ko_heterogeneity_before_after.csv')); m = a.merge(R, on='gene', suffixes=('_120', '_library'))
pd.Series({'knockouts in both runs': len(m),
           'AUC before, r between runs': m.auc_before_120.corr(m.auc_before_library),
           'AUC after, r between runs': m.auc_after_120.corr(m.auc_after_library),
           'AUC change, r between runs': m.auc_change_120.corr(m.auc_change_library),
           'SD of the run-to-run difference in AUC change': (m.auc_change_120 - m.auc_change_library).std(),
           'SD of AUC change across the library': R.auc_change.std()}).round(3)""")

md("""## 4. Three questions about normalising by the cell cycle

Does normalising or weighting cell-cycle scores (1) reduce field and well heterogeneity, (2) find perturbations that change the
cell-population distribution, or (3) separate those from perturbations that only change the cells?

**Two normalisations:**
* **regress**: remove the cycle-explained part of every morphology dimension;
* **reweight**: weight reference cells so that their cycle distribution matches the target.

**Two datasets:**
* **OPS0077**: phase contrast; cycle predicted from morphology, with the measured 4i clock as a check.
* **Element**: imaging features; cycle **measured from RNA**, so it cannot be circular with morphology. It includes a TNFa stimulus and 5 knockouts.

### Q1. Field and well heterogeneity: not reduced

**How.** For control cells we measure the share of morphology variance that lies **between wells or between fields** (tiles with at
least 30 cells), minus what random sampling alone gives (permuted labels).

**Result.**
* **With a measured clock, normalisation changes almost nothing:**
  * Element wells: 1.67% before, 1.72% after regressing, 1.67% after reweighting.
  * Element fields: 7.2%, 7.4%, 7.2%.
  * OPS0077 fields: 7.9%, 7.8%, 7.9%.
* **The apparent reduction with the predicted cycle (OPS0077 fields 12.4% to 10.7%) is circular.** Wells do not differ in measured
  cycle composition beyond chance (spread across wells of the share of cells in the second half of the clock, 0.34% vs 0.32% if random), but they do in predicted
  composition (2.8% vs 0.2%). The predictor reads a technical well difference as "cycle".
* **Why:** fields do differ somewhat in measured cycle composition (local density), but that difference explains at most about 2% of the field effect.""")

code("""display(Image(filename=find('q1_field_well_heterogeneity.png')))
pd.read_csv(find('q1_field_well_heterogeneity.csv'))""")

md("""### Q2 and Q3. Cycle mix versus the cells themselves

**How.** Each perturbation's morphology shift (centroid vs control) is split exactly into two parts:
* **cycle-mix part**: the shift control cells would show if they had the perturbation's cycle distribution (reweighting);
* **matched-cycle part**: the remaining shift, at a matched cycle state.

Each part is tested against the same null as above (NTC guides pooled to n for OPS0077). For Element the null is halves of baseline
wells for TNFa, and single control guides for the knockouts.

**Result (OPS0077, 1,000 knockouts).**

| class | knockouts |
|---|---|
| cells change, cycle distribution unchanged | 790 |
| both | 80 |
| cycle distribution only | 1 (HSPA14) |
| neither | 129 |

**Even among knockouts that shift the cycle, the cycle shift explains little of the morphology change** (median share 0.01, 10th to
90th percentile -0.11 to +0.16).

**Result (Element, measured RNA clock).**
* **TNFa** shifts the cycle distribution slightly (TV 0.038 vs largest null 0.028) and moves morphology a lot (shift 17.8), but its
  cycle-mix part is 0.3% of that.
* **CTNNB1 and TGFBR2 knockouts** shift the cycle (CTNNB1 clearly), yet their cycle-mix part is under 2% of their morphology change.

**So:**
* **(2) yes**, cycle distribution changes can be found, best with a measured clock;
* **(3) yes**, the decomposition separates the two, and in both datasets perturbations that change morphology do so at a matched cycle state. None acts only by changing the cycle mix.""")

code("""display(Image(filename=find('element_cycle_decomposition.png')))
pd.read_csv(find('q23_element_decomposition.csv')).round(3)""")

md("""## 5. A cell-cycle representation of the CROP-seq knockouts, mapped to phase contrast

**The data.** The Altair CROP-seq data are public only at knockout level: DESeq2 knockout-vs-control statistics for 1,000 knockouts
x 9,999 genes. There are no single cells, so the representation is per knockout. The CROP-seq cells are not the imaged cells; the
knockout is the link.

**How the representation is built.**
* **Cycling genes from single cells.** Element (A549, single-cell RNA with a measured RNA clock) gives the expression of every gene
  around the cycle. We keep 397 strongly cycling genes: peak-to-trough at least 0.8 log2, and the same profile in two halves of the
  wells (r at least 0.8).
* **Programmes.** Genes are grouped by where they peak: S, G2/M or G1, using Element's own phase calls and DNA content per clock bin.
* **Score per knockout.** The mean DESeq2 statistic over each programme (the knockout's own gene excluded), as a z-score against
  1,000 random gene sets matched for expression.
* **A composition deconvolution was also tried and is a documented negative.** It models the knockout as a re-weighting of cells
  along the control clock. It was not reproducible across halves of the genes (rho 0.05), and 0 of 997 knockouts passed its null.

**Result: the CROP-seq cycle signal is weak and mostly absent.**
* **Few knockouts move a programme.** 14 of 997 move any programme beyond |z| 4.
* **One standout.** SMARCD1 (S z -12, G2/M z -18) stands out. In phase contrast it does not shift the predicted cycle distribution (q 0.30).
* **Reliability is weak across all knockouts** (G2/M programme split-half 0.23), though high among the few strong ones (G1: 0.95).
* **The canonical arrest genes are flat in CROP-seq RNA** (CCNB1, CDK7, AURKB, CDC45, POLA1: all |z| < 3), although several are
  strong in imaging.""")

code("""display(Image(filename=find('crop_cycle_overview.png')))
display(pd.read_csv(find('crop_programme_reliability.csv')))
P = pd.read_csv(find('crop_cycle_programmes.csv'))
P.assign(z_max=P[['z_S', 'z_G2M', 'z_G1']].abs().max(1)).nlargest(15, 'z_max')[['gene', 'z_S', 'z_G2M', 'z_G1']].round(2)""")

md("""**Mapping to phase-contrast morphology, two routes.**
* **Route 1, per cell, no RNA used for training.** The NTC-trained morphology model predicts each knockout's change in % 4N cells, which we compare with its RNA programmes.
* **Route 2, gene level.** A ridge regression from the knockout's mean phase profile to its RNA programme scores, 10-fold cross-validated across knockouts, against permuted labels.
* **Strength control.** Essential-gene knockouts move every readout, so agreement is also given after partialling out overall perturbation strength in each modality.

**Result: phase contrast and CROP-seq agree only weakly on the cell cycle.**
* **Route 1:** RNA G2/M programme vs morphology-predicted change in % 4N gives rho 0.11 (p 7e-4; 0.12 after the strength control).
* **Route 2:** cross-validated rho is 0.10 to 0.16 for the G2/M, G1 and proliferation scores, above the permuted 95th percentile
  (at most 0.03), but R2 is at most 0.09. The S programme is not predicted at all (rho -0.10).
* **The two modalities disagree on which knockouts change the cycle.** None of the 14 strongest RNA knockouts is significant in
  morphology, and the strongest morphology knockouts (RRM1, CDC45, POLR1B, ribosomal genes) show no RNA programme shift.

**Possible reasons** (not tested here): the two screens are different experiments (cells, time point, selection); knockout-level
pseudobulk dilutes a shift carried by part of the cells; and CROP-seq self-knockdown was weak for several targets. **For cell-cycle
readouts of these knockouts, the imaging data (phase contrast, validated against 4i pRb and DNA) are the stronger source.**""")

code("""display(Image(filename=find('crop_to_phase_overview.png')))
display(pd.read_csv(find('route1_per_cell_agreement.csv')))
pd.read_csv(find('route2_gene_level_cv.csv'))""")

md("""## Takeaways

1. **Fair comparisons change the knockout story.** With all NTC cells as an out-of-fold reference and a guide-pooled null, CDK7 and
   CCNB1 shift by about +3 to +4 points (not significant), not +7 to +11. Across the library, 81 of 1,000 knockouts change the
   predicted cycle distribution, with sensible biology (replication and mitosis genes gain 4N cells; ribosome, translation and
   chaperonin genes lose them), validated by measured pRb (rho 0.37) and by **measured DNA** for 111 knockouts (predicted vs measured
   change in % 4N: Spearman 0.76, slope 1.01; 42 of 81 shifts confirmed, 41 with the predicted sign). Gradient boosting on raw phase
   features is the better cycle model (21.8 vs 29.3 degrees) and finds 194 shifting knockouts with the same pRb agreement.
   **Brian's hypertrophy concern is not supported:** knockout 2N cells are larger but not called G2/M more often, and his DNA floor
   changes nothing.
2. **Regressing the cell cycle out does not reduce the heterogeneity of perturbed populations in a meaningful way.** Spread falls by
   about 3% and separation from NTC is unchanged (595 vs 610 knockouts), because the cycle explains only 3 to 6% of phase-morphology
   variance even when measured. Per-knockout AUC changes are not reproducible between runs, so only the aggregate is interpretable.
3. **Normalising by the cycle does not remove field or well effects** (no change with a measured clock). Doing it with a
   morphology-predicted cycle looks helpful only because the predictor absorbs technical differences.
4. **The cycle-mix vs matched-cycle decomposition is the useful output.** It finds perturbations that change the cycle distribution
   and shows that, in both OPS0077 and Element, morphology changes happen at a matched cycle state; none of 1,000 knockouts acts only through the cycle mix.
5. **CROP-seq knockout-level RNA carries little cell-cycle signal** that agrees with imaging (rho about 0.1). The Element single-cell
   data remain the only place where RNA and imaging cell-cycle state are measured in the same cells.""")

nb = new_notebook(); nb['cells'] = c
nbf.write(nb, sys.argv[1]); print('wrote', sys.argv[1], 'with', len(c), 'cells')
