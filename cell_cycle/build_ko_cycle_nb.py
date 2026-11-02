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

code("""import os, pandas as pd
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
  comparison (more late-cycle cells), so the reference is all NTC cells.
* **A null with the right noise.** NTC guides pooled to the knockout's cell count, then the same comparison, 400 draws, BH FDR.

**Result.** 8 of 120 knockouts shift the predicted cycle distribution (FDR 5%; 7 arrest genes, 1 control).
* **CDK7 and CCNB1 are not among them.** Their late-cycle (0.75 to 1) shifts are +3.2 and +3.8 percentage points (q 0.12 and 0.13),
  far smaller than the +10.8 and +7.3 in Brian's panel.
* **NTC composition.** Brian's NTC composition matches the *predicted* NTC composition, not the measured 4i one.
* **The sharp vs smooth curves in his panel B** are most likely a density-smoothing effect: knockouts have a few hundred cells and
  NTC tens of thousands, and the smoothing depends on cell count.

**Validation.** Across the 120 knockouts the predicted shifts track measured pRb on the same cells (last quarter of `cc_phase`
rho -0.40, second quarter +0.39). For the three genes with measured DNA, the predicted late-cycle gain matches the measured 4N gain for
CCNB1 (+4.3 measured vs +4.2 / +3.8 predicted) and MCM3 (+4.2 vs +2.7 / +2.9).""")

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
  cycle composition beyond chance (spread of the late-cycle share across wells 0.34% vs 0.32% if random), but they do in predicted
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
   chaperonin genes lose them), validated by measured pRb (rho 0.37).
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
