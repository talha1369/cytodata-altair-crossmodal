# Cell cycle from label-free phase (OPS0077 NTC)

Predicting single-cell cell-cycle state from label-free **phase** morphology, on the OPS0077 (Altair, A549)
non-targeting (NTC) population, where variation is cell cycle rather than perturbation. Companion to
[`../phase_to_cell_cycle.ipynb`](../phase_to_cell_cycle.ipynb).

## Headline
Held-out (folds grouped by imaging tile), phase morphology recovers cell-cycle state well beyond cell size:

| subset | n | DNA content R2 | G1 vs G2/M AUC | size-only AUC | shuffled null |
|---|---|---|---|---|---|
| all reliable-DNA cells | 36,014 | 0.48 | 0.94 | 0.75 | 0.50 |
| trusted live-to-fixed match (<25 px) | 12,691 | 0.62 | 0.97 | 0.77 | 0.50 |

pRb (restriction point) is partly predictable from phase (R2 ~0.20). p21 is flat across the cycle in NTC cells and is
not used.

## Pipeline (each script reads the public OPS0077 data and writes one table, keyed by `cell_key` = well + 4i segmentation id)
1. `build_ntc_op.py` - 4i OrganelleProfiler panel for NTC cells (pRb, Rb, p21 + cell/nucleus shape), from the atlas
   per-marker `all_cells_fluor_*.h5ad` files.
2. `dapi_cpmeasure.py` - per-cell DAPI features (all 5 rounds) with `cp_measure` (numba), nuclei from `4i_R<k>_nuclear_seg`.
3. `bg_pass.py` - per-cell local background (median DAPI over non-nuclear crop pixels); exists for every cell.
4. `merge_dapi.py` - DNA content from DAPI rounds 2-4 (round 1 saturates), per-cell background subtracted; 2N/4N peaks.
5. `extract_phase_ntc.py` - per-cell phase OrganelleProfiler features (byte-range read of the 870 GB `all_cells_phase.h5ad`).
6. `combine_frame.py` - the aligned per-cell cell-cycle frame + a DNA x pRb `cycle_stage`.
7. `build_multimodal.py` - one multimodal AnnData (4i in `.X`, phase in `obsm['X_phase']`, DNA + stage in `obs`, spatial coords).
8. `build_phase_cycle_nb.py` - emits the analysis notebook.

Data: CZ Biohub OPS Explorer, public S3 `ops-explorer-public` (Biohub_OPS0077 = ops0144_20260406), region us-west-2.
