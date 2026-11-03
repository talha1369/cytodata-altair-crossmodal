# Knockouts, the cell cycle and heterogeneity (notebook 18)

Outputs behind `18_ko_cell_cycle_heterogeneity.ipynb`. Scripts are in `scripts/`; run them in this order from the folder holding the
OPS0077 NTC files (`/srv/team/data/crossmodal/ops0077_ntc/`) and `cell_cycle_scores.csv`.

| step | script | writes |
|---|---|---|
| 1 | `extract_phase_genes.py` (arrest list, `control_genes.csv`) | per-cell phase features for 60 arrest + 60 random library genes |
| 2 | `ko_cycle_heterogeneity.py` | `ko_predicted_cycle_composition.csv`, `ko3_measured_vs_predicted.csv`, `ko_heterogeneity_before_after.csv`, `ko_cycle_overview.png` |
| 3 | `cycle_var_bound.py` | `cycle_variance_bound.txt` (variance of NTC morphology explained by the measured cycle) |
| 4 | `cycle_norm_questions.py` (`NULL_REPS=2000`) | `q1_field_well_heterogeneity.csv`, `q23_ops0077_knockout_decomposition.csv`, `q23_element_decomposition.csv` |
| 5 | `extract_phase_all.py` | per-cell phase features for all 1,000 OPS0077 knockouts (591,136 cells) |
| 6 | `ko_cycle_library.py` | `ko_library_cycle.csv`, `ko_library_mean_profiles.npz` |
| 7 | `cc_ko_score.py`, `crop_cycle_deconv.py`, `crop_cycle_sets.py` (`crop_diag.py`: model-free check) | `crop_cycle_representation.csv` (deconvolution, a documented negative), `crop_cycle_programmes.csv`, `crop_programme_reliability.csv`, `element_cycling_gene_profiles.csv`, `crop_cycle_overview.png` |
| 8 | `map_crop_to_phase.py` | `route1_per_cell_agreement.csv`, `route2_gene_level_cv.csv`, `crop_vs_phase_by_ko.csv`, `crop_to_phase_overview.png` |
| 9 | `ko_cycle_figures.py` | `q1_field_well_heterogeneity.png`, `library_cycle_decomposition.png`, `element_cycle_decomposition.png` |
| 10 | `build_ko_cycle_nb.py` | the notebook (display-only) |
| 11 | `ko_dna_light.py` (`list`, `validate`, `measure`; laptop, anonymous S3) then `ko_dna_hgb_only_stats.py` | `ko_dna_hgb_only/`: measured DNA for the 111 knockouts only the HistGB run calls (`hgb_only_measured_dna.csv`), all 269 measured knockouts with pooled FDR (`all_measured_knockouts.csv`), confirmation by model call (`confirmation_by_model_call.csv`, `hgb_only_measured_dna.png`), logs |
| 12 | `ko_guide_dna.py` | `ko_dna_hgb_only/guide_heterogeneity_measured_dna.csv` and `per_guide_measured_dna.csv`: do the guides of one knockout differ in measured DNA more than NTC guides of the same size (`guide_log.txt`) |

`*_log.txt` are the console logs of the runs that produced these tables. Data: OPS0077 (Biohub OPS Explorer, public S3
`ops-explorer-public`), the Altair CROP-seq knockout-level DE table (same bucket, `leonetti_ops/ops_data_portal_submission/CROPseq/`),
and the Element multiomics object on the team VM.
