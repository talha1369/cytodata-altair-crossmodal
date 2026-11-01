# Cross-modal analysis of the Altair multimodal perturbation atlas

Phase-imaging morphology ↔ scRNA-seq, at the gene level, on the CZ Biohub **Altair** atlas
(1,000 pooled CRISPR knockouts in A549; fluorescence + **label-free quantitative phase** imaging +
**scRNA-seq**, ~57M cells). Prepared for the **CytoData hackathon 2026** (multimodal-representations workstream).

**Core question:** do morphology embeddings and transcriptome embeddings encode the same gene-function structure —
and where do they diverge?

## Notebooks
- **`altair_inspection.ipynb`** — what's in each embedding; sanity-check UMAPs; three tests of cross-modal
  structure (gene–gene similarity correlation, kNN neighbour overlap, CCA); CORUM protein-complex validation.
- **`cross_modal_prediction.ipynb`** — held-out morphology↔expression prediction (5-fold out-of-fold ridge,
  per-fold scaling), cross-modal retrieval metrics, cross-validated CCA, and per-gene predictability.
- **`guide_level_cross_modal.ipynb`** — the same held-out predictor at **single-guide** resolution, built on
  Tim Treis's [`cytodata-hackathon-prep-ops`](https://github.com/timtreis/cytodata-hackathon-prep-ops) phase-guide
  files joined to CROP-seq by guide sequence; guide→gene aggregation; and a test of what the **4i fixed panel
  (OPS0077: γH2AX/p21/p53/pRb/Rb/RPS6)** adds over phase for predicting RNA (answer: phase ≫ 4i; phase+4i ≈ phase).
- **`altair_vs_posh.ipynb`** — within-A549 cross-screen reproducibility (Altair phase vs CellPainting-POSH, both A549), the clean within-cell-line version of the cross-screen comparison. Manifest-driven and format-robust: it auto-detects POSH via `download_manifest.json` or `data/*posh*`, auto-detects the gene column and features, and prints 'awaiting POSH' until the data lands. Validated end-to-end on a synthetic POSH file.
- **`element_matching_calibration.ipynb`** + **`element_calibration.py`** — single-cell RNA to imaging-morphology matching calibration on the Element true-pair screen (ground truth for a future pooled screen). Per-cell link is weak but real (R2 ~0.04, ~25x chance); **whole-cell features carry ~2.3x the RNA signal of nuclear-region** (perinuclear crops cost >half); interpretable 26-feature panel recovers ~half the full R2; matching scales smoothly with link strength. Axes: pooling (k=1..400), feature panel, extent, link strength; shuffled null each.
- **`altair_vs_aconcagua.ipynb`** — cross-screen reproducibility: builds the guide/gene-level **Aconcagua** table
  (a `groupby` on the 323 GB Funk/Brieflow cell table; full DuckDB recipe for the VM + an offline sampled demo) and
  compares its gene-gene structure to Altair phase. Two OPS screens in different cell lines (Altair A549, Aconcagua HeLa) share real gene-function structure
  (significant vs null) across different imaging modalities; modest in the lean demo, expected to strengthen at full scale.

## Key results (1,000 shared genes)
- Gene–gene similarity correlates **ρ ≈ 0.47** across phase morphology and scRNA-seq (z ≈ 100 vs shuffled null).
- Held-out **morphology→expression R² ≈ 0.37–0.46**; retrieves the true gene into the **top-10 of 1,000 ~36%** of
  the time (chance ~1%). Reverse direction symmetric in R².
- **Cross-validated** CCA keeps ~4–5 shared axes at r ≈ 0.8–0.95 (honest, not the inflated in-sample value).
- **Learned Cell-DINO embedding beats hand-crafted OrganelleProfiler phase**
- **Gene-level representation ranking: Cell-DINO > phase > 4i** (held-out R² ≈ 0.47 / 0.34 / 0.13 → RNA); Cell-DINO subsumes phase and 4i (combinations do not improve). See `guide_level_cross_modal.ipynb` §6. for predicting RNA (gene-level held-out R² ≈ 0.45 vs 0.37, top-10 retrieval ≈ 46% vs 36%; Cell-DINO is `atlas/aggregated_data.h5ad`).
- Same-CORUM-complex genes are close in **both** modalities → the shared structure reflects real biology.
- **~80% of each gene's neighbourhood is modality-specific** → imaging and transcriptome are complementary.
  Ribosomal / COPI-secretory genes are best cross-predicted; cytoskeleton/actin, membrane trafficking, and
  cell-cycle genes are worst (their morphological phenotype does not reduce to their transcriptome).

## Data
Precomputed **gene-level** embeddings from Zenodo record [20495192](https://zenodo.org/records/20495192)
(`altair.zip`; no raw data — raw imaging + RNA are at biohub.ai). Paper: *A multimodal perturbation atlas
defines the phenotypic resolution of cellular morphology* (CZ Biohub). Upstream analysis repo:
[`czbiohub-sf/ops-paper-analysis`](https://github.com/czbiohub-sf/ops-paper-analysis).

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.11 .venv
pip install -r requirements.txt
python -m ipykernel install --user --name altair
mkdir -p data && curl -sL -o data/altair.zip "https://zenodo.org/records/20495192/files/altair.zip?download=1"
cd data && unzip -o altair.zip && cd ..
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=altair altair_inspection.ipynb cross_modal_prediction.ipynb
```
`aggregate_aconcagua.py` runs the full Aconcagua gene/guide aggregation on the VM (DuckDB, out-of-core). `analysis_core.py` / `analysis_prediction.py` are headless versions of the stats. Figures are written to `figures/`.
