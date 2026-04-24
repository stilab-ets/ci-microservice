# Replication Package

This folder contains the replication package for the research work:

**Understanding and Predicting CI Workflow Duration: Temporal Dynamics and Regime Shifts in Microservices**

It includes the archived study inputs, the preprocessing and analysis scripts, and the manuscript assets needed to regenerate the reported study outputs.

## Package Structure

- `data/raw/` contains the archived workflow-run histories collected for the study.
- `data/collected_commit_features/` contains the archived commit-level feature exports used during preprocessing.
- `data/collected_patch_semantic_features/` contains the archived patch-semantic feature exports used during preprocessing.
- `data/collected_metadata/` contains archived metadata used to render the project-overview table.
- `data/frozen_paper_inputs/` contains the processed input snapshot used by the main replication workflow.
- `scripts/clean_data/`, `scripts/process_data/`, `scripts/prepare_data/`, `scripts/pq/`, `scripts/rq1/`, and `scripts/rq2/` contain the package scripts.
- `paper/` contains the manuscript source (`main.tex`), bibliography/class files, and the static study-design figure. Running the scripts regenerates the paper figures and table fragments under this folder.

## Replication Workflows

Two workflows are provided.

### 1. Main replication workflow

This is the primary workflow for regenerating the study outputs reported in the manuscript.

It uses the archived processed snapshot in `data/frozen_paper_inputs/`.

### 2. Preprocessing rebuild workflow

This workflow reruns the cleaning and preprocessing chain from the archived collected inputs in `data/raw/`, `data/collected_commit_features/`, and `data/collected_patch_semantic_features/`.

It is included to document the preprocessing pipeline and to support provenance-oriented inspection of the intermediate datasets.

## Requirements

- Python 3.11+ is recommended.
- Install the Python dependencies with:

```powershell
python -m pip install -r requirements.txt
```

- Optional: to compile `paper/main.tex`, install a LaTeX distribution that provides `pdflatex` and `bibtex`.

## Main Replication Workflow

From inside `replication_package`, run:

```powershell
.\run_exact_replication.ps1
```

This script executes the following steps:

1. Render the selected-project overview table from the archived metadata CSV.
2. Run the PQ stable-region analysis from the archived workflow histories.
3. Copy the PQ paper figures and generate the regime-shift cases table fragment.
4. Generate the motivating `daos` figure.
5. Run the full RQ1 model evaluation on the archived modeling snapshot.
6. Generate the RQ1 full metrics table and the retained-feature summary table.
7. Run the local regime-shift explanation analysis used for RQ2.
8. Generate the RQ2 top-10 increase/decrease table fragments.

Generated outputs will appear in:

- `results/pq/stable_regions/`
- `results/rq1/`
- `results/rq2_local_explanations/`
- `paper/figures/`
- `paper/generated_tables/`

The main manuscript-facing generated files are:

- `paper/generated_tables/project_overview_table.tex`
- `paper/generated_tables/feature_summary_retained.tex`
- `paper/generated_tables/rq1_full_metrics_table.tex`
- `paper/generated_tables/rq2_regime_shift_cases_table.tex`
- `paper/generated_tables/rq2_increase_top10.tex`
- `paper/generated_tables/rq2_decrease_top10.tex`
- `paper/figures/motivating_example_daos_boundary_shift.png`
- `paper/figures/prelim_stable_region_counts.png`
- `paper/figures/prq_example_bruce_no_shift.png`
- `paper/figures/prq_example_radare2_three_shifts.png`
- `paper/figures/prq_example_ouds_android_five_regions.png`
- `paper/figures/rq2_shift_window_example_bmad.png`
- `paper/figures/rq2_increase_shift_feature_heatmap.png`
- `paper/figures/rq2_decrease_shift_feature_heatmap.png`

## Manual Command Sequence

If the replication steps should be run manually instead of through the PowerShell helper script:

```powershell
python .\scripts\prepare_data\render_project_overview_table.py
python .\scripts\pq\detect_stable_regions.py
python .\scripts\pq\prepare_pq_paper_assets.py
python .\scripts\pq\generate_motivating_example_daos_figure.py
python .\scripts\rq1\run_rq1_models.py
python .\scripts\rq1\generate_rq1_full_table_tex.py
python .\scripts\rq1\generate_feature_summary_retained_tex.py
python .\scripts\rq2\run_rq2_regime_shift_local_explanations.py
python .\scripts\rq2\generate_rq2_top10_tables.py
```

## Preprocessing Rebuild Workflow

To rerun the cleaning and preprocessing stages from the archived collected inputs:

```powershell
.\run_data_pipeline.ps1
```

That command runs:

```powershell
python .\scripts\clean_data\data_cleaning.py
python .\scripts\process_data\merge_commit_features.py
python .\scripts\process_data\merge_patch_semantic_features.py
python .\scripts\prepare_data\prepare_modeling_data.py
```

These rebuilt datasets are written to:

- `data/intermediate/cleaned/`
- `data/intermediate/cleaned_commit_enriched/`
- `data/intermediate/cleaned_commit_patch_semantic_enriched/`
- `data/prepared/modeling/`

## Optional Paper Compilation

If `pdflatex` and `bibtex` are installed, the paper can be compiled after running the main replication workflow:

```powershell
Set-Location .\paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`main.tex` is included as part of the package. Some tables are written directly in the manuscript source, so the generated `.tex` files under `paper/generated_tables/` are provided as reproducible companion assets for the reported results.

## Notes

- The package does not include precomputed result folders; running the scripts regenerates them locally.
- The main replication workflow is the intended path for regenerating the manuscript outputs, while the preprocessing rebuild workflow documents the archived preprocessing chain.
