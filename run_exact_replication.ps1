$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python .\scripts\prepare_data\render_project_overview_table.py
python .\scripts\pq\detect_stable_regions.py
python .\scripts\pq\prepare_pq_paper_assets.py
python .\scripts\pq\generate_motivating_example_daos_figure.py
python .\scripts\rq1\run_rq1_models.py
python .\scripts\rq1\generate_rq1_full_table_tex.py
python .\scripts\rq1\generate_feature_summary_retained_tex.py
python .\scripts\rq2\run_rq2_regime_shift_local_explanations.py
python .\scripts\rq2\generate_rq2_top10_tables.py

Write-Host "Exact paper replication finished. Generated outputs are under .\\results, .\\paper\\figures, and .\\paper\\generated_tables."
