$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python .\scripts\clean_data\data_cleaning.py
python .\scripts\process_data\merge_commit_features.py
python .\scripts\process_data\merge_patch_semantic_features.py
python .\scripts\prepare_data\prepare_modeling_data.py

Write-Host "Collected-input rebuild finished. Rebuilt datasets are under .\\data\\intermediate and .\\data\\prepared\\modeling."
