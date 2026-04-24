from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELING_INPUT_DIR = ROOT / "data" / "intermediate" / "cleaned_commit_patch_semantic_enriched"

# Conservative predictive-feature pruning approved for the final paper reruns.
# These features were retained previously, but they are both sparse in the
# modeling data and negligible in the current RQ2 summaries.
PRUNED_PREDICTIVE_FEATURES = [
    "ct_cross_service_change",
    "ct_db_migration_touched",
    "ct_helm_touched",
    "ps_message_broker_changed",
    "ps_feature_flag_changed",
    "ps_port_changed",
    "ps_workflow_concurrency_changed",
    "ps_wait_or_timeout_changed",
    "ps_workflow_cache_changed",
    "ps_test_task_lines_changed",
    "ps_docker_copy_instruction_changes",
    "ps_workflow_secret_changed",
    "ps_mock_changed",
    "ps_workflow_runner_changed",
]

BASE_EXCLUDED_FEATURES = [
    "window_std_7",
    "run_attempt",
    "secs_since_prev",
    "hour",
    "day_or_night",
    "dow",
    "month",
]

EXCLUDED_PREDICTIVE_FEATURES = BASE_EXCLUDED_FEATURES + PRUNED_PREDICTIVE_FEATURES
