from __future__ import annotations


def infer_feature_group(feature: str) -> str:
    if feature.startswith("ps_workflow_"):
        return "Patch workflow semantics"
    if feature.startswith("ps_build_") or feature.startswith("ps_dependency_") or feature.startswith("ps_lockfile_"):
        return "Patch build and dependency semantics"
    if feature.startswith("ps_docker_") or feature.startswith("ps_k8s_") or feature.startswith("ps_compose_"):
        return "Patch deployment semantics"
    if feature.startswith("ps_test_") or feature in {
        "ps_assert_changed",
        "ps_mock_changed",
        "ps_skip_test_changed",
        "ps_wait_or_timeout_changed",
    }:
        return "Patch test semantics"
    if feature.startswith("ps_api_") or feature.startswith("ps_config_") or feature in {
        "ps_http_endpoint_changed",
        "ps_message_broker_changed",
        "ps_env_key_changed",
        "ps_url_or_host_changed",
        "ps_port_changed",
        "ps_feature_flag_changed",
    }:
        return "Patch microservice and configuration semantics"
    if feature.startswith("ps_patch_") or feature in {
        "ps_import_changed_lines",
        "ps_logging_changed",
        "ps_exception_changed",
        "ps_todo_fixme_changed",
    }:
        return "Patch change shape"
    if feature.startswith("ct_"):
        return "Commit timeline structure"
    if feature in {"duration_lag_1", "window_avg_7", "window_std_7"}:
        return "Temporal persistence"
    if feature in {"secs_since_prev", "hour", "day_or_night", "dow", "month"}:
        return "Calendar context"
    if feature.startswith("ft_ms_"):
        return "Curated microservice file groups"
    if feature.startswith("yaml_"):
        return "Workflow YAML structure"
    if feature in {"total_jobs", "tests_ran", "run_attempt", "gh_is_pr", "branch", "workflow_run_count"}:
        return "Workflow context"
    if feature in {
        "gh_files_added",
        "gh_files_deleted",
        "gh_files_modified",
        "gh_lines_added",
        "gh_num_pr_comments",
        "gh_test_churn",
        "gh_test_lines_per_kloc",
        "gh_commits_on_files_touched",
        "gh_other_files",
        "gh_src_files",
        "gh_doc_files",
        "dockerfile_changed",
        "docker_compose_changed",
    }:
        return "Change activity and review context"
    if feature in {
        "gh_sloc",
        "dependencies_count",
        "total_builds",
        "project_age_days",
        "issuer_name",
    }:
        return "Developer and repository context"
    return "Other"


def display_feature_name(feature: str) -> str:
    aliases = {
        "ft_ms_workflow_yaml": "file_type_workflow_yaml",
        "ft_ms_api_contracts": "file_type_api_contracts",
        "ft_ms_jvm_build_and_code": "file_type_jvm_build_and_code",
        "ft_ms_frontend_web": "file_type_frontend_web",
        "ft_ms_scripts_ops": "file_type_scripts_ops",
        "ft_ms_config_data": "file_type_config_data",
    }
    return aliases.get(feature, feature)


FAMILY_LITERATURE_RATIONALE = {
    "Temporal persistence": "Time-aware CI and build-duration prediction studies motivate explicit recent-history signals.",
    "Developer and repository context": "Prior CI prediction work shows that who triggers a run and the surrounding repository state shape build behavior.",
    "Workflow context": "Workflow event, branch, and execution-shape context capture the immediate CI setting of each run.",
    "Change activity and review context": "Change-set size, review activity, and touched-file context are common explanatory signals in CI and JIT-style models.",
    "Workflow YAML structure": "Workflow-definition structure reflects execution parallelism, caching, and other pipeline-design decisions.",
    "Curated microservice file groups": "Microservice-oriented artifacts such as API contracts, workflow YAML, scripts, and config files proxy service coordination work.",
    "Commit timeline structure": "Commit-diff breadth and directory/extension dispersion approximate how widely a change spans the repository.",
    "Patch change shape": "Patch-shape signals summarize the kind of code edits being introduced inside the changed files.",
    "Patch workflow semantics": "Workflow-patch signals capture concrete edits to GitHub Actions behavior and automation logic.",
    "Patch build and dependency semantics": "Build-task, dependency, and lockfile edits capture changes to the build graph and dependency resolution.",
    "Patch deployment semantics": "Deployment-oriented edits capture infrastructure and container-level changes around execution and packaging.",
    "Patch microservice and configuration semantics": "Configuration, endpoint, host, and service-facing edits approximate operational changes that can alter workflow effort.",
    "Patch test semantics": "Test-oriented patch signals approximate how much the triggering change affects validation logic.",
}
