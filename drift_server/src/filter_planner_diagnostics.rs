use std::sync::OnceLock;

pub const FILTER_PLANNER_DIAGNOSTICS_ENV: &str = "DRIFT_FILTER_PLANNER_DIAGNOSTICS";

#[derive(Debug, Clone, Copy, Default)]
pub struct FilterPlannerDiagnosticsSnapshot {
    pub enabled: bool,
    pub query_has_filters: bool,
    pub global_exact_preselect_eligible_query: bool,
    pub global_exact_preselect_input_bucket_count: usize,
    pub global_exact_preselect_pruned_bucket_count: usize,
    pub catalog_exact_clause_eligible_query: bool,
    pub catalog_preselect_input_bucket_count: usize,
    pub catalog_preselect_pruned_bucket_count: usize,
    pub catalog_preselect_complete_may_match_bucket_count: usize,
    pub catalog_preselect_incomplete_bucket_count: usize,
    pub catalog_preselect_stale_bucket_count: usize,
    pub catalog_preselect_missing_bucket_count: usize,
    pub probed_bucket_count: usize,
    pub kept_bucket_count: usize,
    pub pruned_bucket_count: usize,
    pub candidate_produced_bucket_count: usize,
    pub candidate_applied_bucket_count: usize,
    pub candidate_id_count: usize,
    pub candidate_gated_broad_selectivity_bucket_count: usize,
    pub candidate_disabled_probe_error_bucket_count: usize,
    pub candidate_empty_exact_match_bucket_count: usize,
    pub candidate_no_indexed_exact_bucket_count: usize,
    pub candidate_range_stats_only_bucket_count: usize,
    pub candidate_other_absence_bucket_count: usize,
}

pub fn parse_truthy_env_value(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub fn diagnostics_enabled_from_env() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(FILTER_PLANNER_DIAGNOSTICS_ENV)
            .ok()
            .map(|raw| parse_truthy_env_value(&raw))
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::parse_truthy_env_value;

    #[test]
    fn parse_truthy_env_value_accepts_truthy_inputs() {
        for raw in ["1", "true", "TRUE", "yes", "on", " On "] {
            assert!(parse_truthy_env_value(raw), "expected truthy for '{raw}'");
        }
    }

    #[test]
    fn parse_truthy_env_value_rejects_non_truthy_inputs() {
        for raw in ["", "0", "false", "no", "off", "2", "maybe"] {
            assert!(
                !parse_truthy_env_value(raw),
                "expected non-truthy for '{raw}'"
            );
        }
    }
}
