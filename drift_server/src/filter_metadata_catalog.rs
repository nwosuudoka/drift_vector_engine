use drift_storage::unified_format::{UnifiedLogicalType, UnifiedPayloadValue};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub const MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExactValueMembershipKey {
    pub field_id: u32,
    pub logical_type_tag: u16,
    pub encoded_value: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactValuePresence {
    Present,
    Absent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactFieldQueryClause {
    pub field_id: u32,
    pub value_keys: Vec<ExactValueMembershipKey>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeFieldQueryClause {
    pub field_id: u32,
    pub lower: Option<UnifiedPayloadValue>,
    pub lower_inclusive: bool,
    pub upper: Option<UnifiedPayloadValue>,
    pub upper_inclusive: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeFieldZoneMap {
    pub has_non_null_values: bool,
    pub min: Option<UnifiedPayloadValue>,
    pub max: Option<UnifiedPayloadValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketExactClauseCoverage {
    MissingBucket,
    StaleBucketPath,
    StaleBucketStats,
    Incomplete,
    CompleteMayMatch,
    CompleteNoMatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketRangeClauseCoverage {
    MissingBucket,
    StaleBucketPath,
    StaleBucketStats,
    Incomplete,
    CompleteMayMatch,
    CompleteNoMatch,
}

#[derive(Debug, Clone, Default)]
pub struct BucketProbeObservation {
    pub bucket_path: String,
    pub bucket_live_count: Option<u32>,
    pub indexed_exact_fields: HashSet<u32>,
    pub range_stats_fields: HashSet<u32>,
    pub exact_value_presence: HashMap<ExactValueMembershipKey, ExactValuePresence>,
    pub range_field_zone_maps: HashMap<u32, RangeFieldZoneMap>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FilterMetadataCatalogStats {
    pub bucket_count: usize,
    pub indexed_exact_field_memberships: usize,
    pub range_stats_field_memberships: usize,
    pub exact_value_memberships: usize,
}

#[derive(Debug, Clone, Default)]
pub struct FilterMetadataCatalog {
    buckets: HashMap<u32, BucketCatalogEntry>,
}

#[derive(Debug, Clone, Default)]
struct BucketCatalogEntry {
    bucket_path: String,
    bucket_live_count: Option<u32>,
    indexed_exact_fields: HashSet<u32>,
    range_stats_fields: HashSet<u32>,
    exact_value_presence: HashMap<ExactValueMembershipKey, ExactValuePresence>,
    range_field_zone_maps: HashMap<u32, RangeFieldZoneMap>,
}

impl FilterMetadataCatalog {
    pub fn observe_bucket_probe(&mut self, bucket_id: u32, observation: BucketProbeObservation) {
        let entry = self.buckets.entry(bucket_id).or_default();
        if !entry.bucket_path.is_empty() && entry.bucket_path != observation.bucket_path {
            // Bucket IDs are reused across maintenance operations. Reset stale memberships when
            // the backing bucket path changes.
            entry.bucket_live_count = None;
            entry.indexed_exact_fields.clear();
            entry.range_stats_fields.clear();
            entry.exact_value_presence.clear();
            entry.range_field_zone_maps.clear();
        }
        entry.bucket_path = observation.bucket_path;
        entry.bucket_live_count = observation.bucket_live_count;
        entry
            .indexed_exact_fields
            .extend(observation.indexed_exact_fields);
        entry
            .range_stats_fields
            .extend(observation.range_stats_fields);

        for (key, presence) in observation.exact_value_presence {
            if entry.exact_value_presence.len() >= MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET
                && !entry.exact_value_presence.contains_key(&key)
            {
                continue;
            }
            entry.exact_value_presence.insert(key, presence);
        }

        for (field_id, zone_map) in observation.range_field_zone_maps {
            entry.range_stats_fields.insert(field_id);
            entry
                .range_field_zone_maps
                .entry(field_id)
                .and_modify(|existing| merge_range_zone_map(existing, &zone_map))
                .or_insert(zone_map);
        }
    }

    pub fn classify_bucket_exact_clauses(
        &self,
        bucket_id: u32,
        expected_bucket_path: &str,
        expected_bucket_live_count: Option<u32>,
        clauses: &[ExactFieldQueryClause],
    ) -> BucketExactClauseCoverage {
        let Some(entry) = self.buckets.get(&bucket_id) else {
            return BucketExactClauseCoverage::MissingBucket;
        };
        if entry.bucket_path != expected_bucket_path {
            return BucketExactClauseCoverage::StaleBucketPath;
        }
        if expected_bucket_live_count.is_some()
            && entry.bucket_live_count != expected_bucket_live_count
        {
            return BucketExactClauseCoverage::StaleBucketStats;
        }
        if clauses.is_empty() {
            return BucketExactClauseCoverage::CompleteMayMatch;
        }

        let mut has_incomplete_clause = false;
        for clause in clauses {
            if clause.value_keys.is_empty() {
                has_incomplete_clause = true;
                continue;
            }

            let mut clause_has_present = false;
            let mut clause_has_unknown = false;
            for key in &clause.value_keys {
                match entry.exact_value_presence.get(key).copied() {
                    Some(ExactValuePresence::Present) => {
                        clause_has_present = true;
                        break;
                    }
                    Some(ExactValuePresence::Absent) => {}
                    None => {
                        clause_has_unknown = true;
                    }
                }
            }

            if clause_has_present {
                continue;
            }
            if clause_has_unknown {
                has_incomplete_clause = true;
                continue;
            }
            return BucketExactClauseCoverage::CompleteNoMatch;
        }

        if has_incomplete_clause {
            BucketExactClauseCoverage::Incomplete
        } else {
            BucketExactClauseCoverage::CompleteMayMatch
        }
    }

    pub fn classify_bucket_range_clauses(
        &self,
        bucket_id: u32,
        expected_bucket_path: &str,
        expected_bucket_live_count: Option<u32>,
        clauses: &[RangeFieldQueryClause],
    ) -> BucketRangeClauseCoverage {
        let Some(entry) = self.buckets.get(&bucket_id) else {
            return BucketRangeClauseCoverage::MissingBucket;
        };
        if entry.bucket_path != expected_bucket_path {
            return BucketRangeClauseCoverage::StaleBucketPath;
        }
        if expected_bucket_live_count.is_some()
            && entry.bucket_live_count != expected_bucket_live_count
        {
            return BucketRangeClauseCoverage::StaleBucketStats;
        }
        if clauses.is_empty() {
            return BucketRangeClauseCoverage::CompleteMayMatch;
        }

        let mut has_incomplete_clause = false;
        for clause in clauses {
            let Some(zone_map) = entry.range_field_zone_maps.get(&clause.field_id) else {
                has_incomplete_clause = true;
                continue;
            };
            match range_clause_overlaps_zone_map(zone_map, clause) {
                Some(true) => {}
                Some(false) => return BucketRangeClauseCoverage::CompleteNoMatch,
                None => has_incomplete_clause = true,
            }
        }

        if has_incomplete_clause {
            BucketRangeClauseCoverage::Incomplete
        } else {
            BucketRangeClauseCoverage::CompleteMayMatch
        }
    }

    pub fn stats(&self) -> FilterMetadataCatalogStats {
        let mut indexed_exact_field_memberships = 0usize;
        let mut range_stats_field_memberships = 0usize;
        let mut exact_value_memberships = 0usize;
        for entry in self.buckets.values() {
            indexed_exact_field_memberships += entry.indexed_exact_fields.len();
            range_stats_field_memberships += entry.range_stats_fields.len();
            exact_value_memberships += entry.exact_value_presence.len();
        }
        FilterMetadataCatalogStats {
            bucket_count: self.buckets.len(),
            indexed_exact_field_memberships,
            range_stats_field_memberships,
            exact_value_memberships,
        }
    }

    pub fn invalidate_bucket(&mut self, bucket_id: u32) -> bool {
        self.buckets.remove(&bucket_id).is_some()
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
    }
}

fn numeric_payload_value(value: &UnifiedPayloadValue) -> Option<f64> {
    match value {
        UnifiedPayloadValue::Int64(v) => Some(*v as f64),
        UnifiedPayloadValue::Float32(v) => Some(*v as f64),
        UnifiedPayloadValue::Float64(v) => Some(*v),
        UnifiedPayloadValue::TimestampMicros(v) => Some(*v as f64),
        _ => None,
    }
}

fn compare_payload_values(
    lhs: &UnifiedPayloadValue,
    rhs: &UnifiedPayloadValue,
) -> Option<Ordering> {
    if let (Some(a), Some(b)) = (numeric_payload_value(lhs), numeric_payload_value(rhs)) {
        return a.partial_cmp(&b);
    }

    match (lhs, rhs) {
        (UnifiedPayloadValue::Bool(a), UnifiedPayloadValue::Bool(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Keyword(a), UnifiedPayloadValue::Keyword(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Text(a), UnifiedPayloadValue::Text(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Keyword(a), UnifiedPayloadValue::Text(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Text(a), UnifiedPayloadValue::Keyword(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Bytes(a), UnifiedPayloadValue::Bytes(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::TimestampMicros(a), UnifiedPayloadValue::TimestampMicros(b)) => {
            Some(a.cmp(b))
        }
        _ => None,
    }
}

fn merge_range_min(
    lhs: Option<UnifiedPayloadValue>,
    rhs: Option<UnifiedPayloadValue>,
) -> Option<UnifiedPayloadValue> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => match compare_payload_values(&lhs, &rhs) {
            Some(Ordering::Greater) => Some(rhs),
            Some(_) => Some(lhs),
            None => None,
        },
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

fn merge_range_max(
    lhs: Option<UnifiedPayloadValue>,
    rhs: Option<UnifiedPayloadValue>,
) -> Option<UnifiedPayloadValue> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => match compare_payload_values(&lhs, &rhs) {
            Some(Ordering::Less) => Some(rhs),
            Some(_) => Some(lhs),
            None => None,
        },
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

fn merge_range_zone_map(existing: &mut RangeFieldZoneMap, incoming: &RangeFieldZoneMap) {
    existing.has_non_null_values |= incoming.has_non_null_values;
    existing.min = merge_range_min(existing.min.take(), incoming.min.clone());
    existing.max = merge_range_max(existing.max.take(), incoming.max.clone());
}

fn range_clause_overlaps_zone_map(
    zone_map: &RangeFieldZoneMap,
    clause: &RangeFieldQueryClause,
) -> Option<bool> {
    if !zone_map.has_non_null_values {
        return Some(false);
    }

    let (Some(min), Some(max)) = (zone_map.min.as_ref(), zone_map.max.as_ref()) else {
        return None;
    };

    if let Some(lower_bound) = clause.lower.as_ref() {
        let ordering = compare_payload_values(max, lower_bound)?;
        if clause.lower_inclusive {
            if ordering == Ordering::Less {
                return Some(false);
            }
        } else if ordering != Ordering::Greater {
            return Some(false);
        }
    }

    if let Some(upper_bound) = clause.upper.as_ref() {
        let ordering = compare_payload_values(min, upper_bound)?;
        if clause.upper_inclusive {
            if ordering == Ordering::Greater {
                return Some(false);
            }
        } else if ordering != Ordering::Less {
            return Some(false);
        }
    }

    Some(true)
}

pub fn logical_type_tag(logical_type: &UnifiedLogicalType) -> u16 {
    match logical_type {
        UnifiedLogicalType::Bool => 1,
        UnifiedLogicalType::Int64 => 2,
        UnifiedLogicalType::Float32 => 3,
        UnifiedLogicalType::Float64 => 4,
        UnifiedLogicalType::TimestampMicros => 5,
        UnifiedLogicalType::Keyword => 6,
        UnifiedLogicalType::Text => 7,
        UnifiedLogicalType::Bytes => 8,
        UnifiedLogicalType::LobRef => 9,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BucketExactClauseCoverage, BucketProbeObservation, BucketRangeClauseCoverage,
        ExactFieldQueryClause, ExactValueMembershipKey, ExactValuePresence, FilterMetadataCatalog,
        MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET, RangeFieldQueryClause, RangeFieldZoneMap,
    };
    use drift_storage::unified_format::{UnifiedLogicalType, UnifiedPayloadValue};
    use std::collections::{HashMap, HashSet};

    fn exact_key(field_id: u32, value: u32) -> ExactValueMembershipKey {
        ExactValueMembershipKey {
            field_id,
            logical_type_tag: super::logical_type_tag(&UnifiedLogicalType::Keyword),
            encoded_value: value.to_le_bytes().to_vec(),
        }
    }

    fn exact_presence(
        entries: impl IntoIterator<Item = (ExactValueMembershipKey, ExactValuePresence)>,
    ) -> HashMap<ExactValueMembershipKey, ExactValuePresence> {
        entries.into_iter().collect()
    }

    fn range_zone_map(
        min: Option<UnifiedPayloadValue>,
        max: Option<UnifiedPayloadValue>,
    ) -> RangeFieldZoneMap {
        RangeFieldZoneMap {
            has_non_null_values: min.is_some() || max.is_some(),
            min,
            max,
        }
    }

    #[test]
    fn observe_bucket_probe_merges_memberships() {
        let mut catalog = FilterMetadataCatalog::default();

        catalog.observe_bucket_probe(
            7,
            BucketProbeObservation {
                bucket_path: "bucket_a".to_string(),
                bucket_live_count: Some(10),
                indexed_exact_fields: HashSet::from([10]),
                range_stats_fields: HashSet::from([20]),
                exact_value_presence: exact_presence([(
                    exact_key(10, 1),
                    ExactValuePresence::Present,
                )]),
                range_field_zone_maps: HashMap::new(),
            },
        );
        catalog.observe_bucket_probe(
            7,
            BucketProbeObservation {
                bucket_path: "bucket_a".to_string(),
                bucket_live_count: Some(10),
                indexed_exact_fields: HashSet::from([11]),
                range_stats_fields: HashSet::from([21]),
                exact_value_presence: exact_presence([(
                    exact_key(11, 2),
                    ExactValuePresence::Present,
                )]),
                range_field_zone_maps: HashMap::new(),
            },
        );

        let stats = catalog.stats();
        assert_eq!(stats.bucket_count, 1);
        assert_eq!(stats.indexed_exact_field_memberships, 2);
        assert_eq!(stats.range_stats_field_memberships, 2);
        assert_eq!(stats.exact_value_memberships, 2);
    }

    #[test]
    fn observe_bucket_probe_resets_on_bucket_path_change() {
        let mut catalog = FilterMetadataCatalog::default();

        catalog.observe_bucket_probe(
            42,
            BucketProbeObservation {
                bucket_path: "bucket_v1".to_string(),
                bucket_live_count: Some(2),
                indexed_exact_fields: HashSet::from([1]),
                range_stats_fields: HashSet::from([2]),
                exact_value_presence: exact_presence([(
                    exact_key(1, 9),
                    ExactValuePresence::Present,
                )]),
                range_field_zone_maps: HashMap::new(),
            },
        );
        catalog.observe_bucket_probe(
            42,
            BucketProbeObservation {
                bucket_path: "bucket_v2".to_string(),
                bucket_live_count: Some(3),
                indexed_exact_fields: HashSet::from([3]),
                range_stats_fields: HashSet::from([4]),
                exact_value_presence: exact_presence([(
                    exact_key(3, 8),
                    ExactValuePresence::Present,
                )]),
                range_field_zone_maps: HashMap::new(),
            },
        );

        let stats = catalog.stats();
        assert_eq!(stats.bucket_count, 1);
        assert_eq!(stats.indexed_exact_field_memberships, 1);
        assert_eq!(stats.range_stats_field_memberships, 1);
        assert_eq!(stats.exact_value_memberships, 1);
    }

    #[test]
    fn observe_bucket_probe_caps_exact_value_memberships_per_bucket() {
        let mut catalog = FilterMetadataCatalog::default();
        for idx in 0..(MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET + 32) {
            catalog.observe_bucket_probe(
                1,
                BucketProbeObservation {
                    bucket_path: "bucket_v1".to_string(),
                    bucket_live_count: Some(1_000),
                    indexed_exact_fields: HashSet::new(),
                    range_stats_fields: HashSet::new(),
                    exact_value_presence: exact_presence([(
                        exact_key(1, idx as u32),
                        ExactValuePresence::Present,
                    )]),
                    range_field_zone_maps: HashMap::new(),
                },
            );
        }
        let stats = catalog.stats();
        assert_eq!(
            stats.exact_value_memberships,
            MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET
        );
    }

    #[test]
    fn classify_bucket_exact_clauses_returns_complete_no_match_when_all_known_absent() {
        let mut catalog = FilterMetadataCatalog::default();
        let key_a = exact_key(1, 10);
        let key_b = exact_key(1, 20);
        catalog.observe_bucket_probe(
            5,
            BucketProbeObservation {
                bucket_path: "bucket_5".to_string(),
                bucket_live_count: Some(64),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::new(),
                exact_value_presence: exact_presence([
                    (key_a.clone(), ExactValuePresence::Absent),
                    (key_b.clone(), ExactValuePresence::Absent),
                ]),
                range_field_zone_maps: HashMap::new(),
            },
        );
        let clauses = vec![ExactFieldQueryClause {
            field_id: 1,
            value_keys: vec![key_a, key_b],
        }];
        assert_eq!(
            catalog.classify_bucket_exact_clauses(5, "bucket_5", Some(64), &clauses),
            BucketExactClauseCoverage::CompleteNoMatch
        );
    }

    #[test]
    fn classify_bucket_exact_clauses_returns_complete_may_match_when_any_present() {
        let mut catalog = FilterMetadataCatalog::default();
        let key_a = exact_key(2, 100);
        let key_b = exact_key(2, 200);
        catalog.observe_bucket_probe(
            8,
            BucketProbeObservation {
                bucket_path: "bucket_8".to_string(),
                bucket_live_count: Some(64),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::new(),
                exact_value_presence: exact_presence([
                    (key_a.clone(), ExactValuePresence::Absent),
                    (key_b.clone(), ExactValuePresence::Present),
                ]),
                range_field_zone_maps: HashMap::new(),
            },
        );
        let clauses = vec![ExactFieldQueryClause {
            field_id: 2,
            value_keys: vec![key_a, key_b],
        }];
        assert_eq!(
            catalog.classify_bucket_exact_clauses(8, "bucket_8", Some(64), &clauses),
            BucketExactClauseCoverage::CompleteMayMatch
        );
    }

    #[test]
    fn classify_bucket_exact_clauses_returns_incomplete_when_values_unknown() {
        let mut catalog = FilterMetadataCatalog::default();
        let key_known_absent = exact_key(3, 1);
        let key_unknown = exact_key(3, 2);
        catalog.observe_bucket_probe(
            9,
            BucketProbeObservation {
                bucket_path: "bucket_9".to_string(),
                bucket_live_count: Some(10),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::new(),
                exact_value_presence: exact_presence([(
                    key_known_absent.clone(),
                    ExactValuePresence::Absent,
                )]),
                range_field_zone_maps: HashMap::new(),
            },
        );
        let clauses = vec![ExactFieldQueryClause {
            field_id: 3,
            value_keys: vec![key_known_absent, key_unknown],
        }];
        assert_eq!(
            catalog.classify_bucket_exact_clauses(9, "bucket_9", Some(10), &clauses),
            BucketExactClauseCoverage::Incomplete
        );
    }

    #[test]
    fn classify_bucket_exact_clauses_returns_stale_for_bucket_path_mismatch() {
        let mut catalog = FilterMetadataCatalog::default();
        catalog.observe_bucket_probe(
            11,
            BucketProbeObservation {
                bucket_path: "bucket_11_v1".to_string(),
                bucket_live_count: Some(7),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::new(),
                exact_value_presence: HashMap::new(),
                range_field_zone_maps: HashMap::new(),
            },
        );
        let clauses = vec![ExactFieldQueryClause {
            field_id: 1,
            value_keys: vec![exact_key(1, 7)],
        }];
        assert_eq!(
            catalog.classify_bucket_exact_clauses(11, "bucket_11_v2", Some(7), &clauses),
            BucketExactClauseCoverage::StaleBucketPath
        );
    }

    #[test]
    fn classify_bucket_exact_clauses_returns_stale_for_bucket_stats_mismatch() {
        let mut catalog = FilterMetadataCatalog::default();
        catalog.observe_bucket_probe(
            12,
            BucketProbeObservation {
                bucket_path: "bucket_12".to_string(),
                bucket_live_count: Some(100),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::new(),
                exact_value_presence: HashMap::new(),
                range_field_zone_maps: HashMap::new(),
            },
        );
        assert_eq!(
            catalog.classify_bucket_exact_clauses(12, "bucket_12", Some(101), &[]),
            BucketExactClauseCoverage::StaleBucketStats
        );
    }

    #[test]
    fn classify_bucket_range_clauses_returns_complete_no_match_outside_zone_map() {
        let mut catalog = FilterMetadataCatalog::default();
        catalog.observe_bucket_probe(
            13,
            BucketProbeObservation {
                bucket_path: "bucket_13".to_string(),
                bucket_live_count: Some(50),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::from([9]),
                exact_value_presence: HashMap::new(),
                range_field_zone_maps: HashMap::from([(
                    9,
                    range_zone_map(
                        Some(UnifiedPayloadValue::Int64(10)),
                        Some(UnifiedPayloadValue::Int64(20)),
                    ),
                )]),
            },
        );

        let clauses = vec![RangeFieldQueryClause {
            field_id: 9,
            lower: Some(UnifiedPayloadValue::Int64(30)),
            lower_inclusive: true,
            upper: None,
            upper_inclusive: true,
        }];
        assert_eq!(
            catalog.classify_bucket_range_clauses(13, "bucket_13", Some(50), &clauses),
            BucketRangeClauseCoverage::CompleteNoMatch
        );
    }

    #[test]
    fn classify_bucket_range_clauses_returns_complete_may_match_for_overlap() {
        let mut catalog = FilterMetadataCatalog::default();
        catalog.observe_bucket_probe(
            14,
            BucketProbeObservation {
                bucket_path: "bucket_14".to_string(),
                bucket_live_count: Some(60),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::from([8]),
                exact_value_presence: HashMap::new(),
                range_field_zone_maps: HashMap::from([(
                    8,
                    range_zone_map(
                        Some(UnifiedPayloadValue::Int64(100)),
                        Some(UnifiedPayloadValue::Int64(200)),
                    ),
                )]),
            },
        );

        let clauses = vec![RangeFieldQueryClause {
            field_id: 8,
            lower: Some(UnifiedPayloadValue::Int64(150)),
            lower_inclusive: true,
            upper: Some(UnifiedPayloadValue::Int64(160)),
            upper_inclusive: true,
        }];
        assert_eq!(
            catalog.classify_bucket_range_clauses(14, "bucket_14", Some(60), &clauses),
            BucketRangeClauseCoverage::CompleteMayMatch
        );
    }

    #[test]
    fn classify_bucket_range_clauses_returns_incomplete_when_zone_map_missing() {
        let mut catalog = FilterMetadataCatalog::default();
        catalog.observe_bucket_probe(
            15,
            BucketProbeObservation {
                bucket_path: "bucket_15".to_string(),
                bucket_live_count: Some(10),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::new(),
                exact_value_presence: HashMap::new(),
                range_field_zone_maps: HashMap::new(),
            },
        );

        let clauses = vec![RangeFieldQueryClause {
            field_id: 42,
            lower: Some(UnifiedPayloadValue::Int64(1)),
            lower_inclusive: true,
            upper: Some(UnifiedPayloadValue::Int64(2)),
            upper_inclusive: true,
        }];
        assert_eq!(
            catalog.classify_bucket_range_clauses(15, "bucket_15", Some(10), &clauses),
            BucketRangeClauseCoverage::Incomplete
        );
    }

    #[test]
    fn invalidate_bucket_removes_entry() {
        let mut catalog = FilterMetadataCatalog::default();
        catalog.observe_bucket_probe(
            99,
            BucketProbeObservation {
                bucket_path: "bucket_99".to_string(),
                bucket_live_count: Some(1),
                indexed_exact_fields: HashSet::new(),
                range_stats_fields: HashSet::new(),
                exact_value_presence: HashMap::new(),
                range_field_zone_maps: HashMap::new(),
            },
        );
        assert!(catalog.invalidate_bucket(99));
        assert!(!catalog.invalidate_bucket(99));
    }

    #[test]
    fn clear_removes_all_entries() {
        let mut catalog = FilterMetadataCatalog::default();
        for bucket_id in [1, 2] {
            catalog.observe_bucket_probe(
                bucket_id,
                BucketProbeObservation {
                    bucket_path: format!("bucket_{}", bucket_id),
                    bucket_live_count: Some(5),
                    indexed_exact_fields: HashSet::from([7]),
                    range_stats_fields: HashSet::new(),
                    exact_value_presence: HashMap::new(),
                    range_field_zone_maps: HashMap::new(),
                },
            );
        }
        assert_eq!(catalog.stats().bucket_count, 2);

        catalog.clear();
        let stats = catalog.stats();
        assert_eq!(stats.bucket_count, 0);
        assert_eq!(stats.indexed_exact_field_memberships, 0);
        assert_eq!(stats.exact_value_memberships, 0);
    }
}
