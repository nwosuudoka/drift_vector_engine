use drift_storage::unified_format::UnifiedLogicalType;
use std::collections::{HashMap, HashSet};

pub const MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExactValueMembershipKey {
    pub field_id: u32,
    pub logical_type_tag: u16,
    pub encoded_value: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct BucketProbeObservation {
    pub bucket_path: String,
    pub indexed_exact_fields: HashSet<u32>,
    pub range_stats_fields: HashSet<u32>,
    pub exact_value_hits: HashSet<ExactValueMembershipKey>,
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
    indexed_exact_fields: HashSet<u32>,
    range_stats_fields: HashSet<u32>,
    exact_value_hits: HashSet<ExactValueMembershipKey>,
}

impl FilterMetadataCatalog {
    pub fn observe_bucket_probe(&mut self, bucket_id: u32, observation: BucketProbeObservation) {
        let entry = self.buckets.entry(bucket_id).or_default();
        if !entry.bucket_path.is_empty() && entry.bucket_path != observation.bucket_path {
            // Bucket IDs are reused across maintenance operations. Reset stale memberships when
            // the backing bucket path changes.
            entry.indexed_exact_fields.clear();
            entry.range_stats_fields.clear();
            entry.exact_value_hits.clear();
        }
        entry.bucket_path = observation.bucket_path;
        entry
            .indexed_exact_fields
            .extend(observation.indexed_exact_fields);
        entry
            .range_stats_fields
            .extend(observation.range_stats_fields);

        if entry.exact_value_hits.len() < MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET {
            let remaining =
                MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET.saturating_sub(entry.exact_value_hits.len());
            entry
                .exact_value_hits
                .extend(observation.exact_value_hits.into_iter().take(remaining));
        }
    }

    pub fn stats(&self) -> FilterMetadataCatalogStats {
        let mut indexed_exact_field_memberships = 0usize;
        let mut range_stats_field_memberships = 0usize;
        let mut exact_value_memberships = 0usize;
        for entry in self.buckets.values() {
            indexed_exact_field_memberships += entry.indexed_exact_fields.len();
            range_stats_field_memberships += entry.range_stats_fields.len();
            exact_value_memberships += entry.exact_value_hits.len();
        }
        FilterMetadataCatalogStats {
            bucket_count: self.buckets.len(),
            indexed_exact_field_memberships,
            range_stats_field_memberships,
            exact_value_memberships,
        }
    }
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
        BucketProbeObservation, ExactValueMembershipKey, FilterMetadataCatalog,
        MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET,
    };
    use drift_storage::unified_format::UnifiedLogicalType;
    use std::collections::HashSet;

    fn exact_key(field_id: u32, value: u32) -> ExactValueMembershipKey {
        ExactValueMembershipKey {
            field_id,
            logical_type_tag: super::logical_type_tag(&UnifiedLogicalType::Keyword),
            encoded_value: value.to_le_bytes().to_vec(),
        }
    }

    #[test]
    fn observe_bucket_probe_merges_memberships() {
        let mut catalog = FilterMetadataCatalog::default();

        catalog.observe_bucket_probe(
            7,
            BucketProbeObservation {
                bucket_path: "bucket_a".to_string(),
                indexed_exact_fields: HashSet::from([10]),
                range_stats_fields: HashSet::from([20]),
                exact_value_hits: HashSet::from([exact_key(10, 1)]),
            },
        );
        catalog.observe_bucket_probe(
            7,
            BucketProbeObservation {
                bucket_path: "bucket_a".to_string(),
                indexed_exact_fields: HashSet::from([11]),
                range_stats_fields: HashSet::from([21]),
                exact_value_hits: HashSet::from([exact_key(11, 2)]),
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
                indexed_exact_fields: HashSet::from([1]),
                range_stats_fields: HashSet::from([2]),
                exact_value_hits: HashSet::from([exact_key(1, 9)]),
            },
        );
        catalog.observe_bucket_probe(
            42,
            BucketProbeObservation {
                bucket_path: "bucket_v2".to_string(),
                indexed_exact_fields: HashSet::from([3]),
                range_stats_fields: HashSet::from([4]),
                exact_value_hits: HashSet::from([exact_key(3, 8)]),
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
                    indexed_exact_fields: HashSet::new(),
                    range_stats_fields: HashSet::new(),
                    exact_value_hits: HashSet::from([exact_key(1, idx as u32)]),
                },
            );
        }
        let stats = catalog.stats();
        assert_eq!(
            stats.exact_value_memberships,
            MAX_EXACT_VALUE_MEMBERSHIPS_PER_BUCKET
        );
    }
}
