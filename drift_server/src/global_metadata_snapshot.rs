use crate::filter_metadata_catalog::{
    ExactValueMembershipKey, ExactValuePresence, RangeFieldZoneMap,
};
use bincode::{Decode, Encode, config, decode_from_slice, encode_to_vec};
use std::io;

pub const GLOBAL_METADATA_SNAPSHOT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct GlobalMetadataSnapshot {
    pub format_version: u32,
    pub routing: GlobalRoutingSnapshot,
    pub catalog: FilterCatalogSnapshot,
}

impl Default for GlobalMetadataSnapshot {
    fn default() -> Self {
        Self {
            format_version: GLOBAL_METADATA_SNAPSHOT_FORMAT_VERSION,
            routing: GlobalRoutingSnapshot::default(),
            catalog: FilterCatalogSnapshot::default(),
        }
    }
}

impl GlobalMetadataSnapshot {
    pub fn encode_to_bytes(&self) -> io::Result<Vec<u8>> {
        encode_to_vec(self, config::standard()).map_err(io::Error::other)
    }

    pub fn decode_from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let (snapshot, consumed) = decode_from_slice::<Self, _>(bytes, config::standard())
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("failed to decode global metadata snapshot: {e}"),
                )
            })?;
        if consumed != bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "global metadata snapshot had trailing bytes: consumed={} total={}",
                    consumed,
                    bytes.len()
                ),
            ));
        }
        if snapshot.format_version == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "global metadata snapshot format_version must be non-zero",
            ));
        }
        Ok(snapshot)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Encode, Decode)]
pub struct GlobalRoutingSnapshot {
    pub bucket_tokens: Vec<RoutingBucketTokenSnapshot>,
    pub id_entries: Vec<GlobalRoutingIdEntry>,
    pub complete_exact_fields_by_bucket: Vec<BucketExactFieldCoverageSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct RoutingBucketTokenSnapshot {
    pub bucket_id: u32,
    pub bucket_path: String,
    pub bucket_live_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct GlobalRoutingIdEntry {
    pub id: u64,
    pub bucket_id: u32,
    pub value_keys: Vec<ExactValueMembershipKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct BucketExactFieldCoverageSnapshot {
    pub bucket_id: u32,
    pub field_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Default, Encode, Decode)]
pub struct FilterCatalogSnapshot {
    pub buckets: Vec<FilterCatalogBucketSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct FilterCatalogBucketSnapshot {
    pub bucket_id: u32,
    pub bucket_path: String,
    pub bucket_live_count: Option<u32>,
    pub indexed_exact_fields: Vec<u32>,
    pub range_stats_fields: Vec<u32>,
    pub exact_value_presence: Vec<ExactValuePresenceSnapshotEntry>,
    pub range_field_zone_maps: Vec<RangeFieldZoneMapSnapshotEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct ExactValuePresenceSnapshotEntry {
    pub key: ExactValueMembershipKey,
    pub presence: ExactValuePresence,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct RangeFieldZoneMapSnapshotEntry {
    pub field_id: u32,
    pub zone_map: RangeFieldZoneMap,
}

#[cfg(test)]
mod tests {
    use super::{
        BucketExactFieldCoverageSnapshot, ExactValuePresenceSnapshotEntry,
        FilterCatalogBucketSnapshot, GlobalMetadataSnapshot, GlobalRoutingIdEntry,
        RangeFieldZoneMapSnapshotEntry, RoutingBucketTokenSnapshot,
    };
    use crate::filter_metadata_catalog::{
        ExactValueMembershipKey, ExactValuePresence, RangeFieldZoneMap,
    };
    use drift_storage::unified_format::UnifiedPayloadValue;

    #[test]
    fn snapshot_roundtrip_preserves_routing_and_catalog_data() {
        let mut snapshot = GlobalMetadataSnapshot::default();
        snapshot
            .routing
            .bucket_tokens
            .push(RoutingBucketTokenSnapshot {
                bucket_id: 7,
                bucket_path: "bucket_7_run_1.driftu".to_string(),
                bucket_live_count: 128,
            });
        snapshot.routing.id_entries.push(GlobalRoutingIdEntry {
            id: 42,
            bucket_id: 7,
            value_keys: vec![ExactValueMembershipKey {
                field_id: 1,
                logical_type_tag: 6,
                encoded_value: b"tenant_a".to_vec(),
            }],
        });
        snapshot
            .routing
            .complete_exact_fields_by_bucket
            .push(BucketExactFieldCoverageSnapshot {
                bucket_id: 7,
                field_ids: vec![1, 2],
            });
        snapshot.catalog.buckets.push(FilterCatalogBucketSnapshot {
            bucket_id: 7,
            bucket_path: "bucket_7_run_1.driftu".to_string(),
            bucket_live_count: Some(128),
            indexed_exact_fields: vec![1],
            range_stats_fields: vec![2],
            exact_value_presence: vec![ExactValuePresenceSnapshotEntry {
                key: ExactValueMembershipKey {
                    field_id: 1,
                    logical_type_tag: 6,
                    encoded_value: b"tenant_a".to_vec(),
                },
                presence: ExactValuePresence::Present,
            }],
            range_field_zone_maps: vec![RangeFieldZoneMapSnapshotEntry {
                field_id: 2,
                zone_map: RangeFieldZoneMap {
                    has_non_null_values: true,
                    min: Some(UnifiedPayloadValue::Int64(10)),
                    max: Some(UnifiedPayloadValue::Int64(20)),
                },
            }],
        });

        let encoded = snapshot
            .encode_to_bytes()
            .expect("snapshot should encode successfully");
        let decoded = GlobalMetadataSnapshot::decode_from_bytes(&encoded)
            .expect("snapshot should decode successfully");
        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn decode_rejects_zero_format_version() {
        let mut snapshot = GlobalMetadataSnapshot::default();
        snapshot.format_version = 0;
        let encoded = snapshot
            .encode_to_bytes()
            .expect("snapshot should encode successfully");
        let err = GlobalMetadataSnapshot::decode_from_bytes(&encoded)
            .expect_err("decode should reject zero format_version");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
