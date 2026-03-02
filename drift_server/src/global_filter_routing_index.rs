use crate::filter_metadata_catalog::logical_type_tag;
use crate::filter_metadata_catalog::{ExactFieldQueryClause, ExactValueMembershipKey};
use crate::global_metadata_snapshot::{
    BucketExactFieldCoverageSnapshot, GlobalRoutingIdEntry, GlobalRoutingSnapshot,
};
use drift_storage::unified_format::{UnifiedPayloadRow, UnifiedPayloadSchema, encode_exact_key};
use std::collections::{HashMap, HashSet};
use std::io;

#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalFilterRoutingIndexStats {
    pub id_entry_count: usize,
    pub value_entry_count: usize,
    pub value_bucket_pair_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct GlobalFilterRoutingIndex {
    id_entries: HashMap<u64, IdRoutingEntry>,
    value_bucket_counts: HashMap<ExactValueMembershipKey, HashMap<u32, u32>>,
    complete_exact_fields_by_bucket: HashMap<u32, HashSet<u32>>,
}

#[derive(Debug, Clone)]
struct IdRoutingEntry {
    bucket_id: u32,
    value_keys: HashSet<ExactValueMembershipKey>,
}

impl GlobalFilterRoutingIndex {
    pub fn upsert_id_values(
        &mut self,
        id: u64,
        bucket_id: u32,
        value_keys: impl IntoIterator<Item = ExactValueMembershipKey>,
    ) {
        if let Some(previous) = self.id_entries.remove(&id) {
            self.remove_entry_counts(&previous);
        }

        let value_keys: HashSet<ExactValueMembershipKey> = value_keys.into_iter().collect();
        if value_keys.is_empty() {
            return;
        }

        let entry = IdRoutingEntry {
            bucket_id,
            value_keys,
        };
        self.apply_entry_counts(&entry);
        self.id_entries.insert(id, entry);
    }

    pub fn remove_id(&mut self, id: u64) -> bool {
        let Some(entry) = self.id_entries.remove(&id) else {
            return false;
        };
        self.remove_entry_counts(&entry);
        true
    }

    pub fn invalidate_bucket(&mut self, bucket_id: u32) -> usize {
        let ids_to_remove: Vec<u64> = self
            .id_entries
            .iter()
            .filter_map(|(id, entry)| (entry.bucket_id == bucket_id).then_some(*id))
            .collect();
        let removed = ids_to_remove.len();
        for id in ids_to_remove {
            let _ = self.remove_id(id);
        }
        self.complete_exact_fields_by_bucket.remove(&bucket_id);
        removed
    }

    pub fn buckets_for_exact_value(&self, key: &ExactValueMembershipKey) -> HashSet<u32> {
        self.value_bucket_counts
            .get(key)
            .map(|counts| counts.keys().copied().collect())
            .unwrap_or_default()
    }

    pub fn buckets_for_exact_clause(&self, clause: &ExactFieldQueryClause) -> HashSet<u32> {
        let mut buckets = HashSet::new();
        for key in &clause.value_keys {
            buckets.extend(self.buckets_for_exact_value(key));
        }
        buckets
    }

    pub fn buckets_for_exact_clauses(
        &self,
        clauses: &[ExactFieldQueryClause],
    ) -> Option<HashSet<u32>> {
        let mut clause_iter = clauses.iter();
        let first = clause_iter.next()?;
        let mut acc = self.buckets_for_exact_clause(first);
        for clause in clause_iter {
            if acc.is_empty() {
                return Some(acc);
            }
            let clause_buckets = self.buckets_for_exact_clause(clause);
            acc.retain(|bucket_id| clause_buckets.contains(bucket_id));
        }
        Some(acc)
    }

    pub fn value_bucket_live_count(&self, key: &ExactValueMembershipKey, bucket_id: u32) -> u32 {
        self.value_bucket_counts
            .get(key)
            .and_then(|counts| counts.get(&bucket_id).copied())
            .unwrap_or(0)
    }

    pub fn set_bucket_complete_exact_fields(
        &mut self,
        bucket_id: u32,
        exact_field_ids: HashSet<u32>,
    ) {
        if exact_field_ids.is_empty() {
            self.complete_exact_fields_by_bucket.remove(&bucket_id);
            return;
        }
        self.complete_exact_fields_by_bucket
            .insert(bucket_id, exact_field_ids);
    }

    pub fn mark_bucket_exact_field_complete(&mut self, bucket_id: u32, field_id: u32) {
        self.complete_exact_fields_by_bucket
            .entry(bucket_id)
            .or_default()
            .insert(field_id);
    }

    pub fn replace_bucket_exact_field_values(
        &mut self,
        bucket_id: u32,
        field_id: u32,
        values_by_id: HashMap<u64, HashSet<ExactValueMembershipKey>>,
    ) {
        let mut normalized_values_by_id: HashMap<u64, HashSet<ExactValueMembershipKey>> =
            HashMap::new();
        for (id, keys) in values_by_id {
            let filtered: HashSet<ExactValueMembershipKey> = keys
                .into_iter()
                .filter(|key| key.field_id == field_id)
                .collect();
            if !filtered.is_empty() {
                normalized_values_by_id.insert(id, filtered);
            }
        }

        let existing_bucket_ids: HashSet<u64> = self
            .id_entries
            .iter()
            .filter_map(|(id, entry)| (entry.bucket_id == bucket_id).then_some(*id))
            .collect();
        let mut touched_ids = existing_bucket_ids;
        touched_ids.extend(normalized_values_by_id.keys().copied());

        for id in touched_ids {
            let previous = self.id_entries.remove(&id);
            let mut next_keys = HashSet::new();
            if let Some(entry) = previous {
                self.remove_entry_counts(&entry);
                if entry.bucket_id == bucket_id {
                    next_keys.extend(
                        entry
                            .value_keys
                            .into_iter()
                            .filter(|key| key.field_id != field_id),
                    );
                }
            }

            if let Some(field_keys) = normalized_values_by_id.remove(&id) {
                next_keys.extend(field_keys);
            }

            if !next_keys.is_empty() {
                let next = IdRoutingEntry {
                    bucket_id,
                    value_keys: next_keys,
                };
                self.apply_entry_counts(&next);
                self.id_entries.insert(id, next);
            }
        }
    }

    pub fn bucket_exact_field_complete(&self, bucket_id: u32, field_id: u32) -> bool {
        self.complete_exact_fields_by_bucket
            .get(&bucket_id)
            .map(|fields| fields.contains(&field_id))
            .unwrap_or(false)
    }

    pub fn stats(&self) -> GlobalFilterRoutingIndexStats {
        let value_bucket_pair_count = self
            .value_bucket_counts
            .values()
            .map(std::collections::HashMap::len)
            .sum();
        GlobalFilterRoutingIndexStats {
            id_entry_count: self.id_entries.len(),
            value_entry_count: self.value_bucket_counts.len(),
            value_bucket_pair_count,
        }
    }

    pub fn clear(&mut self) {
        self.id_entries.clear();
        self.value_bucket_counts.clear();
        self.complete_exact_fields_by_bucket.clear();
    }

    pub fn export_snapshot(&self) -> GlobalRoutingSnapshot {
        let mut id_entries: Vec<GlobalRoutingIdEntry> = self
            .id_entries
            .iter()
            .map(|(id, entry)| {
                let mut value_keys: Vec<ExactValueMembershipKey> =
                    entry.value_keys.iter().cloned().collect();
                value_keys.sort_by(|lhs, rhs| {
                    lhs.field_id
                        .cmp(&rhs.field_id)
                        .then_with(|| lhs.logical_type_tag.cmp(&rhs.logical_type_tag))
                        .then_with(|| lhs.encoded_value.cmp(&rhs.encoded_value))
                });
                GlobalRoutingIdEntry {
                    id: *id,
                    bucket_id: entry.bucket_id,
                    value_keys,
                }
            })
            .collect();
        id_entries.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));

        let mut complete_exact_fields_by_bucket: Vec<BucketExactFieldCoverageSnapshot> = self
            .complete_exact_fields_by_bucket
            .iter()
            .map(|(bucket_id, fields)| {
                let mut field_ids: Vec<u32> = fields.iter().copied().collect();
                field_ids.sort_unstable();
                BucketExactFieldCoverageSnapshot {
                    bucket_id: *bucket_id,
                    field_ids,
                }
            })
            .collect();
        complete_exact_fields_by_bucket.sort_by(|lhs, rhs| lhs.bucket_id.cmp(&rhs.bucket_id));

        GlobalRoutingSnapshot {
            bucket_tokens: Vec::new(),
            id_entries,
            complete_exact_fields_by_bucket,
        }
    }

    pub fn import_snapshot(&mut self, snapshot: &GlobalRoutingSnapshot) {
        self.clear();
        for entry in &snapshot.id_entries {
            self.upsert_id_values(entry.id, entry.bucket_id, entry.value_keys.iter().cloned());
        }
        for entry in &snapshot.complete_exact_fields_by_bucket {
            self.set_bucket_complete_exact_fields(
                entry.bucket_id,
                entry.field_ids.iter().copied().collect(),
            );
        }
    }

    fn apply_entry_counts(&mut self, entry: &IdRoutingEntry) {
        for key in &entry.value_keys {
            let per_bucket = self.value_bucket_counts.entry(key.clone()).or_default();
            *per_bucket.entry(entry.bucket_id).or_insert(0) += 1;
        }
    }

    fn remove_entry_counts(&mut self, entry: &IdRoutingEntry) {
        for key in &entry.value_keys {
            let mut remove_key = false;
            if let Some(per_bucket) = self.value_bucket_counts.get_mut(key) {
                if let Some(count) = per_bucket.get_mut(&entry.bucket_id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        per_bucket.remove(&entry.bucket_id);
                    }
                }
                remove_key = per_bucket.is_empty();
            }
            if remove_key {
                self.value_bucket_counts.remove(key);
            }
        }
    }
}

pub fn indexed_exact_field_ids(schema: &UnifiedPayloadSchema) -> HashSet<u32> {
    schema
        .fields
        .iter()
        .filter_map(|field| field.indexed.then_some(field.field_id))
        .collect()
}

pub fn extract_indexed_exact_value_keys(
    schema: &UnifiedPayloadSchema,
    row: &UnifiedPayloadRow,
) -> io::Result<Vec<ExactValueMembershipKey>> {
    let mut keys = Vec::new();
    for field in &schema.fields {
        if !field.indexed {
            continue;
        }
        let Some(value) = row.get(&field.field_id) else {
            continue;
        };
        let Some(encoded_value) = encode_exact_key(&field.logical_type, value)? else {
            continue;
        };
        keys.push(ExactValueMembershipKey {
            field_id: field.field_id,
            logical_type_tag: logical_type_tag(&field.logical_type),
            encoded_value,
        });
    }
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::GlobalFilterRoutingIndex;
    use super::{extract_indexed_exact_value_keys, indexed_exact_field_ids};
    use crate::filter_metadata_catalog::{ExactFieldQueryClause, ExactValueMembershipKey};
    use drift_storage::unified_format::{
        UnifiedFieldSchema, UnifiedLogicalType, UnifiedPayloadRow, UnifiedPayloadSchema,
        UnifiedPayloadValue,
    };
    use std::collections::{HashMap, HashSet};

    fn key(field_id: u32, value: &str) -> ExactValueMembershipKey {
        ExactValueMembershipKey {
            field_id,
            logical_type_tag: 1,
            encoded_value: value.as_bytes().to_vec(),
        }
    }

    #[test]
    fn upsert_tracks_bucket_memberships() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");
        let tenant_b = key(1, "tenant_b");

        index.upsert_id_values(100, 7, vec![tenant_a.clone()]);
        index.upsert_id_values(101, 7, vec![tenant_a.clone(), tenant_b.clone()]);

        assert_eq!(index.value_bucket_live_count(&tenant_a, 7), 2);
        assert_eq!(index.value_bucket_live_count(&tenant_b, 7), 1);
        assert_eq!(index.stats().id_entry_count, 2);
    }

    #[test]
    fn upsert_replaces_previous_entry_counts_for_same_id() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");
        let tenant_b = key(1, "tenant_b");

        index.upsert_id_values(100, 7, vec![tenant_a.clone()]);
        index.upsert_id_values(100, 9, vec![tenant_b.clone()]);

        assert_eq!(index.value_bucket_live_count(&tenant_a, 7), 0);
        assert_eq!(index.value_bucket_live_count(&tenant_b, 9), 1);
        assert_eq!(index.stats().id_entry_count, 1);
    }

    #[test]
    fn remove_id_updates_value_bucket_counts() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");

        index.upsert_id_values(100, 7, vec![tenant_a.clone()]);
        index.upsert_id_values(101, 7, vec![tenant_a.clone()]);

        assert!(index.remove_id(100));
        assert_eq!(index.value_bucket_live_count(&tenant_a, 7), 1);
        assert!(index.remove_id(101));
        assert_eq!(index.value_bucket_live_count(&tenant_a, 7), 0);
        assert!(!index.remove_id(999));
    }

    #[test]
    fn invalidate_bucket_removes_all_bucket_entries() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");
        let tenant_b = key(1, "tenant_b");

        index.upsert_id_values(100, 7, vec![tenant_a.clone()]);
        index.upsert_id_values(101, 7, vec![tenant_b.clone()]);
        index.upsert_id_values(102, 8, vec![tenant_b.clone()]);

        assert_eq!(index.invalidate_bucket(7), 2);
        assert_eq!(index.value_bucket_live_count(&tenant_a, 7), 0);
        assert_eq!(index.value_bucket_live_count(&tenant_b, 7), 0);
        assert_eq!(index.value_bucket_live_count(&tenant_b, 8), 1);
    }

    #[test]
    fn buckets_for_exact_clauses_intersects_clause_memberships() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");
        let tenant_b = key(1, "tenant_b");
        let region_us = key(2, "us");

        index.upsert_id_values(100, 7, vec![tenant_a.clone(), region_us.clone()]);
        index.upsert_id_values(101, 8, vec![tenant_a.clone()]);
        index.upsert_id_values(102, 9, vec![tenant_b.clone(), region_us.clone()]);

        let clauses = vec![
            ExactFieldQueryClause {
                field_id: 1,
                value_keys: vec![tenant_a.clone(), tenant_b.clone()],
            },
            ExactFieldQueryClause {
                field_id: 2,
                value_keys: vec![region_us.clone()],
            },
        ];
        let buckets = index
            .buckets_for_exact_clauses(&clauses)
            .expect("clauses should return a set");

        assert_eq!(buckets.len(), 2);
        assert!(buckets.contains(&7));
        assert!(buckets.contains(&9));
    }

    #[test]
    fn buckets_for_exact_clauses_handles_empty_clause_list() {
        let index = GlobalFilterRoutingIndex::default();
        assert!(index.buckets_for_exact_clauses(&[]).is_none());
    }

    #[test]
    fn upsert_deduplicates_duplicate_value_keys() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");

        index.upsert_id_values(100, 7, vec![tenant_a.clone(), tenant_a.clone()]);

        assert_eq!(index.value_bucket_live_count(&tenant_a, 7), 1);
        assert_eq!(index.stats().value_entry_count, 1);
    }

    #[test]
    fn bucket_exact_field_complete_tracks_configured_fields() {
        let mut index = GlobalFilterRoutingIndex::default();
        index.set_bucket_complete_exact_fields(8, HashSet::from([1, 2]));
        assert!(index.bucket_exact_field_complete(8, 1));
        assert!(index.bucket_exact_field_complete(8, 2));
        assert!(!index.bucket_exact_field_complete(8, 3));

        index.set_bucket_complete_exact_fields(8, HashSet::new());
        assert!(!index.bucket_exact_field_complete(8, 1));
    }

    #[test]
    fn mark_bucket_exact_field_complete_extends_existing_coverage() {
        let mut index = GlobalFilterRoutingIndex::default();
        index.set_bucket_complete_exact_fields(5, HashSet::from([1]));

        index.mark_bucket_exact_field_complete(5, 2);

        assert!(index.bucket_exact_field_complete(5, 1));
        assert!(index.bucket_exact_field_complete(5, 2));
    }

    #[test]
    fn replace_bucket_exact_field_values_replaces_only_target_field() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");
        let tenant_b = key(1, "tenant_b");
        let tenant_c = key(1, "tenant_c");
        let region_us = key(2, "us");

        index.upsert_id_values(100, 7, vec![tenant_a.clone(), region_us.clone()]);
        index.upsert_id_values(101, 7, vec![tenant_b.clone()]);
        index.upsert_id_values(102, 8, vec![tenant_a.clone()]);

        let mut replacement = HashMap::new();
        replacement.insert(100, HashSet::from([tenant_a.clone()]));
        replacement.insert(103, HashSet::from([tenant_c.clone()]));
        index.replace_bucket_exact_field_values(7, 1, replacement);

        assert_eq!(
            index.buckets_for_exact_value(&tenant_a),
            HashSet::from([7_u32, 8_u32])
        );
        assert_eq!(index.buckets_for_exact_value(&tenant_b), HashSet::new());
        assert_eq!(
            index.buckets_for_exact_value(&tenant_c),
            HashSet::from([7_u32])
        );
        assert_eq!(
            index.buckets_for_exact_value(&region_us),
            HashSet::from([7_u32])
        );
        assert_eq!(index.stats().id_entry_count, 3);
    }

    #[test]
    fn invalidate_bucket_clears_completeness_tracking() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");
        index.upsert_id_values(100, 7, vec![tenant_a]);
        index.set_bucket_complete_exact_fields(7, HashSet::from([1]));
        assert!(index.bucket_exact_field_complete(7, 1));

        index.invalidate_bucket(7);
        assert!(!index.bucket_exact_field_complete(7, 1));
    }

    #[test]
    fn indexed_exact_field_ids_returns_only_indexed_fields() {
        let schema = UnifiedPayloadSchema::new(vec![
            UnifiedFieldSchema {
                field_id: 1,
                name: "tenant".to_string(),
                logical_type: UnifiedLogicalType::Keyword,
                nullable: false,
                indexed: true,
            },
            UnifiedFieldSchema {
                field_id: 2,
                name: "note".to_string(),
                logical_type: UnifiedLogicalType::Text,
                nullable: true,
                indexed: false,
            },
        ]);
        let fields = indexed_exact_field_ids(&schema);
        assert_eq!(fields, HashSet::from([1]));
    }

    #[test]
    fn extract_indexed_exact_value_keys_uses_indexed_schema_fields() {
        let schema = UnifiedPayloadSchema::new(vec![
            UnifiedFieldSchema {
                field_id: 1,
                name: "tenant".to_string(),
                logical_type: UnifiedLogicalType::Keyword,
                nullable: false,
                indexed: true,
            },
            UnifiedFieldSchema {
                field_id: 2,
                name: "note".to_string(),
                logical_type: UnifiedLogicalType::Text,
                nullable: true,
                indexed: false,
            },
        ]);
        let row = UnifiedPayloadRow::from([
            (1, UnifiedPayloadValue::Keyword("tenant_a".to_string())),
            (2, UnifiedPayloadValue::Text("hello".to_string())),
        ]);
        let keys = extract_indexed_exact_value_keys(&schema, &row).expect("keys should encode");
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].field_id, 1);
    }

    #[test]
    fn routing_snapshot_roundtrip_preserves_entries() {
        let mut index = GlobalFilterRoutingIndex::default();
        let tenant_a = key(1, "tenant_a");
        let region_us = key(2, "us");
        index.upsert_id_values(100, 7, vec![tenant_a.clone(), region_us.clone()]);
        index.set_bucket_complete_exact_fields(7, HashSet::from([1, 2]));

        let snapshot = index.export_snapshot();
        let mut restored = GlobalFilterRoutingIndex::default();
        restored.import_snapshot(&snapshot);

        assert_eq!(restored.value_bucket_live_count(&tenant_a, 7), 1);
        assert_eq!(restored.value_bucket_live_count(&region_us, 7), 1);
        assert!(restored.bucket_exact_field_complete(7, 1));
        assert!(restored.bucket_exact_field_complete(7, 2));
    }
}
