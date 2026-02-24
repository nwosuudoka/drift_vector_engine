#[cfg(test)]
mod tests {
    use crate::manifest::pb::Centroid;
    use crate::math::Metric;
    use crate::partitioner::IncrementalPartitioner;
    use crate::payload::{
        PayloadFieldSchema, PayloadLogicalType, PayloadRow, PayloadSchema, PayloadValue,
    };
    use crate::router::Router;
    use std::collections::BTreeMap;

    #[test]
    fn test_partitioning_logic() {
        // 1. Setup Router (2 Buckets)
        // Bucket 1: [0, 0]
        // Bucket 2: [10, 10]
        let centroids = vec![
            Centroid {
                id: 1,
                vector: vec![0.0, 0.0],
            },
            Centroid {
                id: 2,
                vector: vec![10.0, 10.0],
            },
        ];
        let counts = vec![0, 0]; // Counts don't affect simple routing
        let router = Router::new(&centroids, &counts, 2, Metric::L2).unwrap();

        // 2. Create Mixed Batch
        let batch_ids = vec![100, 101, 102];
        let flat_vecs = vec![0.1, 0.1, 9.9, 9.9, 0.2, 0.2];

        // 3. Partition
        let result = IncrementalPartitioner::partition(&batch_ids, &flat_vecs, 2, &router);

        // 4. Verify
        assert_eq!(result.len(), 2);

        // Check Bucket 1
        let g1 = result.get(&1).unwrap();
        assert_eq!(g1.count, 2);
        assert_eq!(g1.ids, vec![100, 102]);
        assert_eq!(g1.flat_vectors, vec![0.1, 0.1, 0.2, 0.2]);

        // Check Bucket 2
        let g2 = result.get(&2).unwrap();
        assert_eq!(g2.count, 1);
        assert_eq!(g2.ids, vec![101]);
        assert!(g1.payload_rows.is_none());
        assert!(g2.payload_rows.is_none());
    }

    #[test]
    fn test_partitioning_carries_payload_rows() {
        let centroids = vec![
            Centroid {
                id: 1,
                vector: vec![0.0, 0.0],
            },
            Centroid {
                id: 2,
                vector: vec![10.0, 10.0],
            },
        ];
        let router = Router::new(&centroids, &[0, 0], 2, Metric::L2).unwrap();

        let batch_ids = vec![100, 101, 102];
        let flat_vecs = vec![0.1, 0.1, 9.9, 9.9, 0.2, 0.2];
        let schema = PayloadSchema::new(vec![PayloadFieldSchema {
            field_id: 1,
            name: "tenant".to_string(),
            logical_type: PayloadLogicalType::Keyword,
            nullable: false,
            indexed: true,
        }]);
        let rows: Vec<PayloadRow> = vec![
            BTreeMap::from([(1, PayloadValue::Keyword("acme".to_string()))]),
            BTreeMap::from([(1, PayloadValue::Keyword("globex".to_string()))]),
            BTreeMap::from([(1, PayloadValue::Keyword("initech".to_string()))]),
        ];

        let result = IncrementalPartitioner::partition_with_payload(
            &batch_ids,
            &flat_vecs,
            2,
            &router,
            Some(&schema),
            Some(&rows),
        )
        .unwrap();

        let g1 = result.get(&1).unwrap();
        let g2 = result.get(&2).unwrap();

        assert_eq!(g1.ids, vec![100, 102]);
        assert_eq!(g2.ids, vec![101]);
        assert_eq!(g1.payload_schema, Some(schema.clone()));
        assert_eq!(g2.payload_schema, Some(schema));

        let g1_rows = g1.payload_rows.as_ref().unwrap();
        assert_eq!(g1_rows.len(), 2);
        assert_eq!(
            g1_rows[0].get(&1),
            Some(&PayloadValue::Keyword("acme".to_string()))
        );
        assert_eq!(
            g1_rows[1].get(&1),
            Some(&PayloadValue::Keyword("initech".to_string()))
        );

        let g2_rows = g2.payload_rows.as_ref().unwrap();
        assert_eq!(g2_rows.len(), 1);
        assert_eq!(
            g2_rows[0].get(&1),
            Some(&PayloadValue::Keyword("globex".to_string()))
        );
    }
}
