#[cfg(test)]
mod tests {
    use crate::manifest::pb::Centroid;
    use crate::partitioner::IncrementalPartitioner;
    use crate::router::Router;

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
        let router = Router::new(&centroids, &counts, 2, "L2").unwrap();

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
    }
}
