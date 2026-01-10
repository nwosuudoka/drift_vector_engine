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
        let router = Router::new(&centroids, 2, "L2").unwrap();

        // 2. Create Mixed Batch
        let batch = vec![
            (100, vec![0.1, 0.1]), // -> Bucket 1
            (101, vec![9.9, 9.9]), // -> Bucket 2
            (102, vec![0.2, 0.2]), // -> Bucket 1
        ];

        // 3. Partition
        let result = IncrementalPartitioner::partition(&batch, &router);

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
