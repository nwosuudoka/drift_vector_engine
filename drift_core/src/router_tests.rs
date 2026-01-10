#[cfg(test)]
mod tests {
    use crate::{manifest::pb::Centroid, router::Router};

    #[test]
    fn test_router_l2() {
        // Setup: 2 Centroids
        // C1 at [0, 0] (ID 1)
        // C2 at [10, 10] (ID 2)
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

        // Query: [1, 1] -> Should be closer to C1
        let q1 = vec![1.0, 1.0];
        assert_eq!(router.route(&q1), 1);

        // Query: [9, 9] -> Should be closer to C2
        let q2 = vec![9.0, 9.0];
        assert_eq!(router.route(&q2), 2);
    }

    #[test]
    fn test_router_cosine() {
        // Setup: 2 Centroids
        // C1: [1, 0] (X-axis) -> ID 1
        // C2: [0, 1] (Y-axis) -> ID 2
        let centroids = vec![
            Centroid {
                id: 1,
                vector: vec![1.0, 0.0],
            },
            Centroid {
                id: 2,
                vector: vec![0.0, 1.0],
            },
        ];

        let router = Router::new(&centroids, 2, "COSINE").unwrap();

        // Query: [0.9, 0.1] -> Closer angle to X-axis (C1)
        assert_eq!(router.route(&vec![0.9, 0.1]), 1);

        // Query: [0.1, 0.9] -> Closer angle to Y-axis (C2)
        assert_eq!(router.route(&vec![0.1, 0.9]), 2);
    }

    #[test]
    #[should_panic]
    fn test_router_dim_mismatch_init() {
        let centroids = vec![
            Centroid {
                id: 1,
                vector: vec![0.0],
            }, // Dim 1
        ];
        // Init with Dim 2 -> Panic
        Router::new(&centroids, 2, "L2");
    }
}
