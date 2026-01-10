#[cfg(test)]
mod tests {
    use crate::maintainance::MaintenanceCalculator;

    fn mock_flat_data(start_id: u64, count: usize, dim: usize, val: f32) -> (Vec<u64>, Vec<f32>) {
        let mut ids = Vec::with_capacity(count);
        let mut vecs = Vec::with_capacity(count * dim);
        for i in 0..count {
            ids.push(start_id + i as u64);
            for _ in 0..dim {
                vecs.push(val);
            }
        }
        (ids, vecs)
    }

    #[test]
    fn test_split_with_loopback_logic() {
        let dim = 2;

        // 1. Setup Data: Two massive clusters
        // Cluster Left: 100 vectors at [0.0]
        let (mut ids, mut vecs) = mock_flat_data(0, 100, dim, 0.0);

        // Cluster Right: 100 vectors at [20.0] (Far away)
        let (ids_r, vecs_r) = mock_flat_data(100, 100, dim, 20.0);
        ids.extend(ids_r);
        vecs.extend(vecs_r);

        // 2. Add "Defectors": 5 vectors at [9.0]
        // These are geometrically in the middle (0..9..20), closer to 0 (Left).
        // K-Means will essentially put centroids at 0.0 and 20.0.
        // Defector distance to Local (0.0) ~= 9.0 (squared distance ~ 162 for dim 2? No, Euclidean).
        let (ids_d, vecs_d) = mock_flat_data(1000, 5, dim, 9.0);
        ids.extend(ids_d);
        vecs.extend(vecs_d);

        // 3. Define Neighbors
        // Neighbor A: [9.0] (Perfect match for defectors!)
        // Neighbor B: [500.0]
        let neighbors = vec![vec![9.0, 9.0], vec![500.0, 500.0]];

        // 4. Run Split
        let result = MaintenanceCalculator::split_with_loopback(&ids, &vecs, dim, &neighbors)
            .expect("Split failed");

        // 5. Assertions
        // Defector dist to Neighbor = 0.
        // Defector dist to Local (roughly 0.0) = ~12.7.
        // 0 < 12.7 * 0.9 -> True! Defect!

        assert_eq!(result.loopback.len(), 5, "Should identify 5 defectors");
        assert_eq!(result.loopback[0].0, 1000, "Defector ID mismatch");

        // Verify remaining split
        // 200 items remaining
        assert_eq!(result.left.ids.len() + result.right.ids.len(), 200);
    }

    #[test]
    fn test_split_safety_cap() {
        let dim = 2;
        // Scenario: Everyone is unhappy.
        // We have 50 points at [10.0].
        // Neighbor is at [10.0].
        // K-Means will find centroids at [10.0].
        // Local dist = 0. Neighbor dist = 0. No defection.

        // TRICK: Force K-Means to be "bad" for these points by making them secondary.
        // 10 points at [0.0] (Anchor A)
        // 10 points at [20.0] (Anchor B)
        // 50 points at [9.0] (The Mob)
        // Neighbor at [9.0]

        // Actually, simpler: Just make the neighbor *better* than local can possibly be?
        // No, local is 0 if K-Means converges.

        // Let's use the logic: "Safety cap is count / 5".
        // Total 25 points. Cap = 5.
        // 5 points at [0.0].
        // 5 points at [20.0].
        // 15 points at [9.0] -> Want to defect to Neighbor [9.0].
        // K-Means will likely split [0,9] vs [20] or [0] vs [9,20].
        // If [0] vs [9,20] -> Centroid roughly (9*15 + 20*5)/20 = 11.75.
        // Defector [9.0] dist to 11.75 is ~2.75. Neighbor is 0.
        // 0 < 2.75 * 0.9. Defect!
        // But 15 want to defect. Cap is 25/5 = 5.
        // Result: Loopback len should be 5.

        let (mut ids, mut vecs) = mock_flat_data(0, 5, dim, 0.0);
        let (ids_r, vecs_r) = mock_flat_data(5, 5, dim, 20.0);
        ids.extend(ids_r);
        vecs.extend(vecs_r);

        let (ids_d, vecs_d) = mock_flat_data(100, 15, dim, 9.0);
        ids.extend(ids_d);
        vecs.extend(vecs_d);

        let neighbors = vec![vec![9.0, 9.0]];

        let result = MaintenanceCalculator::split_with_loopback(&ids, &vecs, dim, &neighbors)
            .expect("Split failed");

        // Cap check
        let cap = (5 + 5 + 15) / 5; // 5
        assert_eq!(result.loopback.len(), cap, "Should be capped at {}", cap);
    }
}
