#[cfg(test)]
mod tests {
    use crate::tombstone_v2::InMemoryTombstoneTracker;
    use drift_traits::TombstoneTracker;

    #[test]
    fn test_mark_and_check_delete() {
        let tracker = InMemoryTombstoneTracker::new();

        // 1. Initially empty
        assert!(!tracker.is_deleted(1));

        // 2. Mark delete (goes to Delta)
        tracker.mark_delete(1);

        // 3. Verify visibility
        assert!(tracker.is_deleted(1));
        assert!(!tracker.is_deleted(2));
    }

    #[test]
    fn test_snapshot_isolation() {
        let tracker = InMemoryTombstoneTracker::new();

        // 1. Mark ID 100
        tracker.mark_delete(100);

        // 2. Take Snapshot (View A)
        // This triggers internal flush_delta() -> base
        let view_a = tracker.get_view();

        assert!(view_a.contains(100));
        assert!(!view_a.contains(200));

        // 3. Mark ID 200 (goes to Delta)
        tracker.mark_delete(200);

        // 4. Verify View A is UNCHANGED (Snapshot Isolation)
        assert!(view_a.contains(100));
        assert!(!view_a.contains(200), "Snapshot should not see new writes");

        // 5. Verify Tracker sees new write
        assert!(tracker.is_deleted(200));

        // 6. Take New Snapshot (View B)
        let view_b = tracker.get_view();
        assert!(view_b.contains(100));
        assert!(view_b.contains(200));
    }

    #[test]
    fn test_dissolve_batch_lifecycle() {
        let tracker = InMemoryTombstoneTracker::new();

        // Scenario:
        // We have deleted vectors 1, 2, 3.
        // We compact bucket A (containing 1 and 2).
        // We want to dissolve 1 and 2 from RAM, but keep 3.

        tracker.mark_delete(1);
        tracker.mark_delete(2);
        tracker.mark_delete(3);

        // Force flush to base (simulate get_view usage)
        let _ = tracker.get_view();

        // Dissolve 1 and 2
        tracker.dissolve_batch(&[1, 2]);

        // Verify 1 and 2 are gone from tracker (RAM freed)
        assert!(!tracker.is_deleted(1));
        assert!(!tracker.is_deleted(2));

        // Verify 3 is still there
        assert!(tracker.is_deleted(3));
    }

    #[test]
    fn test_dissolve_race_condition_safety() {
        let tracker = InMemoryTombstoneTracker::new();

        // Scenario:
        // ID 1 is in Base.
        // ID 1 is ALSO marked again in Delta (race condition or redundant delete).
        // Dissolve should clear BOTH.

        tracker.mark_delete(1);
        let _ = tracker.get_view(); // 1 moves to Base

        tracker.mark_delete(1); // 1 added to Delta again

        // Verify state
        assert!(tracker.is_deleted(1));

        // Dissolve
        tracker.dissolve_batch(&[1]);

        // Verify completely gone
        assert!(!tracker.is_deleted(1));
    }
}
