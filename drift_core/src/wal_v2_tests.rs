#[cfg(test)]
mod tests {
    use crate::wal_v2::{WalEntry, WalReader, WalWriter};
    use std::fs::OpenOptions;
    use std::io::Write;
    use tempfile::tempdir;

    // --- Helper to verify contents ---
    fn assert_wal_contents(reader: WalReader, expected: &[WalEntry]) {
        let actual = reader.read_committed();
        assert_eq!(
            actual.len(),
            expected.len(),
            "WAL entry count mismatch. Got {:?}, expected {:?}",
            actual,
            expected
        );
        for (a, e) in actual.iter().zip(expected.iter()) {
            // We implement PartialEq for WalEntry so this works
            assert_eq!(a, e, "WAL entry mismatch");
        }
    }

    #[test]
    fn test_wal_transaction_atomicity_commit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("atomicity.wal");

        // 1. Write a full transaction
        {
            let mut writer = WalWriter::new(&path).unwrap();
            let tx = writer.begin_transaction().unwrap();
            writer.write_insert(1, &vec![1.0, 1.0]).unwrap();
            writer.write_delete(2).unwrap();
            writer.commit_transaction(tx).unwrap();
        }

        // 2. Verify: Should see Insert and Delete.
        // Begin/Commit markers are handled internally by the reader.
        let reader = WalReader::open(&path).unwrap();
        let expected = vec![
            WalEntry::Insert {
                id: 1,
                vector: vec![1.0, 1.0],
            },
            WalEntry::Delete { id: 2 },
        ];
        assert_wal_contents(reader, &expected);
    }

    #[test]
    fn test_wal_transaction_atomicity_rollback() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rollback.wal");

        // 1. Start transaction but CRASH (no commit)
        {
            let mut writer = WalWriter::new(&path).unwrap();
            let _tx = writer.begin_transaction().unwrap();
            writer.write_insert(1, &vec![1.0, 1.0]).unwrap();
            // DROP writer without commit
        }

        // 2. Verify Recovery ignores uncommitted data
        let reader = WalReader::open(&path).unwrap();
        let entries = reader.read_committed();
        assert!(
            entries.is_empty(),
            "Uncommitted transaction should be discarded"
        );
    }

    #[test]
    fn test_wal_interleaved_success_and_failure() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("interleaved.wal");

        {
            let mut writer = WalWriter::new(&path).unwrap();

            // TX 1: Success
            let tx1 = writer.begin_transaction().unwrap();
            writer.write_insert(100, &vec![1.0]).unwrap();
            writer.commit_transaction(tx1).unwrap();

            // TX 2: Failure (Crash halfway)
            let _tx2 = writer.begin_transaction().unwrap();
            writer.write_insert(200, &vec![2.0]).unwrap();
            // ... Crash ... (No commit)
        }

        // Re-open to append TX 3 (Simulate restart)
        {
            let mut writer = WalWriter::new(&path).unwrap();
            let tx3 = writer.begin_transaction().unwrap();
            writer.write_insert(300, &vec![3.0]).unwrap();
            writer.commit_transaction(tx3).unwrap();
        }

        // Verify: TX 1 and TX 3 should exist. TX 2 should be gone.
        let reader = WalReader::open(&path).unwrap();
        let entries = reader.read_committed();

        assert_eq!(entries.len(), 2);

        // Check TX 1 presence
        assert!(matches!(entries[0], WalEntry::Insert { id: 100, .. }));
        // Check TX 3 presence
        assert!(matches!(entries[1], WalEntry::Insert { id: 300, .. }));
    }

    #[test]
    fn test_wal_corruption_handling_truncation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupt.wal");

        // 1. Write Valid Data
        {
            let mut writer = WalWriter::new(&path).unwrap();
            let tx = writer.begin_transaction().unwrap();
            writer.write_insert(1, &vec![1.0]).unwrap();
            writer.commit_transaction(tx).unwrap();
        }

        // 2. Corrupt the file (Append garbage bytes)
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            // Write a valid-looking frame header but random junk payload
            // CRC will fail
            file.write_all(&[0xBE, 0xEF, 0x00, 0x00]).unwrap(); // Bad CRC
            file.write_all(&[0x05, 0x00, 0x00, 0x00]).unwrap(); // Length 5
            file.write_all(&[0x01, 0x02, 0x03, 0x04, 0x05]).unwrap(); // Junk
        }

        // 3. Read
        let reader = WalReader::open(&path).unwrap();
        let entries = reader.read_committed();

        // Should successfully read TX 1 and stop at corruption
        assert_eq!(entries.len(), 1); // Only the Insert
        if let WalEntry::Insert { id, .. } = &entries[0] {
            assert_eq!(*id, 1);
        } else {
            panic!("Expected Insert");
        }
    }

    #[test]
    fn test_wal_truncate_operation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("truncate.wal");

        // 1. Fill WAL
        {
            let mut writer = WalWriter::new(&path).unwrap();
            let tx = writer.begin_transaction().unwrap();
            writer.write_insert(1, &vec![0.0]).unwrap();
            writer.commit_transaction(tx).unwrap();

            // 2. Truncate
            writer.truncate().unwrap();
        }

        // 3. Verify Empty
        let file_len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(file_len, 0, "File should be 0 bytes after truncate");

        let reader = WalReader::open(&path).unwrap();
        assert!(reader.read_committed().is_empty());
    }

    #[test]
    fn test_batch_write_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("batch.wal");
        let batch_size = 1000;

        {
            let mut writer = WalWriter::new(&path).unwrap();
            let tx = writer.begin_transaction().unwrap();
            for i in 0..batch_size {
                writer.write_insert(i, &vec![i as f32]).unwrap();
            }
            writer.commit_transaction(tx).unwrap();
        }

        let reader = WalReader::open(&path).unwrap();
        let entries = reader.read_committed();

        // Expect 1000 inserts. Begin/Commit are hidden.
        assert_eq!(entries.len(), batch_size as usize);

        // Check first and last
        assert!(matches!(entries[0], WalEntry::Insert { id: 0, .. }));
        assert!(matches!(entries[999], WalEntry::Insert { id: 999, .. }));
    }

    #[test]
    fn test_nested_begin_handling() {
        // Scenario: Begin(1), Insert(1), Begin(2) [Implicit Rollback of 1], Insert(2), Commit(2)

        let dir = tempdir().unwrap();
        let path = dir.path().join("nested.wal");

        {
            let mut writer = WalWriter::new(&path).unwrap();
            let _tx1 = writer.begin_transaction().unwrap();
            writer.write_insert(1, &vec![1.0]).unwrap();

            // Oops, forgot to commit tx1, start tx2
            // This happens on crash/restart usually, but if called sequentially:
            let tx2 = writer.begin_transaction().unwrap();
            writer.write_insert(2, &vec![2.0]).unwrap();
            writer.commit_transaction(tx2).unwrap();
        }

        let reader = WalReader::open(&path).unwrap();
        let entries = reader.read_committed();

        // Expectation:
        // Begin(1), Insert(1) -> Discarded because it was never committed (or overridden)
        // Begin(2), Insert(2), Commit(2) -> Kept

        assert_eq!(entries.len(), 1); // Only Insert(2)
        match &entries[0] {
            WalEntry::Insert { id, .. } => assert_eq!(*id, 2),
            _ => panic!("Expected insert 2"),
        }
    }
}
