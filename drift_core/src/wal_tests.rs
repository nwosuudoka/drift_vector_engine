// drift_storage/src/wal_tests.rs

#[cfg(test)]
mod tests {
    use crate::wal::{WalEntry, WalReader, WalWriter};
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_wal_write_and_recover() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // 1. Write
        {
            let mut writer = WalWriter::new(&path).unwrap();
            writer.write_insert(1, &vec![1.0, 2.0, 3.0]).unwrap();
            writer.write_insert(2, &vec![4.0, 5.0, 6.0]).unwrap();
            writer.sync().unwrap();
        }

        // 2. Recover
        let reader = WalReader::open(&path).unwrap();
        let entries = reader.read_all();

        // 3. Verify
        assert_eq!(entries.len(), 2);
        match &entries[0] {
            WalEntry::Insert { id, vector } => {
                assert_eq!(*id, 1);
                assert_eq!(vector, &vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Unexpected entry type"),
        }
    }

    #[test]
    fn test_wal_handles_truncation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("truncated.wal");

        // 1. Write valid data
        {
            let mut writer = WalWriter::new(&path).unwrap();
            writer.write_insert(1, &vec![1.0]).unwrap();
            writer.sync().unwrap();
        }

        // 2. Corrupt the file (append garbage bytes representing a half-written entry)
        {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            // Write a fake CRC and Length, but no Payload
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap(); // Fake CRC
            file.write_all(&[0x0A, 0x00, 0x00, 0x00]).unwrap(); // Length 10
            file.write_all(&[0x01]).unwrap(); // 1 byte of payload
            // Missing 9 bytes
        }

        // 3. Recover
        let reader = WalReader::open(&path).unwrap();
        let entries = reader.read_all();

        // 4. Verify (Should ignore the garbage at the end)
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            WalEntry::Insert { id, .. } => assert_eq!(*id, 1),
            _ => panic!("Unexpected entry type"),
        }
    }
}
