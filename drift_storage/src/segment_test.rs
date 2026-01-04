#[cfg(test)]
mod tests {
    use crate::DriftFooter;
    use crate::disk_manager::DiskManager;
    use crate::segment_reader::SegmentReader;
    use crate::segment_writer::SegmentWriter;
    use bit_set::BitSet;
    use drift_core::aligned::AlignedBytes;
    use drift_core::bucket::BucketData;
    use opendal::{Operator, services};
    use std::io::{Seek, SeekFrom, Write};
    use tempfile::tempdir;

    // --- Helpers ---
    fn create_op(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn create_dummy_bucket(count: usize, dim: usize) -> (BucketData, Vec<Vec<f32>>) {
        let mut codes = AlignedBytes::new(count * dim);
        unsafe {
            codes.set_len(count * dim);
        }
        codes.as_mut_slice().fill(0xAA);

        let vids: Vec<u64> = (0..count as u64).collect();
        let bucket = BucketData {
            codes,
            vids: vids.clone(),
            tombstones: BitSet::new(),
        };
        let raw_vecs = vec![vec![0.0; dim]; count];
        (bucket, raw_vecs)
    }

    // --- Test 1: Happy Path ---
    #[tokio::test]
    async fn test_segment_full_roundtrip_with_footer() {
        let dir = tempdir().unwrap();
        let op = create_op(dir.path());
        let filename = "happy_footer.drift";
        let dim = 128;

        // 1. Write
        {
            let manager = DiskManager::new(op.clone(), filename.to_string());
            let mut writer = SegmentWriter::new(manager, vec![1, 2, 3]).await.unwrap();
            let (b1, v1) = create_dummy_bucket(10, dim);
            writer.write_partition(1, &b1, &v1, dim).await.unwrap();
            writer.finalize().await.unwrap();
        }

        // 2. Read
        let reader = SegmentReader::open_with_op(op.clone(), filename)
            .await
            .unwrap();

        assert_eq!(reader.quantizer, vec![1, 2, 3]);
        let (ids1, codes1) = reader.read_bucket(1).await.unwrap();
        assert_eq!(ids1.len(), 10);
        assert_eq!(codes1[0], 0xAA);
    }

    // --- Test 2: File Magic Corruption (Footer) ---
    #[tokio::test]
    async fn test_corrupt_file_magic() {
        let dir = tempdir().unwrap();
        let op = create_op(dir.path());
        let filename = "corrupt_magic.drift";

        // 1. Create valid file
        {
            let manager = DiskManager::new(op.clone(), filename.to_string());
            let writer = SegmentWriter::new(manager, vec![]).await.unwrap();
            writer.finalize().await.unwrap();
        }

        // 2. Corrupt footer
        {
            let path = dir.path().join(filename);
            let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            let len = file.metadata().unwrap().len();
            file.seek(SeekFrom::Start(len - 1)).unwrap();
            file.write_all(&[0x00]).unwrap();
            // ⚡ Drop file ensuring flush
        }

        // 3. Attempt Read
        match SegmentReader::open_with_op(op.clone(), filename).await {
            Err(e) => {
                let err = e.to_string();
                assert!(err.contains("Invalid Magic Bytes"), "Got: {}", err);
            }
            Ok(_) => panic!("Should have failed"),
        }
    }

    // --- Test 3: Truncated File ---
    #[tokio::test]
    async fn test_truncated_file_footer_check() {
        let dir = tempdir().unwrap();
        let op = create_op(dir.path());
        let filename = "truncated.drift";

        // 1. Write file
        {
            let manager = DiskManager::new(op.clone(), filename.to_string());
            let writer = SegmentWriter::new(manager, vec![]).await.unwrap();
            writer.finalize().await.unwrap();
        }

        // 2. Truncate
        {
            use std::fs::OpenOptions;

            let path = dir.path().join(filename);
            let file = OpenOptions::new().write(true).open(&path).unwrap();

            file.set_len(50).unwrap();
            file.sync_all().unwrap(); // optional, but helps on some FSs
        }

        // 3. Attempt Read
        match SegmentReader::open_with_op(op.clone(), filename).await {
            Ok(_) => panic!("Should have failed"),
            Err(e) => {
                let err = e.to_string();
                assert!(err.contains("File too small"), "Got: {}", err);
            }
        }
    }

    // --- Test 4: Bucket Data Integrity ---
    #[tokio::test]
    async fn test_bucket_magic_integrity_check() {
        let dir = tempdir().unwrap();
        let op = create_op(dir.path());
        let filename = "bucket_integrity.drift";
        let dim = 16;

        // 1. Write Valid
        {
            let manager = DiskManager::new(op.clone(), filename.to_string());
            let mut writer = SegmentWriter::new(manager, vec![]).await.unwrap();
            let (b1, v1) = create_dummy_bucket(10, dim);
            writer.write_partition(1, &b1, &v1, dim).await.unwrap();
            writer.finalize().await.unwrap();
        }

        // 2. Corrupt Bucket Header
        {
            let path = dir.path().join(filename);
            let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            // Start of file is bucket 1
            file.seek(SeekFrom::Start(0)).unwrap();
            file.write_all(&[0x00, 0x00, 0x00, 0x00]).unwrap();
        }

        // 3. Attempt Read
        let reader = SegmentReader::open_with_op(op.clone(), filename)
            .await
            .unwrap();
        let err = reader.read_bucket(1).await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid Bucket Magic"),
            "Got: {}",
            err
        );
    }

    // --- Test 5: Footer Version Check ---
    #[tokio::test]
    async fn test_footer_version_check() {
        let dir = tempdir().unwrap();
        let op = create_op(dir.path());
        let filename = "version_check.drift";

        // 1. Manually write bad version footer
        {
            let mut footer = DriftFooter::new(0, 0, 0, 0, 0, 0);
            footer.version = 99;

            let path = dir.path().join(filename);
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&footer.to_bytes()).unwrap();
            // Explicit drop to close file
        }

        // 2. Attempt Read
        match SegmentReader::open_with_op(op.clone(), filename).await {
            Err(e) => {
                let err = e.to_string();
                assert!(err.contains("Bad Version"), "Got: {}", err);
            }
            Ok(_) => panic!("Should have failed"),
        }
    }
}
