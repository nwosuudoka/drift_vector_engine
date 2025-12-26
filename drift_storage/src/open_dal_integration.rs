use crate::disk_manager::DiskManager;
use crate::segment_reader::SegmentReader;
use crate::segment_writer::SegmentWriter;
use rand::Rng;
use tempfile::tempdir;

#[tokio::test]
async fn test_opendal_local_fs_flow() {
    let dir = tempdir().unwrap();
    // Use absolute path for file:// URI
    let abs_path = dir.path().join("test_seg.drift");
    let uri = format!("file://{}", abs_path.to_str().unwrap());

    // 1. Write (using scratch file + atomic upload)
    let manager = DiskManager::open(&uri).await.unwrap();
    let mut writer = SegmentWriter::new(manager, vec![0x01, 0x02]).await.unwrap();

    writer
        .write_bucket(1, &[100, 200], &vec![vec![1.0], vec![2.0]])
        .await
        .unwrap();
    writer.finalize().await.unwrap();

    // 2. Read (using range requests)
    let reader = SegmentReader::open(&uri).await.unwrap();

    assert_eq!(reader.read_metadata(), &[0x01, 0x02]);
    assert!(reader.might_contain(100));

    let (ids, vecs) = reader.read_bucket(1).await.unwrap();
    assert_eq!(ids, vec![100, 200]);
    assert_eq!(vecs[0][0], 1.0);
}

/// Helper to generate random vectors
fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
        .collect()
}

#[tokio::test]
async fn test_heavy_integration_scenario() {
    let dir = tempdir().unwrap();
    let abs_path = dir.path().join("heavy_segment.drift");
    let uri = format!("file://{}", abs_path.to_str().unwrap());

    // Setup Data
    let dim = 128;
    let quantizer_config = vec![1, 2, 3, 4, 5]; // Mock quantizer data

    // Bucket 1: Small, sequential IDs
    let b1_id = 10;
    let b1_ids = vec![100, 101, 102];
    let b1_vecs = generate_vectors(3, dim);

    // Bucket 2: Large, random IDs
    let b2_id = 20;
    let b2_ids: Vec<u64> = (2000..3000).collect();
    let b2_vecs = generate_vectors(1000, dim);

    // Bucket 3: Edge case (Single vector)
    let b3_id = 30;
    let b3_ids = vec![99999];
    let b3_vecs = generate_vectors(1, dim);

    // --- WRITE PHASE ---
    {
        let manager = DiskManager::open(&uri).await.unwrap();
        let mut writer = SegmentWriter::new(manager, quantizer_config.clone())
            .await
            .unwrap();

        writer.write_bucket(b1_id, &b1_ids, &b1_vecs).await.unwrap();
        writer.write_bucket(b2_id, &b2_ids, &b2_vecs).await.unwrap();
        writer.write_bucket(b3_id, &b3_ids, &b3_vecs).await.unwrap();

        writer.finalize().await.unwrap();
    }

    // --- READ PHASE ---
    let reader = SegmentReader::open(&uri).await.unwrap();

    // 1. Verify Metadata
    assert_eq!(
        reader.read_metadata(),
        &quantizer_config,
        "Quantizer config corrupted"
    );

    // 2. Verify Bloom Filter (Probabilistic check)
    // Should definitely contain these
    assert!(reader.might_contain(100));
    assert!(reader.might_contain(2500));
    assert!(reader.might_contain(99999));
    // Should NOT contain these
    assert!(!reader.might_contain(1));
    assert!(!reader.might_contain(500000));

    // 3. Verify Bucket 1 (Small)
    let (ids, vecs) = reader.read_bucket(b1_id).await.unwrap();
    assert_eq!(ids, b1_ids);
    assert_eq!(vecs.len(), b1_vecs.len());
    // Check first vector values
    for i in 0..dim {
        assert!(
            (vecs[0][i] - b1_vecs[0][i]).abs() < 1e-5,
            "Vector data mismatch in Bucket 1"
        );
    }

    // 4. Verify Bucket 2 (Large)
    let (ids, vecs) = reader.read_bucket(b2_id).await.unwrap();
    assert_eq!(ids.len(), 1000);
    assert_eq!(vecs.len(), 1000);
    assert_eq!(ids[500], 2500); // Check mid-point ID

    // 5. Verify Bucket 3 (Edge case)
    let (ids, vecs) = reader.read_bucket(b3_id).await.unwrap();
    assert_eq!(ids, b3_ids);
    assert_eq!(vecs.len(), 1);

    // 6. Verify Error Handling (Missing Bucket)
    let err = reader.read_bucket(999).await;
    assert!(err.is_err());
    assert_eq!(err.unwrap_err().kind(), std::io::ErrorKind::NotFound);
}

#[tokio::test]
async fn test_empty_write_handling() {
    let dir = tempdir().unwrap();
    let uri = format!(
        "file://{}",
        dir.path().join("empty.drift").to_str().unwrap()
    );

    // Write a segment with NO buckets
    {
        let manager = DiskManager::open(&uri).await.unwrap();
        let writer = SegmentWriter::new(manager, vec![]).await.unwrap();
        writer.finalize().await.unwrap();
    }

    // Read it back
    let reader = SegmentReader::open(&uri).await.unwrap();
    assert!(reader.read_metadata().is_empty());

    // Should handle looking up non-existent bucket gracefully
    assert!(reader.read_bucket(1).await.is_err());
}
