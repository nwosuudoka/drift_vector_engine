use crate::format::{
    DriftFooter, FOOTER_SIZE, HEADER_SIZE, MAGIC_V2, ROW_GROUP_HEADER_SIZE, RowGroupHeader,
};
use byteorder::{LittleEndian, ReadBytesExt};
use drift_core::bucket::compute_distance_lut; // SIMD Kernel
use drift_core::quantizer::Quantizer;
use drift_traits::{PageId, PageManager};
use std::io::{self, Cursor};
use std::sync::Arc;
use zerocopy::FromBytes;

pub struct BucketFileReader {
    storage: Arc<dyn PageManager>,
    file_id: u32,
    quantizer: Option<Quantizer>,
}

impl BucketFileReader {
    pub fn new(storage: Arc<dyn PageManager>, file_id: u32) -> Self {
        Self {
            storage,
            file_id,
            quantizer: None,
        }
    }

    /// Loads the Quantizer from the file footer.
    pub async fn load_quantizer(&mut self) -> io::Result<()> {
        if self.quantizer.is_some() {
            return Ok(());
        }

        let file_len = self.storage.len(self.file_id).await?;
        if file_len < FOOTER_SIZE as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "File too small"));
        }

        // 1. Read Footer
        let footer_page = PageId {
            file_id: self.file_id,
            offset: file_len - FOOTER_SIZE as u64,
            length: FOOTER_SIZE as u32,
        };
        let footer_bytes = self.storage.read_page(footer_page).await?;

        let footer = DriftFooter::read_from_bytes(&footer_bytes).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid Footer {e}"))
        })?;

        if footer.magic != MAGIC_V2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid Magic"));
        }

        // 2. Read Quantizer Blob
        let q_page = PageId {
            file_id: self.file_id,
            offset: footer.quantizer_offset,
            length: footer.quantizer_length,
        };
        let q_bytes = self.storage.read_page(q_page).await?;

        let (q, _): (Quantizer, usize) =
            bincode::decode_from_slice(&q_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        self.quantizer = Some(q);
        Ok(())
    }

    pub async fn scan(&mut self, query: &[f32], _k: usize) -> io::Result<Vec<(u64, f32)>> {
        if self.quantizer.is_none() {
            self.load_quantizer().await?;
        }
        let q = self.quantizer.as_ref().unwrap();
        let lut = q.precompute_lut(query);
        let dim = query.len();

        let mut results = Vec::new();
        let mut current_offset = HEADER_SIZE as u64;

        loop {
            // A. Read RG Header
            let head_page = PageId {
                file_id: self.file_id,
                offset: current_offset,
                length: ROW_GROUP_HEADER_SIZE as u32,
            };

            // Check file bounds implicitly via read failure or partial read?
            // PageManager returns error on out of bounds usually.
            let rg_bytes = match self.storage.read_page(head_page).await {
                Ok(b) if b.len() == ROW_GROUP_HEADER_SIZE => b,
                _ => break, // EOF
            };

            let rg = RowGroupHeader::read_from_bytes(&rg_bytes).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Bad RG Header {e}"))
            })?;

            // B. Read Hot Index (IDs + SQ8)
            let hot_len = rg.hot_length;
            let hot_page = PageId {
                file_id: self.file_id,
                offset: rg.hot_offset,
                length: hot_len,
            };
            let hot_blob = self.storage.read_page(hot_page).await?;

            // C. Scan
            let chunk_res = self.scan_row_group(&rg, &hot_blob, &lut, dim);
            results.extend(chunk_res);

            // D. Advance
            current_offset = rg.cold_offset + rg.cold_length as u64;
        }

        Ok(results)
    }

    fn scan_row_group(
        &self,
        header: &RowGroupHeader,
        hot_blob: &[u8],
        lut: &[f32],
        dim: usize,
    ) -> Vec<(u64, f32)> {
        let count = header.vector_count as usize;
        let mut results = Vec::with_capacity(count);

        // 1. Decode IDs
        let mut cursor = Cursor::new(hot_blob);
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            if let Ok(id) = cursor.read_u64::<LittleEndian>() {
                ids.push(id);
            }
        }

        // 2. Decode SQ8 Codes
        let ids_size = cursor.position() as usize;
        if ids_size >= hot_blob.len() {
            return results;
        }

        let codes = &hot_blob[ids_size..];
        if codes.len() < count * dim {
            return results;
        }

        // 3. Run ADC Kernel (Unsafe SIMD)
        let lut_ptr = lut.as_ptr();
        let codes_ptr = codes.as_ptr();

        for i in 0..count {
            let offset = i * dim;
            let dist = unsafe { compute_distance_lut(codes_ptr.add(offset), lut_ptr, dim) };
            results.push((ids[i], dist));
        }

        results
    }
}
