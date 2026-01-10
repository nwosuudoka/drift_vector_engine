use crate::compression::wrapper::{
    CompressionStrategy, decompress_vectors, transpose_from_columns,
};
use crate::format::{DriftHeader, HEADER_SIZE, ROW_GROUP_HEADER_SIZE, RowGroupHeader};
use byteorder::{LittleEndian, ReadBytesExt};
use crc32fast::Hasher;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use zerocopy::{FromBytes, FromZeros, IntoBytes};

/// A reader for .drift v2 files.
/// Optimized for "Stream-First" access but supports random access via offsets.
pub struct BucketFileReader<R: Read + Seek> {
    reader: R,
    #[allow(dead_code)]
    header: DriftHeader,
    current_offset: u64,
    next_header_offset: u64,
}

impl<R: Read + Seek> BucketFileReader<R> {
    pub fn new(mut reader: R) -> io::Result<Self> {
        // 1. Read File Header
        // let mut header_buf = [0u8; HEADER_SIZE];
        // reader.read_exact(&mut header_buf)?;
        let mut header = DriftHeader::new_zeroed();

        reader.read_exact(header.as_mut_bytes())?;

        if !header.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Magic/Version",
            ));
        }

        Ok(Self {
            reader,
            header,
            current_offset: HEADER_SIZE as u64,
            next_header_offset: HEADER_SIZE as u64,
        })
    }

    /// Reads the next Row Group from the stream.
    /// Returns `None` if EOF.
    ///
    /// # Flow
    /// 1. Seek to start of this group (Skipping previous cold data if unread).
    /// 2. Read RG Header.
    /// 3. Read Hot Index (SQ8 + IDs).
    /// 4. Advance internal state to point to start of Cold Data.
    /// 5. Calculate `next_header_offset` for the future.
    pub fn read_next_group(&mut self) -> io::Result<Option<RowGroup<'_, R>>> {
        // 1. Fast-Forward if needed (Skip previous unread cold data)
        if self.current_offset < self.next_header_offset {
            let skip = self.next_header_offset - self.current_offset;
            self.reader.seek(SeekFrom::Current(skip as i64))?;
            self.current_offset = self.next_header_offset;
        }

        // 2. Read RG Header
        let mut buf = [0u8; ROW_GROUP_HEADER_SIZE];
        let mut bytes_read = 0;
        // Robust read loop for Header
        while bytes_read < ROW_GROUP_HEADER_SIZE {
            let n = self.reader.read(&mut buf[bytes_read..])?;
            if n == 0 {
                if bytes_read == 0 {
                    return Ok(None); // Clean EOF
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Partial RG Header",
                    ));
                }
            }
            bytes_read += n;
        }

        let rg_header = RowGroupHeader::read_from_bytes(&buf).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid RG Header {e}"))
        })?;

        self.current_offset += ROW_GROUP_HEADER_SIZE as u64;

        // 3. Update State for NEXT call
        // The next header starts after this group's cold data.
        self.next_header_offset = rg_header.cold_offset + rg_header.cold_length as u64;

        // 4. Read Hot Index (IDs + SQ8)
        let hot_len = rg_header.hot_length as usize;
        let mut hot_buffer = vec![0u8; hot_len];
        self.reader.read_exact(&mut hot_buffer)?;
        self.current_offset += hot_len as u64;

        // 5. Seek to Cold Start (Skip Padding)
        // This puts the file pointer exactly at the start of Cold Data.
        let next_cold_start = rg_header.cold_offset;
        if next_cold_start > self.current_offset {
            let pad = next_cold_start - self.current_offset;
            self.reader.seek(SeekFrom::Current(pad as i64))?;
            self.current_offset += pad;
        }

        Ok(Some(RowGroup {
            header: rg_header,
            hot_data: hot_buffer,
            reader_ref: self,
        }))
    }
}

/// Represents a single Row Group being processed.
/// Holds the Hot Data in RAM, and a reference to the Reader to fetch Cold Data on demand.
pub struct RowGroup<'a, R: Read + Seek> {
    pub header: RowGroupHeader,
    pub hot_data: Vec<u8>,
    reader_ref: &'a mut BucketFileReader<R>,
}

impl<'a, R: Read + Seek> RowGroup<'a, R> {
    /// Decodes the Hot Index (IDs + SQ8 Codes).
    /// Returns (IDs, Flat SQ8 Codes).
    pub fn decode_hot_index(&self, dim: usize) -> io::Result<(Vec<u64>, Vec<u8>)> {
        let count = self.header.vector_count as usize;
        // Hot Buffer Layout: [IDs (u64 array)] [SQ8 Codes (u8 array)] [Tombstone Len] [Tombstones]
        let id_section_len = count * 8;

        if self.hot_data.len() < id_section_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Hot buffer too small for IDs",
            ));
        }

        let mut cursor = Cursor::new(&self.hot_data);

        // 1. Read IDs
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(cursor.read_u64::<LittleEndian>()?);
        }

        // 2. Read SQ8 Codes
        let code_len = count * dim;
        let mut codes = vec![0u8; code_len];
        cursor.read_exact(&mut codes)?;

        // 3. (Optional) Read Tombstones - logic would go here

        Ok((ids, codes))
    }

    /// Fetches and decompresses the Cold Data (High-Fidelity Vectors).
    /// CAUTION: This performs I/O.
    pub fn fetch_cold_vectors(&mut self, dim: usize) -> io::Result<Vec<f32>> {
        let cold_len = self.header.cold_length as usize;
        let mut cold_buffer = vec![0u8; cold_len];

        // Ensure reader is at the right spot (should be, unless seeking happened externally)
        // For safety, we seek absolute.
        self.reader_ref
            .reader
            .seek(SeekFrom::Start(self.header.cold_offset))?;
        self.reader_ref.reader.read_exact(&mut cold_buffer)?;

        // Update reader offset tracker
        self.reader_ref.current_offset = self.header.cold_offset + cold_len as u64;

        // Verify Checksum
        let mut hasher = Hasher::new();
        hasher.update(&self.hot_data);
        hasher.update(&cold_buffer);
        if hasher.finalize() != self.header.checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Checksum Mismatch",
            ));
        }

        // Decompress
        // Format: [Len][Bytes] repeated for each column
        let mut cursor = Cursor::new(&cold_buffer);
        let mut columns = Vec::with_capacity(dim);

        for _ in 0..dim {
            let chunk_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut chunk = vec![0u8; chunk_len];
            cursor.read_exact(&mut chunk)?;

            let col_floats = decompress_vectors(
                &chunk,
                self.header.vector_count as usize,
                CompressionStrategy::AlpRd,
            );
            columns.push(col_floats);
        }

        // Transpose back to Row-Major (Flat Buffer)
        let rows = transpose_from_columns(&columns, self.header.vector_count as usize);
        // Flatten
        Ok(rows.into_iter().flatten().collect())
    }
}
