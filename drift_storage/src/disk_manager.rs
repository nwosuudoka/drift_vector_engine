use std::path::Path;
use tokio::fs::File;
use tokio::io;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};

pub struct DiskManager {
    file: File,
}

impl DiskManager {
    pub async fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .await?;
        Ok(Self { file })
    }

    /// Write a compressed blob to a specific offset.
    /// Ensures we write full pages if possible (O_DIRECT friendly).
    pub async fn write_page(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(offset)).await?;
        self.file.write_all(data).await?;
        // In a real DB, we would align `data` to PageBlock::SIZE here
        Ok(())
    }

    /// Read a compressed blob from a specific offset.
    pub async fn read_page(&mut self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let mut buffer = vec![0u8; length];
        self.file.seek(SeekFrom::Start(offset)).await?;
        self.file.read_exact(&mut buffer).await?;
        Ok(buffer)
    }

    /// Write raw bytes to the current cursor position.
    /// This is used for streaming compressed columns.
    pub async fn write_raw(&mut self, data: &[u8]) -> io::Result<()> {
        self.file.write_all(data).await
    }

    /// Seek to a specific offset.
    pub async fn seek(&mut self, offset: u64) -> io::Result<u64> {
        self.file.seek(SeekFrom::Start(offset)).await
    }

    pub async fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all().await
    }

    // Add this helper
    pub async fn read_exact(&mut self, len: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf).await?;
        Ok(buf)
    }

    // Also expose file metadata for size check
    pub fn file(&self) -> &File {
        &self.file
    }

    pub async fn file_len(&self) -> io::Result<u64> {
        Ok(self.file.metadata().await?.len())
    }
}
