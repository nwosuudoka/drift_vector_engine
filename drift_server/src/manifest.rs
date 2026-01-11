use drift_core::manifest::ManifestWrapper;
use drift_traits::IoContext;
use parking_lot::RwLock;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Manages the on-disk persistence of the Manifest.
/// Wraps the core ManifestWrapper to add Atomic I/O operations.
pub struct ServerManifestManager {
    base_path: PathBuf,
    manifest_path: PathBuf,
    // The core wrapper holds the actual data (Buckets, Centroids, Version)
    state: RwLock<ManifestWrapper>,
}

impl ServerManifestManager {
    /// Opens or creates a manifest in the given directory.
    pub fn new(base_path: impl AsRef<Path>, dim: u32) -> io::Result<Self> {
        let base = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base).context("Failed to create manifest dir")?;
        let path = base.join("manifest.pb");

        let wrapper = if path.exists() {
            let bytes = fs::read(&path).context("Failed to read manifest file")?;
            ManifestWrapper::from_bytes(&bytes)
                .map_err(|e| io::Error::other(format!("Protobuf decode error: {}", e)))?
        } else {
            // Initialize new manifest
            ManifestWrapper::new(dim, "L2")
        };

        Ok(Self {
            base_path: base,
            manifest_path: path,
            state: RwLock::new(wrapper),
        })
    }

    /// Returns a clone of the inner wrapper (cheap if using Arc internally,
    /// but ManifestWrapper might be clone-heavy if large.
    /// For V2 MVP, cloning the metadata is acceptable).
    pub fn get_state(&self) -> ManifestWrapper {
        self.state.read().clone()
    }

    /// Applies a set of changes to the manifest in memory, then persists ONCE.
    /// This ensures complex ops (Split/Merge) are atomic on disk.
    pub fn apply_atomic<F>(&self, op: F) -> io::Result<()>
    where
        F: FnOnce(&mut ManifestWrapper),
    {
        let mut guard = self.state.write();

        // 1. Apply Changes In-Memory
        op(&mut guard);

        // 2. Bump Version & Persist
        guard.bump_version();
        self.save_atomic(&guard)?;

        Ok(())
    }

    /// Serializes and writes to disk atomically.
    fn save_atomic(&self, wrapper: &ManifestWrapper) -> io::Result<()> {
        let bytes = wrapper.to_bytes();

        // Write to "manifest.pb.tmp"
        let tmp_path = self.base_path.join("manifest.pb.tmp");
        fs::write(&tmp_path, bytes).context("Failed to write tmp manifest")?;

        // Rename "manifest.pb.tmp" -> "manifest.pb" (Atomic on POSIX)
        fs::rename(&tmp_path, &self.manifest_path).context("Failed to swap manifest")?;

        // Directory Sync to ensure directory entry update is flushed
        if let Ok(dir) = fs::File::open(&self.base_path) {
            let _ = dir.sync_all();
        }

        Ok(())
    }
}
