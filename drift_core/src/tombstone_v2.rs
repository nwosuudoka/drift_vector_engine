use drift_traits::{TombstoneTracker, TombstoneView};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

/// The Concrete View implementation for our In-Memory Tracker.
/// It wraps the Arc to keep the HashSet implementation details private.
#[derive(Clone)]
pub struct HashSetView {
    inner: Arc<HashSet<u64>>,
}

impl Default for HashSetView {
    fn default() -> Self {
        Self {
            inner: Arc::new(HashSet::new()),
        }
    }
}

impl fmt::Debug for HashSetView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashSetView(items: {})", self.inner.len())
    }
}

impl TombstoneView for HashSetView {
    fn contains(&self, id: u64) -> bool {
        self.inner.contains(&id)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// The Tracker Implementation
#[derive(Debug)]
pub struct InMemoryTombstoneTracker {
    base: RwLock<Arc<HashSet<u64>>>,
    delta: RwLock<HashSet<u64>>,
}

impl InMemoryTombstoneTracker {
    pub fn new() -> Self {
        Self {
            base: RwLock::new(Arc::new(HashSet::new())),
            delta: RwLock::new(HashSet::new()),
        }
    }

    fn flush_delta(&self) {
        let mut delta_guard = self.delta.write();
        if delta_guard.is_empty() {
            return;
        }

        let mut base_guard = self.base.write();
        let mut new_base = (**base_guard).clone();
        for id in delta_guard.drain() {
            new_base.insert(id);
        }
        *base_guard = Arc::new(new_base);
    }
}

#[async_trait::async_trait]
impl TombstoneTracker for InMemoryTombstoneTracker {
    fn mark_delete(&self, id: u64) {
        self.delta.write().insert(id);
    }

    fn is_deleted(&self, id: u64) -> bool {
        if self.delta.read().contains(&id) {
            return true;
        }
        self.base.read().contains(&id)
    }

    fn get_view(&self) -> Arc<dyn TombstoneView> {
        self.flush_delta();

        // Wrap the Arc<HashSet> in our Concrete View Struct
        let snapshot = self.base.read().clone();
        Arc::new(HashSetView { inner: snapshot })
    }

    fn dissolve_batch(&self, compacted_ids: &[u64]) {
        let mut base_guard = self.base.write();
        let mut delta_guard = self.delta.write();

        let mut new_base = (**base_guard).clone();
        for id in compacted_ids {
            new_base.remove(id);
            delta_guard.remove(id);
        }
        *base_guard = Arc::new(new_base);
    }

    fn unmark_delete(&self, id: u64) {
        // Remove from write buffer
        self.delta.write().remove(&id);

        // Remove from base (requires clone-on-write if present)
        // Optimization: Check if base contains it first to avoid unnecessary clone
        let base_read = self.base.read();
        if base_read.contains(&id) {
            drop(base_read); // Drop read lock before write
            let mut base_guard = self.base.write();
            let mut new_base = (**base_guard).clone();
            new_base.remove(&id);
            *base_guard = Arc::new(new_base);
        }
    }
}
