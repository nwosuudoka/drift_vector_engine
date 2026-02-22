use async_trait::async_trait;
use drift_traits::{PageId, PageManager};
use opendal::{Metadata, Operator};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, warn};

const CACHE_OBJECT_FILE: &str = "object.cache";
const CACHE_META_FILE: &str = "cache.meta";

#[derive(Debug, Clone, Copy, Default)]
pub struct NvmeCacheMetricsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub remote_fetches: u64,
    pub singleflight_waits: u64,
    pub evictions: u64,
    pub bytes_cached: u64,
    pub bytes_evicted: u64,
    pub invalidations: u64,
    pub fingerprint_mismatches: u64,
    pub recovered_entries: u64,
}

impl NvmeCacheMetricsSnapshot {
    fn add(&mut self, other: &Self) {
        self.hits += other.hits;
        self.misses += other.misses;
        self.remote_fetches += other.remote_fetches;
        self.singleflight_waits += other.singleflight_waits;
        self.evictions += other.evictions;
        self.bytes_cached += other.bytes_cached;
        self.bytes_evicted += other.bytes_evicted;
        self.invalidations += other.invalidations;
        self.fingerprint_mismatches += other.fingerprint_mismatches;
        self.recovered_entries += other.recovered_entries;
    }
}

#[derive(Default)]
struct NvmeCacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    remote_fetches: AtomicU64,
    singleflight_waits: AtomicU64,
    evictions: AtomicU64,
    bytes_cached: AtomicU64,
    bytes_evicted: AtomicU64,
    invalidations: AtomicU64,
    fingerprint_mismatches: AtomicU64,
    recovered_entries: AtomicU64,
}

impl NvmeCacheMetrics {
    fn snapshot(&self) -> NvmeCacheMetricsSnapshot {
        NvmeCacheMetricsSnapshot {
            hits: self.hits.load(AtomicOrdering::Relaxed),
            misses: self.misses.load(AtomicOrdering::Relaxed),
            remote_fetches: self.remote_fetches.load(AtomicOrdering::Relaxed),
            singleflight_waits: self.singleflight_waits.load(AtomicOrdering::Relaxed),
            evictions: self.evictions.load(AtomicOrdering::Relaxed),
            bytes_cached: self.bytes_cached.load(AtomicOrdering::Relaxed),
            bytes_evicted: self.bytes_evicted.load(AtomicOrdering::Relaxed),
            invalidations: self.invalidations.load(AtomicOrdering::Relaxed),
            fingerprint_mismatches: self.fingerprint_mismatches.load(AtomicOrdering::Relaxed),
            recovered_entries: self.recovered_entries.load(AtomicOrdering::Relaxed),
        }
    }
}

#[derive(Clone, Debug)]
struct CacheConfig {
    max_total_bytes: Option<u64>,
    max_total_files: Option<usize>,
    max_cached_file_bytes: Option<u64>,
    fingerprint_verify_interval: Duration,
}

impl CacheConfig {
    fn from_env() -> Self {
        let verify_ms =
            parse_env_u64("DRIFT_NVME_FINGERPRINT_VERIFY_INTERVAL_MS").unwrap_or(60_000);
        Self {
            max_total_bytes: parse_env_u64("DRIFT_NVME_CACHE_MAX_BYTES"),
            max_total_files: parse_env_usize("DRIFT_NVME_CACHE_MAX_FILES"),
            max_cached_file_bytes: parse_env_u64("DRIFT_NVME_CACHE_MAX_FILE_BYTES"),
            fingerprint_verify_interval: Duration::from_millis(verify_ms),
        }
    }

    fn registry_key(&self, root: &Path) -> String {
        format!(
            "{}|bytes={}|files={}|file_max={}|verify_ms={}",
            root.to_string_lossy(),
            self.max_total_bytes.unwrap_or(0),
            self.max_total_files.unwrap_or(0),
            self.max_cached_file_bytes.unwrap_or(0),
            self.fingerprint_verify_interval.as_millis()
        )
    }
}

fn parse_env_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn object_fingerprint_from_metadata(meta: &Metadata) -> String {
    let mut fields = Vec::with_capacity(5);
    fields.push(format!("len={}", meta.content_length()));

    if let Some(v) = meta.version() {
        fields.push(format!("version={}", v));
    }
    if let Some(etag) = meta.etag() {
        fields.push(format!("etag={}", etag));
    }
    if let Some(md5) = meta.content_md5() {
        fields.push(format!("md5={}", md5));
    }
    if let Some(ts) = meta.last_modified() {
        fields.push(format!("last_modified={:?}", ts));
    }

    fields.join("|")
}

#[derive(Clone)]
struct ObjectReadCache {
    runtime: Arc<DiskCacheRuntime>,
    namespace: String,
}

#[derive(Clone)]
struct CachePaths {
    dir: PathBuf,
    object_file: PathBuf,
    meta_file: PathBuf,
}

#[derive(Clone)]
struct CacheEntry {
    paths: CachePaths,
    size_bytes: u64,
    last_access_seq: u64,
    access_count: u32,
    fingerprint: String,
    last_verified_ms: u64,
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<String, CacheEntry>,
    total_bytes: u64,
    access_seq: u64,
}

struct DiskCacheRuntime {
    root: PathBuf,
    config: CacheConfig,
    metrics: NvmeCacheMetrics,
    state: std::sync::Mutex<CacheState>,
    inflight: std::sync::Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
}

static RUNTIME_REGISTRY: OnceLock<std::sync::Mutex<HashMap<String, Arc<DiskCacheRuntime>>>> =
    OnceLock::new();

impl DiskCacheRuntime {
    fn runtime_for_root(root: PathBuf, config: CacheConfig) -> Arc<Self> {
        let key = config.registry_key(&root);
        let registry = RUNTIME_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));

        {
            let guard = registry.lock().unwrap();
            if let Some(existing) = guard.get(&key) {
                return existing.clone();
            }
        }

        let runtime = Arc::new(Self::new(root, config));
        let mut guard = registry.lock().unwrap();
        guard.entry(key).or_insert_with(|| runtime.clone()).clone()
    }

    fn global_metrics_snapshot() -> Option<NvmeCacheMetricsSnapshot> {
        let registry = RUNTIME_REGISTRY.get()?;
        let guard = registry.lock().unwrap();
        if guard.is_empty() {
            return None;
        }

        let mut out = NvmeCacheMetricsSnapshot::default();
        for runtime in guard.values() {
            out.add(&runtime.metrics.snapshot());
        }
        Some(out)
    }

    #[cfg(test)]
    fn reset_registry_for_tests() {
        if let Some(registry) = RUNTIME_REGISTRY.get() {
            registry.lock().unwrap().clear();
        }
    }

    fn new(root: PathBuf, config: CacheConfig) -> Self {
        let _ = std::fs::create_dir_all(&root);
        let runtime = Self {
            root,
            config,
            metrics: NvmeCacheMetrics::default(),
            state: std::sync::Mutex::new(CacheState::default()),
            inflight: std::sync::Mutex::new(HashMap::new()),
        };
        runtime.load_existing_entries();
        runtime
    }

    fn cache_paths_for_key(&self, object_key: &str) -> CachePaths {
        let mut hasher = DefaultHasher::new();
        object_key.hash(&mut hasher);
        let hash = hasher.finish();

        let dir = self.root.join(format!("{hash:016x}"));
        CachePaths {
            object_file: dir.join(CACHE_OBJECT_FILE),
            meta_file: dir.join(CACHE_META_FILE),
            dir,
        }
    }

    fn load_existing_entries(&self) {
        let mut state = CacheState::default();
        let mut recovered = 0u64;

        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return;
        };

        for dir_entry in entries.flatten() {
            let dir = dir_entry.path();
            if !dir.is_dir() {
                continue;
            }

            let object_file = dir.join(CACHE_OBJECT_FILE);
            let meta_file = dir.join(CACHE_META_FILE);
            if !object_file.exists() || !meta_file.exists() {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            }

            let Ok((object_key, fingerprint)) = read_cache_meta(&meta_file) else {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            };
            let Ok(meta) = std::fs::metadata(&object_file) else {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            };

            let size = meta.len();
            if size == 0 {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            }

            state.access_seq += 1;
            state.total_bytes = state.total_bytes.saturating_add(size);

            state.entries.insert(
                object_key,
                CacheEntry {
                    paths: CachePaths {
                        dir,
                        object_file,
                        meta_file,
                    },
                    size_bytes: size,
                    last_access_seq: state.access_seq,
                    access_count: 1,
                    fingerprint,
                    last_verified_ms: 0,
                },
            );
            recovered += 1;
        }

        *self.state.lock().unwrap() = state;
        self.metrics
            .recovered_entries
            .fetch_add(recovered, AtomicOrdering::Relaxed);
        self.enforce_budget(None);
    }

    fn should_evict(state: &CacheState, config: &CacheConfig) -> bool {
        let over_bytes = config
            .max_total_bytes
            .map(|max| state.total_bytes > max)
            .unwrap_or(false);
        let over_files = config
            .max_total_files
            .map(|max| state.entries.len() > max)
            .unwrap_or(false);
        over_bytes || over_files
    }

    fn pick_eviction_victim(state: &CacheState, protected_key: Option<&str>) -> Option<String> {
        state
            .entries
            .iter()
            .filter(|(k, _)| Some(k.as_str()) != protected_key)
            .min_by(|(_, a), (_, b)| {
                let a_hot = a.access_count > 1;
                let b_hot = b.access_count > 1;

                // S3FIFO-inspired preference: evict one-hit entries before recurring entries.
                let hot_cmp = match (a_hot, b_hot) {
                    (false, true) => Ordering::Less,
                    (true, false) => Ordering::Greater,
                    _ => Ordering::Equal,
                };
                if hot_cmp != Ordering::Equal {
                    return hot_cmp;
                }

                a.last_access_seq.cmp(&b.last_access_seq)
            })
            .map(|(k, _)| k.clone())
    }

    fn enforce_budget(&self, protected_key: Option<&str>) {
        let victims = {
            let mut guard = self.state.lock().unwrap();
            let mut victims = Vec::new();

            while Self::should_evict(&guard, &self.config) {
                let Some(victim_key) = Self::pick_eviction_victim(&guard, protected_key) else {
                    break;
                };

                if let Some(entry) = guard.entries.remove(&victim_key) {
                    guard.total_bytes = guard.total_bytes.saturating_sub(entry.size_bytes);
                    victims.push(entry);
                } else {
                    break;
                }
            }

            victims
        };

        for victim in victims {
            if let Err(e) = std::fs::remove_dir_all(&victim.paths.dir)
                && e.kind() != io::ErrorKind::NotFound
            {
                warn!(
                    "DiskManager: failed to remove evicted cache dir {}: {}",
                    victim.paths.dir.display(),
                    e
                );
            }
            self.metrics.evictions.fetch_add(1, AtomicOrdering::Relaxed);
            self.metrics
                .bytes_evicted
                .fetch_add(victim.size_bytes, AtomicOrdering::Relaxed);
        }
    }

    fn touch_entry(&self, object_key: &str) {
        let mut guard = self.state.lock().unwrap();
        let next = guard.access_seq.saturating_add(1);
        guard.access_seq = next;

        if let Some(entry) = guard.entries.get_mut(object_key) {
            entry.last_access_seq = next;
            entry.access_count = entry.access_count.saturating_add(1);
        }
    }

    fn get_entry(&self, object_key: &str) -> Option<CacheEntry> {
        self.state.lock().unwrap().entries.get(object_key).cloned()
    }

    async fn read_cached_range(
        &self,
        object_key: &str,
        offset: u64,
        length: usize,
    ) -> io::Result<Option<Vec<u8>>> {
        let Some(entry) = self.get_entry(object_key) else {
            self.metrics.misses.fetch_add(1, AtomicOrdering::Relaxed);
            return Ok(None);
        };

        let bytes = read_range_from_file(&entry.paths.object_file, offset, length).await?;
        if bytes.is_some() {
            self.metrics.hits.fetch_add(1, AtomicOrdering::Relaxed);
            self.touch_entry(object_key);
            return Ok(bytes);
        }

        self.metrics.misses.fetch_add(1, AtomicOrdering::Relaxed);
        let _ = self.remove_entry(object_key, false);
        Ok(None)
    }

    fn should_verify_fingerprint(&self, object_key: &str) -> bool {
        let interval = self.config.fingerprint_verify_interval;
        if interval.is_zero() {
            return false;
        }

        let guard = self.state.lock().unwrap();
        let Some(entry) = guard.entries.get(object_key) else {
            return false;
        };

        now_millis().saturating_sub(entry.last_verified_ms) >= interval.as_millis() as u64
    }

    fn cached_fingerprint(&self, object_key: &str) -> Option<String> {
        self.state
            .lock()
            .unwrap()
            .entries
            .get(object_key)
            .map(|e| e.fingerprint.clone())
    }

    fn mark_verified(&self, object_key: &str, fingerprint: &str) {
        let mut guard = self.state.lock().unwrap();
        if let Some(entry) = guard.entries.get_mut(object_key) {
            entry.last_verified_ms = now_millis();
            if entry.fingerprint.is_empty() {
                entry.fingerprint = fingerprint.to_string();
            }
        }
    }

    async fn store_object(
        &self,
        object_key: &str,
        payload: &[u8],
        fingerprint: &str,
    ) -> io::Result<()> {
        let paths = self.cache_paths_for_key(object_key);

        tokio::fs::create_dir_all(&paths.dir).await?;
        write_file_atomic(&paths.object_file, payload).await?;
        write_cache_meta_atomic(&paths.meta_file, object_key, fingerprint).await?;

        let size = payload.len() as u64;
        {
            let mut guard = self.state.lock().unwrap();
            if let Some(prev) = guard.entries.remove(object_key) {
                guard.total_bytes = guard.total_bytes.saturating_sub(prev.size_bytes);
            }

            let next = guard.access_seq.saturating_add(1);
            guard.access_seq = next;
            guard.total_bytes = guard.total_bytes.saturating_add(size);

            guard.entries.insert(
                object_key.to_string(),
                CacheEntry {
                    paths,
                    size_bytes: size,
                    last_access_seq: next,
                    access_count: 1,
                    fingerprint: fingerprint.to_string(),
                    last_verified_ms: now_millis(),
                },
            );
        }

        self.metrics
            .bytes_cached
            .fetch_add(size, AtomicOrdering::Relaxed);
        self.enforce_budget(None);
        Ok(())
    }

    fn remove_entry(&self, object_key: &str, count_invalidation: bool) -> io::Result<()> {
        let removed = {
            let mut guard = self.state.lock().unwrap();
            let removed = guard.entries.remove(object_key);
            if let Some(entry) = &removed {
                guard.total_bytes = guard.total_bytes.saturating_sub(entry.size_bytes);
            }
            removed
        };

        if let Some(entry) = removed {
            if count_invalidation {
                self.metrics
                    .invalidations
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }

            match std::fs::remove_dir_all(&entry.paths.dir) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(e),
            }
        } else {
            // Fallback cleanup for orphaned dirs if they exist.
            let paths = self.cache_paths_for_key(object_key);
            match std::fs::remove_dir_all(paths.dir) {
                Ok(()) => {
                    if count_invalidation {
                        self.metrics
                            .invalidations
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    Ok(())
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(e),
            }
        }
    }

    fn inflight_lock_for_key(&self, object_key: &str) -> Arc<AsyncMutex<()>> {
        let mut guard = self.inflight.lock().unwrap();
        guard
            .entry(object_key.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    fn prune_inflight_lock(&self, object_key: &str, lock: &Arc<AsyncMutex<()>>) {
        if Arc::strong_count(lock) > 2 {
            return;
        }

        let mut guard = self.inflight.lock().unwrap();
        if let Some(current) = guard.get(object_key)
            && Arc::ptr_eq(current, lock)
            && Arc::strong_count(current) <= 2
        {
            guard.remove(object_key);
        }
    }
}

async fn write_file_atomic(path: &Path, payload: &[u8]) -> io::Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp_path = path.with_extension(format!("tmp-{nonce}"));

    tokio::fs::write(&tmp_path, payload).await?;
    tokio::fs::rename(tmp_path, path).await
}

async fn write_cache_meta_atomic(
    path: &Path,
    object_key: &str,
    fingerprint: &str,
) -> io::Result<()> {
    let body = format!("{}\n{}\n", object_key, fingerprint);
    write_file_atomic(path, body.as_bytes()).await
}

fn read_cache_meta(path: &Path) -> io::Result<(String, String)> {
    let data = std::fs::read_to_string(path)?;
    let mut lines = data.lines();

    let object_key = lines.next().unwrap_or_default().trim().to_string();
    if object_key.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Missing object key in cache meta",
        ));
    }
    let fingerprint = lines.next().unwrap_or_default().trim().to_string();
    Ok((object_key, fingerprint))
}

async fn read_range_from_file(
    path: &Path,
    offset: u64,
    length: usize,
) -> io::Result<Option<Vec<u8>>> {
    let mut file = match tokio::fs::File::open(path).await {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };

    let end = offset
        .checked_add(length as u64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Read overflow"))?;

    let file_len = file.metadata().await?.len();
    if end > file_len {
        return Ok(None);
    }

    file.seek(SeekFrom::Start(offset)).await?;
    let mut out = vec![0u8; length];
    match file.read_exact(&mut out).await {
        Ok(_) => Ok(Some(out)),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

// --- DiskManager (Flat File Access) ---
#[derive(Clone)]
pub struct DiskManager {
    op: Operator,
    pub path: String, // Relative path inside the bucket/root
    cache: Option<ObjectReadCache>,
}

impl DiskManager {
    /// Create a manager using an existing Operator.
    /// The 'path' is relative to the Operator's root.
    pub fn new(op: Operator, path: String) -> Self {
        tracing::debug!("DiskManager: Attached to {}", path);
        Self {
            cache: Self::build_object_cache_with_overrides(&op, None, false, None),
            op,
            path,
        }
    }

    #[cfg(test)]
    fn new_with_cache_for_test(
        op: Operator,
        path: String,
        cache_root: PathBuf,
        config: CacheConfig,
    ) -> Self {
        Self {
            cache: Self::build_object_cache_with_overrides(
                &op,
                Some(cache_root),
                true,
                Some(config),
            ),
            op,
            path,
        }
    }

    fn object_namespace(op: &Operator) -> String {
        let info = op.info();
        format!("{}://{}{}", info.scheme(), info.name(), info.root())
    }

    fn object_cache_key(namespace: &str, path: &str) -> String {
        format!("{}::{}", namespace, path)
    }

    fn nvme_cache_root() -> Option<PathBuf> {
        let root = std::env::var("DRIFT_NVME_CACHE_DIR").ok()?;
        let root = PathBuf::from(root);
        std::fs::create_dir_all(&root).ok()?;
        Some(root)
    }

    fn is_remote_scheme(scheme: &str) -> bool {
        if std::env::var("DRIFT_NVME_CACHE_FORCE_ALL_SCHEMES")
            .ok()
            .as_deref()
            == Some("1")
        {
            return true;
        }

        !matches!(scheme, "fs")
    }

    fn build_object_cache_with_overrides(
        op: &Operator,
        cache_root_override: Option<PathBuf>,
        force_cache: bool,
        config_override: Option<CacheConfig>,
    ) -> Option<ObjectReadCache> {
        let scheme = op.info().scheme();
        if !force_cache && !Self::is_remote_scheme(scheme) {
            return None;
        }

        let root = if let Some(root) = cache_root_override {
            std::fs::create_dir_all(&root).ok()?;
            root
        } else {
            Self::nvme_cache_root()?
        };

        let config = config_override.unwrap_or_else(CacheConfig::from_env);
        let runtime = DiskCacheRuntime::runtime_for_root(root, config);
        let namespace = Self::object_namespace(op);

        Some(ObjectReadCache { runtime, namespace })
    }

    pub fn global_nvme_cache_metrics() -> Option<NvmeCacheMetricsSnapshot> {
        DiskCacheRuntime::global_metrics_snapshot()
    }

    #[cfg(test)]
    fn reset_cache_runtime_registry_for_tests() {
        DiskCacheRuntime::reset_registry_for_tests();
    }

    pub fn nvme_cached_fingerprint_for_object(op: &Operator, path: &str) -> Option<String> {
        if !Self::is_remote_scheme(op.info().scheme()) {
            return None;
        }

        let root = Self::nvme_cache_root()?;
        let runtime = DiskCacheRuntime::runtime_for_root(root, CacheConfig::from_env());
        let namespace = Self::object_namespace(op);
        let object_key = Self::object_cache_key(&namespace, path);
        runtime.cached_fingerprint(&object_key)
    }

    pub async fn invalidate_nvme_cache_for_object(op: &Operator, path: &str) -> io::Result<()> {
        if !Self::is_remote_scheme(op.info().scheme()) {
            return Ok(());
        }

        let Some(root) = Self::nvme_cache_root() else {
            return Ok(());
        };

        let runtime = DiskCacheRuntime::runtime_for_root(root, CacheConfig::from_env());
        let namespace = Self::object_namespace(op);
        let object_key = Self::object_cache_key(&namespace, path);
        runtime.remove_entry(&object_key, true)
    }

    async fn read_remote_range(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let end = offset
            .checked_add(length as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Read overflow"))?;

        let range = offset..end;
        let data = self
            .op
            .read_with(&self.path)
            .range(range)
            .await
            .map_err(io::Error::other)?;

        if data.len() != length {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Short read"));
        }

        Ok(data.to_vec())
    }

    // Removed: pub async fn open(uri: &str) -> ...

    pub async fn read_at(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }

        if let Some(cache) = &self.cache {
            let object_key = Self::object_cache_key(&cache.namespace, &self.path);

            if let Some(bytes) = cache
                .runtime
                .read_cached_range(&object_key, offset, length)
                .await?
            {
                if cache.runtime.should_verify_fingerprint(&object_key) {
                    match self.op.stat(&self.path).await {
                        Ok(meta) => {
                            let remote_fp = object_fingerprint_from_metadata(&meta);
                            let cached_fp = cache
                                .runtime
                                .cached_fingerprint(&object_key)
                                .unwrap_or_default();

                            if !cached_fp.is_empty() && cached_fp != remote_fp {
                                cache
                                    .runtime
                                    .metrics
                                    .fingerprint_mismatches
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                                let _ = cache.runtime.remove_entry(&object_key, false);
                            } else {
                                cache.runtime.mark_verified(&object_key, &remote_fp);
                                return Ok(bytes);
                            }
                        }
                        Err(e) => {
                            debug!(
                                "DiskManager: remote stat failed while verifying cache {}: {}",
                                self.path, e
                            );
                            return Ok(bytes);
                        }
                    }
                } else {
                    return Ok(bytes);
                }
            }

            let inflight = cache.runtime.inflight_lock_for_key(&object_key);
            let guard = match inflight.try_lock() {
                Ok(g) => g,
                Err(_) => {
                    cache
                        .runtime
                        .metrics
                        .singleflight_waits
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    inflight.lock().await
                }
            };

            if let Some(bytes) = cache
                .runtime
                .read_cached_range(&object_key, offset, length)
                .await?
            {
                drop(guard);
                cache.runtime.prune_inflight_lock(&object_key, &inflight);
                return Ok(bytes);
            }

            let remote_meta = self.op.stat(&self.path).await.map_err(io::Error::other)?;
            if let Some(max_bytes) = cache.runtime.config.max_cached_file_bytes
                && remote_meta.content_length() > max_bytes
            {
                cache
                    .runtime
                    .metrics
                    .remote_fetches
                    .fetch_add(1, AtomicOrdering::Relaxed);

                drop(guard);
                cache.runtime.prune_inflight_lock(&object_key, &inflight);
                return self.read_remote_range(offset, length).await;
            }

            cache
                .runtime
                .metrics
                .remote_fetches
                .fetch_add(1, AtomicOrdering::Relaxed);

            let full = self.op.read(&self.path).await.map_err(io::Error::other)?;
            let payload = full.to_vec();

            let end = offset
                .checked_add(length as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Read overflow"))?;
            if end > payload.len() as u64 {
                drop(guard);
                cache.runtime.prune_inflight_lock(&object_key, &inflight);
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Short read"));
            }

            let start_idx = offset as usize;
            let end_idx = end as usize;
            let out = payload[start_idx..end_idx].to_vec();

            let fingerprint = object_fingerprint_from_metadata(&remote_meta);
            if let Err(e) = cache
                .runtime
                .store_object(&object_key, &payload, &fingerprint)
                .await
            {
                warn!(
                    "DiskManager: failed to cache object {} on NVMe: {}",
                    self.path, e
                );
            }

            drop(guard);
            cache.runtime.prune_inflight_lock(&object_key, &inflight);
            return Ok(out);
        }

        self.read_remote_range(offset, length).await
    }

    pub async fn len(&self) -> io::Result<u64> {
        let meta = self.op.stat(&self.path).await.map_err(io::Error::other)?;
        Ok(meta.content_length())
    }

    pub async fn upload(&self, data: Vec<u8>) -> io::Result<()> {
        self.op
            .write(&self.path, data)
            .await
            .map_err(io::Error::other)?;
        Ok(())
    }
}

// --- DriftPageManager (Page-Based Access) ---
#[derive(Clone)]
pub struct DriftPageManager {
    op: Operator,
    // Maps FileID -> Relative Path (e.g., 1 -> "segment_1.driftu")
    files: Arc<RwLock<HashMap<u32, String>>>,
}

impl DriftPageManager {
    /// Creates a new PageManager wrapping an OpenDAL Operator.
    /// The Operator should already be configured with the correct Root/Bucket.
    pub fn new(op: Operator) -> Self {
        Self {
            op,
            files: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl PageManager for DriftPageManager {
    fn register_file(&self, file_id: u32, path: PathBuf) {
        // We only care about the filename (relative path) since Operator is rooted.
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.driftu")
            .to_string();
        let mut map = self.files.write().unwrap();
        map.insert(file_id, filename);
    }

    async fn read_page(&self, page_id: PageId) -> io::Result<Vec<u8>> {
        let path = {
            let map = self.files.read().unwrap();
            map.get(&page_id.file_id).cloned()
        };

        if let Some(p) = path {
            let range = page_id.offset..page_id.offset + page_id.length as u64;
            let data = self.op.read_with(&p).range(range).await?;
            if data.len() != page_id.length as usize {
                return Err(io::Error::other("Short read"));
            }
            Ok(data.to_vec())
        } else {
            Err(io::Error::other(format!(
                "File ID {} not registered",
                page_id.file_id,
            )))
        }
    }

    async fn write_page(&self, file_id: u32, offset: u64, data: &[u8]) -> io::Result<()> {
        let filename = {
            let mut map = self.files.write().unwrap();
            map.entry(file_id)
                .or_insert_with(|| format!("page_{}", file_id))
                .clone()
        };

        // RMW Strategy (Simulated for Object Storage)
        let exists = self.op.exists(&filename).await?;
        let mut full_data = if exists {
            self.op.read(&filename).await?.to_vec()
        } else {
            Vec::new()
        };

        let end = offset as usize + data.len();
        if full_data.len() < end {
            full_data.resize(end, 0);
        }
        full_data[offset as usize..end].copy_from_slice(data);
        self.op.write(&filename, full_data).await?;
        Ok(())
    }

    /// Used by Compactor to determine liveness.
    /// Returns None if the ID is not registered or purely in-memory.
    fn get_physical_path(&self, file_id: u32) -> Option<String> {
        self.files.read().unwrap().get(&file_id).cloned()
    }

    async fn len(&self, file_id: u32) -> std::io::Result<u64> {
        let path = {
            let map = self.files.read().unwrap();
            map.get(&file_id).cloned()
        };

        if let Some(p) = path {
            let meta = self.op.stat(&p).await?;
            Ok(meta.content_length())
        } else {
            Err(io::Error::other(format!(
                "File ID {} not registered",
                file_id
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opendal::services;
    use tempfile::tempdir;

    fn fs_op(root: &Path) -> Operator {
        let builder = services::Fs::default().root(root.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn config(
        max_total_bytes: Option<u64>,
        max_total_files: Option<usize>,
        max_cached_file_bytes: Option<u64>,
        verify_ms: u64,
    ) -> CacheConfig {
        CacheConfig {
            max_total_bytes,
            max_total_files,
            max_cached_file_bytes,
            fingerprint_verify_interval: Duration::from_millis(verify_ms),
        }
    }

    #[tokio::test]
    async fn test_full_file_cache_then_range_slice() {
        DiskManager::reset_cache_runtime_registry_for_tests();

        let remote_root = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let op = fs_op(remote_root.path());

        let path = "obj_a.bin";
        let bytes: Vec<u8> = (0..128u8).collect();
        op.write(path, bytes.clone()).await.unwrap();

        let mgr = DiskManager::new_with_cache_for_test(
            op.clone(),
            path.to_string(),
            cache_root.path().to_path_buf(),
            config(None, None, None, 0),
        );

        let first = mgr.read_at(10, 16).await.unwrap();
        assert_eq!(first, bytes[10..26].to_vec());

        op.write(path, vec![7u8; 128]).await.unwrap();
        let second = mgr.read_at(10, 16).await.unwrap();
        assert_eq!(second, bytes[10..26].to_vec());

        let stats = mgr.cache.as_ref().unwrap().runtime.metrics.snapshot();
        assert!(stats.hits >= 1);
        assert!(stats.remote_fetches >= 1);
    }

    #[tokio::test]
    async fn test_singleflight_prevents_duplicate_downloads() {
        DiskManager::reset_cache_runtime_registry_for_tests();

        let remote_root = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let op = fs_op(remote_root.path());

        let path = "obj_sf.bin";
        let payload = vec![3u8; 256 * 1024];
        op.write(path, payload.clone()).await.unwrap();

        let mgr = Arc::new(DiskManager::new_with_cache_for_test(
            op.clone(),
            path.to_string(),
            cache_root.path().to_path_buf(),
            config(None, None, None, 0),
        ));

        let before = mgr.cache.as_ref().unwrap().runtime.metrics.snapshot();

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let m = mgr.clone();
            tasks.push(tokio::spawn(
                async move { m.read_at(1024, 64).await.unwrap() },
            ));
        }

        for t in tasks {
            let out = t.await.unwrap();
            assert_eq!(out, payload[1024..1088].to_vec());
        }

        let after = mgr.cache.as_ref().unwrap().runtime.metrics.snapshot();

        assert_eq!(
            after.remote_fetches.saturating_sub(before.remote_fetches),
            1
        );
        assert!(
            after
                .singleflight_waits
                .saturating_sub(before.singleflight_waits)
                >= 1
        );
    }

    #[tokio::test]
    async fn test_budget_eviction_by_file_count() {
        DiskManager::reset_cache_runtime_registry_for_tests();

        let remote_root = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let op = fs_op(remote_root.path());

        op.write("obj1.bin", vec![1u8; 64]).await.unwrap();
        op.write("obj2.bin", vec![2u8; 64]).await.unwrap();

        let m1 = DiskManager::new_with_cache_for_test(
            op.clone(),
            "obj1.bin".to_string(),
            cache_root.path().to_path_buf(),
            config(None, Some(1), None, 0),
        );
        let m2 = DiskManager::new_with_cache_for_test(
            op.clone(),
            "obj2.bin".to_string(),
            cache_root.path().to_path_buf(),
            config(None, Some(1), None, 0),
        );

        let _ = m1.read_at(0, 8).await.unwrap();
        let _ = m2.read_at(0, 8).await.unwrap();

        let rt = m2.cache.as_ref().unwrap().runtime.clone();
        let state = rt.state.lock().unwrap();

        assert_eq!(state.entries.len(), 1);
        let ns = &m2.cache.as_ref().unwrap().namespace;
        let key2 = DiskManager::object_cache_key(ns, "obj2.bin");
        assert!(state.entries.contains_key(&key2));
    }

    #[tokio::test]
    async fn test_budget_eviction_by_total_bytes() {
        DiskManager::reset_cache_runtime_registry_for_tests();

        let remote_root = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let op = fs_op(remote_root.path());

        op.write("obj1.bin", vec![1u8; 64]).await.unwrap();
        op.write("obj2.bin", vec![2u8; 64]).await.unwrap();

        let m1 = DiskManager::new_with_cache_for_test(
            op.clone(),
            "obj1.bin".to_string(),
            cache_root.path().to_path_buf(),
            config(Some(100), None, None, 0),
        );
        let m2 = DiskManager::new_with_cache_for_test(
            op.clone(),
            "obj2.bin".to_string(),
            cache_root.path().to_path_buf(),
            config(Some(100), None, None, 0),
        );

        let _ = m1.read_at(0, 8).await.unwrap();
        let _ = m2.read_at(0, 8).await.unwrap();

        let rt = m2.cache.as_ref().unwrap().runtime.clone();
        let state = rt.state.lock().unwrap();

        assert_eq!(state.entries.len(), 1);
        assert!(state.total_bytes <= 100);
        let ns = &m2.cache.as_ref().unwrap().namespace;
        let key2 = DiskManager::object_cache_key(ns, "obj2.bin");
        assert!(state.entries.contains_key(&key2));
    }

    #[tokio::test]
    async fn test_fingerprint_mismatch_invalidates_and_refreshes() {
        DiskManager::reset_cache_runtime_registry_for_tests();

        let remote_root = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let op = fs_op(remote_root.path());

        let path = "obj_fp.bin";
        op.write(path, vec![10u8; 64]).await.unwrap();

        let mgr = DiskManager::new_with_cache_for_test(
            op.clone(),
            path.to_string(),
            cache_root.path().to_path_buf(),
            config(None, None, None, 1),
        );

        let first = mgr.read_at(0, 16).await.unwrap();
        assert_eq!(first, vec![10u8; 16]);

        tokio::time::sleep(Duration::from_millis(3)).await;
        op.write(path, vec![11u8; 80]).await.unwrap();

        let before = mgr.cache.as_ref().unwrap().runtime.metrics.snapshot();

        let second = mgr.read_at(0, 16).await.unwrap();
        assert_eq!(second, vec![11u8; 16]);

        let after = mgr.cache.as_ref().unwrap().runtime.metrics.snapshot();
        assert!(
            after
                .fingerprint_mismatches
                .saturating_sub(before.fingerprint_mismatches)
                >= 1
        );
    }

    #[tokio::test]
    async fn test_metadata_recovery_after_runtime_restart() {
        DiskManager::reset_cache_runtime_registry_for_tests();

        let remote_root = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let op = fs_op(remote_root.path());

        let path = "obj_recover.bin";
        op.write(path, vec![42u8; 48]).await.unwrap();

        let mgr1 = DiskManager::new_with_cache_for_test(
            op.clone(),
            path.to_string(),
            cache_root.path().to_path_buf(),
            config(None, None, None, 60_000),
        );

        let first = mgr1.read_at(4, 8).await.unwrap();
        assert_eq!(first, vec![42u8; 8]);

        op.delete(path).await.unwrap();

        DiskManager::reset_cache_runtime_registry_for_tests();
        let mgr2 = DiskManager::new_with_cache_for_test(
            op.clone(),
            path.to_string(),
            cache_root.path().to_path_buf(),
            config(None, None, None, 60_000),
        );

        let second = mgr2.read_at(4, 8).await.unwrap();
        assert_eq!(second, vec![42u8; 8]);
    }
}
