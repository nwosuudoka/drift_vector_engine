use crate::recovery::{RecoveryGuardMetricsSnapshot, RecoveryManager};
use axum::{Router, http::header::CONTENT_TYPE, response::IntoResponse, routing::get};
use drift_storage::disk_manager::{DiskManager, NvmeCacheMetricsSnapshot};
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use tracing::{info, warn};

const METRICS_ADDR_ENV: &str = "DRIFT_METRICS_ADDR";
const METRICS_PORT_ENV: &str = "DRIFT_METRICS_PORT";
const PROM_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub fn metrics_addr_from_env() -> Option<SocketAddr> {
    if let Ok(raw) = std::env::var(METRICS_ADDR_ENV) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.parse::<SocketAddr>() {
            Ok(addr) => return Some(addr),
            Err(_) => {
                warn!(
                    "Metrics exporter disabled: invalid {} value '{}'",
                    METRICS_ADDR_ENV, raw
                );
                return None;
            }
        }
    }

    let raw_port = std::env::var(METRICS_PORT_ENV).ok()?;
    let trimmed = raw_port.trim();
    if trimmed.is_empty() {
        return None;
    }

    match trimmed.parse::<u16>() {
        Ok(port) if port > 0 => Some(SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::UNSPECIFIED,
            port,
        ))),
        _ => {
            warn!(
                "Metrics exporter disabled: invalid {} value '{}'",
                METRICS_PORT_ENV, raw_port
            );
            None
        }
    }
}

pub fn snapshot_prometheus_metrics() -> String {
    let (cache_enabled, cache_snapshot) = match DiskManager::global_nvme_cache_metrics() {
        Some(snapshot) => (true, snapshot),
        None => (false, NvmeCacheMetricsSnapshot::default()),
    };
    let recovery_snapshot = RecoveryManager::global_fingerprint_guard_metrics();
    render_prometheus_metrics(cache_enabled, cache_snapshot, recovery_snapshot)
}

pub fn render_prometheus_metrics(
    cache_enabled: bool,
    cache_snapshot: NvmeCacheMetricsSnapshot,
    recovery_snapshot: RecoveryGuardMetricsSnapshot,
) -> String {
    format!(
        concat!(
            "# HELP drift_nvme_cache_enabled NVMe cache runtime enabled flag.\n",
            "# TYPE drift_nvme_cache_enabled gauge\n",
            "drift_nvme_cache_enabled {}\n",
            "# HELP drift_nvme_cache_hits_total Total NVMe cache hits.\n",
            "# TYPE drift_nvme_cache_hits_total counter\n",
            "drift_nvme_cache_hits_total {}\n",
            "# HELP drift_nvme_cache_misses_total Total NVMe cache misses.\n",
            "# TYPE drift_nvme_cache_misses_total counter\n",
            "drift_nvme_cache_misses_total {}\n",
            "# HELP drift_nvme_cache_remote_fetches_total Remote fetch operations for cache misses.\n",
            "# TYPE drift_nvme_cache_remote_fetches_total counter\n",
            "drift_nvme_cache_remote_fetches_total {}\n",
            "# HELP drift_nvme_cache_singleflight_waits_total Waits behind in-flight cache downloads.\n",
            "# TYPE drift_nvme_cache_singleflight_waits_total counter\n",
            "drift_nvme_cache_singleflight_waits_total {}\n",
            "# HELP drift_nvme_cache_evictions_total Evicted cache entries.\n",
            "# TYPE drift_nvme_cache_evictions_total counter\n",
            "drift_nvme_cache_evictions_total {}\n",
            "# HELP drift_nvme_cache_bytes_cached_total Total bytes cached on NVMe.\n",
            "# TYPE drift_nvme_cache_bytes_cached_total counter\n",
            "drift_nvme_cache_bytes_cached_total {}\n",
            "# HELP drift_nvme_cache_bytes_evicted_total Total bytes evicted from NVMe cache.\n",
            "# TYPE drift_nvme_cache_bytes_evicted_total counter\n",
            "drift_nvme_cache_bytes_evicted_total {}\n",
            "# HELP drift_nvme_cache_invalidations_total Cache invalidations triggered by lifecycle events.\n",
            "# TYPE drift_nvme_cache_invalidations_total counter\n",
            "drift_nvme_cache_invalidations_total {}\n",
            "# HELP drift_nvme_cache_fingerprint_mismatches_total Fingerprint mismatches detected during cache verification.\n",
            "# TYPE drift_nvme_cache_fingerprint_mismatches_total counter\n",
            "drift_nvme_cache_fingerprint_mismatches_total {}\n",
            "# HELP drift_nvme_cache_recovered_entries_total Cache entries recovered from disk metadata at startup.\n",
            "# TYPE drift_nvme_cache_recovered_entries_total counter\n",
            "drift_nvme_cache_recovered_entries_total {}\n",
            "# HELP drift_recovery_guard_mismatches_detected_total Manifest/cache fingerprint mismatches detected during recovery.\n",
            "# TYPE drift_recovery_guard_mismatches_detected_total counter\n",
            "drift_recovery_guard_mismatches_detected_total {}\n",
            "# HELP drift_recovery_guard_invalidations_performed_total Recovery-time cache invalidations performed.\n",
            "# TYPE drift_recovery_guard_invalidations_performed_total counter\n",
            "drift_recovery_guard_invalidations_performed_total {}\n",
            "# HELP drift_recovery_guard_fail_fast_aborts_total Recovery startup aborts caused by fail-fast mismatch policy.\n",
            "# TYPE drift_recovery_guard_fail_fast_aborts_total counter\n",
            "drift_recovery_guard_fail_fast_aborts_total {}\n",
        ),
        u8::from(cache_enabled),
        cache_snapshot.hits,
        cache_snapshot.misses,
        cache_snapshot.remote_fetches,
        cache_snapshot.singleflight_waits,
        cache_snapshot.evictions,
        cache_snapshot.bytes_cached,
        cache_snapshot.bytes_evicted,
        cache_snapshot.invalidations,
        cache_snapshot.fingerprint_mismatches,
        cache_snapshot.recovered_entries,
        recovery_snapshot.mismatches_detected,
        recovery_snapshot.invalidations_performed,
        recovery_snapshot.fail_fast_aborts
    )
}

async fn metrics_handler() -> impl IntoResponse {
    (
        [(CONTENT_TYPE, PROM_CONTENT_TYPE)],
        snapshot_prometheus_metrics(),
    )
}

pub async fn serve_metrics(addr: SocketAddr) -> io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    info!(
        "Metrics exporter listening on http://{}/metrics",
        local_addr
    );

    let app = Router::new().route("/metrics", get(metrics_handler));
    axum::serve(listener, app)
        .await
        .map_err(|err| io::Error::other(format!("metrics server error: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: guarded by ENV_LOCK in tests mutating process env.
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }

        fn unset(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: guarded by ENV_LOCK in tests mutating process env.
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                // SAFETY: guarded by ENV_LOCK in tests mutating process env.
                unsafe { std::env::set_var(self.key, prev) };
            } else {
                // SAFETY: guarded by ENV_LOCK in tests mutating process env.
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    #[test]
    fn test_render_prometheus_metrics_contains_expected_counters() {
        let payload = render_prometheus_metrics(
            true,
            NvmeCacheMetricsSnapshot {
                hits: 10,
                misses: 11,
                remote_fetches: 12,
                singleflight_waits: 13,
                evictions: 14,
                bytes_cached: 15,
                bytes_evicted: 16,
                invalidations: 17,
                fingerprint_mismatches: 18,
                recovered_entries: 19,
            },
            RecoveryGuardMetricsSnapshot {
                mismatches_detected: 20,
                invalidations_performed: 21,
                fail_fast_aborts: 22,
            },
        );

        assert!(payload.contains("drift_nvme_cache_enabled 1"));
        assert!(payload.contains("drift_nvme_cache_hits_total 10"));
        assert!(payload.contains("drift_nvme_cache_misses_total 11"));
        assert!(payload.contains("drift_nvme_cache_recovered_entries_total 19"));
        assert!(payload.contains("drift_recovery_guard_mismatches_detected_total 20"));
        assert!(payload.contains("drift_recovery_guard_invalidations_performed_total 21"));
        assert!(payload.contains("drift_recovery_guard_fail_fast_aborts_total 22"));
    }

    #[test]
    fn test_metrics_addr_from_env_prefers_explicit_addr() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");
        let _addr = EnvVarGuard::set(METRICS_ADDR_ENV, "127.0.0.1:19100");
        let _port = EnvVarGuard::set(METRICS_PORT_ENV, "19101");

        let addr = metrics_addr_from_env();
        assert_eq!(
            addr,
            Some("127.0.0.1:19100".parse().expect("valid metrics addr"))
        );
    }

    #[test]
    fn test_metrics_addr_from_env_uses_port_when_addr_unset() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");
        let _addr = EnvVarGuard::unset(METRICS_ADDR_ENV);
        let _port = EnvVarGuard::set(METRICS_PORT_ENV, "19102");

        let addr = metrics_addr_from_env();
        assert_eq!(
            addr,
            Some(SocketAddr::V4(SocketAddrV4::new(
                Ipv4Addr::UNSPECIFIED,
                19102
            )))
        );
    }

    #[test]
    fn test_metrics_addr_from_env_invalid_value_disables_exporter() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");
        let _addr = EnvVarGuard::set(METRICS_ADDR_ENV, "invalid_addr");
        let _port = EnvVarGuard::unset(METRICS_PORT_ENV);

        assert_eq!(metrics_addr_from_env(), None);
    }
}
