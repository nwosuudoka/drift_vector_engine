use crate::math::{Metric, cosine_distance_simd_friendly, l2_sq_simd_friendly};

pub trait MetricStrategy: Send + Sync {
    fn metric(&self) -> Metric;
    /// Returns a canonical score where lower is better.
    fn score(&self, query: &[f32], candidate: &[f32]) -> f32;
}

#[derive(Debug)]
pub struct L2MetricStrategy;

impl MetricStrategy for L2MetricStrategy {
    fn metric(&self) -> Metric {
        Metric::L2
    }

    #[inline(always)]
    fn score(&self, query: &[f32], candidate: &[f32]) -> f32 {
        l2_sq_simd_friendly(query, candidate)
    }
}

#[derive(Debug)]
pub struct CosineMetricStrategy;

impl MetricStrategy for CosineMetricStrategy {
    fn metric(&self) -> Metric {
        Metric::COSINE
    }

    #[inline(always)]
    fn score(&self, query: &[f32], candidate: &[f32]) -> f32 {
        cosine_distance_simd_friendly(query, candidate)
    }
}

static L2_STRATEGY: L2MetricStrategy = L2MetricStrategy;
static COSINE_STRATEGY: CosineMetricStrategy = CosineMetricStrategy;

#[inline]
pub fn strategy_for(metric: Metric) -> &'static dyn MetricStrategy {
    match metric {
        Metric::L2 => &L2_STRATEGY,
        Metric::COSINE => &COSINE_STRATEGY,
    }
}
