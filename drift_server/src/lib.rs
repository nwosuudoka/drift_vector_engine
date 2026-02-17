pub mod config;
pub mod janitor;
pub mod local_staging;
pub mod manager;
pub mod manifest;
pub mod persistence;
pub mod reaper;
pub mod recovery;
pub mod server;
pub mod storage_factory;

pub mod drift_proto {
    tonic::include_proto!("drift");
}

// #[cfg(test)]
// mod s3_integration_test;

#[cfg(test)]
mod chaos_test;
#[cfg(test)]
mod janitor_tests;
#[cfg(test)]
mod local_staging_test;
#[cfg(test)]
mod manager_tests;
#[cfg(test)]
mod manifest_tests;
#[cfg(test)]
mod persistence_tests;
#[cfg(test)]
mod reaper_test;
#[cfg(test)]
mod recovery_test;
#[cfg(test)]
mod server_integration_tests;
