pub mod compactor;
pub mod compactor_test;
pub mod config;
pub mod janitor;
pub mod manager;
pub mod persistence;
pub mod server;
pub mod storage_factory;

#[cfg(test)]
mod janitor_stress_test;
#[cfg(test)]
mod janitor_tests;
#[cfg(test)]
mod manager_tests; // Add this line
#[cfg(test)]
mod server_tests;

#[cfg(test)]
mod persistence_tests;

#[cfg(test)]
mod tombstone_test;

#[cfg(test)]
mod scatter_budget_test;

#[cfg(test)]
mod s3_integration_test;

#[cfg(test)]
mod scavenger_test;

#[cfg(test)]
#[cfg(feature = "stress-test")]
mod server_heavy_load_test;

// Export the generated protobuf code so binaries (client) can use it
pub mod drift_proto {
    tonic::include_proto!("drift");
}
