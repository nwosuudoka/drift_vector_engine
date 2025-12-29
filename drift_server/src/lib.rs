pub mod config;
pub mod janitor;
pub mod manager;
pub mod persistence;
pub mod server;

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
#[cfg(feature = "stress-test")]
mod server_heavy_load_test;

// Export the generated protobuf code so binaries (client) can use it
pub mod drift_proto {
    tonic::include_proto!("drift");
}
