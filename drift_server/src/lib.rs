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

// Export the generated protobuf code so binaries (client) can use it
pub mod drift_proto {
    tonic::include_proto!("drift");
}
