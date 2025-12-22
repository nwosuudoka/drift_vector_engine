pub mod janitor;
pub mod manager;
pub mod persistence;

#[cfg(test)]
mod janitor_tests;
mod manager_tests; // Add this line

// Export the generated protobuf code so binaries (client) can use it
pub mod drift_proto {
    tonic::include_proto!("drift");
}
