use drift_server::drift_proto::{InsertRequest, SearchRequest, Vector, drift_client::DriftClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "http://127.0.0.1:50051";
    let mut client = DriftClient::connect(addr).await?;
    let dim = 128;

    println!("--- TEST 3: MULTI-COLLECTION ISOLATION ---");

    // 1. Insert into 'users'
    println!("Inserting ID 1 into 'users'...");
    client
        .insert(tonic::Request::new(InsertRequest {
            collection_name: "users".to_string(), // <--- Targeted
            vector: Some(Vector {
                id: 1,
                values: vec![0.1; dim],
            }),
        }))
        .await?;

    // 2. Insert into 'products' (Same ID, different data)
    println!("Inserting ID 1 into 'products'...");
    client
        .insert(tonic::Request::new(InsertRequest {
            collection_name: "products".to_string(), // <--- Targeted
            vector: Some(Vector {
                id: 1,
                values: vec![0.9; dim],
            }),
        }))
        .await?;

    // 3. Search 'users'
    println!("Searching 'users' for 0.1...");
    let res_users = client
        .search(tonic::Request::new(SearchRequest {
            collection_name: "users".to_string(),
            vector: vec![0.1; dim],
            k: 1,
        }))
        .await?
        .into_inner();

    let user_score = res_users.results[0].score;
    println!(
        "Found User ID: {}, Score: {:.4}",
        res_users.results[0].id, user_score
    );

    // 4. Search 'products'
    println!("Searching 'products' for 0.9...");
    let res_products = client
        .search(tonic::Request::new(SearchRequest {
            collection_name: "products".to_string(),
            vector: vec![0.9; dim],
            k: 1,
        }))
        .await?
        .into_inner();

    let prod_score = res_products.results[0].score;
    println!(
        "Found Product ID: {}, Score: {:.4}",
        res_products.results[0].id, prod_score
    );

    // 5. Verify Isolation
    if user_score < 0.001 && prod_score < 0.001 {
        println!("✅ SUCCESS: Collections are isolated correctly.");
    } else {
        println!("❌ FAILURE: Cross-contamination detected.");
    }

    Ok(())
}
