mod data;
mod inference;
mod model;
mod training;

use std::{net::SocketAddr, path::Path, sync::Arc};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use burn::{optim::AdamConfig, prelude::*};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::wgpu(Default::default());
    let artifact_dir = "guide";

    if !Path::new(artifact_dir).join("config.json").is_file()
        || !Path::new(artifact_dir).join("model.bpk").is_file()
    {
        println!("No trained model found; training the MNIST model first.");
        training::train(
            artifact_dir,
            training::TrainingConfig::new(model::ModelConfig::new(10, 512), AdamConfig::new()),
            device.clone(),
        )?;
    }

    let model = Arc::new(inference::load_model(artifact_dir, &device)?);
    let state = inference::AppState { model, device };
    let app = Router::new()
        .route("/", get(inference::index))
        .route("/predict", post(inference::predict))
        .layer(DefaultBodyLimit::max(20 * 1024 * 1024))
        .with_state(state);

    let address = SocketAddr::from(([127, 0, 0, 1], 3000));
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("Open http://{address} to classify images.");
    axum::serve(listener, app).await?;

    Ok(())
}
