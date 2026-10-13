use std::io::Cursor;

use crate::{model::Model, training::TrainingConfig};
use axum::{
    Json,
    extract::{Multipart, State},
    http::StatusCode,
    response::Html,
};
use burn::{prelude::*, store::ModuleRecord};
use image::{DynamicImage, ImageReader, Limits, imageops::FilterType};
use serde::Serialize;
use std::sync::Arc;

const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const MAX_IMAGE_DIMENSION: u32 = 4096;
const MAX_FILES_PER_UPLOAD: usize = 20;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Model>,
    pub device: Device,
}

#[derive(Serialize)]
pub struct Prediction {
    filename: String,
    digit: u8,
}

#[derive(Serialize)]
pub struct ApiError {
    error: String,
}

pub async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

pub fn load_model(
    artifact_dir: &str,
    device: &Device,
) -> Result<Model, Box<dyn std::error::Error>> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))?;
    let record = ModuleRecord::load(format!("{artifact_dir}/model"))?;

    Ok(config.model.init(device).load_record(record).valid())
}

pub async fn predict(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Vec<Prediction>>, (StatusCode, Json<ApiError>)> {
    let mut predictions = Vec::new();

    loop {
        let field = multipart.next_field().await.map_err(|error| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!("Could not read upload: {error}"),
            )
        })?;
        let Some(field) = field else {
            break;
        };
        if field.name() != Some("images") {
            continue;
        }

        if predictions.len() == MAX_FILES_PER_UPLOAD {
            return Err(api_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Upload no more than {MAX_FILES_PER_UPLOAD} images at once."),
            ));
        }

        let filename = field
            .file_name()
            .map(str::to_owned)
            .unwrap_or_else(|| "uploaded image".to_owned());
        let bytes = field.bytes().await.map_err(|error| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!("Could not read image: {error}"),
            )
        })?;
        if bytes.is_empty() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!("{filename} is empty."),
            ));
        }
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(api_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("{filename} is larger than 10 MiB."),
            ));
        }

        let model = Arc::clone(&state.model);
        let device = state.device.clone();
        let digit = tokio::task::spawn_blocking(move || predict_digit(&model, &device, &bytes))
            .await
            .map_err(|error| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Prediction task failed: {error}"),
                )
            })?
            .map_err(|error| api_error(StatusCode::BAD_REQUEST, error))?;

        predictions.push(Prediction { filename, digit });
    }

    if predictions.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Choose at least one image to upload.".to_owned(),
        ));
    }

    Ok(Json(predictions))
}

fn api_error(status: StatusCode, message: String) -> (StatusCode, Json<ApiError>) {
    (status, Json(ApiError { error: message }))
}

fn predict_digit(model: &Model, device: &Device, bytes: &[u8]) -> Result<u8, String> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| format!("Could not identify image format: {error}"))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| format!("Could not decode image: {error}"))?;
    let pixels = preprocess_image(image)?;
    let tensor =
        Tensor::<2>::from_data(TensorData::new(pixels, [28, 28]), device).reshape([1, 28, 28]);
    let output = model.forward(tensor);
    Ok(output.argmax(1).flatten::<1>(0, 1).into_scalar())
}

fn preprocess_image(image: DynamicImage) -> Result<Vec<f32>, String> {
    let mut grayscale = image.to_luma8();
    let (width, height) = grayscale.dimensions();

    let border_pixels = (0..width)
        .map(|x| grayscale.get_pixel(x, 0)[0] as u64 + grayscale.get_pixel(x, height - 1)[0] as u64)
        .chain((0..height).map(|y| {
            grayscale.get_pixel(0, y)[0] as u64 + grayscale.get_pixel(width - 1, y)[0] as u64
        }))
        .sum::<u64>();
    let border_count = 2 * (width as u64 + height as u64);
    if border_pixels / border_count > 127 {
        for pixel in grayscale.pixels_mut() {
            pixel[0] = 255 - pixel[0];
        }
    }

    let mut bounds = (width, height, 0, 0);
    for (x, y, pixel) in grayscale.enumerate_pixels() {
        if pixel[0] > 24 {
            bounds.0 = bounds.0.min(x);
            bounds.1 = bounds.1.min(y);
            bounds.2 = bounds.2.max(x);
            bounds.3 = bounds.3.max(y);
        }
    }
    if bounds.0 > bounds.2 || bounds.1 > bounds.3 {
        return Err("The image appears blank; upload an image with a visible digit.".to_owned());
    }

    let crop = image::imageops::crop_imm(
        &grayscale,
        bounds.0,
        bounds.1,
        bounds.2 - bounds.0 + 1,
        bounds.3 - bounds.1 + 1,
    )
    .to_image();
    let resized = DynamicImage::ImageLuma8(crop)
        .resize(20, 20, FilterType::Lanczos3)
        .to_luma8();
    let offset_x = (28 - resized.width()) / 2;
    let offset_y = (28 - resized.height()) / 2;

    let mut pixels = vec![0.0; 28 * 28];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let normalized = (pixel[0] as f32 / 255.0 - 0.1307) / 0.3081;
        pixels[((offset_y + y) * 28 + offset_x + x) as usize] = normalized;
    }
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::preprocess_image;
    use image::{DynamicImage, GrayImage, Luma};

    #[test]
    fn preprocesses_light_and_dark_backgrounds_consistently() {
        let mut light_background = GrayImage::from_pixel(28, 28, Luma([255]));
        for y in 5..23 {
            for x in 12..16 {
                light_background.put_pixel(x, y, Luma([0]));
            }
        }
        let dark_background = GrayImage::from_fn(28, 28, |x, y| {
            Luma([255 - light_background.get_pixel(x, y)[0]])
        });

        let light = preprocess_image(DynamicImage::ImageLuma8(light_background)).unwrap();
        let dark = preprocess_image(DynamicImage::ImageLuma8(dark_background)).unwrap();

        assert_eq!(light.len(), 28 * 28);
        assert_eq!(light, dark);
        assert!(light.iter().any(|pixel| *pixel > 0.0));
    }

    #[test]
    fn rejects_blank_images() {
        let blank = GrayImage::from_pixel(28, 28, Luma([255]));
        let error = preprocess_image(DynamicImage::ImageLuma8(blank)).unwrap_err();

        assert!(error.contains("appears blank"));
    }
}
