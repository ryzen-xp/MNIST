# MNIST Digit Classifier

This Rust application trains a Burn model on MNIST and serves a small image-upload website.
The first `cargo run` trains and saves the model if `target/guide` does not already contain it.
Later runs reuse the saved model.

```sh
cargo run
```

Open <http://127.0.0.1:3000>, select one or more PNG or JPEG images, and click **Predict digits**.
The model works best with a single handwritten digit on a plain background. Images are converted
to centered 28x28 grayscale inputs before prediction. Each image can be up to 10 MiB, with a
20 MiB request limit.

The backend and model are Rust. The page loads Tailwind CSS from its CDN, so internet access is
needed for Tailwind styling; no Node.js packages or build step are required.
