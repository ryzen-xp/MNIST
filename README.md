# MNIST Digit Classifier

This Rust application serves an image-upload website and uses a Burn model to classify handwritten
digits. It includes a trained model in `guide/`; if the model files are missing, the application
trains a replacement on startup.

```sh
cargo run
```

Open <http://127.0.0.1:3000>, select one or more PNG or JPEG images, and click **Predict digits**.
The model works best with a single handwritten digit on a plain background. Images are converted
to centered 28x28 grayscale inputs before prediction. Each image can be up to 10 MiB, with a
20 MiB request limit.

## Deploy on Render

This repository includes a Render Blueprint (`render.yaml`) and a Dockerfile. In Render, create a
new Blueprint Instance and select this repository. Render builds the Rust app into a Docker image
and starts the web service; open the service's `onrender.com` URL to load the HTML page. The server
listens on Render's `PORT` environment variable and binds to all network interfaces.

The model runs on the CPU, so a GPU is not required. The included trained weights are copied into
the image so the service does not need to download MNIST or train the model during startup. The page
loads Tailwind CSS from its CDN, so internet access is needed for Tailwind styling; no Node.js
packages or build step are required.
