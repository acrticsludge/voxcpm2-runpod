# VoxCPM2 Runpod Serverless

This project packages the official VoxCPM2 inference path into a Runpod Serverless endpoint. It uses the published Python package `voxcpm` and the official `VoxCPM.from_pretrained(...)` loading flow rather than Hugging Face Spaces or a custom demo stack.

## Architecture

- A Runpod Serverless worker starts a lightweight Python HTTP handler.
- The handler validates the request and optionally decodes a base64 WAV speaker clip.
- The model is loaded once per worker and reused across requests.
- Inference happens through the official `voxcpm` package and returns a base64-encoded WAV file.

## Folder layout

- `Dockerfile` – GPU runtime image for Runpod Serverless.
- `requirements.txt` – Python dependencies.
- `handler.py` – Runpod serverless entrypoint.
- `model.py` – lazy model loading and inference orchestration.
- `config.py` – environment-based configuration.
- `utils.py` – audio helpers.
- `download_model.py` – explicit model cache preload script.
- `README.md` – deployment instructions.

## How the model works

VoxCPM2 is a tokenizer-free text-to-speech model. The official package exposes a `VoxCPM` class that loads the checkpoint from Hugging Face using the standard `from_pretrained` API.

The serverless service supports:

- plain text-to-speech
- optional speaker audio cloning via base64 WAV input
- optional control instructions
- configurable guidance scale

## Requirements

- Python 3.10+
- CUDA 12.0+
- PyTorch 2.5+
- NVIDIA GPU with at least 8 GB VRAM for reliable inference

## Docker build

```bash
docker build -t voxcpm2-runpod:latest .
```

## Docker push

```bash
docker tag voxcpm2-runpod:latest <your-registry>/voxcpm2-runpod:latest
docker push <your-registry>/voxcpm2-runpod:latest
```

## Local Docker testing

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e MODEL_ID=openbmb/VoxCPM2 \
  -e CACHE_DIR=/cache/voxcpm \
  -e HF_TOKEN=<optional-token> \
  voxcpm2-runpod:latest
```

Then send a request:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"input":{"text":"Hello world","cfgValue":3.0}}'
```

## Runpod deployment

1. Create a new Runpod Serverless template.
2. Use the container image you built and pushed.
3. Set the GPU type to one of the supported options: RTX A4000, RTX 3090, RTX A5000, or L4.
4. Set the minimum workers to 0 to allow scale to zero.
5. Configure the environment variables below.

## Environment variables

- `HF_TOKEN` – optional Hugging Face access token.
- `MODEL_ID` – model repository id, defaults to `openbmb/VoxCPM2`.
- `LOG_LEVEL` – logging level, defaults to `INFO`.
- `CACHE_DIR` – directory used for model caching, defaults to `/cache/voxcpm`.
- `PRELOAD_MODEL_ON_STARTUP` – set to `true` to load the model during worker boot so the first request is faster; defaults to `true`.
- `DEBUG` – enable verbose behavior for local debugging when set to `true`.

## Choosing GPUs

- RTX A4000 / RTX 3090 / RTX A5000 / L4 are all suitable for this runtime.
- The model is large and benefits from 24 GB VRAM when available.
- The A4000 and L4 are generally adequate for smaller inference workloads, while the 3090/A5000 are more comfortable for steady throughput.

## Cold starts

The first worker start downloads the model into the cache directory and loads it into GPU memory. This is the longest step. Later starts reuse the cache automatically. Set `PRELOAD_MODEL_ON_STARTUP=true` to make the worker warm up during boot rather than on the first request.

## Health checks

The worker exposes a lightweight health response through the same handler entrypoint. For a basic readiness probe, send a payload like:

The container entrypoint is [runpod_entrypoint.py](runpod_entrypoint.py), which calls the Runpod startup hook explicitly.

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"input":{"health":true}}'
```

A healthy worker returns a JSON object with the worker name, model load state, device, and GPU details.

## Logs

The handler logs:

- startup
- model load events
- inference duration
- GPU detection
- failures

## Troubleshooting

### Model download fails

- Verify network access to Hugging Face.
- If the model is gated, set `HF_TOKEN`.
- Check the mounted cache directory.

### CUDA unavailable

- Ensure the container is started with NVIDIA runtime support.
- Confirm the GPU type on Runpod is compatible.

### OOM

- Lower the request rate or use a larger GPU.
- Use a larger VRAM GPU if you expect concurrency.

## Updating images

Use a new tag whenever you change the runtime or dependencies.

For a GitHub-to-Runpod flow, build and publish an image tag that matches the release version, then update the Runpod template to use the new image tag. Keep the old tag around until the new deployment has been verified.

```bash
docker build -t voxcpm2-runpod:<version> .
```

## Versioning

Tag images with the release version or date, for example:

```bash
docker tag voxcpm2-runpod:latest voxcpm2-runpod:2026-07-07
```

## Cost expectations

Costs depend on the GPU type and per-request runtime. Expect cold starts to be expensive but steady-state requests to be cheaper once the model stays warm.

## Performance benchmarks

Performance varies by GPU. In the official project, VoxCPM2 is designed for GPU inference with optimized runtime. Expect slower cold starts on smaller GPUs and faster steady-state generation on RTX 3090/A5000-class hardware.

## Security considerations

- Do not bake secrets into the image.
- Pass tokens through environment variables.
- Keep the cache directory on a persistent volume if you need reuse across image upgrades.
- Prefer short-lived credentials and rotate Hugging Face tokens if you grant access to gated models.

## FAQ

### Can I use this without Hugging Face Spaces?

Yes. This project runs independently of Spaces and uses the official package loader.

### Does it support voice cloning?

Yes. If you pass `speakerAudio` as base64-encoded WAV data, the request will use it as the reference clip.

### Is the model downloaded at startup?

Yes, on first startup it will be cached and reused on future starts.
