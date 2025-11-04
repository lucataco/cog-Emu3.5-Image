# Emu3.5-Image Cog Predictor

This directory packages the **BAAI/Emu3.5-Image** model as a Replicate Cog predictor. It runs the official Emu 3.5 inference stack and supports every task template exposed by the paper (`t2i`, `x2i`, `howto`, `story`, `explore`, `vla`) while returning the generated images directly as PNG/JPEG files. Model weights are downloaded from Hugging Face on first run, so ensure you have adequate disk space and bandwidth [[1]](https://huggingface.co/BAAI/Emu3.5-Image).

## Repository Layout

- `cog.yaml` – Cog build configuration (CUDA 12.4, PyTorch 2.4, FlashAttention 2.8.3).
- `requirements.txt` – Python dependencies mirroring the upstream Emu3.5 repo.
- `predict.py` – Cog `Predictor` implementation that
  1. clones `baaivision/Emu3.5` for the inference helpers,
  2. snapshots the Hugging Face weights/tokenizer into `checkpoints/`, and
  3. orchestrates sampling + multimodal decoding to recover full-resolution images.

## Getting Started

Install Cog (once per machine):

```bash
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
chmod +x /usr/local/bin/cog
```

Then launch the predictor locally:

```bash
cd /home/shadeform/dev/emu
cog predict \
  -i task_type="t2i" \
  -i prompt="A cinematic photo of bioluminescent mushrooms glowing in a misty forest"
```

The first invocation can take several minutes while ~34B parameter weights download into `checkpoints/`. Subsequent runs reuse the cached snapshots.

## Inputs

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `task_type` | string enum | `t2i` | One of `t2i`, `x2i`, `howto`, `story`, `explore`, `vla`. Determines the prompt template. |
| `prompt` | string | required | User prompt (or script) describing the desired output. |
| `reference_image` | file | `null` | Optional image path for image-conditioned flows (required when `task_type=x2i`). |
| `guidance_scale` | float | `5.0` | Classifier-free guidance strength. |
| `temperature` | float | `1.0` | Global sampling temperature. |
| `top_p` / `top_k` | float / int | `1.0` / `131072` | Generic nucleus/top-k filters. |
| `text_top_p`, `text_top_k`, `text_temperature` | floats / int | `0.9`, `1024`, `1.0` | Per-text-token sampling controls. |
| `image_top_p`, `image_top_k`, `image_temperature` | floats / int | `1.0`, `10240`, `1.0` | Per-image-token sampling controls. |
| `max_new_tokens` | int | `4096` | Autoregressive length cap (<= 32768). |
| `seed` | int | `42` | RNG seed for reproducibility. |
| `output_format` | string enum | `png` | Output image format (`png` or `jpeg`). |

## Outputs

- One or more image files saved under a temporary directory (returned as Cog paths).
- Any auxiliary text decoded from the multimodal output is echoed to stdout for inspection.

## Deployment Notes

- Hardware: tuned for a single NVIDIA A100 (80 GB) with bfloat16 weights and FlashAttention disabled if unavailable.
- Dependencies: `flash_attn` is installed inside the container; if build fails, switch `attn_implementation` to `eager` inside `predict.py`.
- Weights live in `checkpoints/` so Cog caches persist between runs; delete the folder to force a fresh download.

## Licensing

- Model weights: Apache-2.0 (per Hugging Face model card) [[1]](https://huggingface.co/BAAI/Emu3.5-Image).
- Code in this wrapper: inherits this repository’s license.

