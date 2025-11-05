# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository packages the **BAAI/Emu3.5-Image** model as a Replicate Cog predictor. Emu3.5 is a native multimodal model that performs unified next-token prediction over interleaved vision-language sequences, supporting various image generation tasks through different prompt templates.

## Key Commands

### Building and Running with Cog

```bash
# Build the Cog container (first time or after dependency changes)
cog build

# Run a simple text-to-image prediction
cog predict \
  -i task_type="t2i" \
  -i prompt="A cinematic photo of bioluminescent mushrooms glowing in a misty forest"

# Run with custom parameters
cog predict \
  -i task_type="t2i" \
  -i prompt="Your prompt here" \
  -i guidance_scale=5.0 \
  -i temperature=1.0 \
  -i max_new_tokens=4096 \
  -i seed=42

# Run image-to-image (x2i) task
cog predict \
  -i task_type="x2i" \
  -i prompt="Transform this into a watercolor painting" \
  -i reference_image=@path/to/image.jpg
```

### Development and Testing

```bash
# Install dependencies locally (for development without Cog)
pip install -r requirements.txt
pip install flash_attn==2.8.3 --no-build-isolation

# The first run will download ~34GB of model weights from Hugging Face
# Weights are cached in checkpoints/ directory
```

## Architecture

### High-Level Structure

The predictor orchestrates three main components:

1. **Emu3.5 Main Model** (`BAAI/Emu3.5-Image`): The ~34B parameter multimodal autoregressive transformer that generates interleaved token sequences
2. **Vision Tokenizer** (`BAAI/Emu3.5-VisionTokenizer`): Vector-quantized (VQ) model that converts images to/from discrete tokens
3. **Inference Helpers** (`baaivision/Emu3.5` GitHub repo): Utilities for generation, multimodal decoding, and prompt templating

### Code Flow in predict.py

1. **Setup phase** (predict.py:47-137):
   - Downloads model weights from Hugging Face or Replicate CDN into `checkpoints/`
   - Clones the official Emu3.5 inference code from GitHub
   - Builds the main model, tokenizer, and VQ model
   - Initializes special tokens (BOS, EOS, BOI, EOI, etc.) and sampling parameters

2. **Prediction phase** (predict.py:149-347):
   - Constructs task-specific prompt template based on `task_type` (t2i, x2i, howto, story, explore, vla)
   - For image-conditioned tasks, encodes reference image into discrete tokens via VQ model
   - Performs autoregressive generation with classifier-free guidance
   - Decodes generated token sequence back into images using multimodal decoder
   - Returns images as PNG/JPEG files

3. **Key helper functions**:
   - `build_unc_and_template()`: Creates unconditional prompt and task template (predict.py:29-43)
   - `_prepare_cfg()`: Merges user parameters with base sampling config (predict.py:349-374)
   - `_split_outputs()`: Separates generated images from text (predict.py:400-408)

### Upstream Dependencies

The predictor dynamically imports from the cloned `checkpoints/Emu3.5/` repo:
- `src.utils.model_utils.build_emu3p5`: Loads model, tokenizer, and VQ model (predict.py:70)
- `src.utils.generation_utils.generate`: Autoregressive token generation (predict.py:71)
- `src.utils.generation_utils.multimodal_decode`: Converts tokens to images/text (predict.py:71)
- `src.utils.input_utils.build_image`: Encodes input images to tokens (predict.py:72)

## Task Types

The model supports 6 task templates (predict.py:29-43):
- **t2i**: Text-to-image generation
- **x2i**: Any-to-image (image editing/transformation, requires `reference_image`)
- **howto**: Step-by-step visual instructions
- **story**: Sequential visual narrative
- **explore**: Spatiotemporal world exploration
- **vla**: Vision-language-action for embodied tasks

Each task type modifies the prompt template structure while using the same underlying generation mechanism.

## Important Configuration

### Sampling Parameters (predict.py:120-137)

The model uses modality-specific sampling controls:
- **Text tokens**: `text_top_k=1024`, `text_top_p=0.9`, `text_temperature=1.0`
- **Image tokens**: `image_top_k=10240`, `image_top_p=1.0`, `image_temperature=1.0`
- **Guidance**: `classifier_free_guidance=5.0` (configurable via `guidance_scale` input)
- **Length**: `max_new_tokens=32768` max, default 4096 for faster generation

### Hardware Requirements

- Tuned for NVIDIA A100 (80GB)
- Uses bfloat16 precision
- FlashAttention 2.8.3 installed in container (fallback to eager attention if unavailable)
- Single GPU for t2i, 2+ GPUs recommended for x2i and narrative tasks

## Weight Management

Model weights are stored in `checkpoints/` and cached between runs:
- `checkpoints/Emu3.5/`: Cloned inference code from GitHub
- `checkpoints/Emu3.5-Image/`: Main model weights from Hugging Face
- `checkpoints/Emu3.5-VisionTokenizer/`: VQ model weights from Hugging Face

To force a fresh download, delete the relevant subdirectory in `checkpoints/`.

The predictor supports two weight download methods:
1. **Replicate CDN** (faster, pre-packaged): Uses `pget` to download `model.tar` from `weights.replicate.delivery` (predict.py:20-27)
2. **Hugging Face** (fallback): Uses `snapshot_download()` from `huggingface-hub` (predict.py:384-397)

## Performance Optimizations

The predictor includes several optimizations for faster inference on A100/H100 GPUs:

### Enabled Optimizations (predict.py:53-110)

1. **TF32 Matrix Multiplication**: `torch.backends.cuda.matmul.allow_tf32 = True` - Enables Tensor Float 32 for faster matmul on Ampere+ GPUs
2. **High Precision Matmul**: `torch.set_float32_matmul_precision("high")` - Optimizes float32 operations
3. **CUDNN Autotuner**: `torch.backends.cudnn.benchmark = True` - Automatically selects optimal convolution algorithms
4. **Inference Mode**: Uses `torch.inference_mode()` context for the entire prediction pipeline - disables autograd tracking for 5-10% speedup
5. **Optimized Attention Backend**: Enables Flash Attention and memory-efficient SDPA, disables slower math fallback
6. **VQ Decoder Compilation**: `torch.compile(decoder, mode="max-autotune")` - Compiles the static vision decoder graph for 30-50% faster image decoding

### Why Main Model is NOT Compiled

torch.compile on the main transformer model causes **excessive recompilations** due to:
- Dynamic sequence lengths during autoregressive generation
- Flash Attention operations causing graph breaks
- CFG (Classifier-Free Guidance) requiring multiple forward passes with different inputs
- Cache limit exhaustion (hitting 256 accumulated cache size limit)

These issues make compiled inference **slower** than the already-optimized Flash Attention 2 implementation. The model uses Flash Attention 2.8.3 which is highly optimized for A100/H100.

### Expected Performance

With these optimizations, typical improvements over baseline:
- VQ decoder: 30-50% faster image decoding
- Overall inference: 10-20% speedup from inference_mode + torch settings
- Memory efficiency: Reduced allocations and better GPU utilization

## Special Tokens

The model uses a rich vocabulary of special tokens (predict.py:97-113):
- `<|extra_203|>`: Beginning of sequence (BOS)
- `<|extra_204|>`: End of sequence (EOS)
- `<|image start|>`, `<|image end|>`: Image region delimiters
- `<|extra_100|>`, `<|extra_101|>`: Begin/end system segment
- `<|IMAGE|>`: Placeholder replaced with actual image tokens during prompt construction

Prompt templates insert these tokens to guide the model's multimodal generation behavior.
