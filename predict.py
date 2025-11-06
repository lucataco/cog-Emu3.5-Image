import os
import sys
import copy
import time
import torch
import random
import tempfile
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
from cog import BasePredictor, Input, Path as CogPath
from typing import Dict, Iterable, List, Optional, Tuple
from huggingface_hub import snapshot_download

MODEL_PATH = "checkpoints"
CODE_REPO_URL = "https://github.com/baaivision/Emu3.5"
MODEL_REPO_ID = "BAAI/Emu3.5-Image"
VQ_REPO_ID = "BAAI/Emu3.5-VisionTokenizer"
MODEL_URL = "https://weights.replicate.delivery/default/BAAI/Emu3.5-Image/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def build_unc_and_template(task: str, with_image: bool) -> Tuple[str, str]:
    task_str = task.lower()
    if with_image:
        unc_prompt = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        template = (
            "<|extra_203|>You are a helpful assistant for %s task. USER: <|IMAGE|>{question} ASSISTANT: <|extra_100|>"
            % task_str
        )
    else:
        unc_prompt = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        template = (
            "<|extra_203|>You are a helpful assistant for %s task. USER: {question} ASSISTANT: <|extra_100|>"
            % task_str
        )
    return unc_prompt, template


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Download weights
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner for A100/H100

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.root = Path(__file__).resolve().parent
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.code_dir = self.checkpoint_dir / "Emu3.5"
        self.model_dir = self.checkpoint_dir / "Emu3.5-Image"
        self.vq_dir = self.checkpoint_dir / "Emu3.5-VisionTokenizer"

        self._ensure_repo()

        if str(self.code_dir) not in sys.path:
            sys.path.insert(0, str(self.code_dir))

        from src.utils.model_utils import build_emu3p5  # type: ignore
        from src.utils.generation_utils import generate, multimodal_decode  # type: ignore
        from src.utils.input_utils import build_image  # type: ignore

        self._build_emu3p5 = build_emu3p5
        self._generate = generate
        self._multimodal_decode = multimodal_decode
        self._build_image = build_image

        self._ensure_weights()

        tokenizer_path = self.code_dir / "src" / "tokenizer_emu3_ibq"

        model_device = 0 if self.device.type == "cuda" else "cpu"
        vq_device = f"cuda:{model_device}" if self.device.type == "cuda" else "cpu"

        self.model, self.tokenizer, self.vq_model = self._build_emu3p5(
            model_path=str(self.model_dir),
            tokenizer_path=str(tokenizer_path),
            vq_path=str(self.vq_dir),
            vq_type="ibq",
            model_device=model_device,
            vq_device=vq_device,
        )

        self.model.eval()

        # Enable inference optimizations for A100/H100
        # Note: torch.compile on the main model causes excessive recompilations with Flash Attention
        # and dynamic generation, making inference slower. The model already uses Flash Attention 2.
        torch.set_grad_enabled(False)  # Ensure gradients are disabled

        # Optimize memory allocation for faster inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Use memory efficient attention patterns
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Slower fallback

        # Compile VQ decoder for faster image decoding (static graph, works well)
        print("Compiling VQ decoder for faster image generation...")
        if hasattr(self.vq_model, 'decoder'):
            try:
                self.vq_model.decoder = torch.compile(
                    self.vq_model.decoder,
                    mode="max-autotune",  # Decoder has static graph, use aggressive optimization
                    fullgraph=True,
                )
                print("VQ decoder compiled successfully")
            except Exception as e:
                print(f"Warning: Could not compile VQ decoder: {e}")
                print("Continuing with uncompiled decoder...")

        self.special_tokens = {
            "BOS": "<|extra_203|>",
            "EOS": "<|extra_204|>",
            "PAD": "<|endoftext|>",
            "EOL": "<|extra_200|>",
            "EOF": "<|extra_201|>",
            "TMS": "<|extra_202|>",
            "IMG": "<|image token|>",
            "BOI": "<|image start|>",
            "EOI": "<|image end|>",
            "BSS": "<|extra_100|>",
            "ESS": "<|extra_101|>",
            "BOG": "<|extra_60|>",
            "EOG": "<|extra_61|>",
            "BOC": "<|extra_50|>",
            "EOC": "<|extra_51|>",
        }

        self.special_token_ids = {
            key: self.tokenizer.encode(value, add_special_tokens=False)[0]
            for key, value in self.special_tokens.items()
        }

        self.base_sampling_params = {
            "use_cache": True,
            "text_top_k": 1024,
            "text_top_p": 0.9,
            "text_temperature": 1.0,
            "image_top_k": 10240,
            "image_top_p": 1.0,
            "image_temperature": 1.0,
            "top_k": 131072,
            "top_p": 1.0,
            "temperature": 1.0,
            "num_beams_per_group": 1,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
            "max_new_tokens": 32768,
            "guidance_scale": 1.0,
            "use_differential_sampling": True,
        }

        self._base_cfg = {
            "sampling_params": self.base_sampling_params,
            "special_tokens": self.special_tokens,
            "special_token_ids": self.special_token_ids,
            "classifier_free_guidance": 5.0,
            "unconditional_type": "no_text",
            "streaming": False,
            "image_area": 518400,
        }

    def predict(
        self,
        task_type: str = Input(
            choices=["t2i", "x2i", "howto", "story", "explore", "vla"],
            default="t2i",
            description="Task template to apply for generation.",
        ),
        prompt: str = Input(description="User prompt to condition the model."),
        reference_image: Optional[List[CogPath]] = Input(
            default=None,
            description="Optional reference image(s) for image-conditioned tasks (required for x2i). Provide as a list of up to 3 images.",
        ),
        max_new_tokens: int = Input(
            default=4096,
            ge=512,
            le=32768,
            description="Maximum number of tokens to autoregressively generate.",
        ),
        seed: int = Input(
            default=42,
            description="Random seed for reproducibility.",
        ),
        output_format: str = Input(
            choices=["png", "jpeg"],
            default="png",
            description="Preferred output image format.",
        ),
        ratio: str = Input(
            choices=["1:1", "16:9", "9:16", "match_input"],
            default="match_input",
            description="Aspect ratio for generated image. Use 'match_input' to match the first reference image's aspect ratio.",
        ),
    ) -> List[CogPath]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")

        task_type = task_type.lower()
        if task_type == "x2i" and (reference_image is None or len(reference_image) == 0):
            raise ValueError("An input image is required for x2i tasks.")

        if reference_image is not None and len(reference_image) > 3:
            raise ValueError("Maximum of 3 reference images are supported.")

        use_image = reference_image is not None and len(reference_image) > 0

        # Handle "match_input" ratio option
        if ratio == "match_input":
            if use_image:
                # Use the first reference image's aspect ratio
                ratio = self._calculate_aspect_ratio_from_image(reference_image[0])
                print(f"[INFO] Using aspect ratio {ratio} from first reference image")
            else:
                # No reference images, fall back to square
                ratio = "1:1"
                print("[WARNING] 'match_input' selected but no reference images provided. Using 1:1 ratio.")

        cfg = self._prepare_cfg(
            task_type=task_type,
            use_image=use_image,
            max_new_tokens=max_new_tokens,
            ratio=ratio,
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Use inference_mode for faster execution (disables autograd tracking)
        with torch.inference_mode():
            unc_prompt = cfg.unc_prompt
            prompt_text = cfg.template.format(question=prompt)

            if use_image:
                image_str = ""
                for img_path in reference_image:
                    pil_image = Image.open(Path(img_path)).convert("RGB")
                    image_tokens = self._build_image(pil_image, cfg, self.tokenizer, self.vq_model)
                    image_str += image_tokens

                prompt_text = prompt_text.replace("<|IMAGE|>", image_str)
                unc_prompt = unc_prompt.replace("<|IMAGE|>", image_str)

            # Calculate and insert output image dimensions
            # This tells the model what size image to generate
            h, w = self.calculate_output_dimensions(cfg.image_area, cfg.ratio)
            output_dim_spec = f"{self.special_tokens['BOI']}{h}*{w}{self.special_tokens['IMG']}"

            # Append dimension specification to both prompts
            prompt_text = prompt_text + output_dim_spec
            unc_prompt = unc_prompt + output_dim_spec

            input_ids = self.tokenizer.encode(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.model.device)

            if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
                bos = torch.tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device)
                input_ids = torch.cat([bos, input_ids], dim=1)

            unconditional_ids = self.tokenizer.encode(
                unc_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.model.device)

            generation_iter = self._generate(
                cfg=cfg,
                model=self.model,
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                unconditional_ids=unconditional_ids,
                full_unconditional_ids=None,
            )

            try:
                generated_tokens = next(generation_iter)
            except StopIteration as exc:  # pragma: no cover - defensive
                raise RuntimeError("Generation did not yield any tokens.") from exc

            for _ in generation_iter:  # exhaust remaining yields if any
                pass

            if isinstance(generated_tokens, torch.Tensor):
                tokens_sequence = generated_tokens.detach().cpu().tolist()
            elif isinstance(generated_tokens, np.ndarray):
                tokens_sequence = generated_tokens.tolist()
            else:
                tokens_sequence = list(generated_tokens)

            # Debug logging: Token sequence analysis
            print(f"[DEBUG] Generated {len(tokens_sequence)} tokens")
            print(f"[DEBUG] First 50 token IDs: {tokens_sequence[:50]}")
            print(f"[DEBUG] Contains BOI token ({self.special_token_ids['BOI']}): {self.special_token_ids['BOI'] in tokens_sequence}")
            print(f"[DEBUG] Contains IMG token ({self.special_token_ids['IMG']}): {self.special_token_ids['IMG'] in tokens_sequence}")
            print(f"[DEBUG] Contains EOI token ({self.special_token_ids['EOI']}): {self.special_token_ids['EOI'] in tokens_sequence}")

            decoded = self.tokenizer.decode(tokens_sequence, skip_special_tokens=False)

            # Debug logging: Decoded output analysis
            print(f"[DEBUG] Decoded output length: {len(decoded)} characters")
            print(f"[DEBUG] Decoded output (first 500 chars): {decoded[:500]}")
            print(f"[DEBUG] Contains '<|image start|>': {'<|image start|>' in decoded}")
            print(f"[DEBUG] Contains '<|image end|>': {'<|image end|>' in decoded}")

            multimodal_output = self._multimodal_decode(decoded, self.tokenizer, self.vq_model)

        images, texts = self._split_outputs(multimodal_output)

        # Debug logging: Extraction results
        print(f"[DEBUG] Extracted {len(images)} images and {len(texts)} text segments")
        print(f"[DEBUG] Multimodal output type: {type(multimodal_output)}")
        if hasattr(multimodal_output, '__len__'):
            print(f"[DEBUG] Multimodal output length: {len(multimodal_output)}")

        if not images:
            print("\n[ERROR] No images were produced by the model!")
            if texts:
                print("\nModel generated text output without images:")
                for idx, text in enumerate(texts, 1):
                    print(f"[{idx}] {text}")
            else:
                print("\nModel generated neither images nor text output.")
            print("\nDiagnostic summary:")
            print(f"  - Generated {len(tokens_sequence)} tokens")
            print(f"  - BOI (image start) token present: {self.special_token_ids['BOI'] in tokens_sequence}")
            print(f"  - Model likely interpreted task as text explanation rather than image generation")
            print(f"  - Consider: simpler prompt, explicit image generation cues, or higher guidance_scale")
            raise RuntimeError("No images were produced by the model.")

        result_dir = Path(tempfile.mkdtemp())
        outputs: List[CogPath] = []
        for idx, image in enumerate(images):
            path = result_dir / f"output_{idx:02d}.{output_format}"
            save_params: Dict[str, object] = {}
            if output_format == "jpeg":
                image = image.convert("RGB")
                save_params["quality"] = 95
            image.save(path, **save_params)
            outputs.append(CogPath(path))

        if texts:
            print("Generated textual outputs:")
            for idx, text in enumerate(texts, 1):
                print(f"[{idx}] {text}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return outputs

    def _prepare_cfg(
        self,
        *,
        task_type: str,
        use_image: bool,
        max_new_tokens: int,
        ratio: str,
    ) -> SimpleNamespace:
        base_cfg = copy.deepcopy(self._base_cfg)
        sampling_params = base_cfg["sampling_params"]

        # Task-specific guidance scale defaults (from official GitHub configs)
        guidance_scale_defaults = {
            "t2i": 5.0,
            "x2i": 2.0,
            "howto": 1.0,
            "story": 3.0,
            "explore": 3.0,
            "vla": 3.0,
        }
        guidance_scale = guidance_scale_defaults.get(task_type, 3.0)

        # Task-specific sampling parameter overrides (from official GitHub configs)
        if task_type in ["t2i", "x2i"]:
            sampling_params["image_top_k"] = 5120
        elif task_type == "howto":
            sampling_params["text_top_k"] = 200
            sampling_params["text_top_p"] = 0.8
            sampling_params["text_temperature"] = 0.7

        # Task-specific image area (from official GitHub configs)
        image_area_defaults = {
            "t2i": 1048576,      # 1024x1024
            "x2i": 1048576,      # 1024x1024
            "howto": 1048576,    # 1024x1024
            "story": 518400,     # 720x720
            "explore": 518400,   # 720x720
            "vla": 518400,       # 720x720
        }
        image_area = image_area_defaults.get(task_type, 518400)

        # Task-specific max_new_tokens defaults (from official GitHub configs)
        # All image generation tasks need enough tokens for 1024x1024 output (64x64 tokens = ~4166 total)
        max_new_tokens_defaults = {
            "t2i": 5120,         # Text-to-image
            "x2i": 5120,         # Any-to-image
            "howto": 5120,       # Multi-step instructions
            "story": 5120,       # Sequential narrative
            "explore": 5120,     # Spatiotemporal exploration
            "vla": 5120,         # Vision-language-action
        }

        # Use task-specific default if user didn't explicitly override
        # (i.e., if still using the predict() function's default of 4096)
        if max_new_tokens == 4096:
            max_new_tokens = max_new_tokens_defaults.get(task_type, 5120)

        # Apply max_new_tokens
        sampling_params["max_new_tokens"] = max_new_tokens

        sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]
        sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1

        base_cfg["sampling_params"] = sampling_params
        base_cfg["classifier_free_guidance"] = guidance_scale
        base_cfg["image_area"] = image_area
        base_cfg["task_type"] = task_type
        base_cfg["use_image"] = use_image
        base_cfg["ratio"] = ratio

        unc_prompt, template = build_unc_and_template(task_type, use_image)
        base_cfg["unc_prompt"] = unc_prompt
        base_cfg["template"] = template

        return SimpleNamespace(**base_cfg)

    def calculate_output_dimensions(
        self, image_area: int, ratio: str = "1:1", spatial_factor: int = 16
    ) -> tuple[int, int]:
        """Calculate output image dimensions in tokens from image_area and aspect ratio.

        Args:
            image_area: Target pixel area (e.g., 1048576 for 1024x1024)
            ratio: Aspect ratio as "H:W" (e.g., "1:1", "16:9", "4:3")
            spatial_factor: VQ model downsampling factor (default: 16)

        Returns:
            (height, width) in tokens
        """
        height, width = map(int, ratio.split(":"))
        current_area = height * width
        target_ratio = (image_area / current_area) ** 0.5
        token_height = int(round(height * target_ratio / spatial_factor))
        token_width = int(round(width * target_ratio / spatial_factor))
        return token_height, token_width

    def _calculate_aspect_ratio_from_image(self, image_path: Path) -> str:
        """Calculate aspect ratio from an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Aspect ratio as "H:W" string (e.g., "16:9", "4:3", "1:1")
        """
        from math import gcd

        pil_image = Image.open(image_path).convert("RGB")
        width, height = pil_image.size

        # Calculate GCD to simplify the ratio
        divisor = gcd(width, height)
        ratio_w = width // divisor
        ratio_h = height // divisor

        # If the ratio is too complex (e.g., 1920:1080), try to match common ratios
        common_ratios = {
            (16, 9): "16:9",
            (9, 16): "9:16",
            (4, 3): "4:3",
            (3, 4): "3:4",
            (1, 1): "1:1",
            (21, 9): "21:9",
            (3, 2): "3:2",
            (2, 3): "2:3",
        }

        # Check if we match a common ratio (with small tolerance)
        aspect = width / height
        for (w, h), ratio_str in common_ratios.items():
            if abs(aspect - (w / h)) < 0.01:  # 1% tolerance
                return ratio_str

        # Return simplified ratio
        return f"{ratio_h}:{ratio_w}"

    def _ensure_repo(self) -> None:
        if self.code_dir.exists():
            return
        subprocess.run(
            ["git", "clone", "--depth", "1", CODE_REPO_URL, str(self.code_dir)],
            check=True,
        )

    def _ensure_weights(self) -> None:
        if not self.model_dir.exists():
            snapshot_download(
                repo_id=MODEL_REPO_ID,
                local_dir=str(self.model_dir),
                local_dir_use_symlinks=False,
            )

        if not self.vq_dir.exists():
            snapshot_download(
                repo_id=VQ_REPO_ID,
                local_dir=str(self.vq_dir),
                local_dir_use_symlinks=False,
            )

    @staticmethod
    def _split_outputs(outputs: Iterable[Tuple[str, object]]) -> Tuple[List[Image.Image], List[str]]:
        images: List[Image.Image] = []
        texts: List[str] = []
        for item_type, payload in outputs:
            if item_type == "image" and isinstance(payload, Image.Image):
                images.append(payload)
            elif isinstance(payload, str):
                texts.append(payload.strip())
        return images, [t for t in texts if t]

