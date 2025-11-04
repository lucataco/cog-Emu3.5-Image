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
            "<|extra_203|>You are a helpful assistant for %s task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>"
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
        reference_image: Optional[CogPath] = Input(
            default=None,
            description="Optional reference image used for image-conditioned tasks (required for x2i).",
        ),
        guidance_scale: float = Input(
            default=5.0,
            ge=0.0,
            le=10.0,
            description="Classifier-free guidance scale.",
        ),
        temperature: float = Input(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Sampling temperature applied to all tokens.",
        ),
        top_p: float = Input(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Nucleus sampling top-p for generic tokens.",
        ),
        top_k: int = Input(
            default=131072,
            ge=0,
            description="Top-k filter for generic tokens.",
        ),
        text_top_k: int = Input(
            default=1024,
            ge=0,
            description="Top-k sampling applied to text tokens.",
        ),
        text_top_p: float = Input(
            default=0.9,
            ge=0.0,
            le=1.0,
            description="Top-p sampling applied to text tokens.",
        ),
        text_temperature: float = Input(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Temperature for text token sampling.",
        ),
        image_top_k: int = Input(
            default=10240,
            ge=0,
            description="Top-k sampling applied to image tokens.",
        ),
        image_top_p: float = Input(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Top-p sampling applied to image tokens.",
        ),
        image_temperature: float = Input(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Temperature for image token sampling.",
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
    ) -> List[CogPath]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")

        task_type = task_type.lower()
        if task_type == "x2i" and reference_image is None:
            raise ValueError("An input image is required for x2i tasks.")

        use_image = reference_image is not None

        cfg = self._prepare_cfg(
            task_type=task_type,
            use_image=use_image,
            guidance_scale=guidance_scale,
            sampling_overrides=dict(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                text_top_k=text_top_k,
                text_top_p=text_top_p,
                text_temperature=text_temperature,
                image_top_k=image_top_k,
                image_top_p=image_top_p,
                image_temperature=image_temperature,
                max_new_tokens=max_new_tokens,
            ),
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        unc_prompt = cfg.unc_prompt
        prompt_text = cfg.template.format(question=prompt)

        if use_image:
            image_path = Path(reference_image)
            pil_image = Image.open(image_path).convert("RGB")
            image_tokens = self._build_image(pil_image, cfg, self.tokenizer, self.vq_model)
            prompt_text = prompt_text.replace("<|IMAGE|>", image_tokens)
            unc_prompt = unc_prompt.replace("<|IMAGE|>", image_tokens)

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

        decoded = self.tokenizer.decode(tokens_sequence, skip_special_tokens=False)
        multimodal_output = self._multimodal_decode(decoded, self.tokenizer, self.vq_model)

        images, texts = self._split_outputs(multimodal_output)

        if not images:
            if texts:
                print("Model generated text output without images:")
                for idx, text in enumerate(texts, 1):
                    print(f"[{idx}] {text}")
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
        guidance_scale: float,
        sampling_overrides: Dict[str, object],
    ) -> SimpleNamespace:
        base_cfg = copy.deepcopy(self._base_cfg)
        sampling_params = base_cfg["sampling_params"]
        for key, value in sampling_overrides.items():
            sampling_params[key] = value

        sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]
        sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1

        base_cfg["sampling_params"] = sampling_params
        base_cfg["classifier_free_guidance"] = guidance_scale
        base_cfg["task_type"] = task_type
        base_cfg["use_image"] = use_image

        unc_prompt, template = build_unc_and_template(task_type, use_image)
        base_cfg["unc_prompt"] = unc_prompt
        base_cfg["template"] = template

        return SimpleNamespace(**base_cfg)

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

