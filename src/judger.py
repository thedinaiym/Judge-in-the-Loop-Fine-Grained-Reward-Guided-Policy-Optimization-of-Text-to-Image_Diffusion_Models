"""
judger.py
---------
Реальная модель Qwen/Qwen-Image-Bench в роли reward-оракула.
Поддерживает INT4 / INT8 / BF16 — выбор через config.yaml (quantization).

Для 40GB карт: quantization: "4bit"  (~14GB VRAM)
Для 80GB карт: quantization: "none"  (~54GB VRAM, полное BF16)
"""
from __future__ import annotations

import json
import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import dataclass, field
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# Официальные промпт-шаблоны и утилиты агрегации скоров
# (вендорировано из github.com/QwenLM/Qwen-Image-Bench, Apache-2.0)
# ---------------------------------------------------------------------------
try:
    from qjudger_official import (
        DIM_TO_CHECKLIST,
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE,
        parse_dims_by_level1,
        extract_json_from_response,
        fix_score_json,
        compute_dimension_score,
        aggregate_total_score,
    )
    OFFICIAL_PROMPTS = True
except ImportError:
    OFFICIAL_PROMPTS = False
    print("[judger] WARN: qjudger_official не найден — используется упрощённый промпт")

ALL_DIMS = ["Quality", "Aesthetics", "Alignment", "Real-world Fidelity", "Creative Generation"]
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

SIMPLE_SYSTEM = (
    "You are an expert image quality evaluator. "
    "Given a prompt and an image, score the image from 0 to 100. "
    "Respond with ONLY a JSON: {\"score\": <int>}"
)


@dataclass
class JudgeResult:
    scalar_reward: float          # итоговый скор 0-100
    per_dim: dict[str, float]     # по направлениям, для логов
    parse_ok: bool = True


@dataclass
class QJudger:
    model_name: str = "Qwen/Qwen-Image-Bench"
    device: str = "cuda:1"
    quantization: str = "4bit"    # "4bit" | "8bit" | "none"
    max_new_tokens: int = 1024
    enable_thinking: bool = True

    def __post_init__(self):
        bnb_cfg = None
        dtype = torch.bfloat16

        if self.quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            dtype = None  # dtype задаётся внутри BnB конфига
            print(f"[judger] Загружаем {self.model_name} в INT4 (~14GB VRAM)")
        elif self.quantization == "8bit":
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            dtype = None
            print(f"[judger] Загружаем {self.model_name} в INT8 (~27GB VRAM)")
        else:
            print(f"[judger] Загружаем {self.model_name} в BF16 (~54GB VRAM)")

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            quantization_config=bnb_cfg,
            device_map=self.device if bnb_cfg else None,
        )
        if bnb_cfg is None:
            self.model = self.model.to(self.device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        print(f"[judger] Загружена на {self.device}")

    @torch.no_grad()
    def score_batch(
        self,
        images: list[Image.Image],
        prompt: str,
        dims_en: str | None = None,
    ) -> list[JudgeResult]:
        results = []
        for img in images:
            r = self._score_one(img, prompt, dims_en)
            results.append(r)
        return results

    def _score_one(self, image: Image.Image, prompt: str, dims_en: str | None) -> JudgeResult:
        if OFFICIAL_PROMPTS:
            return self._score_official(image, prompt, dims_en)
        else:
            return self._score_simple(image, prompt)

    def _score_official(self, image: Image.Image, prompt: str, dims_en: str | None) -> JudgeResult:
        """Официальный протокол: отдельный вызов на каждое L1-направление."""
        if dims_en:
            dims = [d for d in parse_dims_by_level1(dims_en) if d in DIM_TO_CHECKLIST]
        else:
            dims = list(DIM_TO_CHECKLIST.keys())

        if not dims:
            dims = list(DIM_TO_CHECKLIST.keys())

        dim_results = {}
        raw_outputs = {}

        for dim in dims:
            checklist = DIM_TO_CHECKLIST[dim]
            user_text = USER_PROMPT_TEMPLATE.format(
                prompt=prompt,
                level1_dim=dim,
                format_checklist=checklist,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ]},
            ]
            text = self._apply_template(messages)
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.device)

            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
            generated = self.processor.batch_decode(
                out_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0]
            raw_outputs[dim] = generated

            score_json = extract_json_from_response(generated)
            if score_json is not None:
                score_json = fix_score_json(score_json, dim)
                dim_results[dim] = compute_dimension_score(score_json)

        if not dim_results:
            return JudgeResult(scalar_reward=50.0, per_dim={}, parse_ok=False)

        total = aggregate_total_score(dim_results)
        per_dim = {d: r["level1_score"] for d, r in dim_results.items()
                   if r.get("level1_score") is not None}

        return JudgeResult(
            scalar_reward=total if total is not None else 50.0,
            per_dim=per_dim,
            parse_ok=total is not None,
        )

    def _score_simple(self, image: Image.Image, prompt: str) -> JudgeResult:
        """Упрощённый fallback — один вызов, один скаляр."""
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f'Prompt: "{prompt}"\n\n{SIMPLE_SYSTEM}'},
            ]},
        ]
        text = self._apply_template(messages)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        generated = self.processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]
        match = _JSON_RE.search(generated)
        if match:
            try:
                d = json.loads(match.group(0))
                score = float(max(0, min(100, d.get("score", 50))))
                return JudgeResult(scalar_reward=score, per_dim={}, parse_ok=True)
            except Exception:
                pass
        return JudgeResult(scalar_reward=50.0, per_dim={}, parse_ok=False)

    def _apply_template(self, messages):
        try:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )