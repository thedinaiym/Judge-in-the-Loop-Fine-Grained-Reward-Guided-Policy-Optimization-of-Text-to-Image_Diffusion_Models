"""
utils.py — общие утилиты: конфиг, сидирование, логирование, чекпоинты.
"""
from __future__ import annotations

import os
import random
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Prompt pool file not found: {path}. "
            f"Создайте файл с одним промптом на строку перед запуском train.py."
        )
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class MetricsLogger:
    """Простой логгер с поддержкой csv / wandb / tensorboard (по конфигу).

    Сделан легковесным и без хардкода зависимостей при backend="csv" —
    чтобы pipeline можно было гонять даже без wandb/tensorboard установленных.
    """

    def __init__(self, cfg: dict, run_name: str | None = None):
        self.backend = cfg.get("backend", "csv")
        self.output_dir = Path(cfg.get("output_dir", "./outputs")) / "logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self._csv_path = self.output_dir / f"{self.run_name}.csv"
        self._csv_header_written = False
        self._wandb = None

        if self.backend == "wandb":
            try:
                import wandb

                self._wandb = wandb
                self._wandb.init(project=cfg.get("project", "reward-guided-diffusion-rl"),
                                  name=self.run_name)
            except Exception as e:  # noqa: BLE001
                print(f"[MetricsLogger] wandb недоступен ({e}), fallback на csv.")
                self.backend = "csv"

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        metrics = {"step": step, **metrics}
        if self.backend == "wandb" and self._wandb is not None:
            self._wandb.log(metrics, step=step)
        else:
            self._append_csv(metrics)
        print(f"[step {step}] " + " | ".join(f"{k}={v}" for k, v in metrics.items() if k != "step"))

    def _append_csv(self, metrics: dict[str, Any]) -> None:
        keys = list(metrics.keys())
        write_header = not self._csv_header_written and not self._csv_path.exists()
        with open(self._csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(keys) + "\n")
                self._csv_header_written = True
            f.write(",".join(str(metrics[k]) for k in keys) + "\n")


def save_lora_checkpoint(policy, output_dir: str, step: int) -> str:
    """Сохраняет только LoRA-веса (не базовую модель)."""
    ckpt_dir = Path(output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_lora(str(ckpt_dir))
    meta = {"step": step, "timestamp": time.time()}
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return str(ckpt_dir)
