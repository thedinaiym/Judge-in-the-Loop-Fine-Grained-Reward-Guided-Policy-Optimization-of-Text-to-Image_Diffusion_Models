"""
train.py — главная точка входа.

Запуск:
    python train.py                     # использует config.yaml
    python train.py --config my.yaml    # свой конфиг
    python train.py --smoke             # 3 итерации без сохранения, для проверки

Что делает:
  1. Загружает FLUX.1 + LoRA на GPU0
  2. Загружает Q-Judger (INT4/8bit/BF16) на GPU1
  3. Тянет промпты из Qwen-Image-Bench (HF datasets)
  4. Закрытый цикл: generate → score → update LoRA
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import yaml

from judger import QJudger
from policy import DiffusionPolicy
from trainer import grpo_step, online_dpo_step


# ─────────────────────────────────────────────────────────────────────────────
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(cfg: dict) -> list[dict]:
    """Загружает 1000 промптов из Qwen-Image-Bench с dims_en разметкой."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            cfg["prompts"]["dataset"],
            split=cfg["prompts"]["split"],
        )
        lang = cfg["prompts"].get("language", "en")
        col = f"prompt_{lang}"
        prompts = [
            {"prompt": row[col], "dims_en": row.get("dims_en", ""), "id": row["ID"]}
            for row in ds
        ]
        print(f"[data] Загружено {len(prompts)} промптов из {cfg['prompts']['dataset']}")
        return prompts
    except Exception as e:
        print(f"[data] Не удалось загрузить датасет: {e}")
        print("[data] Используем fallback-промпты")
        return [
            {"prompt": p, "dims_en": "", "id": i}
            for i, p in enumerate([
                "a robot walking in Tokyo at night, neon lights",
                "an astronaut planting a flag on a red alien desert",
                "a cozy cabin in a snowy forest, warm light inside",
                "a futuristic city skyline with flying cars, cyberpunk",
                "a chef plating a dessert, close-up, shallow depth of field",
                "a knight in ornate armor in an ancient ruined temple",
                "a watercolor painting of a lighthouse during a storm",
                "a poster with the text OPEN 24 HOURS above a diner",
            ])
        ]


def stratified_sample(prompts: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Балансированная выборка по L1-направлениям из dims_en."""
    buckets: dict[str, list[dict]] = {}
    for p in prompts:
        dims = set()
        for part in p.get("dims_en", "").split(";"):
            part = part.strip()
            if part:
                dims.add(part.split("/")[0].strip())
        if not dims:
            dims = {"General"}
        for d in dims:
            buckets.setdefault(d, []).append(p)

    chosen, seen = [], set()
    keys = list(buckets.keys())
    rng.shuffle(keys)
    i = 0
    while len(chosen) < k and i < k * 30:
        bucket = buckets[keys[i % len(keys)]]
        candidate = rng.choice(bucket)
        if candidate["id"] not in seen:
            chosen.append(candidate)
            seen.add(candidate["id"])
        i += 1
    return chosen


def axis_stats(rollouts: list) -> dict:
    per_dim: dict[str, list] = {}
    for ro in rollouts:
        for jr in ro["judge_results"]:
            for dim, val in jr.per_dim.items():
                if val is not None:
                    per_dim.setdefault(dim, []).append(val)
    return {f"axis/{d}": sum(v) / len(v) for d, v in per_dim.items() if v}


def log_step(step: int, metrics: dict, logfile: str):
    metrics = {"step": step, "time": time.strftime("%H:%M:%S"), **metrics}
    print(f"[step {step:04d}]  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float)
                                               else f"{k}={v}" for k, v in metrics.items()
                                               if k not in ("time",)))
    with open(logfile, "a") as f:
        f.write(json.dumps(metrics) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="3 итерации без сохранения — проверка пайплайна")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    tc = cfg["training"]
    oc = cfg["output"]

    if args.smoke:
        tc["total_steps"] = 3
        oc["save_every"] = 999
        print("[smoke] Режим отладки: 3 итерации")

    set_seed(tc["seed"])
    out_dir = Path(oc["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = str(out_dir / "metrics.jsonl")

    # ── Загрузка моделей ────────────────────────────────────────────────────
    policy = DiffusionPolicy(cfg)
    judger = QJudger(
        model_name=cfg["judger"]["model"],
        device=cfg["judger"]["device"],
        quantization=cfg["judger"].get("quantization", "4bit"),
        max_new_tokens=cfg["judger"].get("max_new_tokens", 1024),
        enable_thinking=cfg["judger"].get("enable_thinking", True),
    )

    # ── Данные ──────────────────────────────────────────────────────────────
    prompts = load_prompts(cfg)
    rng = random.Random(tc["seed"])

    # ── Оптимизатор ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy.trainable_params(),
        lr=tc["lr"],
        weight_decay=tc.get("weight_decay", 0.01),
    )

    algorithm = tc["algorithm"]
    n = cfg["rollout"]["samples_per_prompt"]
    k = cfg["rollout"]["prompts_per_step"]

    print(f"\n[train] Алгоритм: {algorithm}  |  Шагов: {tc['total_steps']}")
    print(f"[train] {k} промптов × {n} сэмплов = {k*n} изображений/шаг\n")

    # ── Основной цикл ────────────────────────────────────────────────────────
    for step in range(tc["total_steps"]):
        t0 = time.time()
        batch = stratified_sample(prompts, k, rng)
        rollouts = []

        for item in batch:
            prompt = item["prompt"]
            dims_en = item.get("dims_en") or None

            # Rollout: генерируем n изображений
            sample = policy.sample(prompt, n=n)

            # Scoring: Q-Judger оценивает каждое изображение
            judge_results = judger.score_batch(
                sample.images, prompt, dims_en=dims_en
            )

            rewards = torch.tensor(
                [jr.scalar_reward for jr in judge_results],
                device=policy.device,
                dtype=torch.float32,
            )
            parse_ok = torch.tensor(
                [jr.parse_ok for jr in judge_results],
                device=policy.device,
            )

            rollouts.append({
                "prompt": prompt,
                "trajectory": sample.trajectory,
                "log_probs_old": sample.log_probs,
                "rewards": rewards,
                "parse_ok": parse_ok,
                "judge_results": judge_results,
            })

        # ── Policy update ────────────────────────────────────────────────────
        if algorithm == "online_dpo":
            metrics = online_dpo_step(
                policy, rollouts, optimizer,
                beta=tc.get("dpo_beta", 0.1),
                grad_clip=tc.get("grad_clip", 1.0),
            )
        elif algorithm == "grpo":
            metrics = grpo_step(
                policy, rollouts, optimizer,
                clip_eps=tc.get("grpo_clip_eps", 0.2),
                kl_beta=tc.get("kl_beta", 0.04),
                grad_clip=tc.get("grad_clip", 1.0),
            )
        else:
            raise ValueError(f"Неизвестный алгоритм: {algorithm}")

        # ── Логирование ──────────────────────────────────────────────────────
        dim_metrics = axis_stats(rollouts)
        all_rewards = [
            jr.scalar_reward
            for ro in rollouts
            for jr in ro["judge_results"]
            if jr.parse_ok
        ]
        parse_rate = sum(
            jr.parse_ok for ro in rollouts for jr in ro["judge_results"]
        ) / (k * n)

        metrics["mean_reward"] = sum(all_rewards) / max(len(all_rewards), 1)
        metrics["parse_rate"] = parse_rate
        metrics["step_sec"] = round(time.time() - t0, 1)
        metrics.update(dim_metrics)

        log_step(step, metrics, logfile)

        # ── Чекпоинт ────────────────────────────────────────────────────────
        if (step + 1) % oc["save_every"] == 0:
            ckpt = str(out_dir / f"lora_step_{step+1:04d}")
            policy.save_lora(ckpt)
            print(f"[ckpt] Сохранено: {ckpt}")

    # Финальный чекпоинт
    final = str(out_dir / "lora_final")
    policy.save_lora(final)
    print(f"\n[done] Финальный чекпоинт: {final}")
    print(f"[done] Метрики: {logfile}")


if __name__ == "__main__":
    main()