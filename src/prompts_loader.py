"""
prompts_loader.py — загрузка промптов и dims_en из датасета
Qwen/Qwen-Image-Bench (1000 промптов, 5 направлений x 23 категории x
56 фасетов разметки, см. README §6.1).

Использование вместо статичного prompts_pool.txt даёт сразу два
преимущества (см. README §6):
  1. Разнообразный, капабилити-размеченный пул промптов вместо 15 примеров
  2. dims_en на каждый промпт -> Q-Judger вызывается только по РЕЛЕВАНТНЫМ
     для этого промпта направлениям (быстрее и точнее, чем гонять все 5
     направлений на каждом промпте без разбора)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchPrompt:
    id: int
    prompt_en: str
    dims_en: str


def load_qwen_image_bench_prompts(
    language: str = "en",
    cache_dir: str | None = None,
) -> list[BenchPrompt]:
    """Загружает все 1000 промптов из датасета Qwen/Qwen-Image-Bench через
    HF `datasets` (требует интернет на момент первого вызова — см.
    docs/gpu_recommendation.md, на vast.ai-инстансе это не проблема).

    Параметр language оставлен для совместимости (prompt_cn доступен в
    датасете), но рекомендуется "en" — на нём проверены чеклисты/промпты
    Q-Judger.
    """
    from datasets import load_dataset

    ds = load_dataset("Qwen/Qwen-Image-Bench", split="test", cache_dir=cache_dir)
    prompt_col = "prompt_en" if language == "en" else "prompt_cn"

    prompts = []
    for row in ds:
        prompts.append(BenchPrompt(
            id=row["ID"],
            prompt_en=row[prompt_col],
            dims_en=row["dims_en"],
        ))
    return prompts


def load_dims_only_fallback(metadata_path: str) -> dict[int, str]:
    """Fallback: только dims_en (без текста промпта) из локально
    забэндленного configs/qwen_image_bench_dims_metadata.json — на случай,
    если нет сетевого доступа к HF Hub. Текст промптов отсюда получить
    НЕЛЬЗЯ (см. README §6.1) — нужен либо load_qwen_image_bench_prompts(),
    либо собственный prompts_pool.txt вместе с dims_en=None (тогда Q-Judger
    оценивает по всем 5 направлениям, см. judger.py: _default_dims_by_level1).
    """
    path = Path(metadata_path)
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["ID"]: r["dims_en"] for r in records}


def stratified_sample(
    prompts: list[BenchPrompt], k: int, by: str = "level1", seed: int | None = None
) -> list[BenchPrompt]:
    """Сэмплирует k промптов, стараясь равномерно покрыть L1-направления
    (Quality/Aesthetics/Alignment/Real-world Fidelity/Creative Generation),
    а не брать случайный срез, в котором какое-то направление может быть
    недопредставлено (важно для §7 ТЗ: рост reward не должен идти за счёт
    одной оси в ущерб другим — для этого нужен баланс уже на этапе rollout).
    """
    rng = random.Random(seed)
    if by != "level1":
        return rng.sample(prompts, min(k, len(prompts)))

    buckets: dict[str, list[BenchPrompt]] = {}
    for p in prompts:
        l1_dims = {part.split("/")[0].strip() for part in p.dims_en.split(";") if part.strip()}
        for d in l1_dims:
            buckets.setdefault(d, []).append(p)

    bucket_names = list(buckets.keys())
    rng.shuffle(bucket_names)
    chosen: list[BenchPrompt] = []
    seen_ids = set()
    i = 0
    while len(chosen) < k and bucket_names:
        bucket = buckets[bucket_names[i % len(bucket_names)]]
        candidate = rng.choice(bucket)
        if candidate.id not in seen_ids:
            chosen.append(candidate)
            seen_ids.add(candidate.id)
        i += 1
        if i > k * 20:  # защита от бесконечного цикла на маленьких k
            break
    return chosen
