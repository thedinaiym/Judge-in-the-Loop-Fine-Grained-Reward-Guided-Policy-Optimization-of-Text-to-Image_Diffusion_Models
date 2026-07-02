"""
reward.py — нормализация reward в group-relative advantage (как в GRPO,
DeepSeekMath, Shao et al. 2024) и вспомогательные функции для агрегации
6-мерного reward от Q-Judger в скаляр (сам взвешенный scalar уже считается
в judger.py; здесь — операции уровня группы/батча).
"""
from __future__ import annotations

import torch


def group_relative_advantage(
    rewards: torch.Tensor, epsilon: float = 1e-6, clip: float | None = 5.0
) -> torch.Tensor:
    """A_i = (R_i - mean(R)) / (std(R) + eps) для одной группы (один промпт,
    n сэмплов). Критик не нужен — это и делает GRPO critic-free.

    rewards: shape (n,) — reward для n сэмплов одного и того же промпта.
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantage = (rewards - mean) / (std + epsilon)
    if clip is not None:
        advantage = advantage.clamp(-clip, clip)
    return advantage


def batch_group_relative_advantage(
    rewards_per_prompt: list[torch.Tensor], epsilon: float = 1e-6, clip: float | None = 5.0
) -> list[torch.Tensor]:
    """То же самое, но для батча из нескольких промптов одновременно —
    нормализация делается ВНУТРИ каждой группы (промпта), не по всему батчу,
    иначе advantage начнёт отражать относительную "трудность" промпта, а не
    относительное качество сэмплов внутри него.
    """
    return [group_relative_advantage(r, epsilon, clip) for r in rewards_per_prompt]


def axis_breakdown_stats(judge_results: list) -> dict[str, float]:
    """Средние значения по каждому из 5 L1-направлений Q-Judger
    (Quality/Aesthetics/Alignment/Real-world Fidelity/Creative Generation)
    в батче — для логирования и контроля метрики из README §7 ("рост не
    должен идти за счёт одного направления в ущерб другим").

    Направление включается в средние только по тем сэмплам, где оно вообще
    было оценено (раз не у каждого промпта релевантны все 5 направлений —
    см. dims_en и prompts_loader.py), иначе результат смещён в сторону 50.0
    "нейтральных" значений там, где направление просто не запрашивалось.
    """
    per_dim_values: dict[str, list[float]] = {}
    for r in judge_results:
        for dim, value in r.raw_scores.items():
            if value is not None:
                per_dim_values.setdefault(dim, []).append(value)

    return {
        f"axis/{dim}": sum(values) / len(values)
        for dim, values in per_dim_values.items()
        if values
    }
