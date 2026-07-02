"""
rollout.py — для батча промптов генерирует n сэмплов на каждый промпт и
получает reward от Q-Judger. Результат группируется по промпту, готовый для
group_relative_advantage (см. reward.py).

Принимает либо BenchPrompt (см. prompts_loader.py, с dims_en из официального
датасета Qwen-Image-Bench — тогда Q-Judger оценивает только релевантные
направления), либо обычную строку (тогда dims_en=None и Q-Judger оценивает
по всем 5 направлениям, см. judger.py: _default_dims_by_level1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from judger import QJudger
from policy import DiffusionPolicy, SampleResult
from prompts_loader import BenchPrompt

PromptLike = Union[str, BenchPrompt]


@dataclass
class PromptRollout:
    prompt: str
    sample_result: SampleResult
    rewards: torch.Tensor          # shape (n,) — scalar reward на каждый сэмпл
    parse_ok_mask: torch.Tensor     # shape (n,) — False там, где judger не распарсился
    judge_results: list             # список JudgeResult, для axis_breakdown_stats


def _unpack(p: PromptLike) -> tuple[str, str | None]:
    if isinstance(p, BenchPrompt):
        return p.prompt_en, p.dims_en
    return p, None


def run_rollout(
    policy: DiffusionPolicy,
    judger: QJudger,
    prompts: list[PromptLike],
    samples_per_prompt: int,
) -> list[PromptRollout]:
    rollouts = []
    for p in prompts:
        prompt_text, dims_en = _unpack(p)
        sample_result = policy.sample(prompt_text, n=samples_per_prompt)
        judge_results = judger.score_batch(sample_result.images, prompt_text, dims_en=dims_en)

        rewards = torch.tensor(
            [r.scalar_reward for r in judge_results],
            device=sample_result.log_probs.device,
            dtype=sample_result.log_probs.dtype,
        )
        parse_ok_mask = torch.tensor([r.parse_ok for r in judge_results], device=rewards.device)

        rollouts.append(
            PromptRollout(
                prompt=prompt_text,
                sample_result=sample_result,
                rewards=rewards,
                parse_ok_mask=parse_ok_mask,
                judge_results=judge_results,
            )
        )
    return rollouts
