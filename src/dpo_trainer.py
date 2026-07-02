"""
dpo_trainer.py — Online-DPO update step.

Рекомендуется как ПЕРВЫЙ работающий замкнутый цикл (см. README §4.3 и
Roadmap §9, этап 4) — в отличие от GRPO не требует доверять стохастической
трактовке rectified-flow ODE для clipping; работает напрямую с разницей
score-matching loss между "выигравшим" (max reward) и "проигравшим" (min
reward) сэмплом внутри каждой группы.

Идея — прямое расширение Diffusion-DPO (Wallace et al. 2023,
arXiv:2311.12945) на ОНЛАЙН-случай: пары (winner, loser) берутся не из
статического датасета человеческих предпочтений, а из rollout текущей
политики на каждой итерации, ранжированные Q-Judger.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from policy import DiffusionPolicy
from rollout import PromptRollout


@dataclass
class DPOStepResult:
    loss: float
    mean_reward: float
    mean_margin: float    # средняя разница reward(winner) - reward(loser)


def online_dpo_step(
    policy: DiffusionPolicy,
    rollouts: list[PromptRollout],
    optimizer: torch.optim.Optimizer,
    beta: float = 0.04,
    grad_clip_norm: float = 1.0,
) -> DPOStepResult:
    """Один шаг обновления LoRA-весов через online-DPO.

    Для каждого промпта в батче: выбираем сэмпл с максимальным и минимальным
    reward внутри группы (n сэмплов), считаем DPO-loss по разнице
    score-matching ошибки winner/loser относительно референсной (frozen,
    pre-LoRA) политики.
    """
    optimizer.zero_grad()
    total_loss = torch.zeros(1, device=policy.device)
    all_rewards = []
    all_margins = []

    for ro in rollouts:
        valid = ro.parse_ok_mask
        if valid.sum() < 2:
            continue  # недостаточно валидных сэмплов для пары winner/loser

        rewards = ro.rewards.clone()
        rewards[~valid] = float("-inf")
        winner_idx = int(torch.argmax(rewards).item())
        rewards_for_loser = ro.rewards.clone()
        rewards_for_loser[~valid] = float("inf")
        loser_idx = int(torch.argmin(rewards_for_loser).item())

        if winner_idx == loser_idx:
            continue

        margin = (ro.rewards[winner_idx] - ro.rewards[loser_idx]).item()
        all_margins.append(margin)
        all_rewards.append(ro.rewards.mean().item())

        # log-prob траектории winner/loser под ТЕКУЩЕЙ политикой (с градиентом)
        n = ro.sample_result.log_probs.shape[0]
        logp_current_winner = policy.recompute_log_prob(
            ro.prompt, ro.sample_result.trajectory, n
        )[winner_idx]
        logp_current_loser = policy.recompute_log_prob(
            ro.prompt, ro.sample_result.trajectory, n
        )[loser_idx]

        # log-prob под референсной (frozen) политикой — без градиента.
        # На практике: временно отключить LoRA-адаптер (peft disable_adapter)
        # на время этого forward-прохода.
        with torch.no_grad(), policy.pipe.transformer.disable_adapter():
            logp_ref_winner = policy.recompute_log_prob(
                ro.prompt, ro.sample_result.trajectory, n
            )[winner_idx]
            logp_ref_loser = policy.recompute_log_prob(
                ro.prompt, ro.sample_result.trajectory, n
            )[loser_idx]

        # DPO loss: -log sigma(beta * [(logp_cur_w - logp_ref_w) - (logp_cur_l - logp_ref_l)])
        logits = beta * (
            (logp_current_winner - logp_ref_winner) - (logp_current_loser - logp_ref_loser)
        )
        loss = -F.logsigmoid(logits)
        total_loss = total_loss + loss

    if len(all_margins) == 0:
        return DPOStepResult(loss=0.0, mean_reward=0.0, mean_margin=0.0)

    total_loss = total_loss / len(all_margins)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), grad_clip_norm)
    optimizer.step()

    return DPOStepResult(
        loss=float(total_loss.item()),
        mean_reward=sum(all_rewards) / len(all_rewards),
        mean_margin=sum(all_margins) / len(all_margins),
    )
