"""
grpo_trainer.py — GRPO update step (Group Relative Policy Optimization,
Shao et al. 2024, arXiv:2402.03300, адаптировано на диффузионные policy в
духе DDPO / T2I-R1).

Переходить на этот trainer рекомендуется ПОСЛЕ того, как online_dpo_step
(dpo_trainer.py) подтвердил, что весь pipeline (rollout → judger → reward →
update) работает корректно — см. README §4.3 и Roadmap §9.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from policy import DiffusionPolicy
from reward import group_relative_advantage
from rollout import PromptRollout


@dataclass
class GRPOStepResult:
    loss: float
    mean_reward: float
    mean_kl: float


def grpo_step(
    policy: DiffusionPolicy,
    rollouts: list[PromptRollout],
    optimizer: torch.optim.Optimizer,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    grad_clip_norm: float = 1.0,
) -> GRPOStepResult:
    """Один шаг GRPO-обновления по батчу из нескольких промптов.

    Для каждого промпта: advantage считается group-relative внутри его n
    сэмплов (без критика), затем берётся clipped surrogate objective как в
    PPO, плюс KL-штраф к референсной (frozen, без LoRA) политике.
    """
    optimizer.zero_grad()
    total_loss = torch.zeros(1, device=policy.device)
    total_kl = torch.zeros(1, device=policy.device)
    all_rewards = []
    n_groups = 0

    for ro in rollouts:
        valid = ro.parse_ok_mask
        if valid.sum() < 2:
            continue
        n_groups += 1
        n = ro.sample_result.log_probs.shape[0]

        advantage = group_relative_advantage(ro.rewards)
        advantage = torch.where(valid, advantage, torch.zeros_like(advantage))

        logp_old = ro.sample_result.log_probs.detach()
        logp_new = policy.recompute_log_prob(ro.prompt, ro.sample_result.trajectory, n)

        ratio = torch.exp(logp_new - logp_old)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
        policy_loss = -torch.min(unclipped, clipped).mean()

        with torch.no_grad(), policy.pipe.transformer.disable_adapter():
            logp_ref = policy.recompute_log_prob(ro.prompt, ro.sample_result.trajectory, n)
        # Приближение KL(pi_theta || pi_ref) через разницу log-prob —
        # стандартный low-variance estimator, используемый в GRPO/PPO-RLHF.
        kl_approx = (logp_new - logp_ref).mean()

        total_loss = total_loss + policy_loss + kl_beta * kl_approx
        total_kl = total_kl + kl_approx.detach()
        all_rewards.append(ro.rewards.mean().item())

    if n_groups == 0:
        return GRPOStepResult(loss=0.0, mean_reward=0.0, mean_kl=0.0)

    total_loss = total_loss / n_groups
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), grad_clip_norm)
    optimizer.step()

    return GRPOStepResult(
        loss=float(total_loss.item()),
        mean_reward=sum(all_rewards) / len(all_rewards),
        mean_kl=float((total_kl / n_groups).item()),
    )
