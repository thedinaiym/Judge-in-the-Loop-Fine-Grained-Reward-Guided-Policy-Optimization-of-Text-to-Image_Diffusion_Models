"""
trainer.py
----------
Online-DPO и GRPO update steps.
Online-DPO: стабильнее, рекомендуется как стартовая точка.
GRPO: critic-free PPO, требует stochastic ODE (см. policy.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def group_relative_advantage(
    rewards: torch.Tensor,
    eps: float = 1e-6,
    clip: float = 5.0,
) -> torch.Tensor:
    """A_i = (R_i - mean) / (std + eps), с обрезкой выбросов."""
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)
    return adv.clamp(-clip, clip)


def online_dpo_step(
    policy,
    rollouts: list,
    optimizer: torch.optim.Optimizer,
    beta: float = 0.1,
    grad_clip: float = 1.0,
) -> dict:
    """
    Online-DPO: внутри каждой группы из n сэмплов на один промпт
    берём winner (max reward) и loser (min reward) как пару предпочтений
    и считаем DPO-loss относительно замороженной референсной политики.
    """
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=policy.device)
    margins = []
    rewards_all = []
    valid_count = 0

    for ro in rollouts:
        r = ro["rewards"]
        mask = ro["parse_ok"]
        if mask.sum() < 2:
            continue

        # выбираем winner / loser только среди валидных
        r_valid = r.clone()
        r_valid[~mask] = float("-inf")
        wi = int(r_valid.argmax())
        r_valid2 = r.clone()
        r_valid2[~mask] = float("inf")
        li = int(r_valid2.argmin())
        if wi == li:
            continue

        margin = (r[wi] - r[li]).item()
        margins.append(margin)
        rewards_all.append(r[mask].mean().item())

        n = ro["log_probs_old"].shape[0]

        # log-prob под ТЕКУЩИМИ весами LoRA
        lp_new = policy.recompute_logp(ro["prompt"], ro["trajectory"], n)
        lp_w_new = lp_new[wi]
        lp_l_new = lp_new[li]

        # log-prob под РЕФЕРЕНСНОЙ (frozen, без LoRA) политикой
        with torch.no_grad():
            with policy.pipe.transformer.disable_adapter():
                lp_ref = policy.recompute_logp(ro["prompt"], ro["trajectory"], n)
        lp_w_ref = lp_ref[wi]
        lp_l_ref = lp_ref[li]

        logit = beta * (
            (lp_w_new - lp_w_ref) - (lp_l_new - lp_l_ref)
        )
        loss = -F.logsigmoid(logit)
        total_loss = total_loss + loss
        valid_count += 1

    if valid_count == 0:
        return {"loss": 0.0, "mean_reward": 0.0, "mean_margin": 0.0}

    (total_loss / valid_count).backward()
    _gn = sum((p.grad.norm()**2).item() for p in policy.trainable_params() if p.grad is not None) ** 0.5
    print(f"[DEBUG] grad_norm = {_gn:.6f}")
    torch.nn.utils.clip_grad_norm_(policy.trainable_params(), grad_clip)
    optimizer.step()

    return {
        "loss": (total_loss / valid_count).item(),
        "mean_reward": sum(rewards_all) / len(rewards_all),
        "mean_margin": sum(margins) / len(margins),
    }


def grpo_step(
    policy,
    rollouts: list,
    optimizer: torch.optim.Optimizer,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    grad_clip: float = 1.0,
) -> dict:
    """
    GRPO: group-relative advantage + PPO-style clipping + KL-штраф.
    Требует, чтобы policy.algorithm == "grpo" (stochastic ODE включён).
    """
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=policy.device)
    total_kl = torch.tensor(0.0, device=policy.device)
    rewards_all = []
    valid_count = 0

    for ro in rollouts:
        r = ro["rewards"]
        mask = ro["parse_ok"]
        if mask.sum() < 2:
            continue
        valid_count += 1

        adv = group_relative_advantage(r)
        adv = torch.where(mask, adv, torch.zeros_like(adv))
        n = ro["log_probs_old"].shape[0]

        lp_new = policy.recompute_logp(ro["prompt"], ro["trajectory"], n)
        lp_old = ro["log_probs_old"].detach()

        ratio = torch.exp(lp_new - lp_old)
        pg_loss = -torch.min(
            ratio * adv,
            ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv,
        ).mean()

        with torch.no_grad():
            with policy.pipe.transformer.disable_adapter():
                lp_ref = policy.recompute_logp(ro["prompt"], ro["trajectory"], n)
        kl = (lp_new - lp_ref).mean()

        total_loss = total_loss + pg_loss + kl_beta * kl
        total_kl = total_kl + kl.detach()
        rewards_all.append(r[mask].mean().item())

    if valid_count == 0:
        return {"loss": 0.0, "mean_reward": 0.0, "mean_kl": 0.0}

    (total_loss / valid_count).backward()
    _gn = sum((p.grad.norm()**2).item() for p in policy.trainable_params() if p.grad is not None) ** 0.5
    print(f"[DEBUG] grad_norm = {_gn:.6f}")
    torch.nn.utils.clip_grad_norm_(policy.trainable_params(), grad_clip)
    optimizer.step()

    return {
        "loss": (total_loss / valid_count).item(),
        "mean_reward": sum(rewards_all) / len(rewards_all),
        "mean_kl": (total_kl / valid_count).item(),
    }