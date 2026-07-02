"""
policy.py
---------
FLUX.1-dev с LoRA-адаптером.
Поддерживает два режима сэмплинга:
  - deterministic ODE  (для Online-DPO — log-prob не нужен)
  - stochastic SDE     (для GRPO — нужен tractable log-prob)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model
from PIL import Image


@dataclass
class SampleResult:
    images: list[Image.Image]
    log_probs: torch.Tensor           # shape (n,) — None если ODE без шума
    trajectory: list[torch.Tensor]    # latent на каждом шаге
    initial_noise: torch.Tensor


class DiffusionPolicy:

    def __init__(self, cfg: dict):
        pc = cfg["policy"]
        lc = cfg["lora"]
        tc = cfg["training"]

        self.device = pc["device"]
        self.dtype = torch.bfloat16 if pc["dtype"] == "bf16" else torch.float32
        self.num_steps = pc["num_steps"]
        self.guidance_scale = pc["guidance_scale"]
        self.height = pc["height"]
        self.width = pc["width"]
        self.sde_noise = tc.get("sde_noise_scale", 0.02)
        self.algorithm = tc["algorithm"]

        print(f"[policy] Загружаем FLUX.1 на {self.device} ...")
        self.pipe = FluxPipeline.from_pretrained(
            pc["model"], torch_dtype=self.dtype
        )
        self.pipe.to(self.device)

        if tc.get("gradient_checkpointing", False):
            self.pipe.transformer.enable_gradient_checkpointing()

        # Замораживаем базовую модель
        self.pipe.transformer.requires_grad_(False)

        # Вешаем LoRA
        lora_cfg = LoraConfig(
            r=lc["rank"],
            lora_alpha=lc["alpha"],
            target_modules=lc["target_modules"],
            lora_dropout=lc.get("dropout", 0.0),
        )
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_cfg)
        self.pipe.transformer.print_trainable_parameters()
        print("[policy] FLUX.1 + LoRA готов")

    # ------------------------------------------------------------------ #
    def trainable_params(self):
        return [p for p in self.pipe.transformer.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    def sample(self, prompt: str, n: int = 8) -> SampleResult:
        use_sde = (self.algorithm == "grpo")

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.num_steps, device=self.device)
        timesteps = scheduler.timesteps

        latents = self._init_latents(n)
        initial_noise = latents.clone()
        prompt_embeds, pooled = self._encode(prompt, n)

        trajectory = [latents.clone()]
        log_prob_sum = torch.zeros(n, device=self.device) if use_sde else None

        for i, t in enumerate(timesteps):
            dt = self._dt(scheduler, i)
            with torch.set_grad_enabled(True):
                vel = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=t.expand(n),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled,
                    return_dict=False,
                )[0]

            mean_next = latents + vel * dt

            if use_sde:
                eps = torch.randn_like(latents)
                next_latents = mean_next + self.sde_noise * eps
                diff = (next_latents - mean_next).flatten(1)
                d = diff.shape[1]
                logp = -0.5 * (
                    diff.pow(2).sum(1) / self.sde_noise**2
                    + d * math.log(2 * math.pi * self.sde_noise**2)
                )
                log_prob_sum = log_prob_sum + logp
            else:
                next_latents = mean_next

            latents = next_latents.detach()
            trajectory.append(latents.clone())

        images = self._decode(latents)
        return SampleResult(
            images=images,
            log_probs=log_prob_sum,
            trajectory=trajectory,
            initial_noise=initial_noise,
        )

    # ------------------------------------------------------------------ #
    def recompute_logp(
        self, prompt: str, trajectory: list[torch.Tensor], n: int
    ) -> torch.Tensor:
        """Пересчёт log-prob той же траектории под текущими весами LoRA."""
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.num_steps, device=self.device)
        timesteps = scheduler.timesteps
        prompt_embeds, pooled = self._encode(prompt, n)

        logp_sum = torch.zeros(n, device=self.device)
        for i, t in enumerate(timesteps):
            dt = self._dt(scheduler, i)
            xt = trajectory[i]
            xt1 = trajectory[i + 1]
            vel = self.pipe.transformer(
                hidden_states=xt,
                timestep=t.expand(n),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]
            mean_next = xt + vel * dt
            diff = (xt1 - mean_next).flatten(1)
            d = diff.shape[1]
            logp = -0.5 * (
                diff.pow(2).sum(1) / self.sde_noise**2
                + d * math.log(2 * math.pi * self.sde_noise**2)
            )
            logp_sum = logp_sum + logp
        return logp_sum

    # ------------------------------------------------------------------ #
    def save_lora(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.pipe.transformer.save_pretrained(path)

    # ------------------------------------------------------------------ #
    def _init_latents(self, n: int) -> torch.Tensor:
        h, w = self.height // 8, self.width // 8
        in_ch = self.pipe.transformer.config.in_channels
        return torch.randn(n, in_ch, h, w, device=self.device, dtype=self.dtype)

    def _encode(self, prompt: str, n: int):
        embeds, pooled = self.pipe.encode_prompt(
            prompt=[prompt] * n,
            prompt_2=[prompt] * n,
            device=self.device,
        )[:2]
        return embeds, pooled

    def _decode(self, latents: torch.Tensor) -> list[Image.Image]:
        with torch.no_grad():
            imgs = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor
            ).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = (imgs.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
        return [Image.fromarray(img) for img in imgs]

    @staticmethod
    def _dt(scheduler, i: int) -> float:
        return float(scheduler.sigmas[i + 1] - scheduler.sigmas[i])