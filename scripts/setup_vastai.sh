#!/usr/bin/env bash
# setup_vastai.sh — bootstrap окружения на арендованной vast.ai машине.
#
# Использование (после SSH в инстанс):
#   git clone <your-repo-url> reward-guided-diffusion-rl
#   cd reward-guided-diffusion-rl
#   bash scripts/setup_vastai.sh
#
# Рекомендуемый образ при создании инстанса на vast.ai:
#   pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel  (или любой свежий PyTorch+CUDA образ)
# Рекомендуемая конфигурация (v2.0, см. docs/gpu_recommendation.md):
#   2x A100 80GB SXM — policy (FLUX) на GPU0, Q-Judger (27B) на GPU1.
#   Бюджетный вариант на 1 GPU: см. README §6.4 (квантование судьи / мок для отладки).
#
# Два чекпоинта потребуются перед первым полным прогоном:
#   1. black-forest-labs/FLUX.1-dev — GATED, нужно принять лицензию на
#      странице модели на HF и использовать токен с этим доступом.
#   2. Qwen/Qwen-Image-Bench (Q-Judger, 27B) — публичный, Apache-2.0,
#      не требует принятия отдельной лицензии, но тоже ~54GB на диск.

set -euo pipefail

echo "== [1/5] Проверка GPU (ожидается 2 устройства для рекомендуемого сетапа) =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo "== [2/5] Системные зависимости =="
apt-get update -qq && apt-get install -y -qq git wget ffmpeg libsm6 libxext6 > /dev/null

echo "== [3/5] Python-зависимости =="
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "== [4/5] Проверка доступа к Hugging Face Hub =="
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ВНИМАНИЕ: переменная HF_TOKEN не установлена."
  echo "Выполните: huggingface-cli login"
  echo "Токен должен иметь принятую лицензию FLUX.1-dev:"
  echo "  https://huggingface.co/black-forest-labs/FLUX.1-dev"
  echo "Qwen/Qwen-Image-Bench (Q-Judger) публичный, отдельного допуска не требует."
else
  huggingface-cli login --token "$HF_TOKEN"
fi

echo "== [5/5] Smoke-test: импорт ключевых библиотек =="
python -c "
import torch, diffusers, transformers, peft, datasets
print('torch:', torch.__version__, '| GPUs:', torch.cuda.device_count())
print('diffusers:', diffusers.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('datasets:', datasets.__version__)
"

echo "== Готово. Следующие шаги: =="
echo "1) Тест загрузки промптов (не требует GPU, только сеть):"
echo "     python -c \"import sys; sys.path.insert(0,'src'); from prompts_loader import load_qwen_image_bench_prompts; p = load_qwen_image_bench_prompts(); print(len(p), p[0])\""
echo "2) Перед полным прогоном — 10-20 тестовых итераций с уменьшенным"
echo "   judger.max_new_tokens (например 1024) в config.yaml, чтобы быстро"
echo "   проверить весь цикл rollout -> judger -> update без ожидания полных"
echo "   4096-токенных thinking-генераций (см. README §6.3)."
echo "3) Полный прогон:"
echo "     python src/train.py --config configs/config.yaml"
