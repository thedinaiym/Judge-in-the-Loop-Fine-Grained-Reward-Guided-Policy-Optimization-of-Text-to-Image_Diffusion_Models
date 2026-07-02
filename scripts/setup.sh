#!/usr/bin/env bash
# Запустить сразу после SSH в инстанс:
#   bash setup.sh
set -euo pipefail

echo "=== GPU check ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "=== Install deps ==="
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "=== HuggingFace login ==="
# Положите свой токен в переменную или введите вручную
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Запустите: huggingface-cli login"
    echo "Нужен токен с принятой лицензией FLUX.1-dev:"
    echo "  https://huggingface.co/black-forest-labs/FLUX.1-dev"
else
    huggingface-cli login --token "$HF_TOKEN"
fi

echo "=== Smoke imports ==="
python3 -c "
import torch, diffusers, transformers, peft, datasets, bitsandbytes
print('torch:', torch.__version__)
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU{i}: {p.name}, {p.total_memory//1024**3}GB')
print('All imports OK')
"
echo "=== Ready. Run: python train.py ==="