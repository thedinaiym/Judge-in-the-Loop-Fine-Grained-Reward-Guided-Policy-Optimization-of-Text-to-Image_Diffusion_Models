This project vendors source code from the following third-party project,
unmodified except for added attribution comments:

  Project: Qwen-Image-Bench
  Source:  https://github.com/QwenLM/Qwen-Image-Bench
  License: Apache License, Version 2.0 (full text in
           Qwen-Image-Bench-LICENSE-Apache-2.0.txt in this directory)
  Files vendored (under src/qjudger_official/):
    - checklists.py        (SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, 5 checklists)
    - score_utils.py        (score parsing + L3->L2->L1->Total aggregation)
    - ms_swift_backend.py   (optional high-throughput ms-swift backend)

  Also used (paths/citation only, not redistributed as a binary):
    - Judge model weights: https://huggingface.co/Qwen/Qwen-Image-Bench
      (downloaded at runtime by the user from Hugging Face Hub, not bundled
      in this repository)
    - Benchmark dataset: https://huggingface.co/datasets/Qwen/Qwen-Image-Bench
      (loaded at runtime via the `datasets` library)

  Citation (paper):
    Li, Niantong, et al. "Qwen-Image-Bench: From Generation to Creation in
    Text-to-Image Evaluation." arXiv:2605.28091 (2026).
