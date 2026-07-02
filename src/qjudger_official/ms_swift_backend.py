"""
Вендорировано без изменений из официального репозитория:
  https://github.com/QwenLM/Qwen-Image-Bench  (Apache License 2.0)
Полный текст лицензии: third_party/Qwen-Image-Bench-LICENSE-Apache-2.0.txt

Опциональный высокопроизводительный backend (ms-swift PtEngine) — то же
окружение, которым продюсировался официальный датасет. Основной путь в
этом проекте — judger.py на чистом transformers (см. README §6.3 про
выбор между ними). Этот backend можно подключить как альтернативу для
максимального throughput при батч-джадже большого числа изображений.

ms-swift PtEngine inference engine.

Mirrors the swift CLI command used to produce this dataset's *_response_* fields:

    swift infer --infer_backend pt --max_batch_size 24 --seed 42 \
                --temperature 0 --top_k 1 --top_p 1 \
                --repetition_penalty 1.05 --max_new_tokens 4096 \
                --enable_thinking true

Requires ms-swift>=4.0.0.
"""

from swift import TransformersEngine, RequestConfig, InferRequest


class MsSwiftJudge:
    def __init__(self, model_path, max_batch_size=24, max_new_tokens=4096):
        self.engine = TransformersEngine(model_path, max_batch_size=max_batch_size)
        self.request_config = RequestConfig(
            max_tokens=max_new_tokens,
            temperature=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.05,
            seed=42,
        )

        # Enable Qwen3 thinking mode on the engine's default template.
        # ms-swift 4.x exposes this via Template.template_meta.template_kwargs.
        try:
            self.engine.default_template.template_meta.template_kwargs = {
                "enable_thinking": True
            }
        except AttributeError:
            pass  # fall back to per-request template_inputs below

    def generate_batch(self, items):
        """
        Batch inference for multiple items.
        Each item: {"system_prompt": str, "user_text": str, "image": PIL.Image}
        Returns list of generated text strings.
        """
        infer_requests = []
        for item in items:
            messages = [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_text"]},
            ]
            infer_requests.append(
                InferRequest(messages=messages, images=[item["image"]])
            )

        resp_list = self.engine.infer(
            infer_requests,
            self.request_config,
        )

        return [r.choices[0].message.content for r in resp_list]
