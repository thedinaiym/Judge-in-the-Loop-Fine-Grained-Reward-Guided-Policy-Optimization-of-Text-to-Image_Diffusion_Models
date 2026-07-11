# ТЗ: Reward-Guided Diffusion Policy Optimization

**Онлайн RL-дообучение text-to-image диффузионной модели (FLUX.1) с РЕАЛЬНОЙ
моделью Q-Judger (Qwen/Qwen-Image-Bench, 27B) в роли reward-оракула и
датасетом Qwen-Image-Bench как источником промптов, через GRPO / Online-DPO
над LoRA-адаптером.**

Версия: 2.0 · Дата: 2026-06-29

> **Что изменилось с v1.0:** в первой версии этого ТЗ "Q-Judger" был
> гипотетическим названием без существующего чекпоинта, и роль судьи играл
> generic Qwen2-VL-7B с самодельной 6-осевой рубрикой. После того, как
> выяснилось, что **Qwen/Qwen-Image-Bench — это настоящая, публично
> доступная модель** (27B, дообучена на Qwen3.6-27B, Apache-2.0), весь
> судейский слой пайплайна переписан на неё: промпт-шаблоны, чеклисты и
> методология агрегации скоров взяты БЕЗ ИЗМЕНЕНИЙ из официального
> репозитория [QwenLM/Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench)
> (вендорированы в `src/qjudger_official/`). Промпт-пул теперь — это
> реальный датасет [Qwen/Qwen-Image-Bench](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench)
> (1000 промптов с разметкой релевантных направлений оценки на каждый).
> Это меняет и оценку требований к VRAM/стоимости — см. §6.

---

## 1. Цель проекта

Построить closed-loop pipeline, в котором text-to-image диффузионная модель
(policy) сама генерирует изображения, а реальная модель-судья Q-Judger
оценивает их по 5 иерархическим направлениям качества с доказанной
корреляцией с человеческими экспертами (Spearman ρ=0.92, см. §6.2) — и
полученный reward используется для онлайн-обновления политики через
policy-gradient методы (GRPO/PPO) или online-DPO, без статического
датасета человеческих предпочтений.

Ключевое отличие от offline-подходов (Diffusion-DPO, SFT): сигнал берётся
из **собственных** сэмплов модели на каждой итерации по промптам из
**реального бенчмарк-датасета** (а не из вручную написанных 15 примеров),
с балансировкой по направлениям оценки — поэтому политика оптимизируется
именно под то распределение, которое она сейчас генерирует, по
репрезентативному срезу capability-таксономии Q-Judger.

---

## 2. Архитектура (pipeline)

```
                         ┌────────────────────────────┐
   Prompt c (из            │   Diffusion Policy π_θ     │
   Qwen-Image-Bench,        │   FLUX.1-dev                │
   1 из 1000, + dims_en)    │   + LoRA adapter (rank 64)  │
                         │   GPU 0                     │
                         └────────────┬───────────────┘
                                      │ rollout: n=8 сэмплов на промпт
                                      ▼
                         ┌────────────────────────────┐
                         │   {x1 ... x8} ~ π_θ(·|c)    │
                         └────────────┬───────────────┘
                                      ▼
                         ┌────────────────────────────┐
                         │   Q-Judger (frozen, 27B)     │
                         │   Qwen/Qwen-Image-Bench      │
                         │   GPU 1 (отдельная карта)    │
                         │   только релевантные для c   │
                         │   L1-направления (dims_en)   │
                         │   -> JSON 0/1/2/N-A по       │
                         │   L3-фасетам -> агрегация    │
                         └────────────┬───────────────┘
                                      ▼
                         ┌────────────────────────────┐
                         │  Total score ∈ [0,100]      │
                         │  group-relative advantage:   │
                         │  A_i = (R_i − μ) / (σ + ε)   │
                         └────────────┬───────────────┘
                                      ▼
                         ┌────────────────────────────┐
                         │  RL update: GRPO / Online   │
                         │  DPO над LoRA-весами         │
                         │  ΔW = B·A, rank r=64         │
                         └────────────┬───────────────┘
                                      ▼
                              π_θ ← π_θ+1   (loop ↺ k итераций)
```

### 2.1 Почему именно Q-Judger (Qwen/Qwen-Image-Bench) как frozen-оракул

Это не общая VLM с самодельным промптом, а **модель, специально дообученная
для judging** T2I-генераций, с опубликованной валидацией против человеческих
экспертов. Замороженная судья-модель не подвержена Goodhart's law в той
степени, в которой был бы подвержен обучаемый reward model — платится это
тем, что верхний предел качества ограничен "вкусом" самого Q-Judger там,
где он слаб (см. §8, риски).

---

## 3. Q-Judger: реальная модель и её таксономия

### 3.1 Модель

| | |
|---|---|
| Чекпоинт | [`Qwen/Qwen-Image-Bench`](https://huggingface.co/Qwen/Qwen-Image-Bench) |
| База | Qwen3.6-27B |
| Размер | 27B, BF16 |
| Вход | текстовый промпт + изображение |
| Режим | thinking ВКЛЮЧЁН (chain-of-thought перед JSON-ответом) |
| Выход | структурированный JSON, 3-уровневая иерархия, score ∈ {0, 1, 2, "N/A"} |
| Лицензия | Apache 2.0 |
| Код инференса | [`github.com/QwenLM/Qwen-Image-Bench`](https://github.com/QwenLM/Qwen-Image-Bench) — вендорирован в `src/qjudger_official/` |

### 3.2 Таксономия: 5 направлений → 23 категории → 56 фасетов

| L1-направление | L2-категории (с числом L3-фасетов) |
|---|---|
| **Quality** | Realism (2), Detail (3), Resolution (1) |
| **Aesthetics** | Composition (1), Color Harmony (1), Lighting (1), Anatomical Portraiture (1), Emotional Expression (1), Style Control (1) |
| **Alignment** | Attributes (6), Actions (3), Layout (2), Relations (3), Scene (2) |
| **Real-world Fidelity** | Fairness (2), Safety & Compliance (1), World Knowledge (5) |
| **Creative Generation** | Imagination (1), Feature Matching (1), Logical Resolution (1), Text Rendering (4), Design Applications (6), Visual Storytelling (7) |

Итого: 5 L1 × 23 L2 × 56 L3 — точное число facets/categories, упомянутое в
исходных заметках проекта. Полные формулировки каждого L3-критерия — в
`src/qjudger_official/checklists.py` (вендорировано без изменений).

Для каждого промпта из датасета Qwen-Image-Bench заранее размечено, какие
именно L3-фасеты релевантны (`dims_en`) — Q-Judger вызывается только по тем
L1-направлениям, которые содержат хотя бы один релевантный фасет для этого
конкретного промпта (см. `parse_dims_by_level1` в checklists.py и
`judger.py: score_batch`). Для промптов **вне** датасета (свои собственные)
такой разметки нет — в этом случае пайплайн по умолчанию прогоняет **все 5**
направлений (см. `judger.py: _default_dims_by_level1`), модель сама
проставит `"N/A"` там, где критерий неприменим.

### 3.3 Методология скоринга (официальная, воспроизведена точно)

| Raw score | Значение | Mapped score |
|---|---|---|
| 0 | Fail | 0 |
| 1 | Pass | 60 |
| 2 | Excel | 100 |
| N/A | Неприменимо | исключается из усреднения |

Агрегация: **L3 → L2** (среднее всех непустых L3 в категории) → **L2 → L1**
(среднее L2 внутри направления) → **L1 → Total** (среднее по всем
оценённым L1-направлениям). Это и есть **reward** в этом пайплайне — без
каких-либо ручных весов (в отличие от v1.0 этого ТЗ, где веса по 6 осям
были придуманы вручную). Реализация — `src/qjudger_official/score_utils.py`,
вендорирована без изменений.

### 3.4 Валидация против человеческих экспертов

Из карточки модели — Spearman ρ между ранжированием Q-Judger и ранжированием
человеческих экспертов по 18 моделям, p<10⁻⁴ для всех направлений:

| Направление | Spearman ρ |
|---|---|
| Quality | 0.89 |
| Aesthetics | 0.89 |
| Alignment | 0.89 |
| Real-world Fidelity | 0.92 |
| Creative Generation | 0.92 |
| **Overall** | **0.92** |

Это прямое эмпирическое обоснование того, что reward от Q-Judger —
осмысленный сигнал для RL, а не шум. Не отменяет риск reward hacking (см.
§8) — корреляция 0.92 на статичном наборе моделей не гарантирует, что
LoRA-политика, оптимизируемая *против* этого самого судьи, не найдёт его
слепые зоны.

---

## 4. Датасет Qwen-Image-Bench как источник промптов

[`Qwen/Qwen-Image-Bench`](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench)
(1000 строк, EN+CN, Apache-2.0) даёт:

1. **1000 промптов**, изначально подобранных под покрытие всей таксономии
   из §3.2 — несравнимо лучше, чем 15 промптов вручную из v1.0 этого ТЗ.
2. **dims_en на каждый промпт** — какие именно L3-фасеты релевантны (см.
   §3.2) — используется и для эффективного вызова Q-Judger, и для
   **стратифицированной выборки батчей** (`prompts_loader.stratified_sample`)
   — каждый RL-шаг старается покрыть разные L1-направления, а не сваливаться
   в случайный перекос (важно для метрики §7: "рост reward не должен идти
   за счёт одного направления").
3. Сгенерированные изображения и judge-ответы 19 коммерческих/открытых
   моделей (Qwen-Image, FLUX.2-pro, GPT-Image-1.5, Imagen-4.0-Ultra и т.д.)
   — это НЕ используется напрямую в обучении (мы оптимизируем свою FLUX.1
   политику), но даёт **референсные точки**: можно посчитать Total score
   вашей FLUX.1+LoRA политики и сравнить с уже размеченными скорами других
   моделей на тех же промптах для калибровки ожиданий ("где мы находимся
   относительно опубликованных моделей").

Загрузка: `src/prompts_loader.py: load_qwen_image_bench_prompts()` — через
`datasets.load_dataset`, требует интернет на момент первого вызова
(на vast.ai-инстансе не проблема, см. §6). Fallback на локальный
`configs/qwen_image_bench_dims_metadata.json` (вендорирован, только
`ID -> dims_en`, без текста промптов) — для сценариев без доступа к Hub,
скомбинировать с собственным `prompts_pool.txt`, см. `prompts.source` в
`config.yaml`.

---

## 5. Алгоритм

### 5.1 Основной цикл (один RL-step)

```
for iteration in range(K):
    prompts = stratified_sample(qwen_image_bench_pool, k=B, by="level1")  # балансировка по 5 направлениям
    for c in prompts:                      # c содержит prompt_en + dims_en
        images, logp_trace = policy.sample(c.prompt_en, n=8)              # rollout
        rewards = judger.score_batch(images, c.prompt_en, dims_en=c.dims_en)  # Total score, только релевантные L1
        advantages = group_relative_advantage(rewards)
        loss = grpo_loss(logp_trace, advantages, clip_eps=0.2)            # или online-DPO, см. §5.3
        loss.backward()
    optimizer.step()  # обновляет только LoRA-параметры
    log_metrics(iteration, rewards, loss)   # включая breakdown по 5 L1-направлениям
    if iteration % save_every == 0:
        save_lora_checkpoint()
```

### 5.2 GRPO loss (critic-free, group-relative)

Без изменений по сравнению с v1.0:

```
A_i = (R_i − mean(R)) / (std(R) + eps)
ratio_i = exp(logp_new_i − logp_old_i)
loss_i = -min(ratio_i * A_i, clip(ratio_i, 1-eps, 1+eps) * A_i)
loss = mean(loss_i) − β * KL(π_θ || π_ref)
```

### 5.3 Проблема log-prob для flow-matching моделей

См. подробности в `src/policy.py` (класс `DiffusionPolicy`, методы
`sample`/`recompute_log_prob`) — стохастическая SDE-релаксация
rectified-flow ODE с управляемым шумом `σ_t` на каждом шаге интегрирования,
дающая вычисляемый log-prob для PPO/GRPO clipping. **Рекомендуется
начинать с Online-DPO** (`src/dpo_trainer.py`) — не требует доверять этой
трактовке для clipping, работает через разницу score-loss winner/loser
внутри группы. GRPO (`src/grpo_trainer.py`) — второй этап, после того как
весь pipeline (rollout → judger → reward → update) подтверждён рабочим.

---

## 6. Требования к железу и рекомендация по vast.ai

### 6.1 Почему GPU-требования выросли по сравнению с v1.0

В v1.0 предполагался generic Qwen2-VL-7B как судья (~15GB). Реальный
Q-Judger — **27B модель** (~54GB в BF16) с включённым thinking-режимом и до
4096 токенов генерации НА КАЖДЫЙ вызов (направление × изображение). Это
существенно меняет расчёт.

| Компонент | Память (BF16) |
|---|---|
| FLUX.1-dev transformer (12B) + T5-XXL + CLIP | ~34 GB |
| LoRA (rank 64) + optimizer states + activations rollout (n=8) | ~10–15 GB |
| **Q-Judger (Qwen/Qwen-Image-Bench, 27B)** | **~54 GB** |
| KV-cache Q-Judger при батче до 24 задач × до 4096 токенов | ~10–20 GB (зависит от батча) |
| **Итого при совместной загрузке на одну карту** | **~108–123 GB** |

Это **превышает одну 80GB-карту**. Рекомендация изменилась относительно
v1.0 — теперь **2 GPU — это основной рекомендуемый вариант, а не опция
масштабирования**.

### 6.2 Три сценария на vast.ai (цены — июнь 2026, диапазоны)

| Уровень | Конфигурация | Цена/ч | Когда использовать |
|---|---|---|---|
| Дешёвая отладка | 1× RTX 4090 24GB | ~$0.30–0.45 | Отладка policy.py/rollout.py на **мок-judger** (без реальной 27B модели) или с FLUX в FP8 — см. §6.4 |
| **Рекомендуемый прогон** | **2× A100 80GB SXM** (policy на GPU0, Q-Judger на GPU1) | **~$1.40–2.60 суммарно** | Полный online-loop без конкуренции за память, без оффлоада — основной режим работы pipeline |
| Производительный | 2× H100 80GB SXM | ~$3.00–4.00 суммарно | Та же память, но 2-3x быстрее transformer-инференс — особенно важно для 27B Q-Judger с thinking-режимом (генерация токенов — основной latency-бутылочный конец, см. §6.3) |

**1–2 часа эксперимента на 2×A100 80GB ≈ $3–6** — дороже, чем оценка v1.0
($1–3), из-за реального размера Q-Judger, но всё ещё бюджетно для проверки
гипотезы целиком.

### 6.3 Главный риск по времени: thinking-режим 27B судьи

Q-Judger генерирует до 4096 токенов **с включённым chain-of-thought** на
каждый вызов (одно L1-направление на одно изображение). Для одного промпта
с n=8 сэмплами и, скажем, 3 релевантных L1-направлениях из 5 (типичный
случай по dims_en) — это **24 вызова генерации** на одну RL-итерацию **до
обновления политики**. Если каждый вызов занимает несколько секунд (27B
модель, длинный CoT) — судейский этап может занимать заметно больше
времени, чем сама генерация изображений FLUX. Варианты митигации:

1. Уменьшить `max_new_tokens` в `config.yaml` (например, до 1024–2048) —
   ускоряет, но отклоняется от официального валидированного сетапа и может
   снижать качество/надёжность парсинга JSON. Тестировать на небольшой
   выборке перед полным прогоном.
2. Перейти на ms-swift backend (`src/qjudger_official/ms_swift_backend.py`,
   тот же сетап, которым продюсировался официальный датасет) — выше
   throughput на батчах, особенно при `max_batch_size=24` по умолчанию.
3. Декомпозировать judging в отдельный сервис (vLLM/ms-swift server) и
   слать запросы асинхронно, пока генерируется следующий rollout — этап 7
   roadmap (Ray-оркестрация), теперь обоснован сильнее, чем в v1.0.

### 6.4 Дешёвая отладка без реального Q-Judger

Для отладки `policy.py`/`rollout.py`/`train.py` (форматы тензоров, шейпы,
основной цикл) **без** затрат на 27B-модель — замените `QJudger` на мок,
возвращающий случайные/константные `JudgeResult` той же формы. Это не
входит в текущий код проекта намеренно (чтобы не плодить параллельную
"тестовую" реализацию, расходящуюся с реальной) — добавляется в несколько
строк по необходимости перед первым прогоном на дешёвой карте.

---

## 7. Метрики успеха

1. **Reward growth curve**: средний Total score по батчу растёт с
   итерациями.
2. **Per-L1-direction breakdown**: рост не должен идти за счёт одного
   направления в ущерб другим — контролируется логированием `axis/<L1>` на
   каждой итерации (`reward.py: axis_breakdown_stats`), с учётом того, что
   не все направления оцениваются на каждом промпте (см. §3.2).
3. **Diversity check**: CLIP-similarity между сэмплами одного промпта не
   должна резко падать (mode collapse / reward hacking).
4. **KL to reference policy**: должен расти плавно, не взрываться.
5. **Held-out eval + сравнение с опубликованными моделями**: помимо
   held-out промптов (не использованных в rollout), можно напрямую
   сравнить Total score своей политики с уже размеченными в датасете
   скорами 19 моделей на тех же ID промптов (см. §4, пункт 3) — это даёт
   внешнюю систему координат, а не только "стало лучше относительно себя".

---

## 8. Риски и ограничения

- **Reward hacking / Goodhart**: даже валидированный (ρ=0.92) Q-Judger
  имеет слепые зоны; митигируется KL-штрафом, multi-dimension reward
  (сложнее обмануть 5 независимых направлений одновременно, чем 1 скаляр).
- **VRAM/латентность Q-Judger 27B**: см. §6.3 — это новый, более серьёзный
  риск по сравнению с v1.0 (где судья был 7B). Бюджетируйте время и
  стоимость с запасом на тестовом прогоне (10-20 итераций) перед полным.
- **Квантование судьи (если идёте на 1 GPU вместо 2)**: FP8/4-bit снижает
  VRAM (27B → ~27GB/~14GB), но валидация ρ=0.92 проводилась в BF16 —
  квантованную версию нужно отдельно сверить на небольшой подвыборке
  датасета (сравнить с уже размеченными `quality_response_*` и т.д.
  колонками) перед тем, как доверять ей как reward-сигналу.
- **Parse failures**: `extract_json_from_response` может не распарситься
  на редких "сорвавшихся" thinking-генерациях — такие сэмплы помечаются
  `parse_ok=False` и исключаются из DPO/GRPO update (см. `dpo_trainer.py`,
  `grpo_trainer.py`), не получая искусственный нулевой reward.
- **Нестабильность flow-matching policy gradient** (§5.3) — начинать с
  Online-DPO.
- **Стоимость**: 2-GPU сетап дороже, чем предполагалось в v1.0. Закладывайте
  чекпоинтинг каждые N итераций для spot-инстансов (см. `docs/gpu_recommendation.md`).

---

## 9. Структура репозитория

```
reward-guided-diffusion-rl/
├── README.md                          # это ТЗ (v2.0)
├── requirements.txt
├── configs/
│   ├── config.yaml                     # все гиперпараметры
│   ├── qwen_image_bench_dims_metadata.json   # вендорированный fallback (ID -> dims_en)
│   ├── prompts_pool.txt                  # fallback-промпты при source: local_file
│   └── prompts_holdout.txt
├── src/
│   ├── judger.py                          # Q-Judger: реальная модель через transformers
│   ├── qjudger_official/                   # вендорировано из QwenLM/Qwen-Image-Bench (Apache-2.0)
│   │   ├── checklists.py                    # SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, 5 чеклистов
│   │   ├── score_utils.py                    # парсинг + L3->L2->L1->Total агрегация
│   │   └── ms_swift_backend.py                 # опциональный high-throughput backend
│   ├── prompts_loader.py                   # загрузка датасета Qwen-Image-Bench + стратификация
│   ├── policy.py                            # FLUX + LoRA + stochastic sampler
│   ├── rollout.py                            # генерация n сэмплов на промпт, передача dims_en
│   ├── reward.py                              # group-relative advantage, axis breakdown
│   ├── grpo_trainer.py                         # GRPO update step
│   ├── dpo_trainer.py                           # Online-DPO update step (рекомендуемый старт)
│   ├── train.py                                  # точка входа, основной цикл
│   └── utils.py                                   # логирование, чекпоинты, seed
├── scripts/
│   └── setup_vastai.sh                       # bootstrap окружения на арендованной машине
├── third_party/
│   ├── NOTICE.md                              # атрибуция вендорированного кода
│   └── Qwen-Image-Bench-LICENSE-Apache-2.0.txt
└── docs/
    └── gpu_recommendation.md                # детальный разбор по vast.ai (обновлён под 27B)
```

---

## 10. Roadmap

| Этап | Статус | Что сделано / осталось |
|---|---|---|
| 0 | ✅ | Структура проекта, конфиг, requirements |
| 1 | ✅ | LoRA-обвязка над FLUX (`policy.py`) |
| 2 | ✅ | **Реальный** Q-Judger (Qwen/Qwen-Image-Bench) вместо generic VLM-заглушки |
| 3 | ✅ | Rollout с dims_en-aware вызовами судьи |
| 4 | ✅ | Промпт-пул из официального датасета + стратифицированная выборка |
| 5 | ✅ | Online-DPO update step |
| 6 | ✅ | GRPO update step со stochastic flow sampler |
| 7 | ⬜ | **Запуск на реальном железе** (2×A100/H100) — это первый шаг, который нужно сделать руками: скачать веса FLUX.1-dev (gated, нужен HF-токен с принятой лицензией) и Qwen/Qwen-Image-Bench, прогнать `scripts/setup_vastai.sh`, затем 10-20 тестовых итераций с уменьшенным `max_new_tokens` для проверки сквозного пайплайна перед полным прогоном |
| 8 | ⬜ | Сравнение Total score своей политики с референсными моделями из датасета (§7, пункт 5) |
| 9 | ⬜ | Масштабирование: декомпозиция Q-Judger в отдельный сервис (vLLM/ms-swift), Ray-оркестрация параллельных rollout-воркеров (см. §6.3, пункт 3) |

**Код полностью готов** (этапы 0-6) — оставшиеся шаги требуют реального GPU
и аккаунта на vast.ai/Hugging Face (принятие лицензии FLUX.1-dev), что
невозможно выполнить в текущей среде. README и весь код намеренно написаны
так, чтобы эти шаги можно было выполнить, просто следуя `scripts/setup_vastai.sh`
и `python src/train.py --config configs/config.yaml`.

---

## 11. Лицензии и атрибуция

- Код проекта: предполагается, что наследует лицензию по выбору автора
  (не специфицирована в этом ТЗ).
- `src/qjudger_official/*` — вендорировано без изменений из
  [QwenLM/Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench),
  Apache License 2.0. Полный текст: `third_party/Qwen-Image-Bench-LICENSE-Apache-2.0.txt`.
  Детали — `third_party/NOTICE.md`.
- Веса модели `Qwen/Qwen-Image-Bench` и датасет `Qwen/Qwen-Image-Bench` —
  Apache 2.0, скачиваются пользователем самостоятельно через Hugging Face
  Hub при первом запуске (не входят в этот репозиторий).
- Веса `black-forest-labs/FLUX.1-dev` — отдельная (non-commercial,
  gated) лицензия Black Forest Labs, требует принятия условий на странице
  модели на Hugging Face и собственного HF-токена. Это ограничение
  накладывается black-forest-labs, не этим проектом.

## Results (preliminary, 10 GRPO steps)

| Metric | Value |
|---|---|
| Parse rate (Q-Judger) | **1.000** (30/30 pairs) |
| Gradient norm | 332 – 764 |
| Policy–reference KL | 0 → 256 |
| Mean reward | 32.11 ± 11.62 |
| OLS slope | +1.34/step (r² = 0.12, *not significant*) |
| Hardware | 1× A100 80GB · 670 s/step · 1.86 h |

⚠️ Per-step reward is **confounded by prompt difficulty** (1 prompt/step) and
must not be read as a learning curve. No baseline comparison yet.

- Metrics: `results/metrics.jsonl`
- Full log: `results/train.log`
- Config: `results/config_final.yaml`
- Trained adapter: https://huggingface.co/thedinaiym/judge-in-the-loop-flux-lora
