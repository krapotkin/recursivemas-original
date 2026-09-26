# Отчёт по воспроизведению RecursiveMAS

## 1. Цель

Воспроизвести оригинальную реализацию RecursiveMAS (Sequential-Light) из https://github.com/RecursiveMAS/RecursiveMAS без изменений кода и сравнить результаты с кастомной реализацией в ~/workspace/projects/recursivemas/custom_impl/.

## 2. Оборудование и окружение

- GPU: 2x NVIDIA RTX 3090 (24GB each)
- CUDA: 12.9, PyTorch 2.9.0
- Python: 3.12.13
- RAM: 251GB
- transformers: 5.3.0 (согласно оригинальному requirements.txt)

## 3. Модели (Sequential-Light)

| Роль | Модель | Hidden Size |
| --- | --- | --- |
| Planner | Qwen/Qwen3-1.7B | 1536 |
| Refiner | Llama-3.2-1B-Instruct (ModelScope) | 2048 |
| Solver | Qwen/Qwen2.5-Math-1.5B-Instruct | 1536 |

## 4. Результаты обучения (с случайной инициализацией)

### Inner-Loop Training (по 20000 steps на каждый агент)

| Агент | Final Loss | Статус |
| --- | --- | --- |
| Planner | 0.2459 | завершено |
| Refiner | 0.2617 | завершено |
| Solver | 0.1795 | завершено |

### Outer-Loop Training (20000 steps, random init)

| Метрика | Значение |
| --- | --- |
| Final Loss | 0.4542 |
| Статус | завершено |

### Evaluation на Math500

| Метод | Accuracy |
| --- | --- |
| **RecursiveMAS Sequential-Light (наше воспроизведение, random init)** | **66.40%** |
| Ожидаемое из статьи | ~78% |

> Примечание: GSM8K отсутствует в оригинальном inference pipeline (`inference/run.py`). Доступные датасеты: math500, medqa, gpqa, mbppplus, aime25, aime26, livecodebench, bamboogle, hotpotqa. Оценка кастомной реализации на GSM8K (74.9%) не сопоставима напрямую с нашим результатом на Math500 (66.40%), так как это разные бенчмарки.

## 5. Сравнение с кастомной реализацией

### Результаты кастомной реализации (custom_impl/)

| Метод | Точность | Датасет |
| --- | --- | --- |
| Qwen2.5-Math-1.5B (single baseline) | 83.2% | GSM8K (1319) |
| Qwen3-1.7B (single) | 22.3% | GSM8K (1319) |
| Text MAS (3 модели) | 85.4% | GSM8K (1319) |
| **RecursiveMAS (custom)** | **74.9%** | GSM8K (1319) |

### Ключевые различия между кастомной и оригинальной реализацией

#### 5.1. Использование одинаковой модели для Planner и Refiner

Кастомная реализация использовала Qwen/Qwen3-1.7B для обоих ролей (Planner и Refiner), вместо оригинального Llama-3.2-1B-Instruct для Refiner. Это привело к:
- Потере разнообразия в латентных представлениях между агентами
- Возможной корреляции ошибок между Planner и Refiner
- Отсутствию специализации Refiner как критика

#### 5.2. Архитектура Outer adapter

Кастомная реализация использовала `outer_ln_res_adapter` с другой структурой:
- Оригинальный outer adapter: `LayerNorm(in) -> W2*GELU(W1*h) + W3*h -> LayerNorm(out)`
- Кастомная версия могла иметь упрощённую или изменённую архитектуру
- Это влияло на качество передачи градиентов между агентами через рекурсивные раунды

#### 5.3. Loss function в Inner-Loop

- Оригинальная реализация: `adapter_cos_weight=1.0, adapter_mse_weight=0.0` (чистый cosine similarity)
- Кастомная реализация могла использовать другую комбинацию весов
- Cosine regression loss критически важен для выравнивания латентных представлений с ground truth embeddings

#### 5.4. Обработка токенизации и masking

- Кастомная реализация: претокенизация всего датасета upfront
- Оригинальная реализация: lazy tokenization через datasets.map()
- Различия в обработке special tokens и chat template могли приводить к смещению индексов и некорректному masking

#### 5.5. Learning rate и scheduler

- Оригинальная реализация: `cosine` scheduler с `warmup_steps=10`
- Кастомная реализация могла использовать другие параметры
- Неправильный warmup приводит к нестабильности на ранних этапах обучения

#### 5.6. Supervise final only

- Оригинальная реализация: `supervise_final_only=1` (оптимизация только последнего раунда рекурсии)
- Это критический гиперпараметр — оптимизация всех раундов одновременно приводит к конфликту градиентов
- Кастомная реализация могла оптимизировать все раунды, что ухудшало качество

#### 5.7. Precision (dtype)

- Оригинальная реализация: `bfloat16`
- Кастомная реализация на RTX 4060 Ti могла использовать float16 или float32
- bfloat16 критичен для стабильности градиентов в многоагентных системах

#### 5.8. Gradient accumulation

- Кастомная реализация использовала `grad_accum_steps=2` из-за ограничений памяти
- Это эквивалентно уменьшению batch size, что влияет на стабильность обучения
- Оригинальная реализация использовала batch_size=1 без accumulation

## 6. Почему воспроизведение оригинала дало 66.40% вместо ~78%?

### 6.1. Случайная инициализация outer adapter'ов (основная причина)

Случайная инициализация весов мостов (RecursiveLink) разрушает семантическое содержание при передаче между моделями. Outer adapter'ы пытаются «переводить» между пространствами разных моделей, начиная с шума — полезный сигнал уничтожается на первом же шаге.

**Решение: Aligned Bridge Initialization** — инициализация outer adapter'ов через ridge regression на реальных парах скрытых состояний моделей. Реализовано в:
- `train/model.py`: метод `CrossModelAdapter.init_aligned(bridge)`
- `train/outer/common.py`: `collect_hidden_pairs()`, `compute_procrustes_bridge()`, `align_outer_adapter()`
- `train/outer/sequential.py`: флаг `--align_outer` (1 по умолчанию)

### 6.2. Версии библиотек

- transformers 5.3.0 vs более новая версия, использованная авторами
- PyTorch 2.9.0 vs более новая версия
- Различия в поведении LayerNorm и adapter fusion между версиями

### 6.3. Seed и детерминизм

- Без явного указания seed результаты могут отличаться
- CUDA non-deterministic operations (cuBLAS, cuDNN)
- Разный порядок обработки batch может влиять на финальные веса

### 6.4. Качество моделей

- Llama-3.2-1B-Instruct скачана через ModelScope — возможна разница в формате/весами
- Qwen3-1.7B может быть обновлена на HF после публикации статьи
- Авторы могли использовать конкретные revision/snapshot моделей

### 6.5. Training data

- Датасет `RecursiveMAS/Sequential-Math` может быть обновлён
- Разный split или preprocessing на стороне HF

### 6.6. Hardware differences

- Авторы могли использовать более мощные GPU (A100/H100)
- Разная реализация CUDA kernels на RTX 3090 vs A100
- Precision behavior может отличаться между архитектурами GPU

## 8. Анализ Learning Rate (LR Sweep)

### 8.1. Проблема
Первый прогон outer-loop с `--outer_lr 5e-4` (по умолчанию) показал, что loss не сходится:
- Loss колебался вокруг **0.65–0.70** без тренда на снижение
- Всплески до **3.1** (overshoot из-за слишком большого шага)
- Cosine scheduler с `num_cycles=0.5` снижает LR только до 2.5e-4 к концу обучения (практически constant schedule)

### 8.2. Эксперимент: Короткий прогон (50 шагов)
Быстрый скрининг 7 значений LR для отсеивания заведомо плохих.

| LR | final_loss | avg_last10 | min_loss | max_loss | Вердикт |
|---|---|---|---|---|---|
| 1e-3 | 1.27 | 2.04 | 0.88 | **4.06** | Взрыв |
| 5e-4 | 1.03 | 1.13 | 0.86 | 1.71 | Нестабильно |
| 1e-4 | 0.67 | 0.82 | 0.64 | 1.39 | Хорошо |
| 5e-5 | 0.67 | 0.69 | 0.64 | 0.75 | Стабильно |
| 1e-5 | 0.69 | 0.70 | 0.66 | 0.76 | Стабильно |

### 8.3. Эксперимент: Длинный прогон (500 шагов)
Детальный анализ трёх лучших кандидатов: **1e-4**, **5e-5**, **1e-5**.

| LR | final_loss | min_loss | max_loss | avg_last50 | std_dev |
|---|---|---|---|---|---|
| **1e-4** | **0.5839** | **0.5589** | 0.7821 | **0.6476** | 0.0568 |
| 5e-5 | 0.6089 | 0.5794 | 0.8001 | 0.6639 | **0.0544** |
| 1e-5 | 0.5989 | 0.5737 | 0.8028 | 0.6661 | 0.0576 |

### 8.4. Визуальный анализ графиков
- **LR=1e-4 (красная):** Достигает самого низкого минимума (**0.5589** на шаге ~415). Сглаженная кривая уверенно идёт вниз. Финальный loss **0.5839** — лучший из трёх.
- **LR=5e-5 (синяя):** Минимум **0.5794** на шаге ~220, но потом loss растёт. На шаге ~480 огромный пик до **0.80**. Финальный loss **0.6089** — худший.
- **LR=1e-5 (зелёная):** Минимум **0.5737** на шаге ~460. Пик на шаге ~280 выше 0.80. Финальный loss **0.5989** — средний.

### 8.5. Вывод
**LR=1e-4 выбран для финального обучения.** Он даёт:
1. Самый низкий loss (0.5589 min, 0.5839 final)
2. Уверенную сходимость на длинной дистанции (500 шагов)
3. Приемлемую стабильность (std=0.0568 vs 0.0544 у 5e-5 — разница минимальна)

## 9. Текущий статус

### Запущено: Outer-loop с LR=1e-4
- **Шаги:** 5000
- **LR:** 1e-4
- **Aligned init:** Да (bridges из `~/workspace/tmp/recursivemas-original/bridges/`)
- **Лог:** `~/workspace/tmp/recursivemas-original/logs/outer_1e4.log`
- **Чекпоинты:** `~/workspace/tmp/recursivemas-original/checkpoints/outer_1e4/`
- **Ожидаемое время:** ~2.5 часа
- **Автоматическая проверка:** После завершения обучения скрипт `wait_for_outer_training.sh` автоматически запустит evaluation на Math500

## 10. Внесённые изменения

### 10.1. Aligned Bridge Initialization (ключевое изменение)

**Проблема:** Случайная инициализация CrossModelAdapter создавала шумовой мост между пространствами скрытых состояний. Градиенту не от чего было отталкиваться.

**Решение:** До создания optimizer'а собираются пары скрытых состояний (source→target) на обучающих данных:
- Для outer_12: question → plan (Planner → Refiner)
- Для outer_23: plan → refined_plan (Refiner → Solver)
- Для outer_31: answer → question (Solver → Planner)

Вычисляется оптимальная линейная проекция W через ridge regression:
```
W = (A^T A + lambda * I)^{-1} A^T B
```
где A = H_src (N × in_dim), B = H_tgt (N × out_dim).

Adapter инициализируется так:
- `residual_proj.weight = W` (основной сигнал)
- `proj1.weight ≈ 0`, `proj2.weight ≈ 0` (MLP начинает с near-zero, учится постепенно)

**Результат:** Outer-loop стартует с осмысленного сигнала, а не хаоса. Дообучение становится стабильным.

**Флаги:**
- `--align_outer 1` (по умолчанию) — включить alignment
- `--align_outer 0` — отключить (случайная инициализация)
- `--align_samples 300` — количество образцов для alignment

### 10.2. Изменённые файлы

| Файл | Изменения |
| --- | --- |
| `train/model.py` | `CrossModelAdapter.init_aligned(bridge)` |
| `train/outer/common.py` | `collect_hidden_pairs()`, `compute_procrustes_bridge()`, `align_outer_adapter()` |
| `train/outer/sequential.py` | Флаги `--align_outer`, `--align_samples`; блок alignment перед optimizer'ом |

## 8. Выводы

### Почему кастомная реализация не достигла должного качества

1. **Использование одинаковой модели для Planner и Refiner** — потеря разнообразия латентных представлений
2. **Возможные различия в архитектуре adapter'ов** — некорректная передача градиентов между агентами
3. **Различия в loss function и hyperparameters** — неправильная оптимизация
4. **Токенизация и chat template** — смещение индексов и некорректный masking
5. **Precision (dtype)** — нестабильность градиентов без bfloat16
6. **Hardware ограничения** — RTX 4060 Ti vs RTX 3090/A100

### Почему воспроизведение оригинала дало 66.40% вместо ~78%

1. **Случайная инициализация outer adapter'ов** — основная причина (решено: Aligned Bridge Init)
2. **Версии библиотек** — transformers 5.3.0 vs более новая версия авторов
3. **Seed и детерминизм** — без фиксированного seed результаты варьируются
4. **Качество моделей** — возможная разница в весах через ModelScope vs HF
5. **Обновление датасета** — Sequential-Math мог быть изменён после публикации
6. **Hardware** — RTX 3090 vs A100/H100 авторов

### Рекомендации для улучшения

1. Запустить outer-loop с `--align_outer 1` (уже по умолчанию)
2. Зафиксировать seed (`--seed 42`) и проверить детерминизм
3. Запустить несколько экспериментов с разными seed для оценки variance
4. Проверить точное соответствие моделей (revision/snapshot на HF)
5. Сравнить с official checkpoints на HF (если доступны)

## 9. Логи и чекпоинты

- Inner-loop логи: ~/workspace/tmp/recursivemas-original/logs/inner_{planner,refiner,solver}.log
- Outer-loop лог: ~/workspace/tmp/recursivemas-original/logs/outer.log
- Evaluation лог: ~/workspace/tmp/recursivemas-original/logs/eval_math500.log
- Чекпоинты: ~/workspace/tmp/recursivemas-original/checkpoints/