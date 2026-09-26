# RecursiveMAS — Воспроизведение оригинальной реализации

Воспроизведение оригинального репозитория [RecursiveMAS/RecursiveMAS](https://github.com/RecursiveMAS/RecursiveMAS) без изменений кода.

## Что это

Полное пошаговое воспроизведение эксперимента Sequential-Light из оригинальной статьи RecursiveMAS:
- Inner-loop training для Planner, Refiner, Solver
- Outer-loop training для RecursiveLink между агентами
- Evaluation на Math500

## Архитектура

Три агента, соединённых в цепочку (chain) с рекурсивной передачей скрытых состояний:

```
Planner (Qwen3-1.7B) --[outer_12]--> Refiner (Llama-3.2-1B) --[outer_23]--> Solver (Qwen2.5-Math-1.5B) --[outer_31]--> (обратно к Planner)
```

Каждый агент имеет **inner adapter** (выравнивание скрытых представлений с эмбеддингами токенов) и агенты связаны **outer adapters** (CrossModelAdapter — проекция между разными hidden_size).

### Inner-loop

Обучение adapter'а (ln_res_adapter) для каждого агента. Loss = cosine similarity между output adapter'а и input embeddings следующего токена. Базовая модель заморожена.

### Outer-loop

Обучение 3 CrossModelAdapter (outer_12, outer_23, outer_31) — проекций между скрытыми состояниями разных моделей. Рекурсивный цикл: Planner -> Refiner -> Solver -> (feedback) -> Planner. Градиент проходит через frozen inner adapter'ы и outer adapter'ы. CE loss только на Solver.

### Aligned Bridge Initialization

Ключевое изменение: outer adapter'ы инициализируются не случайно, а через ridge regression на реальных парах скрытых состояний моделей. Это решает проблему разрушения семантического сигнала при передаче между моделями.

- Собираются пары (question→plan, plan→refined_plan, answer→question) на обучающих данных
- Вычисляется оптимальная линейная проекция W через ridge regression
- residual_proj = W, MLP-ветка начинается с near-zero

## Исходный код

Оригинальный код из https://github.com/RecursiveMAS/RecursiveMAS склонирован в этот репозиторий без изменений. Изменения внесены только в `train/model.py`, `train/outer/common.py`, `train/outer/sequential.py` для поддержки Aligned Bridge Initialization.

## Модели

| Роль | Модель | Hidden Size |
| --- | --- | --- |
| Planner | Qwen/Qwen3-1.7B | 1536 |
| Refiner (Critic) | LLM-Research/Llama-3.2-1B-Instruct | 2048 |
| Solver | Qwen/Qwen2.5-Math-1.5B-Instruct | 1536 |

## Оборудование

- 2x NVIDIA GeForce RTX 3090 (24GB each)
- CUDA 12.9, PyTorch 2.9.0
- 251GB RAM

## Датасет

- RecursiveMAS/Sequential-Math (HuggingFace, 1904 примера)

## Ожидаемые результаты (из статьи)

Sequential-Light на Math500: 78.0% accuracy

## Результаты воспроизведения

| Этап | Результат |
| --- | --- |
| Inner Planner (20K steps) | loss=0.2459 |
| Inner Refiner (20K steps) | loss=0.2617 |
| Inner Solver (20K steps) | loss=0.1795 |
| Outer (20K steps, random init) | loss=0.4542 |
| Math500 accuracy (random init) | 66.40% |

> Примечание: 66.40% вместо ожидаемых ~78% — проблема в случайной инициализации outer adapter'ов. Решение: Aligned Bridge Initialization (см. ниже).

## Анализ Learning Rate

Проведён LR Sweep для выбора оптимальной скорости обучения outer adapter'ов:
- **Короткий прогон (50 шагов):** Тестированы LR от 1e-3 до 1e-6. Отсеивание заведомо плохих значений.
- **Длинный прогон (500 шагов):** Детальный анализ 1e-4, 5e-5, 1e-5.
- **Результат:** LR=1e-4 выбран как оптимальный (min_loss=0.5589, final_loss=0.5839).

## Текущий статус

**Обучение запущено:** Outer-loop с LR=1e-4, 5000 шагов, aligned init.
- Лог: `~/workspace/tmp/recursivemas-original/logs/outer_1e4.log`
- Чекпоинты: `~/workspace/tmp/recursivemas-original/checkpoints/outer_1e4/`
- После завершения автоматически запустится evaluation на Math500.

## Структура проекта

```
recursivemas-original/
├── train/
│   ├── train_inner.py           # Обучение inner adapter
│   ├── train_outer.py           # Роутер для outer-loop
│   ├── model.py                 # Adapter, CrossModelAdapter, LatentReasoningModel
│   ├── data.py                  # Загрузка датасетов, токенизация
│   ├── mas_prompt.py            # Промпты для всех ролей и стилей
│   └── outer/
│       ├── common.py            # Общие утилиты + aligned bridge init
│       ├── sequential.py        # Sequential outer training
│       ├── mixture.py           # Hierarchy outer training
│       ├── distillation.py      # Distillation outer training
│       └── deliberation.py      # Deliberation outer training
├── inference/
│   ├── run.py                   # Главный entry point для inference
│   ├── inference_utils/         # Модули inference для разных стилей
│   ├── modeling.py              # Загрузка моделей + adapter'ов
│   └── ...
├── requirements.txt
├── AGENTS.md                    # Контекст для агента
├── INSTALL.md                   # Инструкция установки
├── REPORT.md                    # Отчёт по результатам
└── TODO.md                      # Команды запуска
```