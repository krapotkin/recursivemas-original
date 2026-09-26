# AGENTS.md — Контекст для агента

## Проект

Воспроизведение оригинальной реализации RecursiveMAS (Sequential-Light) без изменений кода.

## Ключевые факты

- Оригинальный код: склонирован из https://github.com/RecursiveMAS/RecursiveMAS без изменений
- Исходный (неудачный) проект: ~/workspace/projects/recursivemas (custom_impl/)
- Целевая метрика: ~78% accuracy на Math500 (Sequential-Light из статьи)

## Aligned Bridge Initialization (внесено)

Проблема: случайная инициализация outer adapter'ов (RecursiveLink) разрушает семантический сигнал при передаче между моделями.

Решение: перед обучением outer-loop собираются пары скрытых состояний (source→target) на обучающих данных, вычисляется оптимальная линейная проекция через ridge regression, и outer adapter'ы инициализируются этой проекцией.

Реализация:
- `train/model.py`: метод `CrossModelAdapter.init_aligned(bridge)` — задаёт residual_proj = bridge, MLP начинает с near-zero
- `train/outer/common.py`: `collect_hidden_pairs()`, `compute_procrustes_bridge()`, `align_outer_adapter()`
- `train/outer/sequential.py`: флаг `--align_outer` (1 по умолчанию), `--align_samples` (300)

Отключить: `--align_outer 0` (вернуть случайную инициализацию).

## Модели

| Роль | Модель | Путь | Hidden Size |
| --- | --- | --- | --- |
| Planner | Qwen/Qwen3-1.7B | HF cache | 1536 |
| Refiner | Llama-3.2-1B-Instruct | ~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master | 2048 |
| Solver | Qwen/Qwen2.5-Math-1.5B-Instruct | HF cache | 1536 |

## Оборудование

- 2x RTX 3090 (cuda:0, cuda:1), 24GB each
- PyTorch 2.9.0+cu128, CUDA 12.9
- 251GB RAM

## Пути

- Чекпоинты: ~/workspace/tmp/recursivemas-original/checkpoints/
  - `inner_planner/`, `inner_refiner/`, `inner_solver/` — inner-loop адаптеры
  - `outer/` — outer-loop RecursiveLink (outer_12.pt, outer_23.pt, outer_31.pt)
- Логи: ~/workspace/tmp/recursivemas-original/logs/
  - `inner_planner.log`, `inner_refiner.log`, `inner_solver.log`
  - `outer.log`
  - `eval_math500.log`
- Результаты: ~/workspace/data/recursivemas-original/results/

## Команды запуска (из директории train/)

```bash
# Inner-loop (для Planner, Refiner, Solver)
python3 train_inner.py \
  --model_name_or_path <MODEL> \
  --mas_design sequential \
  --mas_role <ROLE> \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --max_steps 20000 \
  --batch_size 16 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/inner_<ROLE>
  2>&1 | tee ~/workspace/tmp/recursivemas-original/logs/inner_<ROLE>.log

# Outer-loop (с gradient accumulation, effective batch = batch_size * grad_accum_steps)
python3 train_outer.py \
  --style sequential_light \
  --agent1_model_name_or_path Qwen/Qwen3-1.7B \
  --agent2_model_name_or_path ~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master \
  --agent3_model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --agent1_inner_aligner_path ~/workspace/tmp/recursivemas-original/checkpoints/inner_planner \
  --agent2_inner_aligner_path ~/workspace/tmp/recursivemas-original/checkpoints/inner_refiner \
  --agent3_inner_aligner_path ~/workspace/tmp/recursivemas-original/checkpoints/inner_solver \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --max_steps 20000 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/outer
  2>&1 | tee ~/workspace/tmp/recursivemas-original/logs/outer.log

# Inference
python3 ../inference/run.py \
  --style sequential_light \
  --dataset math500 \
  --device cuda \
  --ckpt_override planner=~/workspace/tmp/recursivemas-original/checkpoints/inner_planner \
  --ckpt_override critic=~/workspace/tmp/recursivemas-original/checkpoints/inner_refiner \
  --ckpt_override solver=~/workspace/tmp/recursivemas-original/checkpoints/inner_solver \
  --ckpt_override outer=~/workspace/tmp/recursivemas-original/checkpoints/outer
```
