# TODO — Обучение RecursiveMAS Sequential-Light (с gradient accumulation + aligned init)

## Команды запуска (из директории `train/`)

### 1. Inner-loop: Planner

```bash
source ~/workspace/venvs/recursivemas-original/cuda_12_9/bin/activate
cd ~/workspace/projects/recursivemas-original/train

CUDA_VISIBLE_DEVICES=0 python3 train_inner.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --mas_design sequential \
  --mas_role planner \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --max_steps 20000 \
  --batch_size 16 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/inner_planner \
  2>&1 | tee ~/workspace/tmp/recursivemas-original/logs/inner_planner.log
```

### 2. Inner-loop: Refiner

```bash
CUDA_VISIBLE_DEVICES=1 python3 train_inner.py \
  --model_name_or_path ~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master \
  --mas_design sequential \
  --mas_role refiner \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --max_steps 20000 \
  --batch_size 16 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/inner_refiner \
  2>&1 | tee ~/workspace/tmp/recursivemas-original/logs/inner_refiner.log
```

### 3. Inner-loop: Solver

```bash
CUDA_VISIBLE_DEVICES=1 python3 train_inner.py \
  --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --mas_design sequential \
  --mas_role solver \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --max_steps 20000 \
  --batch_size 16 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/inner_solver \
  2>&1 | tee ~/workspace/tmp/recursivemas-original/logs/inner_solver.log
```

### 4. Outer-loop (effective batch = 2 × 8 = 16, aligned init, LR=1e-4, 5k steps)

```bash
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
  --max_steps 5000 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --align_outer 1 \
  --align_samples 300 \
  --load_bridges ~/workspace/tmp/recursivemas-original/bridges \
  --outer_lr 1e-4 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/outer_1e4 \
  2>&1 | tee ~/workspace/tmp/recursivemas-original/logs/outer_1e4.log
```

> **Статус:** Обучение запущено (tmux: `rmas-outer-1e4`). После завершения автоматически запустится evaluation на Math500.

### 5. Evaluation на Math500

```bash
python3 ../inference/run.py \
  --style sequential_light \
  --dataset math500 \
  --device cuda \
  --ckpt_override planner=~/workspace/tmp/recursivemas-original/checkpoints/inner_planner \
  --ckpt_override critic=~/workspace/tmp/recursivemas-original/checkpoints/inner_refiner \
  --ckpt_override solver=~/workspace/tmp/recursivemas-original/checkpoints/inner_solver \
  --ckpt_override outer=~/workspace/tmp/recursivemas-original/checkpoints/outer
```

## Примечания

- Каждый inner-loop занимает ~N часов, запускай по очереди.
- Outer-loop запустить только после завершения всех трёх inner-loop.
- Evaluation — после outer-loop.
- Ключевое отличие от прошлого прогона: `--grad_accum_steps 8` в outer-loop (effective batch = 16).
- `--align_outer 1` (по умолчанию): инициализация outer adapter'ов через ridge regression на скрытых состояниях. Это решает проблему случайной инициализации, которая разрушала семантический сигнал при передаче между моделями. Отключить через `--align_outer 0`.
