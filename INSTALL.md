# INSTALL.md — Установка и развёртывание

## 1. Python и виртуальное окружение

```bash
pyenv local 3.12.13
python3 -m venv ~/workspace/venvs/recursivemas-original/cuda_12_9
source ~/workspace/venvs/recursivemas-original/cuda_12_9/bin/activate
```

## 2. Зависимости

Установить из requirements.txt (версии авторов):

```bash
pip install -r requirements.txt
```

requirements.txt содержит: torch==2.9.0, transformers==5.3.0, datasets==4.4.2, accelerate==1.12.0 и остальные.

## 3. Модели

### Qwen/Qwen3-1.7B (Planner)
Скачивается автоматически при первом запуске через HuggingFace.

### Qwen/Qwen2.5-Math-1.5B-Instruct (Solver)
Скачивается автоматически при первом запуске через HuggingFace.

### Llama-3.2-1B-Instruct (Refiner/Critic)
Gated модель на HF. Скачиваем через ModelScope:

```bash
pip install modelscope
export MODELSCOPE_CACHE=~/workspace/models/modelscope_cache
python3 -c "
from modelscope import snapshot_download
import os
os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/workspace/models/modelscope_cache')
path = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct', cache_dir=os.path.expanduser('~/workspace/models/modelscope_cache'))
print(f'Downloaded to: {path}')
"
```

Путь к модели: ~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master

## 4. Запуск обучения (из директории train/)

Активировать окружение:
```bash
source ~/workspace/venvs/recursivemas-original/cuda_12_9/bin/activate
cd ~/workspace/projects/recursivemas-original/train
```

### Inner-loop: Planner
```bash
python3 train_inner.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --mas_design sequential \
  --mas_role planner \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --max_steps 20000 \
  --batch_size 16 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/inner_planner
```

### Inner-loop: Refiner
```bash
python3 train_inner.py \
  --model_name_or_path ~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master \
  --mas_design sequential \
  --mas_role refiner \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --max_steps 20000 \
  --batch_size 16 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/inner_refiner
```

### Inner-loop: Solver
```bash
python3 train_inner.py \
  --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --mas_design sequential \
  --mas_role solver \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --max_steps 20000 \
  --batch_size 16 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/inner_solver
```

### Outer-loop (effective batch = 2 x 8 = 16, aligned init)

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
  --max_steps 20000 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --align_outer 1 \
  --align_samples 300 \
  --save_dir ~/workspace/tmp/recursivemas-original/checkpoints/outer
```

**Важно:** `--align_outer 1` (по умолчанию) — инициализация outer adapter'ов через ridge regression на скрытых состояниях моделей. Это решает проблему случайной инициализации, которая разрушала семантический сигнал. Для отключения: `--align_outer 0`.

## 5. Evaluation (из директории train/)

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