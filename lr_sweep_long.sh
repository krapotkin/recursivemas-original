#!/usr/bin/env bash
# lr_sweep_long.sh — прогоняет обучение с разными LR по 500 шагов
# Usage: bash lr_sweep_long.sh

set -euo pipefail

PROJECT_DIR="$HOME/workspace/projects/recursivemas-original"
VENV_DIR="$HOME/workspace/venvs/recursivemas-original/cuda_12_9"
LOGS_DIR="$HOME/workspace/tmp/recursivemas-original/logs"
BRIDGES_DIR="$HOME/workspace/tmp/recursivemas-original/bridges"
RESULTS_FILE="$LOGS_DIR/lr_sweep_long_results.csv"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$LOGS_DIR"

# Значения LR для теста
LR_VALUES=(1e-4 5e-5 1e-5)

echo "=== LR Sweep Long: ${#LR_VALUES[@]} значений × 500 шагов ==="
echo "Результаты: $RESULTS_FILE"
echo ""

# Заголовок CSV
echo "lr,final_loss,min_loss,max_loss,avg_last50,std_dev,steps" > "$RESULTS_FILE"

for LR in "${LR_VALUES[@]}"; do
    LOG_FILE="$LOGS_DIR/lr_sweep_long_${LR}.log"
    CKPT_DIR="$LOGS_DIR/lr_sweep_long_ckpt_${LR}"
    echo "Starting LR=${LR} at $(date '+%H:%M:%S')"

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 train/train_outer.py \
      --style sequential_light \
      --agent1_model_name_or_path Qwen/Qwen3-1.7B \
      --agent2_model_name_or_path ~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master \
      --agent3_model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
      --agent1_inner_aligner_path ~/workspace/tmp/recursivemas-original/checkpoints/inner_planner \
      --agent2_inner_aligner_path ~/workspace/tmp/recursivemas-original/checkpoints/inner_refiner \
      --agent3_inner_aligner_path ~/workspace/tmp/recursivemas-original/checkpoints/inner_solver \
      --mas_task math \
      --dataset_name RecursiveMAS/Sequential-Math \
      --max_steps 500 \
      --batch_size 2 \
      --grad_accum_steps 8 \
      --align_outer 1 \
      --align_samples 300 \
      --load_bridges "$BRIDGES_DIR" \
      --outer_lr "$LR" \
      --save_dir "$CKPT_DIR" \
      2>&1 | tee "$LOG_FILE"

    # Парсим результаты
    python3 -c "
import re, sys, statistics

with open('$LOG_FILE') as f:
    text = f.read()

matches = re.findall(r'step=(\d+) loss=([0-9.]+)', text)
if not matches:
    print('  NO DATA')
    sys.exit(0)

steps_losses = [(int(s), float(l)) for s, l in matches]
all_losses = [l for _, l in steps_losses]
final_loss = all_losses[-1]
min_loss = min(all_losses)
max_loss = max(all_losses)

# Среднее за последние 50 шагов
last50 = all_losses[-50:] if len(all_losses) >= 50 else all_losses
avg_last50 = sum(last50) / len(last50)

# Стандартное отклонение (стабильность)
std_dev = statistics.stdev(all_losses) if len(all_losses) > 1 else 0.0

print(f'  final={final_loss:.4f}  min={min_loss:.4f}  max={max_loss:.4f}  avg_last50={avg_last50:.4f}  std={std_dev:.4f}')

with open('$RESULTS_FILE', 'a') as f:
    f.write(f'${LR},{final_loss:.6f},{min_loss:.6f},{max_loss:.6f},{avg_last50:.6f},{std_dev:.6f},{len(steps_losses)}\n')
"
    echo "Finished LR=${LR} at $(date '+%H:%M:%S')"
    echo ""
done

echo "=== Результаты ==="
column -t -s',' < "$RESULTS_FILE"