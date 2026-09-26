#!/usr/bin/env bash
# lr_sweep.sh — прогоняет обучение с разными LR по 50 шагов и сравнивает loss
# Usage: bash lr_sweep.sh

set -euo pipefail

PROJECT_DIR="$HOME/workspace/projects/recursivemas-original"
VENV_DIR="$HOME/workspace/venvs/recursivemas-original/cuda_12_9"
LOGS_DIR="$HOME/workspace/tmp/recursivemas-original/logs"
BRIDGES_DIR="$HOME/workspace/tmp/recursivemas-original/bridges"
RESULTS_FILE="$LOGS_DIR/lr_sweep_results.csv"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

mkdir -p "$LOGS_DIR"

# Проверим что bridges есть
if [ ! -f "$BRIDGES_DIR/bridge_12.pt" ]; then
    echo "[ERROR] Bridges not found at $BRIDGES_DIR"
    echo "Run: bash compute_bridges.sh first"
    exit 1
fi

# Значения LR для теста
LR_VALUES=(1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6)

echo "=== LR Sweep: ${#LR_VALUES[@]} значений × 50 шагов ==="
echo "Bridges: $BRIDGES_DIR"
echo "Результаты: $RESULTS_FILE"
echo ""

# Заголовок CSV
echo "lr,final_loss,avg_last10,min_loss,max_loss,steps" > "$RESULTS_FILE"

for LR in "${LR_VALUES[@]}"; do
    LOG_FILE="$LOGS_DIR/lr_sweep_${LR}.log"
    CKPT_DIR="$LOGS_DIR/lr_sweep_ckpt_${LR}"
    echo -n "LR=${LR} ... "

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
      --max_steps 50 \
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
import re, sys

with open('$LOG_FILE') as f:
    text = f.read()

matches = re.findall(r'step=(\d+) loss=([0-9.]+)', text)
if not matches:
    print('  NO DATA')
    sys.exit(0)

steps_losses = [(int(s), float(l)) for s, l in matches]
final_loss = steps_losses[-1][1]
all_losses = [l for _, l in steps_losses]
last10 = all_losses[-10:] if len(all_losses) >= 10 else all_losses
avg_last10 = sum(last10) / len(last10)
min_loss = min(all_losses)
max_loss = max(all_losses)
total_steps = steps_losses[-1][0]

print(f'  final={final_loss:.4f}  avg_last10={avg_last10:.4f}  min={min_loss:.4f}  max={max_loss:.4f}')

with open('$RESULTS_FILE', 'a') as f:
    f.write(f'${LR},{final_loss:.6f},{avg_last10:.6f},{min_loss:.6f},{max_loss:.6f},{total_steps}\n')
"
done

echo ""
echo "=== Результаты ==="
column -t -s',' < "$RESULTS_FILE"
echo ""
echo "Полный лог каждого прогона: $LOGS_DIR/lr_sweep_*.log"