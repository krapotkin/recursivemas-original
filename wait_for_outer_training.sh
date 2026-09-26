#!/usr/bin/env bash
# wait_for_outer_training.sh — ждёт завершения outer-loop обучения и запускает evaluation
# Usage: bash wait_for_outer_training.sh

set -euo pipefail

LOG_FILE="$HOME/workspace/tmp/recursivemas-original/logs/outer_1e4.log"
CKPT_DIR="$HOME/workspace/tmp/recursivemas-original/checkpoints/outer_1e4"
PROJECT_DIR="$HOME/workspace/projects/recursivemas-original"
VENV_DIR="$HOME/workspace/venvs/recursivemas-original/cuda_12_9"
EVAL_LOG="$HOME/workspace/tmp/recursivemas-original/logs/eval_math500_aligned.log"

echo "[wait] Ожидание завершения обучения..."

# Ждём пока в логе появится "Training complete" или процесс завершится
while true; do
    # Проверим что процесс ещё жив
    if ! pgrep -f "python3 train_outer.py" > /dev/null 2>&1; then
        echo "[wait] Процесс обучения завершён"
        break
    fi
    
    # Проверим лог на завершение
    if grep -q "Training complete\|training complete\|DONE\|Finished training" "$LOG_FILE" 2>/dev/null; then
        echo "[wait] Обучение завершено (по логу)"
        break
    fi
    
    sleep 30
    echo "[wait] $(date '+%H:%M:%S') — обучение ещё идёт..."
done

# Дадим процессу окончательно завершиться
sleep 5

# Проверим последний шаг
echo "[wait] Последний шаг в логе:"
grep "^step=" "$LOG_FILE" | tail -5

# Запускаем evaluation
echo ""
echo "[eval] Запуск evaluation на Math500..."
echo "[eval] Лог: $EVAL_LOG"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 inference/run.py \
  --style sequential_light \
  --dataset math500 \
  --ckpt_override planner="$HOME/workspace/tmp/recursivemas-original/checkpoints/inner_planner" \
  --ckpt_override critic="$HOME/workspace/tmp/recursivemas-original/checkpoints/inner_refiner" \
  --ckpt_override solver="$HOME/workspace/tmp/recursivemas-original/checkpoints/inner_solver" \
  --ckpt_override outer="$CKPT_DIR" \
  2>&1 | tee "$EVAL_LOG"

echo ""
echo "[eval] Evaluation завершена!"
echo "[eval] Результат:"
tail -20 "$EVAL_LOG" | grep -i "accuracy\|pass@" || echo "(результат не найден в последних строках)"