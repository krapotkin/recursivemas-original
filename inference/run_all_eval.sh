#!/usr/bin/env bash
# Run all three eval methods on specified dataset in parallel tmux sessions.
# Usage: bash run_all_eval.sh [math500|gsm8k]
set -euo pipefail

DATASET="${1:-gsm8k}"

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/home/joefox/workspace/tmp/recursivemas-original/logs"
VENV_DIR="/home/joefox/workspace/venvs/recursivemas-original/cuda_12_9"
CKPT_DIR="/home/joefox/workspace/tmp/recursivemas-original/checkpoints"
OUTER_DIR="${CKPT_DIR}/outer_1e4"
INNER_PLANNER="${CKPT_DIR}/inner_planner"
INNER_REFINER="${CKPT_DIR}/inner_refiner"
INNER_SOLVER="${CKPT_DIR}/inner_solver"

mkdir -p "$LOG_DIR"

BASE_CLI="--dataset ${DATASET} --device cuda --batch_size 16 --num_samples -1 --seed 42 --sample_seed 42"
BASE_CLI+=" --num_recursive_rounds 3 --num_rollouts 1 --latent_steps 32"
BASE_CLI+=" --max_new_tokens 1000 --temperature 0.6 --top_p 0.95"
BASE_CLI+=" --dtype auto --outer_dtype auto --trust_remote_code 1 --device cuda"
BASE_CLI+=" --enable_thinking 0 --mas_shape chain"
BASE_CLI+=" --agent1_model_name_or_path Qwen/Qwen3-1.7B"
BASE_CLI+=" --agent2_model_name_or_path /home/joefox/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master"
BASE_CLI+=" --agent3_model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct"
BASE_CLI+=" --agent1_inner_aligner_path ${INNER_PLANNER}"
BASE_CLI+=" --agent2_inner_aligner_path ${INNER_REFINER}"
BASE_CLI+=" --agent3_inner_aligner_path ${INNER_SOLVER}"
BASE_CLI+=" --outer_12_path ${OUTER_DIR}/outer_12.pt"
BASE_CLI+=" --outer_23_path ${OUTER_DIR}/outer_23.pt"
BASE_CLI+=" --outer_31_path ${OUTER_DIR}/outer_31.pt"
BASE_CLI+=" --do_sample"
BASE_CLI+=" --inner_adapter_type_fallback ln_res_adapter"
BASE_CLI+=" --outer_adapter_type_fallback outer_ln_res_adapter"

# --- Text MAS ---
TEXTMAS_CLI="${BASE_CLI} --method text"
tmux new -d -s "textmas-${DATASET}" "source ${VENV_DIR}/bin/activate && cd ${BASE_DIR} && python3 -m inference_utils.inference_mas ${TEXTMAS_CLI} 2>&1 | tee ${LOG_DIR}/textmas_${DATASET}.log"
echo "Text MAS (${DATASET}) launched: tmux attach -t textmas-${DATASET}"

# --- RecursiveMAS ---
RECMAS_CLI="${BASE_CLI} --method ours_recursive"
tmux new -d -s "recmas-${DATASET}" "source ${VENV_DIR}/bin/activate && cd ${BASE_DIR} && python3 -m inference_utils.inference_mas ${RECMAS_CLI} 2>&1 | tee ${LOG_DIR}/recmas_${DATASET}.log"
echo "RecursiveMAS (${DATASET}) launched: tmux attach -t recmas-${DATASET}"

# --- Single Model ---
tmux new -d -s "single-${DATASET}" "source ${VENV_DIR}/bin/activate && cd ${BASE_DIR} && python3 run_single_model.py --dataset ${DATASET} --device cuda --batch_size 16 --num_samples -1 --max_new_tokens 1024 --model Qwen/Qwen2.5-Math-1.5B-Instruct 2>&1 | tee ${LOG_DIR}/single_${DATASET}.log"
echo "Single Model (${DATASET}) launched: tmux attach -t single-${DATASET}"

echo ""
echo "=== All sessions launched ==="
echo "Check progress:        tail -f ${LOG_DIR}/single_${DATASET}.log"
echo "                       tail -f ${LOG_DIR}/textmas_${DATASET}.log"
echo "                       tail -f ${LOG_DIR}/recmas_${DATASET}.log"
echo "See result:            grep 'accuracy' ${LOG_DIR}/*_${DATASET}.log"