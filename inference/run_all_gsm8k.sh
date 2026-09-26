#!/bin/bash
#==============================================================================
# Evaluate all RecursiveMAS baselines on GSM8K
# Usage: source ~/workspace/venvs/recursivemas-original/cuda_12_9/bin/activate
#        bash run_all_gsm8k.sh
#
# Each eval runs in its own tmux session; results are logged and saved.
#==============================================================================
set -e

BASE_DIR="/home/joefox/workspace/projects/recursivemas-original"
VENV_DIR="$HOME/workspace/venvs/recursivemas-original/cuda_12_9"
CKPT_DIR="/home/joefox/workspace/tmp/recursivemas-original/checkpoints"
LOG_DIR="/home/joefox/workspace/tmp/recursivemas-original/logs"
RES_DIR="/home/joefox/workspace/tmp/recursivemas-original/results"

mkdir -p "$LOG_DIR" "$RES_DIR"

ACTIVATE="source $VENV_DIR/bin/activate && cd $BASE_DIR/inference"
DEVICE="cuda"
BATCH=16

# ==============================================================================
# 1. Single Model (Qwen2.5-Math-1.5B-Instruct, baseline)
# ==============================================================================
echo "Launching 1/3 — Single Model on GSM8K..."
tmux new -d -s single-gsm8k "$ACTIVATE && python3 run_single_model.py \
  --dataset gsm8k --device $DEVICE --batch_size $BATCH --num_samples -1 --max_new_tokens 1024 \
  --model Qwen/Qwen2.5-Math-1.5B-Instruct \
  2>&1 | tee $LOG_DIR/single_gsm8k.log"

# ==============================================================================
# 2. Text MAS (3 models, textual communication, no latent)
# ==============================================================================
echo "Launching 2/3 — Text MAS on GSM8K..."
tmux new -d -s textmas-gsm8k "$ACTIVATE && python3 -m inference_utils.inference_mas \
  --dataset gsm8k --dataset_split test \
  --num_samples -1 --seed 42 --sample_seed 42 \
  --num_recursive_rounds 3 --num_rollouts 1 \
  --batch_size $BATCH --latent_steps 32 \
  --max_new_tokens 1000 --temperature 0.6 --top_p 0.95 --top_k -1 \
  --ans_max_new_tokens -1 --mbppplus_timeout_s 10 --mbppplus_num_prompt_tests 3 \
  --dtype auto --outer_dtype auto --trust_remote_code 1 --device $DEVICE \
  --enable_thinking 0 \
  --mas_shape chain \
  --agent1_model_name_or_path Qwen/Qwen3-1.7B \
  --agent2_model_name_or_path ~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master \
  --agent3_model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --agent1_inner_aligner_path $CKPT_DIR/inner_planner \
  --agent2_inner_aligner_path $CKPT_DIR/inner_refiner \
  --agent3_inner_aligner_path $CKPT_DIR/inner_solver \
  --outer_12_path $CKPT_DIR/outer_1e4/outer_12.pt \
  --outer_23_path $CKPT_DIR/outer_1e4/outer_23.pt \
  --outer_31_path $CKPT_DIR/outer_1e4/outer_31.pt \
  --method text \
  --do_sample \
  --solver_pre_question 0 \
  --inner_adapter_type_fallback ln_res_adapter \
  --outer_adapter_type_fallback outer_ln_res_adapter \
  2>&1 | tee $LOG_DIR/textmas_gsm8k.log"

# ==============================================================================
# 3. RecursiveMAS (latent communication, our trained system)
# ==============================================================================
echo "Launching 3/3 — RecursiveMAS on GSM8K..."
tmux new -d -s recmas-gsm8k "$ACTIVATE && python3 -m inference_utils.inference_mas \
  --dataset gsm8k --dataset_split test \
  --num_samples -1 --seed 42 --sample_seed 42 \
  --num_recursive_rounds 3 --num_rollouts 1 \
  --batch_size $BATCH --latent_steps 32 \
  --max_new_tokens 1000 --temperature 0.6 --top_p 0.95 --top_k -1 \
  --ans_max_new_tokens -1 --mbppplus_timeout_s 10 --mbppplus_num_prompt_tests 3 \
  --dtype auto --outer_dtype auto --trust_remote_code 1 --device $DEVICE \
  --enable_thinking 0 \
  --mas_shape chain \
  --agent1_model_name_or_path Qwen/Qwen3-1.7B \
  --agent2_model_name_or_path ~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master \
  --agent3_model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --agent1_inner_aligner_path $CKPT_DIR/inner_planner \
  --agent2_inner_aligner_path $CKPT_DIR/inner_refiner \
  --agent3_inner_aligner_path $CKPT_DIR/inner_solver \
  --outer_12_path $CKPT_DIR/outer_1e4/outer_12.pt \
  --outer_23_path $CKPT_DIR/outer_1e4/outer_23.pt \
  --outer_31_path $CKPT_DIR/outer_1e4/outer_31.pt \
  --method ours_recursive \
  --do_sample \
  --solver_pre_question 0 \
  --inner_adapter_type_fallback ln_res_adapter \
  --outer_adapter_type_fallback outer_ln_res_adapter \
  2>&1 | tee $LOG_DIR/recmas_gsm8k.log"

echo ""
echo "All 3 evals launched! Monitor with:"
echo "  tmux attach -t single-gsm8k"
echo "  tmux attach -t textmas-gsm8k"
echo "  tmux attach -t recmas-gsm8k"
echo ""
echo "Or tail logs:"
echo "  tail -f $LOG_DIR/single_gsm8k.log"
echo "  tail -f $LOG_DIR/textmas_gsm8k.log"
echo "  tail -f $LOG_DIR/recmas_gsm8k.log"