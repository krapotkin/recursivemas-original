#!/usr/bin/env python3
"""Run RecursiveMAS (latent communication) evaluation with trained outer adapters.

Usage:
  # Math500
  python3 run_recursive_mas.py --dataset math500

  # GSM8K
  python3 run_recursive_mas.py --dataset gsm8k
"""

import os
import sys
import argparse
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CHECKPOINTS = Path.home() / "workspace/tmp/recursivemas-original/checkpoints"
OUTER_DIR = CHECKPOINTS / "outer_1e4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math500", choices=["math500", "gsm8k"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--outer_dtype", default="auto")
    parser.add_argument("--inner_adapter_type", default="ln_res_adapter")
    parser.add_argument("--outer_adapter_type", default="outer_ln_res_adapter")
    args = parser.parse_args()

    cli = [
        sys.executable, "-m", "inference_utils.inference_mas",
        "--dataset", args.dataset,
        "--dataset_split", "test",
        "--num_samples", str(args.num_samples),
        "--seed", "42",
        "--sample_seed", "42",
        "--num_recursive_rounds", "3",
        "--num_rollouts", "1",
        "--batch_size", str(args.batch_size),
        "--latent_steps", "32",
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--top_k", "-1",
        "--ans_max_new_tokens", "-1",
        "--mbppplus_timeout_s", "10",
        "--mbppplus_num_prompt_tests", "3",
        "--dtype", args.dtype,
        "--outer_dtype", args.outer_dtype,
        "--trust_remote_code", "1",
        "--device", args.device,
        "--enable_thinking", "0",
        "--mas_shape", "chain",
        "--agent1_model_name_or_path", "Qwen/Qwen3-1.7B",
        "--agent2_model_name_or_path",
        str(Path.home() / "workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master"),
        "--agent3_model_name_or_path", "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "--agent1_inner_aligner_path", str(CHECKPOINTS / "inner_planner"),
        "--agent2_inner_aligner_path", str(CHECKPOINTS / "inner_refiner"),
        "--agent3_inner_aligner_path", str(CHECKPOINTS / "inner_solver"),
        "--outer_12_path", str(OUTER_DIR / "outer_12.pt"),
        "--outer_23_path", str(OUTER_DIR / "outer_23.pt"),
        "--outer_31_path", str(OUTER_DIR / "outer_31.pt"),
        "--method", "ours_recursive",
        "--do_sample",
        "--inner_adapter_type_fallback", args.inner_adapter_type,
        "--outer_adapter_type_fallback", args.outer_adapter_type,
        "--solver_pre_question", "0",
    ]

    print(f"Running RecursiveMAS on {args.dataset}...")
    os.chdir(str(THIS_DIR))
    os.execvp(sys.executable, cli)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())