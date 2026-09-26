#!/usr/bin/env python3
"""Compute and save bridge matrices for LR sweep (avoids recomputing each run)."""
import os, sys, random, torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "train"))

from outer.common import (
    load_model_and_tokenizer,
    load_inner_adapter,
    collect_hidden_pairs,
    compute_procrustes_bridge,
)

def main():
    device = torch.device("cuda:0")
    model_dtype = torch.bfloat16
    outer_dtype = torch.float32

    # Load models
    planner_model, planner_tok = load_model_and_tokenizer(
        "Qwen/Qwen3-1.7B", device, model_dtype, True, "planner", False
    )
    refiner_model, refiner_tok = load_model_and_tokenizer(
        os.path.expanduser("~/workspace/models/modelscope_cache/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master"),
        device, model_dtype, True, "refiner", False
    )
    solver_model, solver_tok = load_model_and_tokenizer(
        "Qwen/Qwen2.5-Math-1.5B-Instruct", device, model_dtype, True, "solver", False
    )

    # Load inner adapters
    inner_1 = load_inner_adapter(
        os.path.expanduser("~/workspace/tmp/recursivemas-original/checkpoints/inner_planner"),
        2048, device, model_dtype, "res_adapter"  # checkpoint says hidden=2048
    )
    inner_2 = load_inner_adapter(
        os.path.expanduser("~/workspace/tmp/recursivemas-original/checkpoints/inner_refiner"),
        2048, device, model_dtype, "res_adapter"  # checkpoint says hidden=2048
    )
    inner_3 = load_inner_adapter(
        os.path.expanduser("~/workspace/tmp/recursivemas-original/checkpoints/inner_solver"),
        1536, device, model_dtype, "res_adapter"  # checkpoint says hidden=1536
    )

    # Load dataset
    from outer.sequential import load_outer_training_dataset
    dataset = load_outer_training_dataset("RecursiveMAS/Sequential-Math", "train", "data")
    dataset = dataset.shuffle(seed=42)

    align_n = min(300, len(dataset))
    align_rows = [dataset[i] for i in range(align_n)]
    questions  = [str(r.get("question", "")).strip() for r in align_rows]
    plans      = [str(r.get("plan", "")).strip() for r in align_rows]
    refined    = [str(r.get("refined_plan", "")).strip() for r in align_rows]
    answers    = [str(r.get("answer", "")).strip() for r in align_rows]

    valid = [i for i in range(len(questions)) if questions[i] and plans[i] and refined[i] and answers[i]]
    questions  = [questions[i] for i in valid]
    plans      = [plans[i] for i in valid]
    refined    = [refined[i] for i in valid]
    answers    = [answers[i] for i in valid]
    print(f"[bridge] {len(questions)} valid samples")

    # Compute bridges
    bridges_dir = os.path.expanduser("~/workspace/tmp/recursivemas-original/bridges")
    os.makedirs(bridges_dir, exist_ok=True)

    # bridge_12: planner -> refiner
    print("[bridge] Computing bridge_12 (planner -> refiner)...")
    H_src, H_tgt = collect_hidden_pairs(
        planner_model, refiner_model, planner_tok, refiner_tok,
        inner_1, inner_2, questions, plans, device, model_dtype, 4096,
    )
    bridge_12 = compute_procrustes_bridge(H_src, H_tgt)
    src_norm = H_src.norm(dim=1, keepdim=True).mean().item()
    tgt_norm = H_tgt.norm(dim=1, keepdim=True).mean().item()
    if src_norm > 1e-8:
        bridge_12 = bridge_12 * (tgt_norm / src_norm)
    torch.save(bridge_12.cpu(), os.path.join(bridges_dir, "bridge_12.pt"))
    print(f"  shape={tuple(bridge_12.shape)}  src_norm={src_norm:.2f}  tgt_norm={tgt_norm:.2f}")

    # bridge_23: refiner -> solver
    print("[bridge] Computing bridge_23 (refiner -> solver)...")
    H_src, H_tgt = collect_hidden_pairs(
        refiner_model, solver_model, refiner_tok, solver_tok,
        inner_2, inner_3, plans, refined, device, model_dtype, 4096,
    )
    bridge_23 = compute_procrustes_bridge(H_src, H_tgt)
    src_norm = H_src.norm(dim=1, keepdim=True).mean().item()
    tgt_norm = H_tgt.norm(dim=1, keepdim=True).mean().item()
    if src_norm > 1e-8:
        bridge_23 = bridge_23 * (tgt_norm / src_norm)
    torch.save(bridge_23.cpu(), os.path.join(bridges_dir, "bridge_23.pt"))
    print(f"  shape={tuple(bridge_23.shape)}  src_norm={src_norm:.2f}  tgt_norm={tgt_norm:.2f}")

    # bridge_31: solver -> planner
    print("[bridge] Computing bridge_31 (solver -> planner)...")
    H_src, H_tgt = collect_hidden_pairs(
        solver_model, planner_model, solver_tok, planner_tok,
        inner_3, inner_1, answers, questions, device, model_dtype, 4096,
    )
    bridge_31 = compute_procrustes_bridge(H_src, H_tgt)
    src_norm = H_src.norm(dim=1, keepdim=True).mean().item()
    tgt_norm = H_tgt.norm(dim=1, keepdim=True).mean().item()
    if src_norm > 1e-8:
        bridge_31 = bridge_31 * (tgt_norm / src_norm)
    torch.save(bridge_31.cpu(), os.path.join(bridges_dir, "bridge_31.pt"))
    print(f"  shape={tuple(bridge_31.shape)}  src_norm={src_norm:.2f}  tgt_norm={tgt_norm:.2f}")

    print(f"\n[Bridge] Saved to {bridges_dir}/")
    print(f"  Use --load_bridges {bridges_dir} to skip alignment phase")

if __name__ == "__main__":
    main()