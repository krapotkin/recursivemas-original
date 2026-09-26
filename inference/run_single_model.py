#!/usr/bin/env python3
"""Run single-model baseline (solver only) on GSM8K or Math500.
Uses unified answer parsing from answer_utils.py for fair comparison."""

import os, sys, json, torch, argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from inference_utils.answer_utils import extract_pred_answer, extract_gold_answer, normalize_answer_string

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def make_chat_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="math500", choices=["math500", "gsm8k"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, "auto")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("Model loaded.")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        questions = [item["question"] for item in ds]
        gold_answers = [item["answer"] for item in ds]
    else:  # math500
        ds = load_dataset("huggingfaceh4/math-500", split="test")
        questions = [item["problem"] for item in ds]
        gold_answers = [item["answer"] for item in ds]
    print(f"Total test samples: {len(questions)}")

    if args.num_samples > 0:
        questions = questions[:args.num_samples]
        gold_answers = gold_answers[:args.num_samples]

    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(questions), args.batch_size), desc="eval"):
        batch_q = questions[i:i+args.batch_size]
        batch_gold = gold_answers[i:i+args.batch_size]

        prompts = [make_chat_prompt(q, tokenizer) for q in batch_q]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_lens = inputs["input_ids"].shape[1]
        gen_ids = outputs[:, prompt_lens:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for j, text in enumerate(texts):
            # Use unified parsers from answer_utils
            pred = extract_pred_answer(text) or ""
            gold = extract_gold_answer(batch_gold[j], args.dataset)

            pred_n = normalize_answer_string(pred)
            gold_n = normalize_answer_string(gold)

            is_correct = pred_n == gold_n
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": batch_q[j],
                "prediction": text.strip(),
                "prediction_parsed": pred,
                "gold_raw": batch_gold[j],
                "gold_parsed": gold,
                "pred_normalized": pred_n,
                "gold_normalized": gold_n,
                "correct": is_correct,
            })

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[result] accuracy={accuracy:.2f}% ({correct}/{total})")

    # Save results
    out_dir = Path("/home/joefox/workspace/tmp/recursivemas-original/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"single_model_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({
            "accuracy": accuracy / 100,
            "correct": correct,
            "total": total,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())