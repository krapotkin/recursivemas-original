#!/usr/bin/env python3
"""Run single-model baseline (solver only) on GSM8K."""

import os, sys, json, re, torch, argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def make_chat_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(text: str) -> str:
    """Extract final answer from model output."""
    # Try \boxed{X}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    # Try "Answer: X" or "The answer is X"
    m = re.search(r'(?:answer|Answer)[:\s]+([\d.,]+)', text)
    if m:
        return m.group(1).strip()
    # Last number in text
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if nums:
        return nums[-1].strip()
    return ""


def normalize_answer(s: str) -> str:
    s = s.strip().rstrip('.')
    s = re.sub(r'^0+', '', s) if s.replace('.', '').isdigit() else s
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(args.dtype, "auto")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("Model loaded.")

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset("gsm8k", "main", split="test")
    questions = [item["question"] for item in ds]
    # GSM8K answer format: "The answer is X" or "X ### ..."
    answers = []
    for item in ds:
        ans = item.get("answer", "")
        # Extract: "X" from "The answer is X" or "X ### ..."
        m = re.search(r'####\s*(-?\d[\d.,]*)', ans)
        if m:
            answers.append(m.group(1))
        else:
            m = re.search(r'(-?\d[\d.,]*)', ans)
            answers.append(m.group(1) if m else ans)
    print(f"Total test samples: {len(questions)}")

    if args.num_samples > 0:
        questions = questions[:args.num_samples]
        answers = answers[:args.num_samples]

    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(questions), args.batch_size), desc="eval"):
        batch_q = questions[i:i+args.batch_size]
        batch_a = answers[i:i+args.batch_size]

        prompts = [make_chat_prompt(q, tokenizer) for q in batch_q]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        prompt_lens = inputs["input_ids"].shape[1]
        gen_ids = outputs[:, prompt_lens:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for j, text in enumerate(texts):
            pred = extract_answer(text)
            gold = normalize_answer(str(batch_a[j]))
            pred_n = normalize_answer(pred)
            is_correct = pred_n == gold
            if is_correct:
                correct += 1
            total += 1
            results.append({
                "question": batch_q[j],
                "prediction": text.strip(),
                "prediction_parsed": pred,
                "answer_parsed": gold,
                "correct": is_correct,
            })

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[result] accuracy={accuracy:.2f}%")

    out_dir = Path("/home/joefox/workspace/tmp/recursivemas-original/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "single_model_gsm8k.json"
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