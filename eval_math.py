#!/usr/bin/env python3
"""
Zero-shot greedy eval for GSM8K / MATH-500 with the SAME prompt used in training.

  python eval_math.py --model out_bandit10 --task gsm8k
  python eval_math.py --model ./OLMoE-1B-7B-0924 --task gsm8k      # base-model reference
  python eval_math.py --model out_bandit10 --task math500

Writes <out>/summary.json and <out>/predictions.jsonl.
MATH answer matching is string/number based after light LaTeX normalisation (approximate).
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adagradselect_moe import PROMPT
from answer_utils import extract_pred, gsm_correct, gsm_gold, math_correct


def load_task(task, max_samples):
    from datasets import load_dataset
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [(r["question"], gsm_gold(r["answer"])) for r in ds]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [(r["problem"], r["answer"]) for r in ds]
    return rows[:max_samples] if max_samples else rows


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dtype == "auto":
        ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        args.dtype = "bf16" if ok else "fp16"
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    out_dir = args.out or f"eval_{os.path.basename(os.path.normpath(args.model))}_{args.task}"
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(args.device).eval()
    model.config.use_cache = True

    rows = load_task(args.task, args.max_samples)
    prompts = [PROMPT.format(q=q) for q, _ in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))   # similar lengths per batch
    check = gsm_correct if args.task == "gsm8k" else math_correct
    results, n_ok, t0 = [None] * len(rows), 0, time.time()

    for s in range(0, len(order), args.batch_size):
        ids = order[s:s + args.batch_size]
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to(args.device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             use_cache=True)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(ids, texts):
            t = t.split("\nQuestion:")[0]            # base model may keep going
            pred = extract_pred(t)
            ok = bool(check(pred, rows[i][1]))
            n_ok += ok
            results[i] = {"question": rows[i][0], "gold": rows[i][1], "pred": pred,
                          "correct": ok, "output": t}
        done = min(s + args.batch_size, len(order))
        print(f"{done}/{len(order)} acc so far {100 * sum(r['correct'] for r in results if r) / done:.2f}% "
              f"({time.time() - t0:.0f}s)", flush=True)

    acc = 100 * n_ok / len(rows)
    summary = {"model": args.model, "task": args.task, "n": len(rows), "accuracy": acc,
               "dtype": args.dtype, "max_new_tokens": args.max_new_tokens,
               "seconds": time.time() - t0}
    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"FINAL {args.task}: {acc:.2f}% on {len(rows)} examples -> {out_dir}")


if __name__ == "__main__":
    main()