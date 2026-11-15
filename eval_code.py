#!/usr/bin/env python3
"""
Zero-shot greedy pass@1 on HumanEval (164) and MBPP-sanitized test (257), same prompt template as training.

  python eval_code.py --model out_x --tasks humaneval,mbpp
Runs generated code in a subprocess sandbox (see code_utils.py warning). Writes <out>/summary.json + predictions_*.jsonl
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adagradselect_moe import PROMPT
from code_utils import (humaneval_program, humaneval_question, mbpp_program, mbpp_question,
                        run_many)


def load_task(name, max_samples):
    from datasets import load_dataset
    if name == "humaneval":
        items = list(load_dataset("openai/openai_humaneval", split="test"))
        mk_q, mk_prog = humaneval_question, humaneval_program
    else:
        items = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
        mk_q, mk_prog = mbpp_question, mbpp_program
    items = items[:max_samples] if max_samples else items
    return items, mk_q, mk_prog


@torch.inference_mode()
def generate(model, tok, prompts, args):
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outs = [None] * len(prompts)
    for s in range(0, len(order), args.batch_size):
        ids = order[s:s + args.batch_size]
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to(args.device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
        for i, t in zip(ids, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            outs[i] = t.split("\nQuestion:")[0]
        print(f"  generated {min(s + args.batch_size, len(order))}/{len(order)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="humaneval,mbpp")
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
    out_dir = args.out or f"evalcode_{os.path.basename(os.path.normpath(args.model))}"
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(args.device).eval()
    model.config.use_cache = True

    summary = {"model": args.model, "dtype": args.dtype}
    for name in args.tasks.split(","):
        t0 = time.time()
        items, mk_q, mk_prog = load_task(name, args.max_samples)
        print(f"{name}: {len(items)} problems")
        outs = generate(model, tok, [PROMPT.format(q=mk_q(it)) for it in items], args)
        res = run_many([mk_prog(it, o) for it, o in zip(items, outs)])
        n_ok = sum(r[0] for r in res)
        with open(os.path.join(out_dir, f"predictions_{name}.jsonl"), "w") as f:
            for it, o, (ok, msg) in zip(items, outs, res):
                f.write(json.dumps({"task_id": it.get("task_id"), "passed": ok, "msg": msg, "output": o}) + "\n")
        summary[name] = {"pass@1": 100 * n_ok / len(items), "n": len(items), "seconds": time.time() - t0}
        print(f"FINAL {name}: pass@1 {100 * n_ok / len(items):.2f}% ({n_ok}/{len(items)})")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()