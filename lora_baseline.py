#!/usr/bin/env python3
"""
LoRA baseline for OLMoE with the SAME data / prompt / masking / schedule / aux-loss as adagradselect_moe.py.
Merges the adapters at the end and saves a plain HF model, so eval_math.py works unchanged.

  pip install peft
  python lora_baseline.py --model ./OLMoE-1B-7B-0924 --data_files metamath40k.jsonl \
      --max_samples 10000 --epochs 1 --rank 128 --lr 2e-4 --output_dir out_lora_r128_s0

--targets attn          : q,k,v,o                      (default, cheap)
--targets attn_experts  : + every expert's gate/up/down (~1.2B adapter params at r=128; check the printout)
LoRA needs a much larger lr than full/selective tuning (1e-4..2e-4); sweep it a little to be fair.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from adagradselect_moe import load_rows, make_collate, tokenize_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--dataset", default="meta-math/MetaMathQA-40K")
    ap.add_argument("--data_files", default=None)
    ap.add_argument("--query_field", default="query")
    ap.add_argument("--response_field", default="response")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--filter_field", default=None, help="e.g. lang (Magicoder)")
    ap.add_argument("--filter_value", default=None, help="e.g. python")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--alpha", type=int, default=None, help="default 2*rank")
    ap.add_argument("--targets", default="attn", choices=["attn", "attn_experts"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    ap.add_argument("--gc", action="store_true", help="enable gradient checkpointing (off by default)")
    ap.add_argument("--no_aux", action="store_true", help="drop router aux loss (default keeps it, like adagradselect per_token)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--output_dir", default="./out_lora")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dtype == "auto":
        ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        args.dtype = "bf16" if ok else "fp16"
    print(f"dtype: {args.dtype}")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(args.device)
    model.config.use_cache = False
    if args.gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.targets == "attn_experts":
        targets += ["gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha or 2 * args.rank, lora_dropout=0.0,
                     target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()            # fp32 adapters
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"trainable LoRA params: {n_tr / 1e6:.1f}M (targets={args.targets}, r={args.rank})")

    rows = load_rows(args)
    examples = tokenize_rows(rows, tok, args.query_field, args.response_field, args.max_len)
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=make_collate(tok.pad_token_id), generator=g)
    steps_per_epoch = len(loader) // args.grad_accum
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_scheduler("cosine", opt, num_warmup_steps=int(0.05 * total), num_training_steps=total)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "fp16"))

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        it = iter(loader)
        for _ in range(steps_per_epoch):
            loss_sum = torch.zeros((), device=args.device)
            for _ in range(args.grad_accum):
                ids, att, lab = (x.to(args.device) for x in next(it))
                out = model(input_ids=ids, attention_mask=att, labels=lab,
                            output_router_logits=not args.no_aux)
                scaler.scale(out.loss / args.grad_accum).backward()
                loss_sum += out.loss.detach()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == 1:
                print(f"ep {epoch} step {step}/{total} loss {loss_sum.item() / args.grad_accum:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} peakmem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB "
                      f"scale {scaler.get_scale():.0f}")
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"done in {secs:.0f}s, peak GPU mem {peak:.2f}GB")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_stats.json"), "w") as f:
        json.dump({"args": vars(args), "trainable_params": n_tr, "seconds": secs, "peak_gb": peak}, f, indent=1)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()