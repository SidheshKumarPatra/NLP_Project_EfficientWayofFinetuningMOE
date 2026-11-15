#!/usr/bin/env python3
"""
AdaGradSelect-MoE: expert-level block-selective fine-tuning of MoE LMs (OLMoE first).

Blocks
  L{l}.E{e}   one routed expert            -> bandit ARM
  L{l}.attn   attention + norms (+shared)  -> always trained (--always attn)
  L{l}.router MoE router                   -> always trained (--always router)
  embed/norm/lm_head                       -> always trained (--always other); as arms their
                                              unnormalised grad norm always beat per-token expert scores

Changes vs the sample script / paper
  * freq counts updated every step; exploit resampled every step (also epoch>=1)
  * exploit steps: selection is made BEFORE forward -> unselected blocks get
    requires_grad=False -> no weight grads / no grad memory for them
    (explore steps still need full grads: that is the memory peak)
  * expert score can be normalised by #tokens routed to the expert (--score per_token)
  * optimizer states created lazily, offloaded to pinned CPU, per-block evict/prefetch
  * bf16/fp16 weights updated in fp32 + stochastic rounding (plain low-precision += lr*upd loses updates)
  * --dtype auto: fp16 + dynamic loss scaling on Turing GPUs (no fast bf16), bf16 on Ampere+
  * labels masked on prompt + padding; --batch_size actually used
  * --policy random / gradtopk ablations; selection counts saved for cross-dataset analysis

Not covered yet: resume from checkpoint (optimizer state not saved), multi-GPU, eval.

Usage
  python adagradselect_moe.py --inspect                      # check block mapping first!
  python adagradselect_moe.py --pct 10 --epochs 2            # OLMoE-1B-7B on MetaMathQA-40K
  python adagradselect_moe.py --policy random --pct 10       # ablation
"""
import argparse
import json
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from block_selector import BlockSelector

# ------------------------------------------------------------------ block mapping
_EXPERT = re.compile(r"^model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.")
_FUSED = re.compile(r"^model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.[A-Za-z_]")
_ROUTER = re.compile(r"^model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.gate\.")
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_ARM_EXPERT = re.compile(r"^L(\d+)\.E(\d+)$")


def build_blocks(model):
    blocks = defaultdict(list)
    for name, p in model.named_parameters():
        if _FUSED.match(name):
            raise RuntimeError(
                f"Fused expert tensor '{name}': per-expert selection needs one module per "
                "expert (mlp.experts.N.*). Use a transformers version that keeps experts "
                "as a ModuleList, or add slicing/masking support.")
        if (m := _EXPERT.match(name)):
            key = f"L{m[1]}.E{m[2]}"
        elif (m := _ROUTER.match(name)):
            key = f"L{m[1]}.router"
        elif (m := _LAYER.match(name)):
            key = f"L{m[1]}.attn"
        else:
            key = name
        blocks[key].append(p)
    return dict(blocks)


def split_blocks(blocks, always, freeze=()):
    """always: kinds trained every step; freeze: kinds never trained; everything else = bandit arms."""
    pred = {"attn": lambda k: k.endswith(".attn"), "router": lambda k: k.endswith(".router"),
            "other": lambda k: not re.match(r"^L\d+\.", k)}   # embed / final norm / lm_head
    assert not set(always) & set(freeze), "a kind cannot be both --always and --freeze"
    always_blocks = [k for k in blocks if any(pred[a](k) for a in always)]
    frozen = {k for k in blocks if any(pred[a](k) for a in freeze)}
    s = set(always_blocks) | frozen
    return always_blocks, [k for k in blocks if k not in s]


# ------------------------------------------------------------------ optimizer
def _copy_stochastic(target, src32, drop_bits):
    """fp32 -> bf16 (drop 16 mantissa bits) / fp16 (drop 13) with stochastic rounding.
    Unbiased; keeps tiny updates alive. (fp16 subnormals fall back to nearest rounding.)"""
    r = torch.randint(0, 1 << drop_bits, src32.shape, device=src32.device, dtype=torch.int32)
    r.add_(src32.contiguous().view(torch.int32))
    r.bitwise_and_(-(1 << drop_bits))
    target.copy_(r.view(torch.float32))


class DynamicLossScaler:
    """Minimal dynamic loss scaler for fp16 (disabled -> scale 1.0, used for bf16)."""

    def __init__(self, enabled, init=2.0 ** 14, growth=2.0, backoff=0.5, interval=100):
        self.enabled = enabled
        self.scale = init if enabled else 1.0
        self.growth, self.backoff, self.interval, self._good = growth, backoff, interval, 0

    def update(self, found_inf):
        if not self.enabled:
            return
        if found_inf:
            self.scale = max(self.scale * self.backoff, 1.0)
            self._good = 0
        else:
            self._good += 1
            if self._good >= self.interval:
                self.scale = min(self.scale * self.growth, 2.0 ** 24)
                self._good = 0


@torch.no_grad()
def grads_finite(params):
    flags = [torch.isfinite(p.grad).all() for p in params if p.grad is not None]
    return bool(torch.stack(flags).all().item()) if flags else True


class BlockAdamW(torch.optim.Optimizer):
    """AdamW whose m/v live on pinned CPU and are moved to GPU only for selected params."""

    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 state_dtype=torch.float32, stochastic_round=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.state_dtype = state_dtype
        self.sr = stochastic_round
        self._pin = torch.cuda.is_available()
        self._gpu, self._cpu, self._steps = {}, {}, defaultdict(int)  # keyed by id(p)

    def _evict(self, k):
        m, v = self._gpu.pop(k)
        if k not in self._cpu:
            self._cpu[k] = tuple(torch.empty(m.shape, dtype=m.dtype, device="cpu",
                                             pin_memory=self._pin) for _ in range(2))
        self._cpu[k][0].copy_(m, non_blocking=True)
        self._cpu[k][1].copy_(v, non_blocking=True)

    def _fetch(self, p):
        k = id(p)
        if k in self._gpu:
            return self._gpu[k]
        if k in self._cpu:
            m = self._cpu[k][0].to(p.device, non_blocking=True)
            v = self._cpu[k][1].to(p.device, non_blocking=True)
        else:
            m = torch.zeros_like(p, dtype=self.state_dtype)
            v = torch.zeros_like(p, dtype=self.state_dtype)
        self._gpu[k] = (m, v)
        return m, v

    @torch.no_grad()
    def step(self, params, grad_mult=1.0):
        keep = {id(p) for p in params}
        for k in [k for k in self._gpu if k not in keep]:
            self._evict(k)
        g = self.param_groups[0]
        lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
        for p in params:
            if p.grad is None:       # e.g. expert that received no tokens
                continue
            m, v = self._fetch(p)
            self._steps[id(p)] += 1
            t = self._steps[id(p)]
            grad = p.grad.float() * grad_mult   # unscale (+clip) in fp32
            m32 = m if m.dtype == torch.float32 else m.float()
            v32 = v if v.dtype == torch.float32 else v.float()
            m32.mul_(b1).add_(grad, alpha=1 - b1)
            v32.mul_(b2).addcmul_(grad, grad, value=1 - b2)
            if m32 is not m:
                m.copy_(m32)
                v.copy_(v32)
            bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
            denom = (v32.sqrt() / math.sqrt(bc2)).add_(eps)
            w = p.float()
            if wd > 0 and p.ndim >= 2:
                w.mul_(1 - lr * wd)
            w.addcdiv_(m32, denom, value=-lr / bc1)
            if p.dtype in (torch.bfloat16, torch.float16) and self.sr:
                _copy_stochastic(p, w, 16 if p.dtype == torch.bfloat16 else 13)
            else:
                p.copy_(w)


# ------------------------------------------------------------------ data
PROMPT = "Question: {q}\nAnswer:"


def tokenize_rows(rows, tok, qf, rf, max_len):
    out = []
    for r in rows:
        prompt = PROMPT.format(q=r[qf])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(prompt + " " + r[rf] + tok.eos_token, add_special_tokens=False)["input_ids"][:max_len]
        n_p = min(len(p_ids), len(ids))
        out.append({"input_ids": ids, "labels": [-100] * n_p + ids[n_p:]})
    return out


def make_collate(pad_id):
    def collate(batch):
        L = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            lab[i, :n] = torch.tensor(x["labels"])
            att[i, :n] = 1
        return ids, att, lab
    return collate


def load_rows(args):
    from datasets import load_dataset
    if args.data_files:
        ds = load_dataset("json", data_files=args.data_files, split="train")
    else:
        ds = load_dataset(args.dataset, split="train")
    if getattr(args, "filter_field", None):
        ds = ds.filter(lambda r: r[args.filter_field] == args.filter_value)
    if args.max_samples:
        ds = ds.select(range(min(len(ds), args.max_samples)))
    return ds


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def routed_counts(router_logits, top_k, attn_mask, n_experts):
    """[L, E] tokens routed to each expert (padding excluded). Assumes every layer is MoE."""
    mask = attn_mask.reshape(-1).bool()
    counts = torch.zeros(len(router_logits), n_experts, device=attn_mask.device)
    for l, lg in enumerate(router_logits):
        if lg is None:
            continue
        idx = lg.topk(top_k, dim=-1).indices[mask].reshape(-1)
        counts[l] += torch.bincount(idx, minlength=n_experts).float()
    return counts


@torch.no_grad()
def arm_scores(blocks, arms, counts, arm_l, arm_e, is_exp, per_token, device):
    """Paper Alg.1: block score = sum of per-param grad L2 norms. Optionally / tokens routed."""
    norms = []
    for a in arms:
        s = torch.zeros((), device=device)
        for p in blocks[a]:
            if p.grad is not None:
                s = s + torch.linalg.vector_norm(p.grad, dtype=torch.float32)
        norms.append(s)
    s = torch.stack(norms)
    if per_token and counts is not None:
        c = torch.where(is_exp, counts[arm_l, arm_e], torch.ones_like(s))
        s = s / c.clamp(min=1.0)
    return s.cpu().numpy()


# ------------------------------------------------------------------ training
def train(model, pad_id, examples, args):
    from transformers import get_scheduler
    device = next(model.parameters()).device
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
    top_k = cfg.num_experts_per_tok

    blocks = build_blocks(model)
    always = [] if args.always == "none" else args.always.split(",")
    freeze = [] if args.freeze == "none" else args.freeze.split(",")
    always_blocks, arms = split_blocks(blocks, always, freeze)
    always_set = set(always_blocks)
    all_params = [p for ps in blocks.values() for p in ps]

    # arm -> (layer, expert) lookup for per-token normalisation
    arm_l, arm_e, is_exp = [], [], []
    for a in arms:
        m = _ARM_EXPERT.match(a)
        arm_l.append(int(m[1]) if m else 0)
        arm_e.append(int(m[2]) if m else 0)
        is_exp.append(bool(m))
    arm_l = torch.tensor(arm_l, device=device)
    arm_e = torch.tensor(arm_e, device=device)
    is_exp = torch.tensor(is_exp, device=device)

    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=make_collate(pad_id), generator=g)
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    sel = BlockSelector(arms, args.pct, args.policy, args.eps0, args.eps_min,
                        decay_steps=steps_per_epoch, delta=args.delta, seed=args.seed)
    opt = BlockAdamW(all_params, lr=args.lr, weight_decay=args.weight_decay,
                     state_dtype=torch.bfloat16 if args.state_dtype == "bf16" else torch.float32,
                     stochastic_round=not args.no_sr)
    sched = get_scheduler("cosine", opt, num_warmup_steps=int(0.05 * total_steps),
                          num_training_steps=total_steps)

    print(f"blocks={len(blocks)} arms={len(arms)} always={len(always_blocks)} "
          f"frozen={len(blocks) - len(arms) - len(always_blocks)} "
          f"k={sel.k} ({100 * sel.k / len(arms):.1f}% of arms) "
          f"steps/epoch={steps_per_epoch} total={total_steps}")

    state = {}

    def set_active(names):
        for b, ps in blocks.items():
            want = b in names
            if state.get(b) is not want:
                for p in ps:
                    p.requires_grad_(want)
                state[b] = want

    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    use_counts = args.score == "per_token"
    scaler = DynamicLossScaler(args.dtype == "fp16")
    n_skipped = 0
    step = 0
    for epoch in range(args.epochs):
        it = iter(loader)
        t0 = time.time()
        for _ in range(steps_per_epoch):
            mode, idx = sel.plan(epoch)
            active = (always_set | set(arms)) if mode == "explore" else always_set | {arms[i] for i in idx}
            set_active(active)

            counts = torch.zeros(n_layers, n_exp, device=device) if use_counts else None
            loss_sum = torch.zeros((), device=device)
            for _ in range(args.grad_accum):
                ids, att, lab = (x.to(device) for x in next(it))
                out = model(input_ids=ids, attention_mask=att, labels=lab,
                            output_router_logits=use_counts)
                (out.loss / args.grad_accum * scaler.scale).backward()
                loss_sum += out.loss.detach()
                if use_counts:
                    if getattr(out, "router_logits", None) is None:
                        print("WARNING: no router_logits returned; falling back to --score raw")
                        use_counts, counts = False, None
                    else:
                        counts += routed_counts(out.router_logits, top_k, att, n_exp)

            check = all_params if mode == "explore" else [p for b in active for p in blocks[b]]
            if scaler.enabled and not grads_finite(check):
                scaler.update(True)
                n_skipped += 1
                if n_skipped <= 20 or n_skipped % 50 == 0:
                    print(f"step {step + 1}: non-finite grads -> skipped (total {n_skipped}), "
                          f"loss scale -> {scaler.scale:.0f}")
                sched.step()
                model.zero_grad(set_to_none=True)
                step += 1
                continue
            if mode == "explore":
                idx = sel.pick_from_scores(arm_scores(blocks, arms, counts, arm_l, arm_e,
                                                      is_exp, use_counts, device))
            sel.record(idx)
            upd = always_set | {arms[i] for i in idx}
            params = [p for b in upd for p in blocks[b]]
            mult = 1.0 / scaler.scale
            gp = [p for p in params if p.grad is not None]
            if args.max_grad_norm > 0 and gp:
                tot = torch.stack([torch.linalg.vector_norm(p.grad, dtype=torch.float32)
                                   for p in gp]).norm().item() * mult
                mult *= min(1.0, args.max_grad_norm / (tot + 1e-6))
            opt.step(params, grad_mult=mult)
            scaler.update(False)
            sched.step()
            model.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == 1:
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                print(f"ep {epoch} step {step}/{total_steps} {mode:7s} "
                      f"loss {loss_sum.item() / args.grad_accum:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"eps {sel.eps():.3f} peakmem {mem:.1f}GB scale {scaler.scale:.0f} skipped {n_skipped}")
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s, peak GPU mem {mem:.2f}GB, "
              f"explore steps so far {sel.n_explore}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "selection_stats.json"), "w") as f:
        json.dump({"args": vars(args), **sel.stats()}, f, indent=1)
    return sel


def inspect(model, args):
    blocks = build_blocks(model)
    always = [] if args.always == "none" else args.always.split(",")
    freeze = [] if args.freeze == "none" else args.freeze.split(",")
    always_blocks, arms = split_blocks(blocks, always, freeze)
    sz = lambda names: sum(p.numel() for n in names for p in blocks[n])
    tot = sz(blocks)
    exp = [a for a in arms if _ARM_EXPERT.match(a)]
    k = max(1, round(args.pct / 100 * len(arms)))
    print(f"total params {tot / 1e9:.2f}B in {len(blocks)} blocks")
    print(f"always-trained blocks {len(always_blocks)}: {sz(always_blocks) / 1e6:.0f}M params")
    print(f"arms {len(arms)} (experts {len(exp)}), k={k} at {args.pct}%")
    if exp:
        print(f"params per expert {sz(exp[:1]) / 1e6:.2f}M | all experts {sz(exp) / 1e9:.2f}B")
    for n in [a for a in arms if not _ARM_EXPERT.match(a)]:
        print(f"  other arm: {n} ({sz([n]) / 1e6:.0f}M)")
    sel_p = (sz(always_blocks) + k * (sz(arms) / len(arms))) / 1e9
    print(f"~GPU optimizer states if fp32: {2 * 4 * sel_p:.1f}GB | "
          f"CPU worst case (all blocks touched): {2 * 4 * tot / 1e9:.0f}GB fp32 / "
          f"{2 * 2 * tot / 1e9:.0f}GB bf16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--dataset", default="meta-math/MetaMathQA-40K")
    ap.add_argument("--data_files", default=None, help="local jsonl instead of --dataset")
    ap.add_argument("--query_field", default="query")
    ap.add_argument("--response_field", default="response")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--filter_field", default=None, help="e.g. lang (Magicoder)")
    ap.add_argument("--filter_value", default=None, help="e.g. python")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--pct", type=float, default=10, help="%% of arms updated per step")
    ap.add_argument("--policy", default="bandit", choices=["bandit", "gradtopk", "random"])
    ap.add_argument("--always", default="attn,router,other",
                    help="comma list of attn,router,other (embed/norm/lm_head) or 'none'")
    ap.add_argument("--freeze", default="none",
                    help="kinds (attn,router,other) never trained; use with --always none for experts-only")
    ap.add_argument("--score", default="per_token", choices=["raw", "per_token"])
    ap.add_argument("--eps0", type=float, default=1.0)
    ap.add_argument("--eps_min", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--state_dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--no_sr", action="store_true", help="disable stochastic rounding")
    ap.add_argument("--no_gc", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"],
                    help="auto: bf16 on Ampere+, fp16 (loss scaling) on Turing/older")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--output_dir", default="./out_adagradselect_moe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--inspect", action="store_true", help="print block mapping + memory estimate, exit")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.dtype == "auto":
        ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        args.dtype = "bf16" if ok else "fp16"
    print(f"dtype: {args.dtype}")
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(args.device)
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"non-finite weights after cast to {args.dtype} (overflow): {bad[:5]}")
    model.config.use_cache = False
    if args.inspect:
        inspect(model, args)
        return
    if not args.no_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    rows = load_rows(args)
    examples = tokenize_rows(rows, tok, args.query_field, args.response_field, args.max_len)
    train(model, tok.pad_token_id, examples, args)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()