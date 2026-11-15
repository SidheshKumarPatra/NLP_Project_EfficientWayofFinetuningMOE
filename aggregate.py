#!/usr/bin/env python3
"""
Build all tables/figures from results/ (GSM8K) and results_code/ (HumanEval+MBPP). CPU only, reads files.
  python aggregate.py            -> prints tables, writes report_out/tables.md and report_out/*.png
Needs numpy (+ matplotlib for figures: pip install matplotlib; scipy optional for p-values).
"""
import collections
import glob
import json
import math
import os
import re
import statistics as st

import numpy as np

OUT = "report_out"
PRETTY = {
    "lora_r128": "LoRA r=128 (attn)", "noexp": "No experts (attn+router+embed/head)",
    "bandit_eo_raw": "Experts-only, bandit raw", "bandit_eo": "Experts-only, bandit per-token",
    "random_eo": "Experts-only, random", "bandit10": "Bandit 10% + attn (per-token)",
    "bandit10raw": "Bandit 10% + attn (raw)", "random10": "Random 10% + attn",
    "gradtopk10": "Grad top-k 10% + attn", "bandit20": "Bandit 20% + attn", "bandit10raw_": "",
}


def parse(name):
    m = re.match(r"^(.*)_s(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else float("nan"), len(v))


def welch(a, b):
    """mean(a)-mean(b), Welch t, p (if scipy available)."""
    if len(a) < 2 or len(b) < 2:
        return None
    d = st.mean(a) - st.mean(b)
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    t = d / se if se > 0 else float("inf")
    p = None
    try:
        from scipy import stats
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
    except Exception:
        pass
    return d, t, p


def load_scores():
    gsm, code = collections.defaultdict(dict), collections.defaultdict(dict)
    for f in glob.glob("results/*/eval_gsm8k/summary.json"):
        meth, seed = parse(f.split("/")[1])
        gsm[meth][seed] = json.load(open(f))["accuracy"]
    for f in glob.glob("results_code/*/eval/summary.json"):
        meth, seed = parse(f.split("/")[1])
        d = json.load(open(f))
        he, mb = d["humaneval"], d["mbpp"]
        comb = (he["pass@1"] * he["n"] + mb["pass@1"] * mb["n"]) / (he["n"] + mb["n"])
        code[meth][seed] = (he["pass@1"], mb["pass@1"], comb)
    return gsm, code


def cost_from_logs(name, code=False):
    cands = [f"logs_code/train_{name}.log"] if code else [f"logs/train_{name}.log", f"train_{name}.log"]
    for p in cands:
        if not os.path.exists(p):
            continue
        txt = open(p, errors="ignore").read()
        m = re.findall(r"done in (\d+)s, peak GPU mem ([\d.]+)GB", txt)
        tot = re.findall(r"step \d+/(\d+)", txt)
        if m and tot:
            return int(m[-1][0]) / int(tot[-1]), float(m[-1][1])
    return None


def table(rows, header):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def fmt(m, s, n):
    return f"{m:.2f} ± {s:.2f} (n={n})" if n > 1 else f"{m:.2f} (n=1)"


def freq(path):
    f = json.load(open(path))["freq"]
    arms = sorted(f)
    return arms, np.array([f[a] for a in arms], float)


def jaccard(a, b, k):
    A, B = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(A & B) / len(A | B)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    os.makedirs(OUT, exist_ok=True)
    gsm, code = load_scores()
    md = []

    # ---- math table
    rows = []
    for meth, d in sorted(gsm.items(), key=lambda kv: -st.mean(kv[1].values())):
        m, s, n = ms(list(d.values()))
        c = cost_from_logs(f"{meth}_s0")
        rows.append([PRETTY.get(meth, meth), fmt(m, s, n), ", ".join(f"{d[k]:.1f}" for k in sorted(d)),
                     f"{c[0]:.2f}" if c else "-", f"{c[1]:.1f}" if c else "-"])
    md += ["## GSM8K (full 1319, zero-shot greedy; MetaMath 10K samples, 1 epoch)\n",
           table(rows, ["Method", "Accuracy %", "per seed", "s/step (seed0)", "peak GB"])]

    # ---- code table
    rows = []
    for meth, d in sorted(code.items(), key=lambda kv: -st.mean(v[2] for v in kv[1].values())):
        he, mb, cb = ([v[i] for v in d.values()] for i in range(3))
        c = cost_from_logs(f"{meth}_s0", code=True)
        rows.append([PRETTY.get(meth, meth), fmt(*ms(he)), fmt(*ms(mb)), fmt(*ms(cb)),
                     f"{c[0]:.2f}" if c else "-", f"{c[1]:.1f}" if c else "-"])
    md += ["\n## Code (Magicoder python 10K, 1 epoch): HumanEval (164) / MBPP-sanitized (257) pass@1 %\n",
           table(rows, ["Method", "HumanEval", "MBPP", "combined (421)", "s/step (seed0)", "peak GB"])]

    # ---- key comparisons
    comps = [("gsm8k", gsm, "bandit_eo_raw", "random_eo"), ("gsm8k", gsm, "bandit_eo_raw", "bandit_eo"),
             ("gsm8k", gsm, "bandit10", "random10"), ("gsm8k", gsm, "bandit_eo_raw", "noexp"),
             ("gsm8k", gsm, "noexp", "lora_r128")]
    rows = []
    for dom, src, a, b in comps:
        if a in src and b in src:
            r = welch(list(src[a].values()), list(src[b].values()))
            if r:
                rows.append([dom, f"{PRETTY.get(a, a)} − {PRETTY.get(b, b)}", f"{r[0]:+.2f}", f"{r[1]:.2f}",
                             "-" if r[2] is None else f"{r[2]:.3f}"])
    for a, b in [("bandit_eo_raw", "random_eo"), ("noexp", "lora_r128"), ("bandit_eo_raw", "lora_r128")]:
        if a in code and b in code:
            for i, nm in enumerate(["HumanEval", "MBPP", "combined"]):
                r = welch([v[i] for v in code[a].values()], [v[i] for v in code[b].values()])
                if r:
                    rows.append([f"code/{nm}", f"{PRETTY.get(a, a)} − {PRETTY.get(b, b)}", f"{r[0]:+.2f}",
                                 f"{r[1]:.2f}", "-" if r[2] is None else f"{r[2]:.3f}"])
    md += ["\n## Seed-level comparisons (Welch; 3 seeds => low power)\n",
           table(rows, ["domain", "difference", "Δ (points)", "t", "p"])]

    # ---- expert-selection analysis (bandit runs)
    sel = collections.defaultdict(dict)
    for dom, pat in (("math", "results/{m}_s*/selection_stats.json"), ("code", "results_code/{m}_s*/selection_stats.json")):
        for meth in ("bandit_eo_raw", "bandit10raw", "bandit_eo", "bandit10"):
            for f in sorted(glob.glob(pat.format(m=meth))):
                sel[(meth, dom)][parse(f.split("/")[1])[1]] = freq(f)
    rows, layer = [], {}
    for meth in ("bandit_eo_raw", "bandit10raw", "bandit_eo", "bandit10"):
        k = 102
        M, C = sel.get((meth, "math"), {}), sel.get((meth, "code"), {})
        def pairs(x, y, same):
            out = []
            for i in x:
                for j in y:
                    if same and j <= i:
                        continue
                    if x[i][0] == y[j][0]:
                        out.append((jaccard(x[i][1], y[j][1], k), spearman(x[i][1], y[j][1])))
            return out
        for label, p in (("math vs math (diff seeds)", pairs(M, M, True)), ("code vs code (diff seeds)", pairs(C, C, True)),
                         ("math vs code", pairs(M, C, False))):
            if p:
                rows.append([PRETTY.get(meth, meth), label, f"{np.mean([a for a, _ in p]):.3f}",
                             f"{np.mean([b for _, b in p]):.3f}", len(p)])
        for dom, D in (("math", M), ("code", C)):
            if D:
                arms = next(iter(D.values()))[0]
                v = np.mean([x[1] for x in D.values()], axis=0)
                lay = collections.Counter()
                for a, c in zip(arms, v):
                    if a.startswith("L") and ".E" in a:
                        lay[int(a[1:].split(".")[0])] += c
                layer[(meth, dom)] = [lay[i] for i in range(max(lay) + 1)] if lay else []
    md += ["\n## Which experts does the bandit pick? (top-102 Jaccard; chance ≈ 0.052; Spearman of pick counts)\n",
           table(rows, ["run", "comparison", "Jaccard@102", "Spearman", "#pairs"])] if rows else []

    text = "\n".join(md)
    print(text)
    open(os.path.join(OUT, "tables.md"), "w").write(text)

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(pip install matplotlib for figures)")
        return

    def bars(data, title, fn, ylabel):
        names = sorted(data, key=lambda k: -st.mean(data[k]))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, n in enumerate(names):
            v = data[n]
            ax.bar(i, st.mean(v), yerr=(st.stdev(v) if len(v) > 1 else 0), capsize=4, alpha=0.75)
            ax.scatter([i] * len(v), v, color="k", s=12, zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([PRETTY.get(n, n) for n in names], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        lo = min(min(v) for v in data.values())
        ax.set_ylim(max(0, lo - 4), None)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fn), dpi=150)
        plt.close(fig)

    if gsm:
        bars({k: list(v.values()) for k, v in gsm.items()}, "GSM8K accuracy (mean ± sd, dots = seeds)", "fig_gsm8k.png", "accuracy %")
    if code:
        bars({k: [x[2] for x in v.values()] for k, v in code.items()}, "Code pass@1, HumanEval+MBPP combined", "fig_code.png", "pass@1 %")
    if layer:
        fig, ax = plt.subplots(figsize=(9, 4))
        for (meth, dom), v in layer.items():
            if meth in ("bandit_eo_raw", "bandit10raw") and v:
                ax.plot(range(len(v)), np.array(v) / sum(v), marker="o", label=f"{PRETTY.get(meth, meth)} / {dom}")
        ax.axhline(1 / 16, color="gray", ls="--", label="uniform")
        ax.set_xlabel("layer")
        ax.set_ylabel("share of expert picks")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_layers.png"), dpi=150)
        plt.close(fig)
    print(f"\nwrote {OUT}/tables.md and figures")


if __name__ == "__main__":
    main()