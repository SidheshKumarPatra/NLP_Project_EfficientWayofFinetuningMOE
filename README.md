# AdaGradSelect-MoE: Block-Selective Fine-Tuning for Mixture-of-Experts

This repository implements **AdaGradSelect-MoE**, an adaptive, parameter-efficient fine-tuning framework for Mixture-of-Experts (MoE) Large Language Models (evaluated on [OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924)).

By treating routed expert modules as arms in an $\epsilon$-decay multi-armed bandit, AdaGradSelect-MoE dynamically identifies and fine-tunes only the top $k\%$ most task-salient experts while keeping unselected experts frozen.

---

## Key Highlights

- **Dynamic Expert Allocation**: Balances exploration ($\epsilon$-greedy via gradient scores/norms) and exploitation (Dirichlet sampling from historical selection counts).
- **Zero-Grad Memory Optimization**: On exploitation steps, unselected experts are frozen *before* the forward pass (`requires_grad = False`), eliminating gradient tracking and drastically lowering peak memory.
- **Offloaded Lazy Optimizer**: Per-block parameter updates with AdamW optimizer states offloaded to pinned host memory (CPU), preventing GPU VRAM exhaustion during training.
- **Cross-Domain Specialization**: The bandit autonomously isolates domain-specific expert sub-networks:
  - Within-domain expert selection Jaccard similarity across random seeds: **> 0.91**
  - Cross-domain (Math vs. Code) expert overlap drops to **~0.33**, demonstrating domain-specific expert specialization.

---

## Empirical Benchmark Results

### 1. GSM8K Mathematics Benchmark
*Trained on 10K MetaMathQA samples (1 epoch), evaluated zero-shot greedy on full GSM8K (1,319 test problems).*

| Method | Accuracy (%) | Per-Seed Accuracy | Peak VRAM | Time/Step |
| :--- | :---: | :---: | :---: | :---: |
| **Bandit 10% + Attn (per-token)** | **33.99 ± 0.36** | 34.3, 33.6, 34.1 | 35.8 GB | 1.86s |
| No Experts (Attn + Router + Embed) | 33.79 ± 0.42 | 33.6, 34.3, 33.5 | 27.4 GB | 1.30s |
| **Experts-Only (Bandit raw)** | **33.79 ± 0.59** | 33.1, 34.0, 34.3 | 30.1 GB | 1.44s |
| Random 10% + Attn | 33.54 ± 0.35 | 33.7, 33.7, 33.1 | 32.5 GB | 1.92s |
| LoRA $r=128$ (Attn) | 32.42 ± 0.65 | 32.3, 33.1, 31.8 | 24.6 GB | 0.83s |
| Experts-Only (Random) | 28.83 ± 0.42 | 29.1, 28.4, 29.0 | 27.4 GB | 1.71s |

> In the experts-only setting (attention/router frozen), **Bandit selection beats Random selection by +4.95 percentage points ($p = 0.001$)**.

---

### 2. Code Generation Benchmark
*Trained on 10K Magicoder-Python samples (1 epoch), evaluated zero-shot pass@1 on HumanEval (164) and MBPP-sanitized (257).*

| Method | HumanEval (%) | MBPP (%) | Combined (421) (%) |
| :--- | :---: | :---: | :---: |
| LoRA $r=128$ (Attn) | 16.06 ± 0.93 | 32.30 ± 1.56 | **25.97 ± 0.76** |
| No Experts (Attn + Router) | 16.46 ± 2.20 | 29.96 ± 2.72 | 24.70 ± 1.56 |
| Experts-Only (Random) | 16.46 ± 1.61 | 28.15 ± 2.97 | 23.59 ± 2.44 |
| Bandit 10% + Attn (raw) | 14.23 ± 1.53 | 27.63 ± 1.70 | 22.41 ± 0.60 |
| Experts-Only (Bandit raw) | 15.04 ± 0.93 | 26.46 ± 3.18 | 22.01 ± 2.02 |

---

### Visualizations

The generated comparison curves and layerwise expert activation patterns are available in `report_out/`:
- `report_out/fig_gsm8k.png`: GSM8K accuracy across methods and ablation settings.
- `report_out/fig_code.png`: Code generation performance across HumanEval and MBPP.
- `report_out/fig_layers.png`: Layerwise distribution of bandit-selected experts.

---

## Repository Structure

```text
├── adagradselect_moe.py    # Main training script (block-selective bandit MoE fine-tuning)
├── block_selector.py       # Multi-armed bandit selector (exploration / exploitation policies)
├── lora_baseline.py        # LoRA baseline implementation for comparative evaluation
├── eval_math.py            # Zero-shot evaluation runner for GSM8K and MATH-500
├── eval_code.py            # Execution-based eval runner for HumanEval and MBPP-sanitized
├── answer_utils.py         # GSM8K / MATH answer parsing and symbolic verification
├── code_utils.py           # Sandboxed code execution and test verification
├── aggregate.py            # Aggregate run results, compute Welch's t-tests, and plot figures
├── run_sweep.sh            # Automated seed/ablation sweep script for GSM8K
├── run_sweep_code.sh       # Automated sweep script for Code generation
├── report_out/             # Summary tables (tables.md) and benchmark figures
└── .gitignore              # Pre-configured rules to ignore virtualenvs, checkpoints, and logs
```

---

## Installation & Requirements

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch transformers accelerate datasets numpy matplotlib scipy
```

### 2. Download Base Model
Download or symlink the OLMoE-1B-7B base checkpoint:
```bash
# Example using huggingface-cli:
huggingface-cli download allenai/OLMoE-1B-7B-0924 --local-dir ./OLMoE-1B-7B-0924
```

---

## Usage

### 1. Inspect Block Mapping
Check how routed experts, attention, router, and embedding layers are partitioned into blocks:
```bash
python adagradselect_moe.py --model ./OLMoE-1B-7B-0924 --inspect
```

### 2. Fine-Tuning with AdaGradSelect-MoE
Train on MetaMathQA with 10% expert budget under the bandit policy:
```bash
python adagradselect_moe.py \
    --model ./OLMoE-1B-7B-0924 \
    --data_files metamath40k.jsonl \
    --max_samples 10000 \
    --epochs 1 \
    --pct 10 \
    --policy bandit \
    --output_dir results/bandit10_s0 \
    --seed 0
```

#### Ablation / Policy Flags:
- `--policy bandit`: AdaGradSelect multi-armed bandit.
- `--policy random`: Uniform random expert selection (ablation baseline).
- `--policy gradtopk`: Pure greedy top-$k$ gradient norm selection (always explore).
- `--pct <int>`: Percentage of routed experts to train (e.g., `10` or `20`). Set `--pct 0` to train attention and router only.
- `--always none --freeze attn,router,other`: Experts-only ablation (freezes attention and router weights).

### 3. Evaluation

#### GSM8K Math Evaluation:
```bash
python eval_math.py \
    --model results/bandit10_s0 \
    --task gsm8k \
    --batch_size 32 \
    --out results/bandit10_s0/eval_gsm8k
```

#### HumanEval & MBPP Code Evaluation:
```bash
python eval_code.py \
    --model results_code/bandit10raw_s0 \
    --tasks humaneval,mbpp \
    --batch_size 32 \
    --out results_code/bandit10raw_s0/eval
```

### 4. Aggregating Results and Generating Plots
```bash
python aggregate.py
```
This reads from `results/` and `results_code/`, executes Welch's t-tests, and updates `report_out/tables.md` and `report_out/*.png`.

---

## Automated Sweeps
To run the full suite across multiple random seeds and ablations sequentially:
```bash
# Run math sweep (GSM8K):
SEEDS="0 1 2" bash run_sweep.sh

# Run code sweep (HumanEval + MBPP):
SEEDS="0 1 2" bash run_sweep_code.sh
```
