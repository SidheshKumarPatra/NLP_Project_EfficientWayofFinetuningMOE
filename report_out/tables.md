## GSM8K (full 1319, zero-shot greedy; MetaMath 10K samples, 1 epoch)

| Method | Accuracy % | per seed | s/step (seed0) | peak GB |
|---|---|---|---|---|
| Bandit 10% + attn (per-token) | 33.99 ± 0.36 (n=3) | 34.3, 33.6, 34.1 | 1.86 | 35.8 |
| No experts (attn+router+embed/head) | 33.79 ± 0.42 (n=3) | 33.6, 34.3, 33.5 | 1.30 | 27.4 |
| Experts-only, bandit raw | 33.79 ± 0.59 (n=3) | 33.1, 34.0, 34.3 | 1.44 | 30.1 |
| Random 10% + attn | 33.54 ± 0.35 (n=3) | 33.7, 33.7, 33.1 | 1.92 | 32.5 |
| Bandit 20% + attn | 33.28 (n=1) | 33.3 | 2.38 | 40.0 |
| Grad top-k 10% + attn | 33.21 (n=1) | 33.2 | 1.92 | 35.1 |
| Bandit 10% + attn (raw) | 32.83 (n=1) | 32.8 | 1.59 | 35.7 |
| LoRA r=128 (attn) | 32.42 ± 0.65 (n=3) | 32.3, 33.1, 31.8 | 0.83 | 24.6 |
| Experts-only, random | 28.83 ± 0.42 (n=3) | 29.1, 28.4, 29.0 | 1.71 | 27.4 |
| Experts-only, bandit per-token | 26.66 ± 1.08 (n=3) | 26.2, 25.9, 27.9 | 1.84 | 30.1 |


## Code (Magicoder python 10K, 1 epoch): HumanEval (164) / MBPP-sanitized (257) pass@1 %

| Method | HumanEval | MBPP | combined (421) | s/step (seed0) | peak GB |
|---|---|---|---|---|---|
| LoRA r=128 (attn) | 16.06 ± 0.93 (n=3) | 32.30 ± 1.56 (n=3) | 25.97 ± 0.76 (n=3) | 3.63 | 24.7 |
| No experts (attn+router+embed/head) | 16.46 ± 2.20 (n=3) | 29.96 ± 2.72 (n=3) | 24.70 ± 1.56 (n=3) | 1.08 | 27.5 |
| Experts-only, random | 16.46 ± 1.61 (n=3) | 28.15 ± 2.97 (n=3) | 23.59 ± 2.44 (n=3) | 1.78 | 27.4 |
| Bandit 10% + attn (raw) | 14.23 ± 1.53 (n=3) | 27.63 ± 1.70 (n=3) | 22.41 ± 0.60 (n=3) | 2.26 | 36.1 |
| Experts-only, bandit raw | 15.04 ± 0.93 (n=3) | 26.46 ± 3.18 (n=3) | 22.01 ± 2.02 (n=3) | 1.70 | 30.1 |


## Seed-level comparisons (Welch; 3 seeds => low power)

| domain | difference | Δ (points) | t | p |
|---|---|---|---|---|
| gsm8k | Experts-only, bandit raw − Experts-only, random | +4.95 | 11.88 | 0.001 |
| gsm8k | Experts-only, bandit raw − Experts-only, bandit per-token | +7.13 | 10.05 | 0.002 |
| gsm8k | Bandit 10% + attn (per-token) − Random 10% + attn | +0.45 | 1.57 | 0.191 |
| gsm8k | Experts-only, bandit raw − No experts (attn+router+embed/head) | +0.00 | 0.00 | 1.000 |
| gsm8k | No experts (attn+router+embed/head) − LoRA r=128 (attn) | +1.36 | 3.05 | 0.047 |
| code/HumanEval | Experts-only, bandit raw − Experts-only, random | -1.42 | -1.32 | 0.272 |
| code/MBPP | Experts-only, bandit raw − Experts-only, random | -1.69 | -0.67 | 0.539 |
| code/combined | Experts-only, bandit raw − Experts-only, random | -1.58 | -0.87 | 0.437 |
| code/HumanEval | No experts (attn+router+embed/head) − LoRA r=128 (attn) | +0.41 | 0.29 | 0.789 |
| code/MBPP | No experts (attn+router+embed/head) − LoRA r=128 (attn) | -2.33 | -1.29 | 0.283 |
| code/combined | No experts (attn+router+embed/head) − LoRA r=128 (attn) | -1.27 | -1.26 | 0.298 |
| code/HumanEval | Experts-only, bandit raw − LoRA r=128 (attn) | -1.02 | -1.34 | 0.252 |
| code/MBPP | Experts-only, bandit raw − LoRA r=128 (attn) | -5.84 | -2.85 | 0.068 |
| code/combined | Experts-only, bandit raw − LoRA r=128 (attn) | -3.96 | -3.18 | 0.062 |


## Which experts does the bandit pick? (top-102 Jaccard; chance ≈ 0.052; Spearman of pick counts)

| run | comparison | Jaccard@102 | Spearman | #pairs |
|---|---|---|---|---|
| Experts-only, bandit raw | math vs math (diff seeds) | 0.919 | 0.714 | 3 |
| Experts-only, bandit raw | code vs code (diff seeds) | 0.943 | 0.643 | 3 |
| Experts-only, bandit raw | math vs code | 0.348 | 0.466 | 9 |
| Bandit 10% + attn (raw) | code vs code (diff seeds) | 0.955 | 0.684 | 3 |
| Bandit 10% + attn (raw) | math vs code | 0.328 | 0.481 | 3 |
| Experts-only, bandit per-token | math vs math (diff seeds) | 0.872 | 0.931 | 3 |
| Bandit 10% + attn (per-token) | math vs math (diff seeds) | 0.849 | 0.936 | 3 |
