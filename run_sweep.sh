#!/usr/bin/env bash
# Sequential sweep: train -> full GSM8K eval -> delete 13GB weights. Resumable (skips finished runs).
# Usage:  tmux new -s sweep ; bash run_sweep.sh            (N=10000 samples, seeds 0 1 2)
#         SEEDS="0" bash run_sweep.sh                       (seed 0 only)
# Existing runs: mv out_bandit10_s0 results/bandit10_s0   (same for random10_s0) -> they get evaluated, not retrained.
set -u
cd "$(dirname "$0")"
MODEL=./OLMoE-1B-7B-0924; DATA=metamath40k.jsonl
N=${N:-10000}; SEEDS=${SEEDS:-"0 1 2"}
C="--model $MODEL --data_files $DATA --max_samples $N --epochs 1 --log_every 50"
mkdir -p results logs

wait_gpu () {   # don't start (and OOM) while another job is using the GPU
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 3000 ]; do
    echo "[wait] GPU busy ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)); retry in 120s"; sleep 120
  done
}

run () {   # run NAME script.py args...
  name=$1; shift
  if [ -f results/$name/eval_gsm8k/summary.json ]; then echo "[skip] $name done"; return; fi
  wait_gpu
  mkdir -p results/$name
  if ! ls results/$name/*.safetensors >/dev/null 2>&1; then
    python "$@" --output_dir results/$name 2>&1 | tee logs/train_$name.log
  fi
  if ! ls results/$name/*.safetensors >/dev/null 2>&1; then echo "[FAIL] $name: no weights"; return; fi
  python eval_math.py --model results/$name --task gsm8k --batch_size 64 \
      --out results/$name/eval_gsm8k 2>&1 | tee logs/eval_$name.log
  rm -f results/$name/*.safetensors
}

for s in $SEEDS; do
  run bandit10_s$s   adagradselect_moe.py $C --no_gc --pct 10 --policy bandit   --seed $s
  run random10_s$s   adagradselect_moe.py $C --no_gc --pct 10 --policy random   --seed $s
  run lora_r128_s$s  lora_baseline.py     $C --rank 128 --lr 2e-4               --seed $s
  if [ "$s" = "0" ]; then   # ablations, seed 0 only
    run gradtopk10_s0   adagradselect_moe.py $C --no_gc --pct 10 --policy gradtopk --seed 0
    run bandit20_s0     adagradselect_moe.py $C --no_gc --pct 20 --policy bandit   --seed 0
    run bandit10raw_s0  adagradselect_moe.py $C --no_gc --pct 10 --policy bandit --score raw --seed 0
  fi
  # control: train ONLY attention+router+embed/head (no experts at all)
  run noexp_s$s       adagradselect_moe.py $C --no_gc --pct 0  --policy random   --seed $s
  # experts-only: attention/router/embed/head FROZEN -> isolates the effect of expert selection
  run bandit_eo_s$s   adagradselect_moe.py $C --no_gc --pct 10 --policy bandit --always none --freeze attn,router,other --seed $s
  run random_eo_s$s   adagradselect_moe.py $C --no_gc --pct 10 --policy random --always none --freeze attn,router,other --seed $s
  # diagnostic: does per-token normalisation (favours rarely-routed experts) hurt the bandit?
  run bandit_eo_raw_s$s adagradselect_moe.py $C --no_gc --pct 10 --policy bandit --score raw --always none --freeze attn,router,other --seed $s
done

echo "=== results ==="
python - <<'PY'
import glob, json
for f in sorted(glob.glob("results/*/eval_gsm8k/summary.json")):
    d = json.load(open(f)); print(f"{f.split('/')[1]:22s} {d['accuracy']:.2f}%  (n={d['n']})")
PY