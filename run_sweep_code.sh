#!/usr/bin/env bash
# Code-domain sweep: train on Magicoder (python only) -> HumanEval+MBPP -> delete weights. Resumable.
#   SEEDS="0" bash run_sweep_code.sh        (default: seed 0 only)
set -u
cd "$(dirname "$0")"
MODEL=./OLMoE-1B-7B-0924
N=${N:-10000}; SEEDS=${SEEDS:-"0"}
C="--model $MODEL --dataset ise-uiuc/Magicoder-OSS-Instruct-75K --query_field problem --response_field solution \
   --filter_field lang --filter_value python --max_samples $N --epochs 1 --log_every 50"
mkdir -p results_code logs_code

wait_gpu () {   # don't start (and OOM) while another job is using the GPU
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 3000 ]; do
    echo "[wait] GPU busy ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)); retry in 120s"; sleep 120
  done
}

run () {
  name=$1; shift
  if [ -f results_code/$name/eval/summary.json ]; then echo "[skip] $name done"; return; fi
  wait_gpu
  mkdir -p results_code/$name
  if ! ls results_code/$name/*.safetensors >/dev/null 2>&1; then
    python "$@" --output_dir results_code/$name 2>&1 | tee logs_code/train_$name.log
  fi
  if ! ls results_code/$name/*.safetensors >/dev/null 2>&1; then echo "[FAIL] $name: no weights"; return; fi
  python eval_code.py --model results_code/$name --tasks humaneval,mbpp --out results_code/$name/eval 2>&1 | tee logs_code/eval_$name.log
  rm -f results_code/$name/*.safetensors
}

EO="--always none --freeze attn,router,other"
for s in $SEEDS; do
  run noexp_s$s          adagradselect_moe.py $C --no_gc --pct 0  --policy random --seed $s
  run bandit_eo_raw_s$s  adagradselect_moe.py $C --no_gc --pct 10 --policy bandit --score raw $EO --seed $s
  run random_eo_s$s      adagradselect_moe.py $C --no_gc --pct 10 --policy random $EO --seed $s
  run lora_r128_s$s      lora_baseline.py     $C --rank 128 --lr 2e-4 --seed $s
  run bandit10raw_s$s    adagradselect_moe.py $C --no_gc --pct 10 --policy bandit --score raw --seed $s
done

echo "=== results (pass@1) ==="
python - <<'PY'
import glob, json
for f in sorted(glob.glob("results_code/*/eval/summary.json")):
    d = json.load(open(f)); print(f"{f.split('/')[1]:22s} HumanEval {d['humaneval']['pass@1']:.2f}%  MBPP {d['mbpp']['pass@1']:.2f}%")
PY