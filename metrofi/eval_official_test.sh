#!/usr/bin/env bash
set -euo pipefail

project_dir="/mnt/lts4/scratch/home/yqin/GraphGeneration/jacobi-graph-diffusion"
cd "$project_dir"
source /mnt/lts4/scratch/home/yqin/conda/etc/profile.d/conda.sh
conda activate graphon

mode="${1:-all}"
out_dir="$project_dir/metrofi/results/official_test"
runtime_dir="$project_dir/results/metrofi-official-runtime"
mkdir -p "$out_dir" "$runtime_dir"

common=(
  --device cuda:0
  --conditional-eval-only
  --conditional-eval-split test
  --conditional-eval-graphs 943
  --conditional-eval-mask-mode true
  --sampler-predictor milstein
  --sampler-eps-time 0.03
  --sampler-snr 0.01
  --sampler-scale-eps 0.1
  --sampler-n-steps 1
  --no-model-use-sampled-features
  --batch-size 256
)

run_conditional() {
  python main.py gen --model metrofi_cond \
    --checkpoint checkpoints/metrofi_cond_pe_nosf_epoch999.ckpt \
    --conditional-eval-condition-mode true \
    --conditional-eval-name official_test \
    --guidance-scale 1.8 \
    --json-out "$out_dir/conditional_metrics.json" \
    --conditional-eval-resume-path "$runtime_dir/conditional_partial.pt" \
    "${common[@]}" 2>&1 | tee "$runtime_dir/conditional.log"
}

run_unconditional() {
  python main.py gen --model metrofi_uncond \
    --checkpoint "checkpoints/metrofi_uncond_pe_nosf_epoch999.ckpt" \
    --conditional-eval-condition-mode true \
    --conditional-eval-name official_test \
    --json-out "$out_dir/unconditional_pe_metrics.json" \
    --conditional-eval-resume-path "$runtime_dir/unconditional_pe_partial.pt" \
    "${common[@]}" 2>&1 | tee "$runtime_dir/unconditional_pe.log"
}

case "$mode" in
  conditional) run_conditional ;;
  unconditional) run_unconditional ;;
  all)
    run_conditional
    run_unconditional
    ;;
  *)
    echo "Usage: $0 [conditional|unconditional|all]" >&2
    exit 2
    ;;
esac
