#!/usr/bin/env bash
set -euo pipefail

# Run PA graph generation for fixed node counts from 80 to 200 in steps of 10.
for nodes in $(seq 20 20 200); do
  echo "Generating graphs with ${nodes} nodes..."
  # CUDA_VISIBLE_DEVICES=0 python gen_pa.py --min-nodes "${nodes}" --max-nodes "$((nodes + 20))"
  CUDA_VISIBLE_DEVICES=0 python gen_tree.py --min-nodes "${nodes}" --max-nodes "$((nodes + 20))"
done
