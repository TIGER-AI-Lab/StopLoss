#!/usr/bin/env bash
# Train every seed of one recipe, one after another (python for 1-GPU recipes, torchrun for 2-GPU recipes).
#   bash training/run_seeds.sh <recipe.json> <output root> <data arguments of train.py>
#   e.g. bash training/run_seeds.sh training/recipes/qwen3-8b.json runs --data train.jsonl --val-data validation.jsonl
# Runs go to <output root>/<recipe name>_s<seed>; set CUDA_VISIBLE_DEVICES to choose the GPUs.
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,5p "$0"; exit 2; }
recipe=$1; root=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd)
field() { python -c "import json, sys; v = json.load(open(sys.argv[1]))[sys.argv[2]]; print(' '.join(map(str, v)) if isinstance(v, list) else v)" "$recipe" "$1"; }
name=$(field name); gpus=$(field gpus); seeds=$(field seeds)
for seed in $seeds; do
  if [ "$gpus" -gt 1 ]; then launch=(torchrun --nproc_per_node "$gpus"); else launch=(python); fi
  echo "== $name seed $seed on $gpus GPU(s)"
  "${launch[@]}" "$here/train.py" --recipe "$recipe" "$@" --out "$root/${name}_s$seed" --seed "$seed"
done
