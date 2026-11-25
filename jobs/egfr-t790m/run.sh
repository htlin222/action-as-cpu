#!/usr/bin/env bash
# run.sh SHARD SHARDS OUTDIR — ray-trace every frame i with i % SHARDS == SHARD at 1080x1920.
set -euo pipefail
python3 work.py --shard "$1" --of "$2" --out "$3"
