#!/usr/bin/env bash
# run.sh SHARD SHARDS OUTDIR
# Do shard number SHARD (0-based) of SHARDS equal pieces of the work and write
# its results into OUTDIR. File names must be unique across shards, because all
# shard outputs are merged into one folder before merge.sh runs.
set -euo pipefail
SHARD=$1; SHARDS=$2; OUT=$3
python3 work.py --shard "$SHARD" --of "$SHARDS" --out "$OUT"
