#!/usr/bin/env bash
# Render every SHARDS-th frame of the 120-frame glass lizard shot, starting at SHARD.
set -euo pipefail
SHARD=$1; SHARDS=$2; OUT=$3
export LIZ_RES=${LIZ_RES:-1280x720} LIZ_SAMPLES=${LIZ_SAMPLES:-256} LIZ_NOISE=${LIZ_NOISE:-0.008} LIZ_MNEE=${LIZ_MNEE:-1}
FRAMES=$(python3 -c "print(','.join(str(f) for f in range(120) if f % $SHARDS == $SHARD))")
python3 scene.py --frames "$FRAMES" --out "$OUT" 2>&1 | grep -E "FRAME|Error|Traceback|fit |eye " || true
ls "$OUT"
test "$(ls "$OUT"/f*.png | wc -l)" -eq "$(echo "$FRAMES" | tr ',' '\n' | wc -l)"
