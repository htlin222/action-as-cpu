#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$2"
for cue in page focus section; do
  test -s "$1/$cue.wav"
  cp "$1/$cue.wav" "$2/$cue.wav"
done
