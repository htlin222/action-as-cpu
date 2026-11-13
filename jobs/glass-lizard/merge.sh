#!/usr/bin/env bash
set -euo pipefail
IN=$1; DIST=$2
sudo apt-get install -y -qq ffmpeg > /dev/null 2>&1 || true
test "$(ls "$IN"/f*.png | wc -l)" -eq 120
ffmpeg -y -loglevel error -framerate 24 -i "$IN/f%04d.png" -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p -movflags +faststart "$DIST/glass-lizard.tmp.mp4"
mv "$DIST/glass-lizard.tmp.mp4" "$DIST/glass-lizard.mp4"
cp "$IN/f0006.png" "$DIST/still-start.png"; cp "$IN/f0060.png" "$DIST/still-middle.png"; cp "$IN/f0114.png" "$DIST/still-end.png"
(cd "$IN" && tar czf "$DIST/frames.tar.gz" f*.png)
