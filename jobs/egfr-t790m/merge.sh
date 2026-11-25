#!/usr/bin/env bash
# merge.sh INDIR DISTDIR — add labels and captions, encode the MP4, keep clean frames for re-compositing.
set -euo pipefail
IN=$1; DIST=$2; NF=420
pip install -q pillow numpy imageio-ffmpeg
test "$(ls "$IN"/f*.png | wc -l)" -eq $NF
test "$(ls "$IN"/f*.json | wc -l)" -eq $NF
python3 post.py "$IN" "$DIST/EGFR_T790M_1080p.mp4"
python3 post.py "$IN" --test 45,150,235,330
for i in 0045 0150 0235 0330; do mv "$IN/comp_$i.jpg" "$DIST/still_$i.jpg"; done
python3 - "$IN" <<'PY'
import sys, os, glob
from PIL import Image
d = sys.argv[1]; os.makedirs(d + '/clean', exist_ok=True)
for f in sorted(glob.glob(d + '/f*.png')):
    Image.open(f).convert('RGB').save(d + '/clean/' + os.path.basename(f)[:-4] + '.jpg', quality=94)
PY
(cd "$IN" && cp f*.json clean/ && tar cf "$DIST/clean_frames.tar" clean)
