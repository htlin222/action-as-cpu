#!/usr/bin/env bash
# merge.sh INDIR DISTDIR (optional)
# INDIR holds every shard's output files together. Write the finished
# deliverables into DISTDIR; each file there becomes a release asset.
# Without a merge.sh the pipeline publishes INDIR as one .tar.gz.
set -euo pipefail
IN=$1; DIST=$2
cat "$IN"/part-*.txt > "$DIST/result.txt"
