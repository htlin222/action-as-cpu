#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update -qq && sudo apt-get install -y -qq libgl1 libglu1-mesa libx11-6 > /dev/null
pip install -q pymol-open-source-whl numpy
