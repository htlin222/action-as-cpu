#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg libxi6 libxkbcommon0 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libsm6 libice6 > /dev/null
pip install -q bpy scipy numpy
