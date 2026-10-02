#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
assets=public/assets/videos
for layout in wide grid; do
  positions='0_0|540_0|1080_0|1620_0'
  if [[ "$layout" == grid ]]; then positions='0_0|540_0|0_540|540_540'; fi
  ffmpeg -hide_banner -loglevel error -y \
    -i "$assets/41586_2026_10900_MOESM6_ESM.mp4" \
    -i "$assets/41586_2026_10900_MOESM7_ESM.mp4" \
    -i "$assets/41586_2026_10900_MOESM8_ESM.mp4" \
    -i "$assets/41586_2026_10900_MOESM9_ESM.mp4" \
    -filter_complex "[0:v]scale=540:540,setsar=1,tpad=stop_mode=clone:stop_duration=3[a];[1:v]scale=540:540,setsar=1,tpad=stop_mode=clone:stop_duration=3[b];[2:v]scale=540:540,setsar=1,tpad=stop_mode=clone:stop_duration=3[c];[3:v]scale=540:540,setsar=1,tpad=stop_mode=clone:stop_duration=3[d];[a][b][c][d]xstack=inputs=4:layout=$positions[v]" \
    -map '[v]' -t 2.6 -an -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -movflags +faststart "$assets/paper-$layout.mp4"
  ffmpeg -hide_banner -loglevel error -y -i "$assets/paper-$layout.mp4" \
    -frames:v 1 -q:v 4 "$assets/paper-$layout.jpg"
done
