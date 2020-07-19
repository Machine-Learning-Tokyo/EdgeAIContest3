#!/usr/bin/env bash

for i in `seq -f  "%02g"  0 24`; do
echo "---$i"
mkdir -pv train_$i
ffmpeg -i train_videos/train_$i.mp4 train_videos/train_$i/%03d.png
done