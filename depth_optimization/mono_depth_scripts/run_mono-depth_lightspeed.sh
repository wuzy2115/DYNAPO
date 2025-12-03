#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

evalset=(
0120_LOFT  0335_BKST      0500_LOFTRAMP    0630_GDZRCK2NDF  0745_DINOPARK
0180_DUST  0340_BKST      0520_LOFTRAMP    0640_GDZRCK2NDF  0750_GODZROCK
0190_DUST  0370_GODZROCK  0530_LOFTRAMP    0650_GDZRCK2NDF  0765_DISCOEND
0200_DUST  0380_GODZROCK  0540_GODZDRIFT   0680_GDZRCK2NDF  0770_DISCOEND
0240_DSRT  0400_LOFT      0560_GODZDRIFT   0700_DINOPARK
0260_DSRT  0410_LOFT      0570_GODZDRIFT   0710_DINOPARK
0270_DSRT  0420_GODZROCK  0600_GODZDRIFT   0720_DINOPARK
0310_BKST  0490_LOFTRAMP  0620_GDZRCK2NDF  0740_DINOPARK

)

DATA_DIR=/data/zhuoyuan/dynpose-100k/lightspeed/frames-24fps

# Run DepthAnything
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=9 python Depth-Anything/run_videos.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path $DATA_DIR/$seq/images \
  --outdir Depth-Anything/video_visualization/$seq
done

# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=9 python UniDepth/scripts/demo_mega-sam.py \
  --scene-name $seq \
  --img-path $DATA_DIR/$seq/images \
  --outdir UniDepth/outputs
done
