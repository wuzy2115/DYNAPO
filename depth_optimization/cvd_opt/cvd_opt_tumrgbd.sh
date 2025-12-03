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
rgbd_dataset_freiburg3_sitting_halfsphere
rgbd_dataset_freiburg3_sitting_rpy
rgbd_dataset_freiburg3_sitting_static
rgbd_dataset_freiburg3_sitting_xyz
rgbd_dataset_freiburg3_walking_halfsphere
rgbd_dataset_freiburg3_walking_rpy
rgbd_dataset_freiburg3_walking_static
rgbd_dataset_freiburg3_walking_xyz
)

DATA_DIR=/data/zhuoyuan/tum_rgbd

# Run Raft Optical Flows
# for seq in ${evalset[@]}; do
#   CUDA_VISIBLE_DEVICES=3 python cvd_opt/preprocess_flow.py \
#   --datapath=$DATA_DIR/$seq/rgb \
#   --model=cvd_opt/raft-things.pth \
#   --scene_name $seq --mixed_precision
# done

# Run CVD optmization
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=2 python cvd_opt/cvd_opt.py \
  --data_dir ./reconstructions_tumrgbd \
  --scene_name $seq \
  --output_dir outputs_cvd_tumrgbd \
  --w_grad 2.0 \
  --w_normal 5.0 \
  --device cpu
done
