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

"""Evaluate depth for DyCheck dataset."""

import glob
import os
import cv2
import numpy as np


if __name__ == "__main__":
  scene_names = ["apple", "block", "creeper", "handwavy"]
  scene_names += ["haru-sit", "mochi-high-five", "paper-windmill"]
  scene_names += ["pillow", "spin", "sriracha-tree", "teddy", "backpack"]
  ate = []
  rte = []
  rre = []

  gt_root_dir = "/data/zhuoyuan/DyCheck"
  pred_root_dir = "./outputs_cvd_dycheck_mask_opt_uncert"

  abs_rel_list = []
  log_rmse_list = []
  threshold_1_list = []
  threshold_2_list = []
  threshold_3_list = []

  for scene_name in scene_names:
    print(scene_name)
    gt_list = sorted(
        glob.glob(
            os.path.join(gt_root_dir, scene_name, "depth", "2x", "0_*.npy")
        )
    )

    gt_depths = []
    for i, gt_path in enumerate(gt_list):
      gt_depth = np.float32(np.load(gt_path))[..., -1]
      h0, w0 = gt_depth.shape
      h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
      w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
      gt_depth = cv2.resize(gt_depth, (w1, h1), interpolation=cv2.INTER_LINEAR)
      gt_depth = gt_depth[: h1 - h1 % 8, : w1 - w1 % 8]
      gt_depth = cv2.resize(
          gt_depth,
          (gt_depth.shape[-1], gt_depth.shape[-2]),
          interpolation=cv2.INTER_LINEAR,
      )
      gt_depths.append(gt_depth)

    gt_depths = np.array(gt_depths)
    gt_depths = np.nan_to_num(
        gt_depths, copy=True, nan=0.0, posinf=1e3, neginf=0.0
    )

    cvd_data = np.load(
        os.path.join(pred_root_dir, "%s_sgd_cvd_hr.npz" % scene_name)
    )
    pred_depths = cvd_data["depths"]

    assert pred_depths.shape == gt_depths.shape
    valid_mask = (gt_depths < 100) & (gt_depths > 0.1)

    pred_depths = np.clip(pred_depths, 0.1, 100.0)
    gt_depths = np.clip(gt_depths, 0.1, 100.0)

    gt_d_ms = gt_depths[valid_mask] - np.median(gt_depths[valid_mask]) + 1e-8
    pred_d_ms = (
        pred_depths[valid_mask] - np.median(pred_depths[valid_mask]) + 1e-8
    )

    scale = np.median(gt_d_ms / pred_d_ms)
    shift = np.median(gt_depths[valid_mask] - scale * pred_depths[valid_mask])

    pred_depths = pred_depths * scale + shift

    abs_rel = np.mean(
        np.abs(pred_depths[valid_mask] - gt_depths[valid_mask])
        / gt_depths[valid_mask]
    )
    log_rmse = np.sqrt(
        np.mean(
            (
                np.log(np.clip(pred_depths[valid_mask], 1e-3, 1e6))
                - np.log(gt_depths[valid_mask])
            )
            ** 2
        )
    )

    # Calculate the accuracy thresholds
    max_ratio = np.maximum(
        pred_depths[valid_mask] / gt_depths[valid_mask],
        gt_depths[valid_mask] / pred_depths[valid_mask],
    )
    threshold_1 = np.mean(max_ratio < 1.25)
    threshold_2 = np.mean(max_ratio < 1.25**2)
    threshold_3 = np.mean(max_ratio < 1.25**3)

    print(scene_name)
    print("abs_rel ", abs_rel)
    print("log_rmse ", log_rmse)
    print("threshold_1 ", threshold_1)

    abs_rel_list.append(abs_rel)
    log_rmse_list.append(log_rmse)
    threshold_1_list.append(threshold_1)
    threshold_2_list.append(threshold_2)
    threshold_3_list.append(threshold_3)

  print("abs_rel: ", np.mean(abs_rel_list))
  print("log_rmse: ", np.mean(log_rmse_list))
  print("threshold_1: ", np.mean(threshold_1_list))
  print("threshold_2: ", np.mean(threshold_2_list))
  print("threshold_3: ", np.mean(threshold_3_list))
