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

"""Evaluate depth for TUM-RGBD dataset."""

import glob
import os
import cv2
import numpy as np


def load_tum_depth_image(depth_path):
  """Load TUM-RGBD depth image and convert to meters.
  
  Args:
    depth_path: Path to the depth PNG file.
    
  Returns:
    Depth image in meters as float32.
  """
  # Load 16-bit PNG depth image
  depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
  # Convert from millimeters to meters (TUM-RGBD uses factor of 5000)
  depth_image = depth_image.astype(np.float32) / 5000.0
  return depth_image


def load_tum_associations(scene_path):
  """Load RGB and depth associations from TUM-RGBD dataset.
  
  Args:
    scene_path: Path to the scene directory.
    
  Returns:
    List of tuples (rgb_timestamp, rgb_path, depth_timestamp, depth_path).
  """
  rgb_txt = os.path.join(scene_path, "rgb.txt")
  depth_txt = os.path.join(scene_path, "depth.txt")
  
  # Load RGB timestamps and paths
  rgb_dict = {}
  with open(rgb_txt, 'r') as f:
    for line in f:
      if line.startswith('#'):
        continue
      parts = line.strip().split()
      if len(parts) >= 2:
        timestamp = float(parts[0])
        rgb_path = parts[1]
        rgb_dict[timestamp] = rgb_path
  
  # Load depth timestamps and paths
  depth_dict = {}
  with open(depth_txt, 'r') as f:
    for line in f:
      if line.startswith('#'):
        continue
      parts = line.strip().split()
      if len(parts) >= 2:
        timestamp = float(parts[0])
        depth_path = parts[1]
        depth_dict[timestamp] = depth_path
  
  # Associate RGB and depth by finding closest timestamps
  # MUST assign a depth for EVERY RGB image
  associations = []
  rgb_timestamps = sorted(rgb_dict.keys())
  depth_timestamps = sorted(depth_dict.keys())
  
  for rgb_ts in rgb_timestamps:
    # Find closest depth timestamp (no tolerance limit - always assign closest)
    closest_depth_ts = min(depth_timestamps, key=lambda x: abs(x - rgb_ts))
    associations.append((
        rgb_ts,
        rgb_dict[rgb_ts],
        closest_depth_ts,
        depth_dict[closest_depth_ts]
    ))
  
  return associations


if __name__ == "__main__":
  # Common TUM-RGBD sequences used for evaluation
  scene_names = [
      "rgbd_dataset_freiburg3_sitting_halfsphere",
      "rgbd_dataset_freiburg3_sitting_rpy",
      "rgbd_dataset_freiburg3_sitting_static",
      "rgbd_dataset_freiburg3_sitting_xyz",
      "rgbd_dataset_freiburg3_walking_halfsphere",
      "rgbd_dataset_freiburg3_walking_rpy",
      "rgbd_dataset_freiburg3_walking_static",
      "rgbd_dataset_freiburg3_walking_xyz"
      # "rgbd_dataset_freiburg3_sitting_static",
      # "rgbd_dataset_freiburg3_walking_static"
  ]
  
  # Adjust paths as needed
  gt_root_dir = "/data/zhuoyuan/tum_rgbd"
  # megasam
#   pred_root_dir = "./outputs_wo_cvd_est_disp_tumrgbd"
  # pred_root_dir = "./outputs_cvd_tumrgbd"
  pred_root_dir = "outputs_cvd_tumrgbd_our_mask_opt_uncert"
  
  abs_rel_list = []
  log_rmse_list = []
  threshold_1_list = []
  threshold_2_list = []
  threshold_3_list = []
  
  for scene_name in scene_names:
    print(scene_name)
    
    # Load RGB-Depth associations
    scene_path = os.path.join(gt_root_dir, scene_name)
    associations = load_tum_associations(scene_path)
    print(f"Found {len(associations)} RGB-Depth pairs")
    
    gt_depth_list = []
    for i, (rgb_ts, rgb_path, depth_ts, depth_path) in enumerate(associations):
      # Load ground truth depth
      gt_depth_full_path = os.path.join(scene_path, depth_path)
      gt_depth = load_tum_depth_image(gt_depth_full_path)
      h0, w0 = gt_depth.shape
      # Resize to target resolution (similar to Sintel)
      h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
      w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
      gt_depth = cv2.resize(gt_depth, (w1, h1), interpolation=cv2.INTER_LINEAR)
      gt_depth = gt_depth[: h1 - h1 % 8, : w1 - w1 % 8]
      gt_depth_list.append(gt_depth)
    
    gt_depths = np.array(gt_depth_list)
    # Handle invalid values (nan, inf, and zeros from missing data)
    gt_depths = np.nan_to_num(
        gt_depths, copy=True, nan=0.0, posinf=1e3, neginf=0.0
    )
    
    # Load predicted depths
    cvd_data = np.load(
        os.path.join(pred_root_dir, "%s_sgd_cvd_hr.npz" % scene_name)
    )
    pred_depths = cvd_data["depths"]
    
    assert pred_depths.shape == gt_depths.shape
    
    # Valid mask: exclude missing data (0) and invalid ranges
    # TUM-RGBD typical depth range is 0.5m to 5m for most scenes
    valid_mask = (gt_depths < 10.0) & (gt_depths > 0.1)
    
    pred_depths = np.clip(pred_depths, 0.1, 10.0)
    gt_depths = np.clip(gt_depths, 0.1, 10.0)
    
    # Scale and shift alignment (similar to Sintel evaluation)
    gt_d_ms = gt_depths[valid_mask] - np.median(gt_depths[valid_mask]) + 1e-6
    pred_d_ms = (
        pred_depths[valid_mask] - np.median(pred_depths[valid_mask]) + 1e-6
    )
    
    scale = np.median(gt_d_ms / pred_d_ms)
    shift = np.median(gt_depths[valid_mask] - scale * pred_depths[valid_mask])
    
    pred_depths = pred_depths * scale + shift
    
    # Calculate metrics
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

