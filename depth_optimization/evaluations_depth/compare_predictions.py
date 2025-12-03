"""Compare two sets of depth predictions for Sintel dataset on a frame-level."""

import glob
import os
import cv2
import numpy as np


def load_sintel_gt_paths(gt_root_dir, scene_name):
  return sorted(
      glob.glob(os.path.join(gt_root_dir, scene_name, "depth", "*.npy"))
  )


def load_dycheck_gt_paths(gt_root_dir, scene_name):
  return sorted(
      glob.glob(os.path.join(gt_root_dir, scene_name, "depth", "2x", "0_*.npy"))
  )


def load_tum_associations(scene_path):
  """Load RGB and depth associations from TUM-RGBD dataset."""
  rgb_txt = os.path.join(scene_path, "rgb.txt")
  depth_txt = os.path.join(scene_path, "depth.txt")
  
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
  
  associations = []
  rgb_timestamps = sorted(rgb_dict.keys())
  depth_timestamps = sorted(depth_dict.keys())
  
  for rgb_ts in rgb_timestamps:
    closest_depth_ts = min(depth_timestamps, key=lambda x: abs(x - rgb_ts))
    associations.append(os.path.join(gt_root_dir, scene_name, depth_dict[closest_depth_ts]))
  
  return associations


def load_gt_paths(gt_root_dir, scene_name):
  if "sintel" in gt_root_dir.lower():
    return load_sintel_gt_paths(gt_root_dir, scene_name)
  elif "dycheck" in gt_root_dir.lower():
    return load_dycheck_gt_paths(gt_root_dir, scene_name)
  elif "tum" in gt_root_dir.lower():
    return load_tum_associations(os.path.join(gt_root_dir, scene_name))
  else:
    print("Assuming Sintel dataset.")
    return load_sintel_gt_paths(gt_root_dir, scene_name)


def load_depth_from_path(path):
    if "tum" in path.lower():
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return depth_image.astype(np.float32) / 5000.0
    elif "dycheck" in path.lower():
        return np.float32(np.load(path))[..., -1]
    else: # Sintel
        return np.float32(np.load(path))


if __name__ == "__main__":
#   gt_root_dir = "/data/zhuoyuan/Sintel"
#   pred_root_dir_1 = "./outputs_cvd_sintel"
#   pred_root_dir_2 = "./outputs_cvd_sintel_out_mask_opt_uncert"

#   gt_root_dir = "/data/zhuoyuan/DyCheck"
#   pred_root_dir_1 = "./outputs_cvd_dycheck"
#   pred_root_dir_2 = "./outputs_cvd_dycheck_mask_opt_uncert"

  gt_root_dir = "/data/zhuoyuan/tum_rgbd"
  pred_root_dir_1 = "./outputs_cvd_tumrgbd"
  pred_root_dir_2 = "./outputs_cvd_tumrgbd_our_mask_opt_uncert"

  if "sintel" in gt_root_dir.lower():
    scene_names = ["alley_1", "alley_2", "temple_2", "temple_3", "market_5",
                   "mountain_1", "bamboo_2", "bamboo_1", "ambush_4", "ambush_5",
                   "ambush_6", "market_2", "market_6", "cave_4", "cave_2",
                   "shaman_3", "sleeping_1", "sleeping_2"]
  elif "dycheck" in gt_root_dir.lower():
    scene_names = ["apple", "block", "creeper", "handwavy",
                   "haru-sit", "mochi-high-five", "paper-windmill",
                   "pillow", "spin", "sriracha-tree", "teddy", "backpack"]
  elif "tum" in gt_root_dir.lower():
    scene_names = [
        # "rgbd_dataset_freiburg3_sitting_halfsphere",
        # "rgbd_dataset_freiburg3_sitting_rpy",
        # "rgbd_dataset_freiburg3_sitting_static",
        # "rgbd_dataset_freiburg3_sitting_xyz",
        # "rgbd_dataset_freiburg3_walking_halfsphere",
        # "rgbd_dataset_freiburg3_walking_rpy",
        # "rgbd_dataset_freiburg3_walking_static",
        # "rgbd_dataset_freiburg3_walking_xyz"
        "rgbd_dataset_freiburg3_sitting_static",
        "rgbd_dataset_freiburg3_walking_static"
    ]
  else:
      scene_names = []
      print("Unknown dataset based on gt_root_dir")
      exit()

  all_scene_improvements = {}

  for scene_name in scene_names:
    print(f"Processing scene: {scene_name}")
    
    scene_improvements = []

    cvd_data_1 = np.load(
        os.path.join(pred_root_dir_1, f"{scene_name}_sgd_cvd_hr.npz")
    )
    pred_depths_1 = cvd_data_1["depths"]

    cvd_data_2 = np.load(
        os.path.join(pred_root_dir_2, f"{scene_name}_sgd_cvd_hr.npz")
    )
    pred_depths_2 = cvd_data_2["depths"]
    

    gt_list = load_gt_paths(gt_root_dir, scene_name)

    if not gt_list:
        print(f"No ground truth found for scene {scene_name}, skipping.")
        continue

    if len(gt_list) != pred_depths_1.shape[0] or len(gt_list) != pred_depths_2.shape[0]:
        print(
            "Mismatch in number of frames for scene"
            f" {scene_name}. GT: {len(gt_list)}, Pred1:"
            f" {pred_depths_1.shape[0]}, Pred2: {pred_depths_2.shape[0]}."
            " Skipping."
        )
        continue


    for frame_idx, gt_path in enumerate(gt_list):
      gt_depth = load_depth_from_path(gt_path)
      h0, w0 = gt_depth.shape
      h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
      w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
      gt_depth = cv2.resize(gt_depth, (w1, h1), interpolation=cv2.INTER_LINEAR)
      gt_depth = gt_depth[: h1 - h1 % 8, : w1 - w1 % 8]

      gt_depth = np.nan_to_num(
          gt_depth, copy=True, nan=0.0, posinf=1e3, neginf=0.0
      )

      pred_depth_1 = pred_depths_1[frame_idx]
      pred_depth_2 = pred_depths_2[frame_idx]

      assert pred_depth_1.shape == gt_depth.shape
      assert pred_depth_2.shape == gt_depth.shape

      valid_mask = (gt_depth < 100) & (gt_depth > 0.1)

      if np.sum(valid_mask) < 100:  # Skip if not enough valid pixels
        continue

      # Process prediction 1
      pred_d1_clipped = np.clip(pred_depth_1, 0.1, 100.0)
      gt_d_clipped = np.clip(gt_depth, 0.1, 100.0)

      gt_d_ms = (
          gt_d_clipped[valid_mask] - np.median(gt_d_clipped[valid_mask]) + 1e-6
      )
      pred_d1_ms = (
          pred_d1_clipped[valid_mask]
          - np.median(pred_d1_clipped[valid_mask])
          + 1e-6
      )

      scale1 = np.median(gt_d_ms / pred_d1_ms)
      shift1 = np.median(
          gt_d_clipped[valid_mask] - scale1 * pred_d1_clipped[valid_mask]
      )

      aligned_pred_1 = pred_d1_clipped * scale1 + shift1

      abs_rel_1 = np.mean(
          np.abs(aligned_pred_1[valid_mask] - gt_d_clipped[valid_mask])
          / gt_d_clipped[valid_mask]
      )

      # Process prediction 2
      pred_d2_clipped = np.clip(pred_depth_2, 0.1, 100.0)

      pred_d2_ms = (
          pred_d2_clipped[valid_mask]
          - np.median(pred_d2_clipped[valid_mask])
          + 1e-6
      )

      scale2 = np.median(gt_d_ms / pred_d2_ms)
      shift2 = np.median(
          gt_d_clipped[valid_mask] - scale2 * pred_d2_clipped[valid_mask]
      )

      aligned_pred_2 = pred_d2_clipped * scale2 + shift2

      abs_rel_2 = np.mean(
          np.abs(aligned_pred_2[valid_mask] - gt_d_clipped[valid_mask])
          / gt_d_clipped[valid_mask]
      )

      improvement = abs_rel_1 - abs_rel_2
      frame_name = os.path.splitext(os.path.basename(gt_path))[0]

      scene_improvements.append({
          "scene": scene_name,
          "frame": frame_name,
          "improvement": improvement,
          "abs_rel_1": abs_rel_1,
          "abs_rel_2": abs_rel_2,
      })
    
    # Sort by improvement for the current scene
    scene_improvements.sort(key=lambda x: x["improvement"], reverse=True)
    all_scene_improvements[scene_name] = scene_improvements

  # Print top 5 for each scene
  for scene_name, improvements in all_scene_improvements.items():
    print(f"\nTop 5 frames for scene '{scene_name}':")
    for i in range(min(5, len(improvements))):
      res = improvements[i]
      print(
          f"{i+1}. Frame: {res['frame']}"
          f" - Improvement: {res['improvement']:.4f} (abs_rel_1:"
          f" {res['abs_rel_1']:.4f}, abs_rel_2: {res['abs_rel_2']:.4f})"
      )
