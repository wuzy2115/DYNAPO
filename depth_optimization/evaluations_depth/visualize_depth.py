import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_sintel_gt_depths(gt_root_dir, scene_name):
  """Loads and preprocesses ground truth depth for a Sintel scene."""
  gt_list = sorted(
      glob.glob(os.path.join(gt_root_dir, scene_name, "depth", "*.npy"))
  )
  if not gt_list:
    print(f"Warning: No GT depth found for scene {scene_name} in {gt_root_dir}")
    return None

  gt_depth_list = []
  for gt_path in gt_list:
    gt_depth = np.float32(np.load(gt_path))
    h0, w0 = gt_depth.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    gt_depth = cv2.resize(gt_depth, (w1, h1), interpolation=cv2.INTER_LINEAR)
    gt_depth = gt_depth[: h1 - h1 % 8, : w1 - w1 % 8]
    gt_depth_list.append(gt_depth)

  gt_depths = np.array(gt_depth_list)
  gt_depths = np.nan_to_num(
      gt_depths, copy=True, nan=0.0, posinf=1e3, neginf=0.0
  )
  return gt_depths


def load_dycheck_gt_depths(gt_root_dir, scene_name):
  """Loads and preprocesses ground truth depth for a DyCheck scene."""
  gt_list = sorted(
      glob.glob(os.path.join(gt_root_dir, scene_name, "depth", "2x", "0_*.npy"))
  )
  if not gt_list:
    print(f"Warning: No GT depth found for scene {scene_name} in {gt_root_dir}")
    return None

  gt_depth_list = []
  for gt_path in gt_list:
    gt_depth = np.float32(np.load(gt_path))[..., -1]
    h0, w0 = gt_depth.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    gt_depth = cv2.resize(gt_depth, (w1, h1), interpolation=cv2.INTER_LINEAR)
    gt_depth = gt_depth[: h1 - h1 % 8, : w1 - w1 % 8]
    gt_depth_list.append(gt_depth)

  gt_depths = np.array(gt_depth_list)
  gt_depths = np.nan_to_num(
      gt_depths, copy=True, nan=0.0, posinf=1e3, neginf=0.0
  )
  return gt_depths


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


def load_tum_gt_depths(gt_root_dir, scene_name):
  """Loads and preprocesses ground truth depth for a TUM-RGBD scene."""
  scene_path = os.path.join(gt_root_dir, scene_name)
  associations = load_tum_associations(scene_path)
  if not associations:
    print(f"Warning: No associations found for scene {scene_name} in {gt_root_dir}")
    return None

  gt_depth_list = []
  for _, _, _, depth_path in associations:
    gt_depth_full_path = os.path.join(scene_path, depth_path)
    gt_depth = load_tum_depth_image(gt_depth_full_path)
    h0, w0 = gt_depth.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    gt_depth = cv2.resize(gt_depth, (w1, h1), interpolation=cv2.INTER_LINEAR)
    gt_depth = gt_depth[: h1 - h1 % 8, : w1 - w1 % 8]
    gt_depth_list.append(gt_depth)

  gt_depths = np.array(gt_depth_list)
  gt_depths = np.nan_to_num(
      gt_depths, copy=True, nan=0.0, posinf=1e3, neginf=0.0
  )
  return gt_depths


def load_gt_depths(gt_root_dir, scene_name):
  """Loads and preprocesses ground truth depth for a Sintel scene."""
  if "sintel" in gt_root_dir.lower():
    return load_sintel_gt_depths(gt_root_dir, scene_name)
  elif "dycheck" in gt_root_dir.lower():
    return load_dycheck_gt_depths(gt_root_dir, scene_name)
  elif "tum" in gt_root_dir.lower():
    return load_tum_gt_depths(gt_root_dir, scene_name)
  else:
    print(
        "Warning: Could not determine dataset from gt_root_dir. Assuming Sintel."
    )
    return load_sintel_gt_depths(gt_root_dir, scene_name)


def save_depth_maps(depth_maps, output_dir_for_scene, valid_mask=None, vmin=0.1, vmax=100.0):
  """Saves a series of depth maps to individual files."""
  if depth_maps is None or len(depth_maps) == 0:
    print(f"No depth maps to save for {output_dir_for_scene}")
    return

  os.makedirs(output_dir_for_scene, exist_ok=True)
  num_frames = depth_maps.shape[0]

  cmap = plt.get_cmap("viridis")
  cmap.set_bad(color="black")

  for i in range(num_frames):
    depth_map = depth_maps[i].astype(np.float32)
    if valid_mask is not None:
      depth_map[~valid_mask[i]] = np.nan

    # Using plt.imsave to save the raw depth map without axes
    plt.imsave(
        os.path.join(output_dir_for_scene, f"{i:04d}.png"),
        depth_map,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

  print(f"Saved {num_frames} depth maps to {output_dir_for_scene}")


def main():
  parser = argparse.ArgumentParser(description="Visualize depth maps.")
  parser.add_argument(
      "--data_type",
      type=str,
      required=True,
      choices=["gt", "pred"],
      help="Type of depth data to visualize: 'gt' or 'pred'.",
  )
  parser.add_argument(
      "--pred_root_dir",
      type=str,
      help="Root directory for predicted depth files (*_sgd_cvd_hr.npz).",
  )
  parser.add_argument(
      "--gt_root_dir",
      type=str,
      default="/data/zhuoyuan/Sintel",
      help="Root directory for Sintel ground truth data.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Directory to save the output visualization.",
  )
  parser.add_argument(
      "--no_scale",
      action="store_true",
      help="Do not scale predicted depth maps for visualization.",
  )
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  if args.data_type == "pred":
    if not args.pred_root_dir:
      raise ValueError("--pred_root_dir must be provided for data_type 'pred'")

    pred_files = sorted(
        glob.glob(os.path.join(args.pred_root_dir, "*_sgd_cvd_hr.npz"))
    )
    for pred_file in pred_files:
      scene_name = os.path.basename(pred_file).replace("_sgd_cvd_hr.npz", "")
      print(f"Processing prediction for {scene_name}...")

      pred_depths = np.load(pred_file)["depths"]
      gt_depths = load_gt_depths(args.gt_root_dir, scene_name)

      if gt_depths is None:
        continue

      valid_mask = (gt_depths < 100) & (gt_depths > 0.1)

      depths_to_visualize = pred_depths
      if not args.no_scale:
        assert pred_depths.shape == gt_depths.shape
        pred_depths_c = np.clip(pred_depths, 0.1, 100.0)
        gt_depths_c = np.clip(gt_depths, 0.1, 100.0)

        gt_d_ms = (
            gt_depths_c[valid_mask] - np.median(gt_depths_c[valid_mask]) + 1e-6
        )
        pred_d_ms = (
            pred_depths_c[valid_mask]
            - np.median(pred_depths_c[valid_mask])
            + 1e-6
        )

        scale = np.median(gt_d_ms / pred_d_ms)
        shift = np.median(
            gt_depths_c[valid_mask] - scale * pred_depths_c[valid_mask]
        )

        depths_to_visualize = pred_depths * scale + shift

      output_scene_dir = os.path.join(args.output_dir, scene_name)
      vmax = np.max(depths_to_visualize[valid_mask])
      save_depth_maps(depths_to_visualize, output_scene_dir, valid_mask=valid_mask, vmax=vmax)

  elif args.data_type == "gt":
    if "sintel" in args.gt_root_dir.lower():
      scene_names = [
          "alley_1",
          "alley_2",
          "temple_2",
          "temple_3",
          "market_5",
          "mountain_1",
          "bamboo_2",
          "bamboo_1",
          "ambush_4",
          "ambush_5",
          "ambush_6",
          "market_2",
          "market_6",
          "cave_4",
          "cave_2",
          "shaman_3",
          "sleeping_1",
          "sleeping_2",
      ]
    elif "dycheck" in args.gt_root_dir.lower():
      scene_names = ["apple", "block", "creeper", "handwavy",
                     "haru-sit", "mochi-high-five", "paper-windmill",
                     "pillow", "spin", "sriracha-tree", "teddy", "backpack"]
    elif "tum" in args.gt_root_dir.lower():
      scene_names = [
          "rgbd_dataset_freiburg3_sitting_halfsphere",
          "rgbd_dataset_freiburg3_sitting_rpy",
          "rgbd_dataset_freiburg3_sitting_static",
          "rgbd_dataset_freiburg3_sitting_xyz",
          "rgbd_dataset_freiburg3_walking_halfsphere",
          "rgbd_dataset_freiburg3_walking_rpy",
          "rgbd_dataset_freiburg3_walking_static",
          "rgbd_dataset_freiburg3_walking_xyz"
      ]
    else:
        print("Could not determine dataset from gt_root_dir, assuming Sintel.")
        scene_names = [
            "alley_1", "alley_2", "temple_2", "temple_3", "market_5",
            "mountain_1", "bamboo_2", "bamboo_1", "ambush_4", "ambush_5",
            "ambush_6", "market_2", "market_6", "cave_4", "cave_2",
            "shaman_3", "sleeping_1", "sleeping_2"
        ]

    for scene_name in scene_names:
      print(f"Processing ground truth for {scene_name}...")
      gt_depths = load_gt_depths(args.gt_root_dir, scene_name)
      if gt_depths is not None:
        vmax = np.max(gt_depths)
        valid_mask = (gt_depths < vmax) & (gt_depths > 0.1)
        output_scene_dir = os.path.join(args.output_dir, f"{scene_name}_gt")
        save_depth_maps(gt_depths, output_scene_dir, valid_mask=valid_mask, vmax=vmax)


if __name__ == "__main__":
  main()
