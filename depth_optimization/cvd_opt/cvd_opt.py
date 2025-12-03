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

"""Consistent video depth optimization."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=redefined-outer-name

import argparse
import os
from pathlib import Path

import imageio
from geometry_utils import NormalGenerator
import kornia
from lietorch import SE3
import numpy as np
import torch
import cv2


def gradient_loss(gt, pred, u):
  """Gradient loss."""
  del u
  diff = pred - gt
  v_gradient = torch.abs(
      diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1]
  )  # * mask_v
  h_gradient = torch.abs(
      diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:]
  )  # * mask_h

  pred_grad = torch.abs(
      pred[..., 0:-2, 1:-1] - (pred[..., 2:, 1:-1])
  ) + torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
  gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - (gt[..., 2:, 1:-1])) + torch.abs(
      gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:]
  )

  grad_diff = torch.abs(pred_grad - gt_grad)
  nearby_mask = (torch.exp(gt[..., 1:-1, 1:-1]) > 1.0).float().detach()
  # weight = (1. - torch.exp(-(grad_diff * 5.)).detach())
  weight = 1.0 - torch.exp(-(grad_diff * 5.0)).detach()
  weight *= nearby_mask

  g_loss = torch.mean(h_gradient * weight) + torch.mean(v_gradient * weight)
  return g_loss


def si_loss(gt, pred):
  log_gt = torch.log(torch.clamp(gt, 1e-3, 1e3)).view(gt.shape[0], -1)
  log_pred = torch.log(torch.clamp(pred, 1e-3, 1e3)).view(pred.shape[0], -1)
  log_diff = log_gt - log_pred
  num_pixels = gt.shape[-2] * gt.shape[-1]
  data_loss = torch.sum(log_diff**2, dim=-1) / num_pixels - torch.sum(
      log_diff, dim=-1
  ) ** 2 / (num_pixels**2)
  return torch.mean(data_loss)


def sobel_fg_alpha(disp, mode="sobel", beta=10.0):
  sobel_grad = kornia.filters.spatial_gradient(
      disp, mode=mode, normalized=False
  )
  sobel_mag = torch.sqrt(
      sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2
  )
  alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

  return alpha


ALPHA_MOTION = 0.25
RESIZE_FACTOR = 0.5


def consistency_loss(
    cam_c2w,
    K,
    K_inv,
    disp_data,
    init_disp,
    uncertainty,
    flows,
    flow_masks,
    ii,
    jj,
    compute_normals,
    fg_alpha,
    device,
    w_ratio=1.0,
    w_flow=0.2,
    w_si=1.0,
    w_grad=2.0,
    w_normal=4.0,
):
  """Consistency loss."""
  _, H, W = disp_data.shape
  # mesh grid
  xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
  yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
  xx = xx.view(1, 1, H, W)  # .repeat(B ,1 ,1 ,1)
  yy = yy.view(1, 1, H, W)  # .repeat(B ,1 ,1 ,1)
  grid = (
      torch.cat((xx, yy), 1).float().to(device).permute(0, 2, 3, 1)
  )  # [None, ...]

  loss_flow = 0.0  # flow reprojection loss
  loss_d_ratio = 0.0  # depth consistency loss

  flows_step = flows.permute(0, 2, 3, 1)
  flow_masks_step = flow_masks.permute(0, 2, 3, 1).squeeze(-1)

  cam_1to2 = torch.bmm(
      torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
      torch.index_select(cam_c2w, dim=0, index=ii),
  )

  # warp disp from target time
  pixel_locations = grid + flows_step
  resize_factor = torch.tensor([W - 1.0, H - 1.0]).to(device)[None, None, None, ...]
  normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

  disp_sampled = torch.nn.functional.grid_sample(
      torch.index_select(disp_data, dim=0, index=jj)[:, None, ...],
      normalized_pixel_locations,
      align_corners=True,
  )

  uu = torch.index_select(uncertainty, dim=0, index=ii).squeeze(1)

  grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(
      -1
  )
  # depth of reference view
  ref_depth = 1.0 / torch.clamp(
      torch.index_select(disp_data, dim=0, index=ii), 1e-3, 1e3
  )

  pts_3d_ref = ref_depth[..., None, None] * (K_inv[None, None, None] @ grid_h)
  rot = cam_1to2[:, None, None, :3, :3]
  trans = cam_1to2[:, None, None, :3, 3:4]

  pts_3d_tgt = (rot @ pts_3d_ref) + trans  # [:, None, None, :, None]
  depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
  disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

  # flow consistency loss
  pts_2D_tgt = K[None, None, None] @ pts_3d_tgt

  flow_masks_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1)
  pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / torch.clamp(
      pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3
  )

  disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
  disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

  ratio = torch.maximum(
      disp_sampled.squeeze() / disp_tgt.squeeze(),
      disp_tgt.squeeze() / disp_sampled.squeeze(),
  )
  ratio_error = torch.abs(ratio - 1.0)  #

  loss_d_ratio += torch.sum(
      (ratio_error * uu + ALPHA_MOTION * torch.log(1.0 / uu)) * flow_masks_step_
  ) / (torch.sum(flow_masks_step_) + 1e-8)

  flow_error = torch.abs(pts_2D_tgt - pixel_locations)
  loss_flow += torch.sum(
      (
          flow_error * uu[..., None]
          + ALPHA_MOTION * torch.log(1.0 / uu[..., None])
      )
      * flow_masks_step_[..., None]
  ) / (torch.sum(flow_masks_step_) * 2.0 + 1e-8)

  # prior mono-depth reg loss
  loss_prior = si_loss(init_disp, disp_data)
  KK = torch.inverse(K_inv)

  # multi gradient consistency
  disp_data_ds = disp_data[:, None, ...]
  init_disp_ds = init_disp[:, None, ...]
  K_rescale = KK.clone()
  K_inv_rescale = torch.inverse(K_rescale)
  pred_normal = compute_normals[0](
      1.0 / torch.clamp(disp_data_ds, 1e-3, 1e3), K_inv_rescale[None]
  )
  init_normal = compute_normals[0](
      1.0 / torch.clamp(init_disp_ds, 1e-3, 1e3), K_inv_rescale[None]
  )

  loss_normal = torch.mean(
      fg_alpha * (1.0 - torch.sum(pred_normal * init_normal, dim=1))
  )  # / (1e-8 + torch.sum(fg_alpha))

  loss_grad = 0.0
  for scale in range(4):
    interval = 2**scale
    disp_data_ds = torch.nn.functional.interpolate(
        disp_data[:, None, ...],
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    init_disp_ds = torch.nn.functional.interpolate(
        init_disp[:, None, ...],
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    uncertainty_rs = torch.nn.functional.interpolate(
        uncertainty,
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    loss_grad += gradient_loss(
        torch.log(disp_data_ds), torch.log(init_disp_ds), uncertainty_rs
    )

  return (
      w_ratio * loss_d_ratio
      + w_si * loss_prior
      + w_flow * loss_flow
      + w_normal * loss_normal
      + loss_grad * w_grad
  )


def _quaternion_to_rotation_matrix(
    qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
  """Convert a quaternion to a rotation matrix."""
  x2 = qx + qx
  y2 = qy + qy
  z2 = qz + qz

  xx2 = qx * x2
  yy2 = qy * y2
  zz2 = qz * z2
  xy2 = qx * y2
  xz2 = qx * z2
  yz2 = qy * z2
  wx2 = qw * x2
  wy2 = qw * y2
  wz2 = qw * z2

  rot = np.empty((3, 3), dtype=np.float32)
  rot[0, 0] = 1.0 - (yy2 + zz2)
  rot[0, 1] = xy2 - wz2
  rot[0, 2] = xz2 + wy2
  rot[1, 0] = xy2 + wz2
  rot[1, 1] = 1.0 - (xx2 + zz2)
  rot[1, 2] = yz2 - wx2
  rot[2, 0] = xz2 - wy2
  rot[2, 1] = yz2 + wx2
  rot[2, 2] = 1.0 - (xx2 + yy2)
  return rot


def _rotation_matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
  """Convert a rotation matrix to a quaternion."""
  # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
  tr = np.trace(m)
  if tr > 0:
    s = np.sqrt(tr + 1.0) * 2
    qw = 0.25 * s
    qx = (m[2, 1] - m[1, 2]) / s
    qy = (m[0, 2] - m[2, 0]) / s
    qz = (m[1, 0] - m[0, 1]) / s
  elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
    s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
    qw = (m[2, 1] - m[1, 2]) / s
    qx = 0.25 * s
    qy = (m[0, 1] + m[1, 0]) / s
    qz = (m[0, 2] + m[2, 0]) / s
  elif m[1, 1] > m[2, 2]:
    s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
    qw = (m[0, 2] - m[2, 0]) / s
    qx = (m[0, 1] + m[1, 0]) / s
    qy = 0.25 * s
    qz = (m[1, 2] + m[2, 1]) / s
  else:
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    qw = (m[1, 0] - m[0, 1]) / s
    qx = (m[0, 2] + m[2, 0]) / s
    qy = (m[1, 2] + m[2, 1]) / s
    qz = 0.25 * s
  return np.array([qx, qy, qz, qw], dtype=np.float32)


def _read_tum_trajectory(tum_path: Path) -> np.ndarray:
  """Read a trajectory file in TUM format."""
  poses = []
  with open(tum_path, 'r') as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      parts = [float(x) for x in line.split()]
      # Accept either 7 (tx ty tz qw qx qy qz) or 8 columns (t tx ty tz qw qx qy qz)
      if len(parts) == 7:
        tx, ty, tz, qw, qx, qy, qz = parts
      elif len(parts) >= 8:
        _, tx, ty, tz, qw, qx, qy, qz = parts[:8]
      else:
        continue
      rot = _quaternion_to_rotation_matrix(qx, qy, qz, qw)
      pose = np.eye(4, dtype=np.float32)
      pose[:3, :3] = rot
      pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
      poses.append(pose)
  if len(poses) == 0:
    return np.zeros((0, 4, 4), dtype=np.float32)
  return np.stack(poses, axis=0).astype(np.float32)


def _read_intrinsics_txt(k_path: Path) -> np.ndarray:
  """Read an intrinsics file."""
  with open(k_path, 'r') as f:
    lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
  values = []
  for ln in lines:
    values.extend([float(x) for x in ln.split()])
  if len(values) >= 9:
    vals = values[:9]
    k = np.array(vals, dtype=np.float32).reshape(3, 3)
    return k
  if len(values) >= 4:
    fx, fy, cx, cy = values[:4]
    k = np.eye(3, dtype=np.float32)
    k[0, 0] = fx
    k[1, 1] = fy
    k[0, 2] = cx
    k[1, 2] = cy
    return k
  # Fallback identity
  return np.eye(3, dtype=np.float32)


def _load_depth_stack(root_dir: Path) -> np.ndarray | None:
  """Load a stack of depth maps."""
  # Prefer .npy if available
  npy_candidates = [
      root_dir / 'depths.npy',
      root_dir / 'depth.npy',
      root_dir / 'pred_depths.npy',
  ]
  for p in npy_candidates:
    if p.exists():
      arr = np.load(p)
      return arr.astype(np.float32)

  # Look for a directory of depth images
  dir_candidates = [
      root_dir / 'depth',
      root_dir / 'depths',
      root_dir / 'depth_maps',
      root_dir / 'Depth',
  ]
  image_exts = ['.png', '.exr', '.pfm', '.jpg', '.jpeg']
  for d in dir_candidates:
    if d.exists() and d.is_dir():
      files = sorted([p for p in d.iterdir() if p.suffix.lower() in image_exts])
      if len(files) == 0:
        continue
      frames = []
      for fp in files:
        if fp.suffix.lower() == '.exr':
          img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
          if img is None:
            continue
          if img.ndim == 3:
            img = img[..., 0]
          frames.append(img.astype(np.float32))
        elif fp.suffix.lower() == '.pfm':
          with open(fp, 'rb') as f:
            header = f.readline().decode('ascii').strip()
            dims = f.readline().decode('ascii').strip()
            scale = f.readline().decode('ascii').strip()
            w, h = map(int, dims.split())
            data = np.fromfile(f, '<f')
            img = np.reshape(data, (h, w))
          frames.append(img.astype(np.float32))
        else:
          img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
          if img is None:
            continue
          if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          frames.append(img.astype(np.float32))
      if len(frames) > 0:
        return np.stack(frames, axis=0).astype(np.float32)

  # Look for per-frame numpy depth files like frame_*.npy or depth_*.npy
  npy_frames = sorted((root_dir).glob('frame_*.npy'))
  if len(npy_frames) == 0:
    npy_frames = sorted((root_dir).glob('depth_*.npy'))
  if len(npy_frames) > 0:
    frames = []
    # Sort by numeric index if possible
    def _frame_index(p: Path) -> int:
      stem = p.stem
      if 'frame_' in stem:
        num = stem.split('frame_')[-1]
      elif 'depth_' in stem:
        num = stem.split('depth_')[-1]
      else:
        num = stem
      try:
        return int(num)
      except:
        return 0

    npy_frames = sorted(npy_frames, key=_frame_index)
    for fp in npy_frames:
      arr = np.load(fp)
      frames.append(arr.astype(np.float32))
    if len(frames) > 0:
      return np.stack(frames, axis=0).astype(np.float32)
  return None


def _load_images(root_dir: Path) -> np.ndarray | None:
  """Load a sequence of images."""
  # Prefer .npy if available
  npy_candidates = [
      root_dir / 'images.npy',
      root_dir / 'rgb.npy',
  ]
  for p in npy_candidates:
    if p.exists():
      arr = np.load(p)
      return arr.astype(np.uint8)

  # Look for a directory of images
  dir_candidates = [
      root_dir / 'images',
      root_dir / 'image',
      root_dir / 'rgb',
  ]
  image_exts = ['.png', '.jpg', '.jpeg']
  for d in dir_candidates:
    if d.exists() and d.is_dir():
      files = sorted([p for p in d.iterdir() if p.suffix.lower() in image_exts])
      if not files:
        continue
      frames = []
      for fp in files:
        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None:
          continue
        if img.ndim == 3 and img.shape[2] == 4:
          img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        frames.append(img)
      if frames:
        return np.stack(frames, axis=0)
  return None


def load_easi3r_data(args):
  """Load data for the easi3r method."""
  data_dir = args.data_dir
  scene_name = args.scene_name
  mask_path = args.mask_path
  rgb_path = args.rgb_dir
  base_dir = Path(data_dir)
  candidates = [
      base_dir / scene_name / scene_name,
      base_dir / scene_name,
      base_dir,
      base_dir / scene_name / 'NULL'
  ]
  root_dir = None
  for cand in candidates:
    if (cand / 'pred_traj.txt').exists() and (
        cand / 'pred_intrinsics.txt'
    ).exists():
      root_dir = cand
      break
  if root_dir is None:
    raise FileNotFoundError(
        f'Easi3R data directory not found for {scene_name} under {data_dir}'
    )

  poses_path = root_dir / 'pred_traj.txt'
  intrinsics_path = root_dir / 'pred_intrinsics.txt'

  poses_np = _read_tum_trajectory(poses_path)
  intrinsics_k = _read_intrinsics_txt(intrinsics_path)
  intrinsics_np = np.array([
      intrinsics_k[0, 0],
      intrinsics_k[1, 1],
      intrinsics_k[0, 2],
      intrinsics_k[1, 2],
  ])

  img_data = _load_images(root_dir)
  if img_data is None:
    img_data = _load_images(Path(os.path.join(rgb_path, scene_name)))

  depths_np = _load_depth_stack(root_dir)
  disps_np = None
  if depths_np is not None:
    depths_np = depths_np.astype(np.float32)
    depths_np = np.clip(depths_np, 1e-6, None)
    disps_np = 1.0 / depths_np

  mot_prob = None
  if mask_path is not None:
    mask_video_path = os.path.join(mask_path, scene_name, 'segmentation_mask.mp4')
    if os.path.exists(mask_video_path):
      video_reader = imageio.get_reader(mask_video_path)
      mot_prob = np.array([im for im in video_reader])
      video_reader.close()
      if mot_prob.ndim == 4:
        mot_prob = mot_prob[..., 0]
      mot_prob = 1.0 - np.float32(mot_prob > 0)
    else:
      mot_prob = np.ones(img_data.shape[:-1], dtype=np.float32)

  # Reshape to match MegaSAM's output
  h, w = 288, 672
  n = img_data.shape[0]
  
  original_height, original_width = depths_np.shape[1], depths_np.shape[2]

  # Reshape images and transpose to (N, 3, H, W)
  img_data_reshaped = np.zeros((n, h, w, 3), dtype=np.uint8)
  for i in range(n):
    img_data_reshaped[i] = cv2.resize(
        img_data[i], (w, h), interpolation=cv2.INTER_LINEAR
    )
  img_data = img_data_reshaped.transpose(0, 3, 1, 2)

  # Reshape disps to (N, H, W)
  if disps_np is not None:
    disp_data_reshaped = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
      disp_data_reshaped[i] = cv2.resize(
          disps_np[i], (w, h), interpolation=cv2.INTER_NEAREST
      )
    disps_np = disp_data_reshaped

  # Scale intrinsics to match resized depth maps
  fx_scale = w / original_width
  fy_scale = h / original_height
  intrinsics_np[0] *= fx_scale  # fx
  intrinsics_np[1] *= fy_scale  # fy
  intrinsics_np[2] *= fx_scale  # cx
  intrinsics_np[3] *= fy_scale  # cy

  # Repeat intrinsics for N frames
  intrinsics_np = np.tile(intrinsics_np, (n, 1))

  # Convert poses to (N, 7) quaternion format
  poses_quat = np.zeros((n, 7), dtype=np.float32)
  for i in range(n):
    tx, ty, tz = poses_np[i, :3, 3]
    quat = _rotation_matrix_to_quaternion(poses_np[i, :3, :3])
    poses_quat[i] = np.array([tx, ty, tz, quat[3], quat[0], quat[1], quat[2]])
  poses_np = poses_quat

  return img_data, disps_np, intrinsics_np, poses_np, mot_prob


def load_cut3r_data(args):
  """Load data for the cut3r method."""
  data_dir = args.data_dir
  scene_name = args.scene_name
  mask_path = args.mask_path
  rgb_path = args.rgb_dir
  base_dir = Path(data_dir)
  
  # Look for cut3r data structure: <data_dir>/<scene_name>/camera, depth, etc.
  root_dir = base_dir / scene_name
  camera_dir = root_dir / 'camera'
  depth_dir = root_dir / 'depth'
  
  if not camera_dir.exists():
    raise FileNotFoundError(
        f'Cut3R camera directory not found: {camera_dir}'
    )
  
  # Load camera parameters and poses from camera/*.npz files
  cam_files = sorted(list(camera_dir.glob('*.npz')))
  if len(cam_files) == 0:
    raise ValueError(f'No camera npz files found in {camera_dir}')
  
  poses_list = []
  intrinsics_list = []
  for f in cam_files:
    data = np.load(f)
    pose = data['pose'].astype(np.float32)  # 4x4 camera-to-world
    intrins = data['intrinsics'].astype(np.float32)  # 3x3
    poses_list.append(pose)
    intrinsics_list.append(intrins)
  
  poses_np = np.stack(poses_list, axis=0)
  intrinsics_k = intrinsics_list[0]  # Use first frame intrinsics
  intrinsics_np = np.array([
      intrinsics_k[0, 0],
      intrinsics_k[1, 1],
      intrinsics_k[0, 2],
      intrinsics_k[1, 2],
  ])
  
  # Load images from rgb_path
  img_data = None
  if rgb_path is not None:
    img_data = _load_images(Path(os.path.join(rgb_path, scene_name)))
  if img_data is None:
    raise FileNotFoundError(
        f'Images not found for scene {scene_name} in {rgb_path}'
    )
  
  # Load depths and convert to disparities
  depths_np = None
  disps_np = None
  if depth_dir.exists():
    depth_files = sorted(list(depth_dir.glob('*.npy')))
    if depth_files:
      depths = []
      for f in depth_files:
        d = np.load(f)
        depths.append(d)
      depths_np = np.stack(depths, axis=0).astype(np.float32)
      depths_np = np.clip(depths_np, 1e-6, None)
      disps_np = 1.0 / depths_np
  
  if disps_np is None:
    raise FileNotFoundError(
        f'Depth data not found for scene {scene_name} in {depth_dir}'
    )
  
  # Load motion probability (masks)
  mot_prob = None
  if mask_path is not None:
    mask_video_path = os.path.join(mask_path, scene_name, 'segmentation_mask.mp4')
    if os.path.exists(mask_video_path):
      video_reader = imageio.get_reader(mask_video_path)
      mot_prob = np.array([im for im in video_reader])
      video_reader.close()
      if mot_prob.ndim == 4:
        mot_prob = mot_prob[..., 0]
      mot_prob = 1.0 - np.float32(mot_prob > 0)
    else:
      mot_prob = np.ones(img_data.shape[:-1], dtype=np.float32)
  
  # Reshape to match MegaSAM's output
  h, w = 288, 672
  n = img_data.shape[0]
  
  # Cut3r intrinsics correspond to depth scale, so we need to rescale
  original_height, original_width = depths_np.shape[1], depths_np.shape[2]
  
  # Reshape images and transpose to (N, 3, H, W)
  img_data_reshaped = np.zeros((n, h, w, 3), dtype=np.uint8)
  for i in range(n):
    img_data_reshaped[i] = cv2.resize(
        img_data[i], (w, h), interpolation=cv2.INTER_LINEAR
    )
  img_data = img_data_reshaped.transpose(0, 3, 1, 2)
  
  # Reshape disps to (N, H, W)
  disp_data_reshaped = np.zeros((n, h, w), dtype=np.float32)
  for i in range(n):
    disp_data_reshaped[i] = cv2.resize(
        disps_np[i], (w, h), interpolation=cv2.INTER_NEAREST
    )
  disps_np = disp_data_reshaped
  
  # Scale intrinsics to match resized depth maps
  # Cut3r intrinsics correspond to the depth resolution
  fx_scale = w / original_width
  fy_scale = h / original_height
  intrinsics_np[0] *= fx_scale  # fx
  intrinsics_np[1] *= fy_scale  # fy
  intrinsics_np[2] *= fx_scale  # cx
  intrinsics_np[3] *= fy_scale  # cy
  
  # Repeat intrinsics for N frames
  intrinsics_np = np.tile(intrinsics_np, (n, 1))
  
  # Convert poses to (N, 7) quaternion format
  poses_quat = np.zeros((n, 7), dtype=np.float32)
  for i in range(n):
    tx, ty, tz = poses_np[i, :3, 3]
    quat = _rotation_matrix_to_quaternion(poses_np[i, :3, :3])
    poses_quat[i] = np.array([tx, ty, tz, quat[3], quat[0], quat[1], quat[2]])
  poses_np = poses_quat
  
  return img_data, disps_np, intrinsics_np, poses_np, mot_prob


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--w_grad", type=float, default=2.0, help="w_grad")
  parser.add_argument("--w_normal", type=float, default=6.0, help="w_normal")
  parser.add_argument("--w_smooth", type=float, default=0.00, help="weight for smoothness loss")
  parser.add_argument(
      "--output_dir", type=str, default="outputs_cvd", help="outputs direcotry"
  )
  parser.add_argument("--mask_path", type=str, default=None)
  parser.add_argument("--scene_name", type=str, help="scene name")
  parser.add_argument(
      "--opt_uncert",
      default=True,
      type=lambda x: (str(x).lower() == "true"),
      help="If False, we do not optimize uncertainty",
  )
  parser.add_argument("--opt_pose", default=False, type=lambda x: (str(x).lower() == "true"), help="If True, we optimize pose")
  parser.add_argument(
      "--enable_phase2",
      default=True,
      type=lambda x: (str(x).lower() == "true"),
      help="If False, we only do phase1 optimization",
  )
  parser.add_argument('--method', type=str, default='megasam', help='Method name')
  parser.add_argument(
      '--data_dir', type=str, default='/data/zhuoyuan/code/mega-sam/reconstructions', help='data dir'
  )
  parser.add_argument(
    '--rgb_dir', type=str, default=None, help='load images from the original dataset'
  )
  parser.add_argument(
    '--device', type=str, default='gpu', choices=['gpu', 'cpu'], help='device to run optimization on (gpu or cpu)'
  )

  args = parser.parse_args()

  # Set up device
  if args.device == 'gpu':
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  cache_dir = "./cache_flow"
  rootdir = args.data_dir

  output_dir = args.output_dir
  scene_name = args.scene_name
  print("***************************** ", scene_name)
  if args.method == 'megasam':
    img_data = np.load(os.path.join(rootdir, scene_name, "images.npy"))[
        :, ::-1, ...
    ]
    disp_data = (
        np.load(
            os.path.join(rootdir, scene_name.replace("_opt", ""), "disps.npy")
        )
        + 1e-6
    )
    intrinsics = np.load(os.path.join(rootdir, scene_name, "intrinsics.npy"))
    poses = np.load(os.path.join(rootdir, scene_name, "poses.npy"))
    if args.mask_path is None:
      mot_prob = np.load(os.path.join(rootdir, scene_name, "motion_prob.npy"))
      is_full_res_mot_prob = False
    else:
      is_full_res_mot_prob = True
      mask_video_path = os.path.join(
          args.mask_path, scene_name, "segmentation_mask.mp4"
      )
      if os.path.exists(mask_video_path):
        video_reader = imageio.get_reader(mask_video_path)
        mot_prob = np.array(
            [im for im in video_reader],
        )
        video_reader.close()
        if mot_prob.ndim == 4:
          mot_prob = mot_prob[..., 0]
        mot_prob = 1.0 - np.float32(mot_prob > 0)
      else:
        mot_prob = np.ones(img_data.shape[:-1], dtype=np.float32)
  elif args.method == 'easi3r':
    img_data, disp_data, intrinsics, poses, mot_prob = load_easi3r_data(args)
    is_full_res_mot_prob = True  # Assume masks are full resolution
  elif args.method == 'cut3r':
    img_data, disp_data, intrinsics, poses, mot_prob = load_cut3r_data(args)
    is_full_res_mot_prob = True  # Assume masks are full resolution
  else:
    raise ValueError(f'Unknown method: {args.method}')

  flows = np.load(
      "%s/%s/flows.npy" % (cache_dir, scene_name), allow_pickle=True
  )
  flow_masks = np.load(
      "%s/%s/flows_masks.npy" % (cache_dir, scene_name), allow_pickle=True
  )
  flow_masks = np.float32(flow_masks)
  iijj = np.load("%s/%s/ii-jj.npy" % (cache_dir, scene_name), allow_pickle=True)

  intrinsics = intrinsics[0]
  poses_th = torch.as_tensor(poses, device="cpu").float().to(device)

  K = np.eye(3)
  K[0, 0] = intrinsics[0]
  K[1, 1] = intrinsics[1]
  K[0, 2] = intrinsics[2]
  K[1, 2] = intrinsics[3]

  img_data_pt = (
      torch.from_numpy(np.ascontiguousarray(img_data)).float().to(device) / 255.0
  )
  flows = torch.from_numpy(np.ascontiguousarray(flows)).float().to(device)
  flow_masks = (
      torch.from_numpy(np.ascontiguousarray(flow_masks)).float().to(device)
  )  # .unsqueeze(1)
  iijj = torch.from_numpy(np.ascontiguousarray(iijj)).float().to(device)
  ii = iijj[0, ...].long()
  jj = iijj[1, ...].long()
  K = torch.from_numpy(K).float().to(device)

  init_disp = torch.from_numpy(disp_data).float().to(device)
  disp_data = torch.from_numpy(disp_data).float().to(device)

  assert init_disp.shape == disp_data.shape

  init_disp = torch.nn.functional.interpolate(
      init_disp.unsqueeze(1),
      scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
      mode="bilinear",
  ).squeeze(1)
  disp_data = torch.nn.functional.interpolate(
      disp_data.unsqueeze(1),
      scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
      mode="bilinear",
  ).squeeze(1)

  fg_alpha = sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
  fg_alpha = fg_alpha.squeeze(1).float() + 0.2

  cvd_prob_tensor = torch.from_numpy(mot_prob).unsqueeze(1).to(device)
  if is_full_res_mot_prob:
    target_h = int(img_data.shape[-2] * RESIZE_FACTOR)
    target_w = int(img_data.shape[-1] * RESIZE_FACTOR)
    cvd_prob = torch.nn.functional.interpolate(
        cvd_prob_tensor,
        size=(target_h, target_w),
        mode="bilinear",
    )
  else:
    cvd_prob = torch.nn.functional.interpolate(
        torch.from_numpy(mot_prob).unsqueeze(1).to(device),
        scale_factor=(4, 4),
        mode="bilinear",
    )

  cvd_prob[cvd_prob > 0.5] = 0.5
  cvd_prob = torch.clamp(cvd_prob, 1e-3, 1.0)

  # rescale intrinsic matrix to small resolution
  K_o = K.clone()
  K[0:2, ...] *= RESIZE_FACTOR
  K_inv = torch.linalg.inv(K)

  disp_data.requires_grad = False
  poses_th.requires_grad = False

  uncertainty = cvd_prob

  # First optimize scale and shift to align them
  log_scale_ = torch.log(torch.ones(init_disp.shape[0]).to(disp_data.device))
  shift_ = torch.zeros(init_disp.shape[0]).to(disp_data.device)
  log_scale_.requires_grad = True
  shift_.requires_grad = True
  uncertainty.requires_grad = True

  adam_params = [
      {"params": log_scale_, "lr": 1e-2},
      {"params": shift_, "lr": 1e-2},
  ]
  if args.opt_uncert:
    adam_params.append({"params": uncertainty, "lr": 1e-2})
  optim = torch.optim.Adam(adam_params)

  compute_normals = []
  compute_normals.append(
      NormalGenerator(disp_data.shape[-2], disp_data.shape[-1])
  )
  init_disp = torch.clamp(init_disp, 1e-3, 1e3)

  for i in range(100):
    optim.zero_grad()
    cam_c2w = SE3(poses_th).inv().matrix()
    scale_ = torch.exp(log_scale_)

    loss = consistency_loss(
        cam_c2w,
        K,
        K_inv,
        torch.clamp(
            disp_data * scale_[..., None, None] + shift_[..., None, None],
            1e-3,
            1e3,
        ),
        init_disp,
        torch.clamp(uncertainty, 1e-4, 1e3),
        flows,
        flow_masks,
        ii,
        jj,
        compute_normals,
        fg_alpha,
        device,
    )

    loss.backward()
    uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
    log_scale_.grad = torch.nan_to_num(log_scale_.grad, nan=0.0)
    shift_.grad = torch.nan_to_num(shift_.grad, nan=0.0)

    optim.step()
    print("step ", i, loss.item())

  # Then optimize depth and uncertainty
  disp_data = (
      disp_data * torch.exp(log_scale_)[..., None, None].detach()
      + shift_[..., None, None].detach()
  )
 
  if args.enable_phase2:
    init_disp = (
        init_disp * torch.exp(log_scale_)[..., None, None].detach()
        + shift_[..., None, None].detach()
    )
    init_disp = torch.clamp(init_disp, 1e-3, 1e3)
 
    disp_data.requires_grad = True
    uncertainty.requires_grad = True
    poses_th.requires_grad = args.opt_pose  # True
 
    adam_params = [
        {"params": disp_data, "lr": 5e-3},
    ]
    if args.opt_uncert:
      adam_params.append({"params": uncertainty, "lr": 5e-3})
    if args.opt_pose:
      adam_params.append({"params": poses_th, "lr": 1e-4})
    optim = torch.optim.Adam(adam_params)
 
    losses = []
    for i in range(400):
      optim.zero_grad()
      cam_c2w = SE3(poses_th).inv().matrix()
      loss = consistency_loss(
          cam_c2w,
          K,
          K_inv,
          torch.clamp(disp_data, 1e-3, 1e3),
          init_disp,
          torch.clamp(uncertainty, 1e-4, 1e3),
          flows,
          flow_masks,
          ii,
          jj,
          compute_normals,
          fg_alpha,
          device,
          w_ratio=1.0,
          w_flow=0.2,
          w_si=1,
          w_grad=args.w_grad,
          w_normal=args.w_normal,
      )
 
      loss.backward()
      disp_data.grad = torch.nan_to_num(disp_data.grad, nan=0.0)
      uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
 
      optim.step()
      print("step ", i, loss.item())
      losses.append(loss)
 
  disp_data_opt = (
      torch.nn.functional.interpolate(
          disp_data.unsqueeze(1), scale_factor=(2, 2), mode="bilinear"
      )
      .squeeze(1)
      .detach()
      .cpu()
      .numpy()
  )

  # poses_ = poses_th.detach().cpu().numpy()

  Path(output_dir).mkdir(parents=True, exist_ok=True)
  np.savez(
      "%s/%s_sgd_cvd_hr.npz" % (output_dir, scene_name),
      images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0),
      depths=np.clip(np.float16(1.0 / disp_data_opt), 1e-3, 1e2),
      intrinsic=K_o.detach().cpu().numpy(),
      cam_c2w=cam_c2w.detach().cpu().numpy(),
  )
