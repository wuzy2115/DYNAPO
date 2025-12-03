"""
Utilities for static-only COLMAP-style BA and metric scale alignment.

This module provides:
- reproject_static_point: project 3D world point to pixel coordinates
- run_colmap_style_ba: optimize static 3D points given fixed camera poses
- align_scale_to_metric: align unscaled BA to metric using a depth model

Notes:
- Camera pose matrices are camera-to-world (C2W) 4x4.
- Intrinsics are standard pixel-space 3x3 matrices.
- We do not optimize camera poses here; only 3D points per static track.
"""

from typing import Dict, List, Tuple, Callable, Optional

import numpy as np


def _invert_pose_c2w_to_w2c(pose_c2w: np.ndarray) -> np.ndarray:
    """Compute world-to-camera extrinsic from camera-to-world pose.

    Args:
        pose_c2w: 4x4 camera-to-world pose matrix.

    Returns:
        4x4 world-to-camera matrix.
    """
    rotation_c2w = pose_c2w[:3, :3]
    translation_c2w = pose_c2w[:3, 3]
    rotation_w2c = rotation_c2w.T
    translation_w2c = -rotation_w2c @ translation_c2w

    pose_w2c = np.eye(4, dtype=np.float64)
    pose_w2c[:3, :3] = rotation_w2c
    pose_w2c[:3, 3] = translation_w2c
    return pose_w2c


def _build_camera_matrix(intrinsics_k: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    """Build 3x4 camera matrix P = K [R|t] from intrinsics and C2W pose.

    Args:
        intrinsics_k: 3x3 intrinsics in pixels.
        pose_c2w: 4x4 camera-to-world matrix.

    Returns:
        3x4 projection matrix in pixel coordinates.
    """
    pose_w2c = _invert_pose_c2w_to_w2c(pose_c2w)
    rotation_w2c = pose_w2c[:3, :3]
    translation_w2c = pose_w2c[:3, 3:4]
    extrinsic_w2c = np.concatenate([rotation_w2c, translation_w2c], axis=1)
    camera_matrix = intrinsics_k @ extrinsic_w2c
    return camera_matrix


def reproject_static_point(point_3d_world: np.ndarray, camera_pose_4x4: np.ndarray, intrinsics: np.ndarray) -> Tuple[float, float]:
    """Project a 3D world point to pixel coordinates for a given camera.

    Args:
        point_3d_world: (3,) xyz in world frame.
        camera_pose_4x4: 4x4 C2W pose.
        intrinsics: 3x3 pixel-space intrinsics.

    Returns:
        (u, v) pixel coordinates (floats).
    """
    camera_matrix = _build_camera_matrix(intrinsics, camera_pose_4x4)
    point_h = np.concatenate([point_3d_world.astype(np.float64), np.array([1.0], dtype=np.float64)])
    proj = camera_matrix @ point_h
    u = float(proj[0] / proj[2])
    v = float(proj[1] / proj[2])
    return u, v


def _triangulate_point_multi_view(observations: List[Tuple[int, float, float]],
                                  intrinsics_k: np.ndarray,
                                  camera_poses_c2w: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """Linear triangulation with multiple views using DLT.

    Args:
        observations: list of (frame_idx, u, v) pixel observations.
        intrinsics_k: 3x3 intrinsics.
        camera_poses_c2w: dict frame_idx -> 4x4 C2W poses.

    Returns:
        (3,) 3D point in world or None if insufficient observations.
    """
    if len(observations) < 2:
        return None

    design_rows: List[np.ndarray] = []
    for frame_idx, u, v in observations:
        if frame_idx not in camera_poses_c2w:
            continue
        P = _build_camera_matrix(intrinsics_k, camera_poses_c2w[frame_idx])  # 3x4
        # Rows: u * P[2,:] - P[0,:] and v * P[2,:] - P[1,:]
        row1 = u * P[2, :] - P[0, :]
        row2 = v * P[2, :] - P[1, :]
        design_rows.append(row1)
        design_rows.append(row2)

    if len(design_rows) < 4:
        return None

    A = np.stack(design_rows, axis=0)  # (2N, 4)
    # Solve Ax = 0, subject to ||x||=1 via SVD
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    X_h = vh[-1]
    if abs(X_h[-1]) < 1e-8:
        return None
    X = X_h[:3] / X_h[3]
    return X.astype(np.float64)


def run_colmap_style_ba(frames: List[np.ndarray],
                        static_2d_tracks: Dict[int, List[Tuple[int, float, float]]],
                        initial_intrinsics: np.ndarray,
                        initial_poses: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Static-only BA: triangulate per-track 3D points with fixed camera poses.

    Args:
        frames: list of images (unused here, kept for API parity).
        static_2d_tracks: mapping track_id -> list of (frame_idx, u, v) pixel samples.
        initial_intrinsics: 3x3 pixel intrinsics.
        initial_poses: mapping frame_idx -> 4x4 C2W pose.

    Returns:
        (unscaled_camera_poses, unscaled_static_3d_points)
        - unscaled_camera_poses: same as input (poses fixed here)
        - unscaled_static_3d_points: mapping track_id -> (3,) world coordinates
    """
    unscaled_camera_poses: Dict[int, np.ndarray] = {k: v.copy() for k, v in initial_poses.items()}
    unscaled_static_3d_points: Dict[int, np.ndarray] = {}

    for track_id, obs_list in static_2d_tracks.items():
        point_world = _triangulate_point_multi_view(obs_list, initial_intrinsics, unscaled_camera_poses)
        if point_world is not None:
            unscaled_static_3d_points[track_id] = point_world

    return unscaled_camera_poses, unscaled_static_3d_points


def align_scale_to_metric(unscaled_camera_poses: Dict[int, np.ndarray],
                          unscaled_static_3d_points: Dict[int, np.ndarray],
                          metric_depth_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
                          camera_intrinsics: np.ndarray,
                          static_2d_tracks: Optional[Dict[int, List[Tuple[int, float, float]]]] = None,
                          reference_frame_idx: int = 0) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], float]:
    """Align scale using metric depth for a reference frame via closed-form s.

    Args:
        unscaled_camera_poses: frame_idx -> 4x4 C2W.
        unscaled_static_3d_points: track_id -> (3,) world.
        metric_depth_model: function(image, intrinsics) -> metric depth map (H, W)
        camera_intrinsics: 3x3 intrinsics in pixels.
        static_2d_tracks: optional mapping track_id -> list of (frame_idx, u, v).
        reference_frame_idx: frame index to sample depths from.

    Returns:
        (scaled_camera_poses, scaled_static_3d_points, s)
    """
    # If no 2D tracks are provided, return identity scale.
    if static_2d_tracks is None or len(unscaled_static_3d_points) == 0:
        return ({k: v.copy() for k, v in unscaled_camera_poses.items()},
                {k: v.copy() for k, v in unscaled_static_3d_points.items()},
                1.0)

    # Build correspondences between BA points and metric 3D points in world.
    # Select those tracks that have an observation in the reference frame.
    P_ba_list: List[np.ndarray] = []
    P_metric_list: List[np.ndarray] = []

    # Fetch (dummy) image for reference frame for API parity; caller ensures model handles it.
    # The metric_depth_model signature is assumed: model(image, intrinsics) -> depth (H, W)
    # The caller should pass image via closure if needed.
    depth_ref = metric_depth_model(None, camera_intrinsics)

    pose_ref_c2w = unscaled_camera_poses[reference_frame_idx]
    pose_ref_w2c = _invert_pose_c2w_to_w2c(pose_ref_c2w)
    rotation_ref_w2c = pose_ref_w2c[:3, :3]
    translation_ref_w2c = pose_ref_w2c[:3, 3]

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    for track_id, point_world_ba in unscaled_static_3d_points.items():
        if track_id not in static_2d_tracks:
            continue

        # Find an observation at the reference frame
        ref_obs = None
        for frame_idx, u, v in static_2d_tracks[track_id]:
            if frame_idx == reference_frame_idx:
                ref_obs = (u, v)
                break
        if ref_obs is None:
            continue
        u_ref, v_ref = ref_obs

        # Sample metric depth
        u_int = int(round(u_ref))
        v_int = int(round(v_ref))
        if v_int < 0 or u_int < 0 or v_int >= depth_ref.shape[0] or u_int >= depth_ref.shape[1]:
            continue
        depth_val = float(depth_ref[v_int, u_int])
        if not np.isfinite(depth_val) or depth_val <= 0:
            continue

        # Unproject to camera frame of reference
        x_cam = (u_ref - cx) * depth_val / fx
        y_cam = (v_ref - cy) * depth_val / fy
        z_cam = depth_val
        point_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

        # Transform to world: X_w = R_cw @ X_c + t_cw
        rotation_c2w = pose_ref_c2w[:3, :3]
        translation_c2w = pose_ref_c2w[:3, 3]
        point_world_metric = rotation_c2w @ point_cam + translation_c2w

        P_ba_list.append(point_world_ba.astype(np.float64))
        P_metric_list.append(point_world_metric)

    if len(P_ba_list) == 0:
        return ({k: v.copy() for k, v in unscaled_camera_poses.items()},
                {k: v.copy() for k, v in unscaled_static_3d_points.items()},
                1.0)

    P_ba = np.stack(P_ba_list, axis=0)
    P_metric = np.stack(P_metric_list, axis=0)

    # Closed-form least squares scale: s = sum(P_ba dot P_metric) / sum(||P_ba||^2)
    numerator = float(np.sum(P_ba * P_metric))
    denominator = float(np.sum(P_ba * P_ba) + 1e-8)
    s = numerator / denominator

    # Apply scale to camera translations and BA points
    scaled_camera_poses: Dict[int, np.ndarray] = {}
    for idx, pose in unscaled_camera_poses.items():
        pose_scaled = pose.copy()
        pose_scaled[:3, 3] = pose_scaled[:3, 3] * s
        scaled_camera_poses[idx] = pose_scaled

    scaled_static_points: Dict[int, np.ndarray] = {tid: pt * s for tid, pt in unscaled_static_3d_points.items()}

    return scaled_camera_poses, scaled_static_points, s


