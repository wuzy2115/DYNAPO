from matplotlib import pyplot as plt
import numpy as np
import torch
import rerun as rr
import torch.nn.functional as F
import einops
import cv2
import glob
import os
from tqdm import tqdm
import jax
import jax.numpy as jnp
from minipytorch3d.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    standardize_quaternion,
)

from anycam.visualization.common import color_tensor


# This does not produce graph breaks when compiled with torch.compile()

def custom_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = (
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    q_abs = torch.sqrt(q_abs.clamp_min(0.0))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    # out = quat_candidates[
    #     F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    # ].reshape(batch_dim + (4,))
    out = torch.gather(quat_candidates, dim=-2, index=q_abs.argmax(dim=-1).view(*batch_dim, 1, 1).expand(*batch_dim, 1, 4)).squeeze(-2)

    return standardize_quaternion(out)

def compute_pixel_tracks(flow_occs, uncert, depth, grid_size = 8, track_len = 8, stride=4, is_backward=False, mask_depth=False, imgs=None, long_tracks=False, track3d=None):
    # flow_occs: (n, f, 3, h, w)
    # uncert: (n, f, 1, h, w)

    n = 1
    f, c, h, w = flow_occs.shape
    device = flow_occs.device

    flow_occs = flow_occs.view(n, f, c, h, w)
    uncert = uncert.view(n, f, 1, h, w)
    depth = depth.view(n, f, 1, h, w)

    if is_backward:
        flow_occs = flow_occs.flip(1)
        uncert = uncert.flip(1)
        depth = depth.flip(1)
    
    initial_depths = []
    pixel_tracks = []
    uncerts = []
    indices = []
    depths = []
    rgbs = []
    visibles = []

    if long_tracks:
        track_until = f - 1
    else:
        track_until = f - track_len + 1

    for frame_idx in range(0, track_until, stride):
        x = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(1, -1).expand(grid_size, -1)
        y = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(-1, 1).expand(-1, grid_size)

        grid = torch.stack([x, y], dim=0).view(1, 2, grid_size, grid_size).expand(n, 2, grid_size, grid_size)
        grid = grid.permute(0, 2, 3, 1)

        grid_uncert = torch.zeros_like(grid[..., 0:1])

        curr_initial_depths = F.grid_sample(depth[:, frame_idx], grid, align_corners=False)
        curr_indices = [torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_idx]
        curr_tracks = [grid]
        curr_uncerts = [grid_uncert]
        # curr_uncerts = [F.grid_sample(uncert[:, frame_idx], grid, align_corners=False).permute(0, 2, 3, 1)]
        curr_depths = [F.grid_sample(depth[:, frame_idx], grid, align_corners=False).permute(0, 2, 3, 1)]
        curr_occs = [torch.ones_like(grid[..., 0:1])]
        if imgs is not None:
            curr_rgbs = [F.grid_sample(imgs[frame_idx, None], grid, align_corners=False).permute(0, 2, 3, 1)]

        for i in range(track_len-1):
            if frame_idx + i < f - 1:
                uncert_ = uncert[:, frame_idx + i, :1]
                flow_ = flow_occs[:, frame_idx + i, :2]
                occ_ = flow_occs[:, frame_idx + i, 2:3]
                depth_ = depth[:, frame_idx + i]


                grid_flow = F.grid_sample(flow_, grid, align_corners=False).permute(0, 2, 3, 1)
                grid_occ = F.grid_sample(occ_, grid, align_corners=False, padding_mode="zeros").permute(0, 2, 3, 1)
                grid_uncert = F.grid_sample(uncert_, grid, align_corners=False).permute(0, 2, 3, 1)
                grid_depth = F.grid_sample(depth_, grid, align_corners=False).permute(0, 2, 3, 1)

                valid = torch.ones_like(grid[..., 0:1], dtype=bool)

                if mask_depth:
                    max_depth = 0.95 * torch.quantile(depth_.reshape(n, 1, -1), 0.95, dim=-1, keepdim=True).reshape(n, 1, 1, 1)
                    grid_valid_depth = grid_depth < max_depth
                    valid = valid & (grid_valid_depth > .5)

                valid = valid & (grid_occ > .5)

                grid = grid + grid_flow

                valid = valid & (grid[..., :1].abs() < .99) & (grid[..., 1:2].abs() < .99)

                grid_uncert = grid_uncert + curr_uncerts[-1]
                grid_uncert[~valid] = float("inf")
                grid_depth[~valid] = 0  # or some invalid value

                curr_tracks.append(grid)
                curr_uncerts.append(grid_uncert)
                curr_indices.append(torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_idx + i + 1)
                curr_depths.append(grid_depth)
                curr_occs.append(grid_occ)

                if imgs is not None:
                    curr_rgbs.append(F.grid_sample(imgs[frame_idx + i, None], grid, align_corners=False).permute(0, 2, 3, 1))

            else:
                invalid_uncerts = grid_uncert + float("inf")
                invalid_indices = torch.zeros_like(grid[..., 0:1], dtype=torch.long) + f - 1
                invalid_depths = torch.zeros_like(grid[..., 0:1])
                invalid_occ = torch.zeros_like(grid[..., 0:1])
                curr_tracks.append(grid.clone())
                curr_uncerts.append(invalid_uncerts)
                curr_indices.append(invalid_indices)
                curr_depths.append(invalid_depths)
                curr_occs.append(invalid_occ)

                if imgs is not None:
                    curr_rgbs.append(curr_rgbs[-1].clone())

        curr_initial_depths = curr_initial_depths.reshape(n, grid_size ** 2)
        curr_tracks = torch.stack(curr_tracks, dim=1).reshape(n, track_len, grid_size ** 2, 2).permute(0, 2, 1, 3)
        curr_uncerts = torch.stack(curr_uncerts, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        curr_indices = torch.stack(curr_indices, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        curr_depths = torch.stack(curr_depths, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        curr_occs = torch.stack(curr_occs, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        if imgs is not None:
            curr_rgbs = torch.stack(curr_rgbs, dim=1).reshape(n, track_len, grid_size ** 2, 3).permute(0, 2, 1, 3)

        initial_depths.append(curr_initial_depths)
        pixel_tracks.append(curr_tracks)
        uncerts.append(curr_uncerts)
        indices.append(curr_indices)
        depths.append(curr_depths)
        visibles.append(curr_occs)
        if imgs is not None:
            rgbs.append(curr_rgbs)

    initial_depths = torch.stack(initial_depths, dim=1)
    pixel_tracks = torch.stack(pixel_tracks, dim=1)
    uncerts = torch.stack(uncerts, dim=1)
    indices = torch.stack(indices, dim=1)
    depths = torch.stack(depths, dim=1)
    visibles = torch.stack(visibles, dim=1)
    if imgs is not None:
        rgbs = torch.stack(rgbs, dim=1)

    if is_backward:
        indices = f - 1 - indices

    if imgs is not None:
        return initial_depths, pixel_tracks, uncerts, indices, depths, rgbs, visibles
    else:
        return initial_depths, pixel_tracks, uncerts, indices, depths, visibles
    
def compute_depth_tracks(depth, track_len = 8, stride=4, grid_size=16, is_backward=False, long_tracks=False):
    # flow_occs: (n, f, 3, h, w)
    # uncert: (n, f, 1, h, w)

    n = 1
    _, f, npt = depth.shape
    if is_backward:
        depth = depth.flip(1)
    
    depths = []

    if long_tracks:
        track_until = f - 1
    else:
        track_until = f - track_len + 1

    for frame_idx in range(0, track_until, stride):
        curr_depths = [depth[:, frame_idx]]
        
        for i in range(track_len-1):
            if frame_idx + i < f - 1:
                depth_ = depth[:, frame_idx + i]
                curr_depths.append(depth_)
            else:
                invalid_depths = torch.zeros_like(depth[..., 0:1])
                curr_depths.append(invalid_depths)

        curr_depths = torch.stack(curr_depths, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        
        depths.append(curr_depths)
        
    depths = torch.stack(depths, dim=1)
    
    return depths

def compute_pixel_tracks_with_track3d(flow_occs, uncert, depth, grid_size = 8, track_len = 8, stride=4, is_backward=False, mask_depth=False, imgs=None, long_tracks=False, track3d=None):
    # flow_occs: (n, f, 3, h, w)
    # uncert: (n, f, 1, h, w)

    n = 1
    f, c, h, w = flow_occs.shape
    device = flow_occs.device

    flow_occs = flow_occs.view(n, f, c, h, w)
    uncert = uncert.view(n, f, 1, h, w)
    depth = depth.view(n, f, 1, h, w)
    all_points = track3d.track3d # npt, f, 3
    visible_list = track3d.visible_list # npt, f
    cameras = track3d.cameras # f
    if is_backward:
        flow_occs = flow_occs.flip(1)
        uncert = uncert.flip(1)
        depth = depth.flip(1)
        all_points = all_points.flip(1)
        visible_list = visible_list.flip(1)
        cameras.reverse()
    
    initial_depths = []
    pixel_tracks = []
    uncerts = []
    indices = []
    depths = []
    rgbs = []
    points_2d = []
    points_3d = []
    points_3d_displacements = []
    visible_lists = []

    if long_tracks:
        track_until = f - 1
    else:
        track_until = f - track_len + 1

    for frame_idx in range(0, track_until, stride):
        x = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(1, -1).expand(grid_size, -1)
        y = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(-1, 1).expand(-1, grid_size)

        grid = torch.stack([x, y], dim=0).view(1, 2, grid_size, grid_size).expand(n, 2, grid_size, grid_size)
        grid = grid.permute(0, 2, 3, 1)

        grid_uncert = torch.zeros_like(grid[..., 0:1])
        
        points_3d_at_frame_idx = all_points[:, frame_idx:frame_idx+1, :]
        visible_list_at_frame_idx = visible_list[:, frame_idx:frame_idx+1]
        npt, nframe, _ = points_3d_at_frame_idx.shape
        points_3d_at_frame_idx = einops.rearrange(
            points_3d_at_frame_idx, 'npt nframe xyz->(npt nframe) xyz'
        )
        points_2d_at_frame_idx, valid_mask_at_frame_idx, _ = cameras[frame_idx].world_2_pix_np(
            points_3d_at_frame_idx,
            track3d.imh,
            track3d.imw,
            )
        points_2d_at_frame_idx, points_3d_at_frame_idx = map(lambda x : einops.rearrange(
            x,
            '(npt nframe) xyz->nframe npt xyz',
            npt=npt,
            nframe=nframe,
            ), (points_2d_at_frame_idx, points_3d_at_frame_idx))
        valid_mask_at_frame_idx = einops.rearrange(
            valid_mask_at_frame_idx, '(npt nframe)-> npt nframe', npt=npt, nframe=nframe
            )
        valid_mask_at_frame_idx = valid_mask_at_frame_idx & visible_list_at_frame_idx

        curr_initial_depths = F.grid_sample(depth[:, frame_idx], grid, align_corners=False)
        curr_indices = [torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_idx]
        curr_tracks = [grid]
        curr_uncerts = [grid_uncert]
        curr_2d_points = [points_2d_at_frame_idx]
        curr_3d_points = [points_3d_at_frame_idx]
        curr_3d_points_displacements = [np.zeros_like(points_3d_at_frame_idx)]
        curr_valid_mask = [valid_mask_at_frame_idx]
        # curr_uncerts = [F.grid_sample(uncert[:, frame_idx], grid, align_corners=False).permute(0, 2, 3, 1)]
        curr_depths = [F.grid_sample(depth[:, frame_idx], grid, align_corners=False).permute(0, 2, 3, 1)]
        if imgs is not None:
            curr_rgbs = [F.grid_sample(imgs[frame_idx, None], grid, align_corners=False).permute(0, 2, 3, 1)]

        for i in range(track_len-1):
            if frame_idx + i < f - 1:
                uncert_ = uncert[:, frame_idx + i, :1]
                flow_ = flow_occs[:, frame_idx + i, :2]
                occ_ = flow_occs[:, frame_idx + i, 2:3]
                depth_ = depth[:, frame_idx + i]
                points_3d_at_frame_i = all_points[:, frame_idx + i:frame_idx + i+1, :]
                visible_list_at_frame_i = visible_list[:, frame_idx + i:frame_idx + i+1]
                npt, nframe, _ = points_3d_at_frame_i.shape
                points_3d_at_frame_i = einops.rearrange(
                    points_3d_at_frame_i, 'npt nframe xyz->(npt nframe) xyz'
                )
                points_2d_at_frame_i, valid_mask_at_frame_i, _ = cameras[frame_idx].world_2_pix_np(
                    points_3d_at_frame_i,
                    track3d.imh,
                    track3d.imw,
                    )
                points_2d_at_frame_i, points_3d_at_frame_i = map(lambda x : einops.rearrange(
                    x,
                    '(npt nframe) xyz->nframe npt xyz',
                    npt=npt,
                    nframe=nframe,
                    ), (points_2d_at_frame_i, points_3d_at_frame_i))
                valid_mask_at_frame_i = einops.rearrange(
                    valid_mask_at_frame_i, '(npt nframe)-> npt nframe', npt=npt, nframe=nframe
                    )
                valid_mask_at_frame_i = valid_mask_at_frame_i & visible_list_at_frame_i
                points_3d_displacements_idx2i = points_3d_at_frame_i - points_3d_at_frame_idx
                grid_flow = F.grid_sample(flow_, grid, align_corners=False).permute(0, 2, 3, 1)
                grid_occ = F.grid_sample(occ_, grid, align_corners=False, padding_mode="zeros").permute(0, 2, 3, 1)
                grid_uncert = F.grid_sample(uncert_, grid, align_corners=False).permute(0, 2, 3, 1)
                grid_depth = F.grid_sample(depth_, grid, align_corners=False).permute(0, 2, 3, 1)

                valid = torch.ones_like(grid[..., 0:1], dtype=bool)

                if mask_depth:
                    max_depth = 0.95 * torch.quantile(depth_.reshape(n, 1, -1), 0.95, dim=-1, keepdim=True).reshape(n, 1, 1, 1)
                    grid_valid_depth = grid_depth < max_depth
                    valid = valid & (grid_valid_depth > .5)

                valid = valid & (grid_occ > .5)

                grid = grid + grid_flow

                valid = valid & (grid[..., :1].abs() < .99) & (grid[..., 1:2].abs() < .99)

                grid_uncert = grid_uncert + curr_uncerts[-1]
                grid_uncert[~valid] = float("inf")
                grid_depth[~valid] = 0  # or some invalid value

                curr_tracks.append(grid)
                curr_uncerts.append(grid_uncert)
                curr_indices.append(torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_idx + i + 1)
                curr_depths.append(grid_depth)
                curr_2d_points.append(points_2d_at_frame_i)
                curr_3d_points.append(points_3d_at_frame_i)
                curr_valid_mask.append(valid_mask_at_frame_i)
                curr_3d_points_displacements.append(points_3d_displacements_idx2i & visible_list_at_frame_i & visible_list_at_frame_idx)
                if imgs is not None:
                    curr_rgbs.append(F.grid_sample(imgs[frame_idx + i, None], grid, align_corners=False).permute(0, 2, 3, 1))

            else:
                invalid_uncerts = grid_uncert + float("inf")
                invalid_indices = torch.zeros_like(grid[..., 0:1], dtype=torch.long) + f - 1
                invalid_depths = torch.zeros_like(grid[..., 0:1])
                points_2d_at_frame_i = torch.zeros_like(points_2d_at_frame_idx)
                points_3d_at_frame_i = torch.zeros_like(points_3d_at_frame_idx)
                valid_mask_at_frame_i = torch.zeros_like(valid_mask_at_frame_idx, dtype=bool)
                points_3d_displacements_idx2i = torch.zeros_like(points_3d_at_frame_idx) + float("inf")

                curr_tracks.append(grid.clone())
                curr_uncerts.append(invalid_uncerts)
                curr_indices.append(invalid_indices)
                curr_depths.append(invalid_depths)
                curr_2d_points.append(points_2d_at_frame_i)
                curr_3d_points.append(points_3d_at_frame_i)
                curr_valid_mask.append(valid_mask_at_frame_i)
                curr_3d_points_displacements.append(points_3d_displacements_idx2i)

                if imgs is not None:
                    curr_rgbs.append(curr_rgbs[-1].clone())

        curr_initial_depths = curr_initial_depths.reshape(n, grid_size ** 2)
        curr_tracks = torch.stack(curr_tracks, dim=1).reshape(n, track_len, grid_size ** 2, 2).permute(0, 2, 1, 3)
        curr_uncerts = torch.stack(curr_uncerts, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        curr_indices = torch.stack(curr_indices, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        curr_depths = torch.stack(curr_depths, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        curr_2d_points = torch.stack(curr_2d_points)
        curr_3d_points = torch.stack(curr_3d_points)
        curr_3d_points_displacements = torch.stack(curr_3d_points_displacements)
        if imgs is not None:
            curr_rgbs = torch.stack(curr_rgbs, dim=1).reshape(n, track_len, grid_size ** 2, 3).permute(0, 2, 1, 3)

        initial_depths.append(curr_initial_depths)
        pixel_tracks.append(curr_tracks)
        uncerts.append(curr_uncerts)
        indices.append(curr_indices)
        depths.append(curr_depths)
        points_2d.append(curr_2d_points)
        points_3d.append(curr_3d_points)
        points_3d_displacements.append(curr_3d_points_displacements)
        if imgs is not None:
            rgbs.append(curr_rgbs)

    initial_depths = torch.stack(initial_depths, dim=1)
    pixel_tracks = torch.stack(pixel_tracks, dim=1)
    uncerts = torch.stack(uncerts, dim=1)
    indices = torch.stack(indices, dim=1)
    depths = torch.stack(depths, dim=1)

    if imgs is not None:
        rgbs = torch.stack(rgbs, dim=1)

    if is_backward:
        indices = f - 1 - indices

    if imgs is not None:
        return initial_depths, pixel_tracks, uncerts, indices, depths, rgbs
    else:
        return initial_depths, pixel_tracks, uncerts, indices, depths
    
def compute_pixel_tracks_full_frames(flow_occs, uncert, depth, track_len=8, stride=4, is_backward=False, mask_depth=False, imgs=None, long_tracks=False):
    # flow_occs: (n, f, 3, h, w)
    # uncert: (n, f, 1, h, w)

    n = 1
    f, c, h, w = flow_occs.shape
    device = flow_occs.device

    flow_occs = flow_occs.view(n, f, c, h, w)
    uncert = uncert.view(n, f, 1, h, w)
    depth = depth.view(n, f, 1, h, w)

    if is_backward:
        flow_occs = flow_occs.flip(1)
        uncert = uncert.flip(1)
        depth = depth.flip(1)
    
    initial_depths = []
    pixel_tracks = []
    uncerts = []
    indices = []
    depths = []
    rgbs = []
    visibles = []

    if long_tracks:
        track_until = f - 1
    else:
        track_until = f - track_len + 1

    for frame_idx in range(0, track_until, stride):
        # Generate full frame grid
        x_coords = (torch.arange(w, device=device, dtype=torch.float) + 0.5) / w * 2 - 1
        y_coords = (torch.arange(h, device=device, dtype=torch.float) + 0.5) / h * 2 - 1
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # Shape (1, 2, h, w)
        grid = grid.expand(n, 2, h, w).permute(0, 2, 3, 1)  # Shape (n, h, w, 2)

        grid_uncert = torch.zeros_like(grid[..., 0:1])

        curr_initial_depths = F.grid_sample(depth[:, frame_idx], grid, align_corners=False)
        curr_indices = [torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_idx]
        curr_tracks = [grid]
        curr_uncerts = [grid_uncert]
        curr_depths = [F.grid_sample(depth[:, frame_idx], grid, align_corners=False).permute(0, 2, 3, 1)]
        curr_occs = [torch.ones_like(grid[..., 0:1])]
        if imgs is not None:
            curr_rgbs = [F.grid_sample(imgs[frame_idx, None], grid, align_corners=False).permute(0, 2, 3, 1)]

        for i in range(track_len-1):
            if frame_idx + i < f - 1:
                uncert_ = uncert[:, frame_idx + i, :1]
                flow_ = flow_occs[:, frame_idx + i, :2]
                occ_ = flow_occs[:, frame_idx + i, 2:3]
                depth_ = depth[:, frame_idx + i]

                grid_flow = F.grid_sample(flow_, grid, align_corners=False).permute(0, 2, 3, 1)
                grid_occ = F.grid_sample(occ_, grid, align_corners=False, padding_mode="zeros").permute(0, 2, 3, 1)
                grid_uncert = F.grid_sample(uncert_, grid, align_corners=False).permute(0, 2, 3, 1)
                grid_depth = F.grid_sample(depth_, grid, align_corners=False).permute(0, 2, 3, 1)

                valid = torch.ones_like(grid[..., 0:1], dtype=bool)

                if mask_depth:
                    max_depth = 0.95 * torch.quantile(depth_.reshape(n, 1, -1), 0.95, dim=-1, keepdim=True).reshape(n, 1, 1, 1)
                    grid_valid_depth = grid_depth < max_depth
                    valid = valid & (grid_valid_depth > .5)

                valid = valid & (grid_occ > .5)

                grid = grid + grid_flow

                valid = valid & (grid[..., :1].abs() < .99) & (grid[..., 1:2].abs() < .99)

                grid_uncert = grid_uncert + curr_uncerts[-1]
                grid_uncert[~valid] = float("inf")
                grid_depth[~valid] = 0  # or some invalid value

                curr_tracks.append(grid)
                curr_uncerts.append(grid_uncert)
                curr_indices.append(torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_idx + i + 1)
                curr_depths.append(grid_depth)
                curr_occs.append(grid_occ)

                if imgs is not None:
                    curr_rgbs.append(F.grid_sample(imgs[frame_idx + i, None], grid, align_corners=False).permute(0, 2, 3, 1))

            else:
                invalid_uncerts = grid_uncert + float("inf")
                invalid_indices = torch.zeros_like(grid[..., 0:1], dtype=torch.long) + f - 1
                invalid_depths = torch.zeros_like(grid[..., 0:1])
                invalid_occ = torch.zeros_like(grid[..., 0:1])

                curr_tracks.append(grid.clone())
                curr_uncerts.append(invalid_uncerts)
                curr_indices.append(invalid_indices)
                curr_depths.append(invalid_depths)
                curr_occs.append(invalid_occ)

                if imgs is not None:
                    curr_rgbs.append(curr_rgbs[-1].clone())

        # Reshape steps for full frame
        curr_initial_depths = curr_initial_depths.reshape(n, h * w)
        curr_tracks = torch.stack(curr_tracks, dim=1).reshape(n, track_len, h * w, 2).permute(0, 2, 1, 3)
        curr_uncerts = torch.stack(curr_uncerts, dim=1).reshape(n, track_len, h * w, 1).permute(0, 2, 1, 3)
        curr_indices = torch.stack(curr_indices, dim=1).reshape(n, track_len, h * w, 1).permute(0, 2, 1, 3)
        curr_depths = torch.stack(curr_depths, dim=1).reshape(n, track_len, h * w, 1).permute(0, 2, 1, 3)
        curr_occs = torch.stack(curr_occs, dim=1).reshape(n, track_len, h * w, 1).permute(0, 2, 1, 3)

        if imgs is not None:
            curr_rgbs = torch.stack(curr_rgbs, dim=1).reshape(n, track_len, h * w, 3).permute(0, 2, 1, 3)

        initial_depths.append(curr_initial_depths)
        pixel_tracks.append(curr_tracks)
        uncerts.append(curr_uncerts)
        indices.append(curr_indices)
        depths.append(curr_depths)
        visibles.append(curr_occs)

        if imgs is not None:
            rgbs.append(curr_rgbs)

    initial_depths = torch.stack(initial_depths, dim=1)
    pixel_tracks = torch.stack(pixel_tracks, dim=1)
    uncerts = torch.stack(uncerts, dim=1)
    indices = torch.stack(indices, dim=1)
    depths = torch.stack(depths, dim=1)
    visibles = torch.stack(visibles, dim=1)

    if imgs is not None:
        rgbs = torch.stack(rgbs, dim=1)

    if is_backward:
        indices = f - 1 - indices

    if imgs is not None:
        return initial_depths, pixel_tracks, uncerts, indices, depths, rgbs, visibles
    else:
        return initial_depths, pixel_tracks, uncerts, indices, depths, visibles


def get_corr_poses(indices, poses):
    n, wc, gs, tl, _ = indices.shape
    n, seq_len, _, _ = poses.shape

    indices = indices.reshape(n, -1, 1, 1).expand(-1, -1, 4, 4)

    corr_poses = torch.gather(poses, dim=1, index=indices)

    return corr_poses


def get_corr_scales_shifts(indices, scales, shifts):
    n, wc, gs, tl, _ = indices.shape
    seq_len, _ = scales.shape
    seq_len, _ = shifts.shape

    indices = indices.reshape(n, -1, 1)

    corr_scales = torch.gather(scales[None, ...], dim=1, index=indices)
    corr_shifts = torch.gather(shifts[None, ...], dim=1, index=indices)

    return corr_scales, corr_shifts


def param_to_pose(rot, t):
    n, seq_len, c = rot.shape

    if c == 3:
        rot_mat = axis_angle_to_matrix(rot)
    else:
        rot_mat = quaternion_to_matrix(rot)

    trans = t.view(n, seq_len, 3, 1)

    pose = torch.cat((rot_mat, trans), dim=-1)
    pose = torch.cat((pose, torch.tensor([[[0, 0, 0, 1]]], device=pose.device).expand(n, seq_len, -1, -1)), dim=-2)

    return pose


def pose_to_param(pose, representation="axis-angle"):
    n, seq_len, _, _ = pose.shape

    rot_mat = pose[:, :, :3, :3]
    t = pose[:, :, :3, 3]

    if representation == "axis-angle":
        rot = matrix_to_axis_angle(rot_mat)
    elif representation == "quaternion":
        rot = custom_matrix_to_quaternion(rot_mat)
    else:
        raise ValueError("Invalid representation")

    return rot, t


def make_normalized_proj(focal_length, aspect_ratio=1.0):
    proj = torch.eye(3, device=focal_length.device) * focal_length

    # proj[0, 0] /= aspect_ratio
    proj[1, 1] *= aspect_ratio

    proj[2, 2] = 1

    inv_proj = torch.eye(3, device=focal_length.device) / focal_length

    # inv_proj[0, 0] *= aspect_ratio
    inv_proj[1, 1] /= aspect_ratio

    inv_proj[2, 2] = 1

    return proj, inv_proj


def log_ba_state(ba_poses_c2w, points=None, imgs=None, timestep=0, point_colors=None, max_dist=-1):
    n, seq_len, _, _ = ba_poses_c2w.shape

    ba_poses_c2w = ba_poses_c2w.detach()

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    rr.set_time_sequence("timestep", timestep)

    
    for i in range(seq_len):
        rr.log(f"world/cam{i:04d}", 
               rr.Transform3D(
                   translation=ba_poses_c2w[0, i, :3, 3].cpu().numpy(), 
                   mat3x3=ba_poses_c2w[0, i, :3, :3].cpu().numpy(),
                   axis_length=0.01
                ),
                )

        if imgs is not None:
            h, w = imgs.shape[-2:]

            rr.log(f"world/cam{i:04d}/pinhole", rr.Pinhole(
                resolution=[w, h],
                focal_length=w,
                image_plane_distance=0.02,
            ), static=True)

            img = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rr.log(f"world/cam{i:04d}/pinhole/img", rr.Image(img).compress(jpeg_quality=95), static=True)
            
    if points is not None:
        points_ = points[0, :].T

        std, mean = torch.std_mean(points_, dim=0, keepdim=True)

        points_[:, 0].clamp_(mean[:, 0] - 10 * std[:, 0], mean[:, 0] + 10 * std[:, 0])
        points_[:, 1].clamp_(mean[:, 1] - 10 * std[:, 1], mean[:, 1] + 10 * std[:, 1])
        points_[:, 2].clamp_(mean[:, 2] - 10 * std[:, 2], mean[:, 2] + 10 * std[:, 2])

        if max_dist > 0:
            points_.clamp_(-max_dist, max_dist)

        points_ = points_.detach().cpu().numpy()

        if point_colors is not None:
            point_colors = (point_colors[0].detach().cpu().numpy() * 255).astype(np.uint8)
            rr.log("world/points", rr.Points3D(points_, colors=point_colors))
        else:
            rr.log("world/points", rr.Points3D(points_, colors=[[0, 255, 0]]))


def log_ba_imgs(imgs, uncertainties=None, tracks=None, timestep=0, frame_idx=0):
    seq_len, c, h, w = imgs.shape

    if tracks is not None:
        cmap = plt.get_cmap('hsv')

        indices, uncerts, pixel_tracks = tracks
        grid_total = indices.shape[2]
        grid_colors = np.array([cmap((i % grid_total) / grid_total) for i in range(grid_total)])[:, :3] * 255
        grid_colors = grid_colors.astype(np.uint8)

        grid_colors = torch.tensor(grid_colors).cuda()
        grid_colors = grid_colors.view(1, 1, grid_total, 1, 3).expand(*indices.shape[:2], -1, indices.shape[3], -1)

        grid_colors = color_tensor(-uncerts.clamp(0, .1), cmap="plasma", norm=True).view(*uncerts.shape[:-1], 3).cpu()
        indices = indices.cpu()

    if timestep == -1:
        for i in range(seq_len):
            img = (imgs[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rr.set_time_sequence("timestep", i)
            rr.log(f"input/img", rr.Image(img).compress(jpeg_quality=95))

            if uncertainties is not None and i < len(uncertainties):
                uncert = uncertainties[i] 
                uncert = uncert / uncert.max()
                uncert = (uncert.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                rr.log(f"input/uncert", rr.Image(uncert).compress(jpeg_quality=95))

            if tracks is not None:

                mask = indices.reshape(-1) == i
                mask = mask & (uncerts.reshape(-1) < 1).cpu()

                tracks = pixel_tracks.reshape(-1, 2)[mask, :].cpu().numpy()
                colors = grid_colors.reshape(-1, 3)[mask, :].cpu().numpy()

                tracks[:, 0] = (tracks[:, 0] * .5 + .5) * w
                tracks[:, 1] = (tracks[:, 1] * .5 + .5) * h

                rr.log("input/tracks", rr.Points2D(tracks, colors=colors, radii=[1]))
    else:
        img = (imgs[frame_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        rr.set_time_sequence("timestep", timestep)
        rr.log(f"input/img", rr.Image(img).compress(jpeg_quality=95))
        
def _quaternion_to_rotation_matrix(q):
    # x, y, z, w = q  # This creates views which can cause issues with backward
    
    # Use indexing instead of unpacking to avoid creating views
    x = q[0].clone()  # Use clone to ensure we're creating a new tensor, not a view
    y = q[1].clone()
    z = q[2].clone()
    w = q[3].clone()
    
    # Create rotation matrix using operations that preserve gradients
    R = torch.zeros((3, 3), device=q.device)
    
    # First row
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*w*z
    R[0, 2] = 2*x*z + 2*w*y
    
    # Second row
    R[1, 0] = 2*x*y + 2*w*z
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*w*x
    
    # Third row
    R[2, 0] = 2*x*z - 2*w*y
    R[2, 1] = 2*y*z + 2*w*x
    R[2, 2] = 1 - 2*x*x - 2*y*y
    
    return R

def world_2_pix_np(xyz_world, ba_proj, ba_param_rot, ba_param_t, min_depth=0.01, normalized=True, imw=None, imh=None):
    """Project points from world frame to image space.
    
    Args:
        xyz_world (torch.Tensor): Points in world frame (N, 3).
        ba_proj (torch.Tensor): Camera intrinsic matrix (3, 3), normalized or unnormalized.
        ba_param_rot (torch.Tensor): Quaternion rotation parameters (4,).
        ba_param_t (torch.Tensor): Translation vector (3,).
        min_depth (float): Minimum valid depth.
        normalized (bool): Whether to return normalized coordinates in [-1, 1] range.
        imw (int): Image width in pixels. Required if converting between coordinate systems.
        imh (int): Image height in pixels. Required if converting between coordinate systems.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - (N, 2) tensor of projected image coordinates (normalized if normalized=True)
            - (N,) boolean mask of valid projections
    """
    # Convert quaternion to rotation matrix - avoid creating views
    # Clone the input parameters to ensure we're not working with views
    ba_param_rot_clone = ba_param_rot.clone()
    ba_param_t_clone = ba_param_t.clone()
    R = _quaternion_to_rotation_matrix(ba_param_rot_clone)
    
    # Transform points from world to camera coordinates
    xyz_cam = (R @ xyz_world.T).T + ba_param_t_clone
    
    # Get depth for z-division
    z = xyz_cam[:, 2:]  # (N, 1)
    
    # Check if we're likely dealing with a normalized projection matrix
    is_normalized_proj = (ba_proj[0, 0] <= 5.0 and ba_proj[1, 1] <= 5.0 and
                          ba_proj[0, 2] >= -2.0 and ba_proj[0, 2] <= 2.0 and
                          ba_proj[1, 2] >= -2.0 and ba_proj[1, 2] <= 2.0)
    
    # Ensure we have image dimensions if needed for coordinate conversion
    if not is_normalized_proj and normalized and (imw is None or imh is None):
        raise ValueError("Image dimensions (imw, imh) must be provided when using unnormalized matrix with normalized output")
    
    if is_normalized_proj and not normalized and (imw is None or imh is None):
        raise ValueError("Image dimensions (imw, imh) must be provided when using normalized matrix with unnormalized output")
    
    if is_normalized_proj:
        # For normalized projection matrices, we're working in NDC space [-1, 1]
        # Project directly to normalized coordinates
        x_norm = (ba_proj[0, 0] * xyz_cam[:, 0] / xyz_cam[:, 2]) + ba_proj[0, 2]
        y_norm = (ba_proj[1, 1] * xyz_cam[:, 1] / xyz_cam[:, 2]) + ba_proj[1, 2]
        xy_norm = torch.stack([x_norm, y_norm], dim=1)
        
        # Create validity mask for normalized coordinates
        valid_mask = (
            (xy_norm[:, 0] >= -1.0) & 
            (xy_norm[:, 1] >= -1.0) & 
            (xy_norm[:, 0] <= 1.0) & 
            (xy_norm[:, 1] <= 1.0) & 
            (z[:, 0] > min_depth)
        )
        
        if normalized:
            return xy_norm, valid_mask
        else:
            # Convert from normalized [-1, 1] to pixel coordinates [0, W/H]
            x_pixel = ((xy_norm[:, 0] + 1) / 2) * imw
            y_pixel = ((xy_norm[:, 1] + 1) / 2) * imh
            xy_pixel = torch.stack([x_pixel, y_pixel], dim=1)
            return xy_pixel, valid_mask
    else:
        # For unnormalized projection matrices, we're working in pixel space
        # Extract intrinsic parameters
        fx = ba_proj[0, 0]
        fy = ba_proj[1, 1]
        cx = ba_proj[0, 2]
        cy = ba_proj[1, 2]
        
        # Project to pixel coordinates
        x_pixel = (fx * xyz_cam[:, 0] / xyz_cam[:, 2]) + cx
        y_pixel = (fy * xyz_cam[:, 1] / xyz_cam[:, 2]) + cy
        xy_pixel = torch.stack([x_pixel, y_pixel], dim=1)
        
        # Use provided image dimensions or infer from intrinsics
        if imw is None:
            imw = int(cx * 2) if cx > 1 else 512  # Fallback to standard dim
        if imh is None:
            imh = int(cy * 2) if cy > 1 else 512  # Fallback to standard dim
        
        # Create validity mask for pixel coordinates
        valid_mask = (
            (xy_pixel[:, 0] >= 0) & 
            (xy_pixel[:, 1] >= 0) & 
            (xy_pixel[:, 0] < imw) & 
            (xy_pixel[:, 1] < imh) & 
            (z[:, 0] > min_depth)
        )
        
        if not normalized:
            return xy_pixel, valid_mask
        else:
            # Convert from pixel coordinates to normalized coordinates [-1, 1]
            x_norm = (x_pixel / imw) * 2 - 1
            y_norm = (y_pixel / imh) * 2 - 1
            xy_norm = torch.stack([x_norm, y_norm], dim=1)
            return xy_norm, valid_mask


def pix_2_world_np(
    xy,
    depth,
    valid_depth_min,
    valid_depth_max,
    ba_proj,
    ba_param_rot, 
    ba_param_t,
    normalized=False,
    imw=None,
    imh=None
):
    """Unproject points from image coordinates to world frame.

    Args:
        xy (torch.Tensor): Points in image space (N, 2).
        depth (torch.Tensor): Depth values for each point (N,).
        valid_depth_min (float): Minimum valid depth.
        valid_depth_max (float): Maximum valid depth.
        ba_proj (torch.Tensor): Camera intrinsic matrix (3, 3), normalized or unnormalized.
        ba_param_rot (torch.Tensor): Quaternion rotation parameters (4,).
        ba_param_t (torch.Tensor): Translation vector (3,).
        normalized (bool): Whether xy is in normalized coordinates [-1, 1] (True) 
                           or pixel coordinates (False).
        imw (int): Image width in pixels (required if coordinates need conversion).
        imh (int): Image height in pixels (required if coordinates need conversion).
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - (N, 3) tensor of unprojected world coordinates
            - (N,) boolean mask of valid projections
    """
    npt = xy.shape[0]
    assert xy.shape == (npt, 2), f"xy shape {xy.shape} is incorrect."
    assert depth.shape == (npt,), f"depth shape {depth.shape} is incorrect."
    assert ba_proj.shape == (3, 3), f"ba_proj shape {ba_proj.shape} is incorrect."
    assert ba_param_rot.shape == (4,), f"ba_param_rot shape {ba_param_rot.shape} is incorrect."
    assert ba_param_t.shape == (3,), f"ba_param_t shape {ba_param_t.shape} is incorrect."
    
    # Check if we're likely dealing with a normalized projection matrix
    is_normalized_proj = (ba_proj[0, 0] <= 20.0 and ba_proj[1, 1] <= 20.0 and
                          ba_proj[0, 2] >= -2.0 and ba_proj[0, 2] <= 2.0 and
                          ba_proj[1, 2] >= -2.0 and ba_proj[1, 2] <= 2.0)
    
    # Ensure we have image dimensions if needed for coordinate conversion
    if not normalized and is_normalized_proj and (imw is None or imh is None):
        raise ValueError("Image dimensions (imw, imh) must be provided when using normalized projection matrix with pixel coordinates")
    
    if normalized and not is_normalized_proj and (imw is None or imh is None):
        raise ValueError("Image dimensions (imw, imh) must be provided when using unnormalized projection matrix with normalized coordinates")
    
    # Extract intrinsic parameters
    fx = ba_proj[0, 0]
    fy = ba_proj[1, 1]
    cx = ba_proj[0, 2]
    cy = ba_proj[1, 2]
    
    # For unnormalized projection matrices, infer image dimensions if not provided
    if not is_normalized_proj and (imw is None or imh is None):
        if imw is None:
            imw = int(cx * 2) if cx > 1 else 512  # Fallback to standard dim
        if imh is None:
            imh = int(cy * 2) if cy > 1 else 512  # Fallback to standard dim
    
    # Handle coordinate conversion based on projection type and input format
    if is_normalized_proj:
        # Working with normalized projection matrix
        if normalized:
            # Input coordinates already in normalized [-1, 1] space
            x_ndc = xy[:, 0]
            y_ndc = xy[:, 1]
        else:
            # Convert from pixel to normalized coordinates using provided dimensions
            x_ndc = (xy[:, 0] / imw) * 2 - 1
            y_ndc = (xy[:, 1] / imh) * 2 - 1
        
        # Compute camera coordinates using normalized intrinsics
        Z = depth
        X_cam = (x_ndc - cx) * Z / fx
        Y_cam = (y_ndc - cy) * Z / fy
    else:
        # Working with unnormalized (pixel-space) projection matrix
        if normalized:
            # Convert from normalized to pixel space using provided dimensions
            x_pixel = ((xy[:, 0] + 1) / 2) * imw
            y_pixel = ((xy[:, 1] + 1) / 2) * imh
        else:
            # Already in pixel coordinates
            x_pixel = xy[:, 0]
            y_pixel = xy[:, 1]
        
        # Compute camera coordinates using pixel-space intrinsics
        Z = depth
        X_cam = (x_pixel - cx) * Z / fx
        Y_cam = (y_pixel - cy) * Z / fy
    
    # Stack camera coordinates
    Z_cam = Z
    xyz_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=1)
    
    # Convert quaternion to rotation matrix - avoid creating views
    # Clone the input quaternion to ensure we're not working with views
    ba_param_rot_clone = ba_param_rot.clone()
    R = _quaternion_to_rotation_matrix(ba_param_rot_clone)
    
    # Transform points from camera to world coordinates
    # For camera-to-world transformation: world_point = R^T * (camera_point - t)
    # This is the inverse of the world-to-camera transform in world_2_pix_np
    xyz_world = torch.matmul(R.transpose(0, 1), (xyz_cam - ba_param_t).T).T
    
    # Compute valid mask
    valid_mask = (Z > valid_depth_min) & (Z < valid_depth_max)
    
    return xyz_world, valid_mask

def compute_motion_scores(adjusted_points, visible_list, window=8, rot_ba_window=None, t_ba_window=None, ba_proj=None, imw=None, imh=None):
    """Compute motion scores based on 2D projected displacements.
    
    Args:
        adjusted_points: (B, T, 3) tensor of 3D points
        visible_list: (B, T) tensor of visibility masks
        window: Number of frames to look back
        rot_ba_window: List of rotation matrices for each frame
        t_ba_window: List of translation vectors for each frame
        ba_proj: Projection matrix (normalized or unnormalized)
        imw: Image width in pixels
        imh: Image height in pixels
    
    Returns:
        Mean of 90th percentile displacements across frames in pixel space,
        ready to be compared with motion_threshold_track3d.
    """
    B, T, _ = adjusted_points.shape
    motion = []
    
    # Check if ba_proj is normalized (assumes format detection works reliably)
    is_normalized_proj = (ba_proj[0, 0] <= 5.0 and ba_proj[1, 1] <= 5.0 and
                          ba_proj[0, 2] >= -2.0 and ba_proj[0, 2] <= 2.0 and
                          ba_proj[1, 2] >= -2.0 and ba_proj[1, 2] <= 2.0)
    
    # Ensure we have image dimensions if working with normalized matrix
    if is_normalized_proj and (imw is None or imh is None):
        raise ValueError("Image dimensions (imw, imh) must be provided when using a normalized projection matrix")
    
    # Ensure visible_list has the right shape
    if visible_list.dim() == 3:
        visible_list = visible_list.squeeze(-1)
    
    for t in range(T):
        if t == 0:  # Skip first frame as there's no previous frame
            continue
            
        start = max(0, t-window)
        if start == t:  # Need at least one previous frame
            continue
            
        # Project all points in the window using current frame's camera
        all_projected_points = []
        all_valid_masks = []
        
        # First project current frame points
        points_t = adjusted_points[:, t]
        # Use normalized=False to work directly in pixel space when using unnormalized matrix
        xy_t, valid_t = world_2_pix_np(
            points_t, ba_proj, rot_ba_window[t], t_ba_window[t], 
            normalized=is_normalized_proj, imw=imw, imh=imh
        )
        
        # Then project points from previous frames
        for s in range(start, t):
            points_s = adjusted_points[:, s]
            # Use normalized=False to work directly in pixel space when using unnormalized matrix
            xy_s, valid_s = world_2_pix_np(
                points_s, ba_proj, rot_ba_window[t], t_ba_window[t], 
                normalized=is_normalized_proj, imw=imw, imh=imh
            )
            all_projected_points.append(xy_s)
            all_valid_masks.append(valid_s)
        
        if len(all_projected_points) > 0:
            # Stack all projected points except current frame
            xy_prev = torch.stack(all_projected_points, dim=1)  # (B, t-start, 2)
            valid_prev = torch.stack(all_valid_masks, dim=1)    # (B, t-start)
            
            # Calculate 2D displacements
            disp_2d = xy_prev - xy_t.unsqueeze(1)               # (B, t-start, 2)
            
            # If using normalized coordinates, convert to pixel space for consistent comparison
            if is_normalized_proj:
                # Convert normalized displacement to pixel displacement
                pixel_disp = torch.zeros_like(disp_2d)
                pixel_disp[..., 0] = disp_2d[..., 0] * imw / 2
                pixel_disp[..., 1] = disp_2d[..., 1] * imh / 2
                distances_2d = torch.norm(pixel_disp, dim=-1)   # (B, t-start)
            else:
                # Already in pixel space, compute distances directly
                distances_2d = torch.norm(disp_2d, dim=-1)      # (B, t-start)
            
            # Combine visibility masks (both frames must be visible and valid projections)
            visibility_window = visible_list[:, start:t]         # (B, t-start)
            visibility_t = visible_list[:, t]                    # (B)
            
            valid = (
                valid_prev & 
                valid_t.unsqueeze(1) & 
                visibility_window & 
                visibility_t.unsqueeze(1)
            )
            
            # Calculate 90th percentile of valid displacements
            if valid.any():
                valid_distances = distances_2d[valid]
                if len(valid_distances) > 0:
                    # Filter out any NaN or Inf values that may have occurred during computation
                    valid_distances = valid_distances[torch.isfinite(valid_distances)]
                    if len(valid_distances) > 0:
                        motion.append(torch.quantile(valid_distances, 0.9))
    
    if len(motion) > 0:
        return torch.stack(motion).mean()
    else:
        # Still return the dependency to maintain gradient flow
        return torch.tensor(0.0, device=adjusted_points.device)

def compute_reg_loss(sigma, initial_inv_depth_ba_window, lambda_reg):
    # Convert inverse depth to depth with improved numerical stability
    epsilon = 1e-6  # Increased epsilon for better stability
    
    # Clamp values to prevent extreme values
    sigma_clamped = torch.clamp(sigma, -1e3, 1e3)
    initial_inv_depth_clamped = torch.clamp(initial_inv_depth_ba_window, 1e-5, 1e5)
    
    # Safely compute depths
    depth_sigma = 1 / (torch.abs(initial_inv_depth_clamped) + epsilon) + sigma_clamped
    initial_depth = 1 / (torch.abs(initial_inv_depth_clamped) + epsilon)
    
    # Use a more stable squared error
    diff = initial_depth - depth_sigma
    # Clip extremely large differences
    diff_clipped = torch.clamp(diff, -1e2, 1e2)
    # Use abs instead of square for very large values to prevent overflow
    large_values_mask = torch.abs(diff) > 10.0
    l = torch.mean(
        torch.where(
            large_values_mask,
            torch.abs(diff_clipped), 
            diff_clipped ** 2
        )
    )
    
    return lambda_reg * l

def normalized_to_pixel_matrix(norm_mat, imw, imh):
    """
    Convert a normalized projection matrix to an unnormalized (pixel-space) matrix.
    
    Args:
        norm_mat (torch.Tensor): Normalized projection matrix (3x3)
        imw (int): Image width in pixels
        imh (int): Image height in pixels
        
    Returns:
        torch.Tensor: Unnormalized projection matrix (3x3) for pixel coordinates
    """
    pixel_mat = norm_mat.clone()
    pixel_mat[0, 0] = norm_mat[0, 0] * (imw / 2)  # fx * (imw/2)
    pixel_mat[1, 1] = norm_mat[1, 1] * (imh / 2)  # fy * (imh/2)
    pixel_mat[0, 2] = (norm_mat[0, 2] + 1) * (imw / 2)  # cx transformed to pixels
    pixel_mat[1, 2] = (norm_mat[1, 2] + 1) * (imh / 2)  # cy transformed to pixels
    return pixel_mat

def pixel_to_normalized_matrix(pixel_mat, imw, imh):
    """
    Convert an unnormalized (pixel-space) projection matrix to a normalized matrix.
    
    Args:
        pixel_mat (torch.Tensor): Unnormalized projection matrix (3x3)
        imw (int): Image width in pixels
        imh (int): Image height in pixels
        
    Returns:
        torch.Tensor: Normalized projection matrix (3x3) for normalized coordinates
    """
    norm_mat = pixel_mat.clone()
    norm_mat[0, 0] = pixel_mat[0, 0] / (imw / 2)  # fx / (imw/2)
    norm_mat[1, 1] = pixel_mat[1, 1] / (imh / 2)  # fy / (imh/2)
    norm_mat[0, 2] = (pixel_mat[0, 2] / (imw / 2)) - 1  # cx transformed to [-1,1]
    norm_mat[1, 2] = (pixel_mat[1, 2] / (imh / 2)) - 1  # cy transformed to [-1,1]
    return norm_mat

# Usage example for coordinate transformation functions
def example_coordinate_transforms():
    """
    Example demonstrating how to use the coordinate transformation functions
    with both normalized and unnormalized intrinsic matrices.
    
    This function shows two main scenarios:
    1. Using a normalized intrinsic matrix (values in [-1,1] range)
    2. Using an unnormalized intrinsic matrix (pixel coordinates)
    
    And demonstrates proper conversion between coordinate systems.
    """
    import torch
    import numpy as np
    
    # Define image dimensions
    imw = 640
    imh = 480
    
    # Create some example 3D points in world space
    xyz_world = torch.tensor([
        [1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
        [-1.0, -1.0, 5.0]
    ], dtype=torch.float32)
    
    # Example camera parameters
    ba_param_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)  # Identity rotation
    ba_param_t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)         # No translation
    
    # Example 1: Using a normalized projection matrix (NDC space)
    # Focal length 1.2 in normalized coordinates
    norm_proj = torch.tensor([
        [1.2, 0.0, 0.0],
        [0.0, 1.2, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    # Project 3D points to 2D image points using normalized matrix
    xy_norm, valid_norm = world_2_pix_np(
        xyz_world, norm_proj, ba_param_rot, ba_param_t,
        normalized=True, imw=imw, imh=imh
    )
    
    print("Normalized projection matrix results:")
    print("Projected 2D points (normalized coords):", xy_norm)
    print("Valid projections:", valid_norm)
    
    # Convert to pixel coordinates
    xy_pixel, _ = world_2_pix_np(
        xyz_world, norm_proj, ba_param_rot, ba_param_t,
        normalized=False, imw=imw, imh=imh
    )
    
    print("Projected 2D points (pixel coords):", xy_pixel)
    
    # Example 2: Using an unnormalized projection matrix (pixel space)
    # Same focal length but in pixel coordinates
    unnorm_proj = normalized_to_pixel_matrix(norm_proj, imw, imh)
    
    # Project 3D points to 2D image points using unnormalized matrix
    xy_pixel2, valid_pixel = world_2_pix_np(
        xyz_world, unnorm_proj, ba_param_rot, ba_param_t,
        normalized=False, imw=imw, imh=imh
    )
    
    print("\nUnnormalized projection matrix results:")
    print("Projected 2D points (pixel coords):", xy_pixel2)
    print("Valid projections:", valid_pixel)
    
    # Convert to normalized coordinates
    xy_norm2, _ = world_2_pix_np(
        xyz_world, unnorm_proj, ba_param_rot, ba_param_t,
        normalized=True, imw=imw, imh=imh
    )
    
    print("Projected 2D points (normalized coords):", xy_norm2)
    
    # Example of unprojection: pixel to world coordinates
    # Create some example depth values
    depths = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)
    
    # Unproject using normalized matrix
    xyz_world_norm, valid_unproj_norm = pix_2_world_np(
        xy_norm, depths, 0.1, 10.0, norm_proj, ba_param_rot, ba_param_t,
        normalized=True, imw=imw, imh=imh
    )
    
    print("\nUnprojection results:")
    print("Unprojected 3D points (from normalized coords):", xyz_world_norm)
    print("Valid unprojections:", valid_unproj_norm)
    
    # Unproject using unnormalized matrix
    xyz_world_pixel, valid_unproj_pixel = pix_2_world_np(
        xy_pixel2, depths, 0.1, 10.0, unnorm_proj, ba_param_rot, ba_param_t,
        normalized=False, imw=imw, imh=imh
    )
    
    print("Unprojected 3D points (from pixel coords):", xyz_world_pixel)
    print("Valid unprojections:", valid_unproj_pixel)
    
    # Test matrix conversion consistency
    pixel_from_norm = normalized_to_pixel_matrix(norm_proj, imw, imh)
    norm_from_pixel = pixel_to_normalized_matrix(unnorm_proj, imw, imh)
    
    print("\nConverting between normalized and unnormalized matrices:")
    print("Original normalized matrix:\n", norm_proj)
    print("Converted to pixel matrix:\n", pixel_from_norm)
    print("Original pixel matrix:\n", unnorm_proj)
    print("Converted to normalized matrix:\n", norm_from_pixel)

@torch.no_grad()
def compute_depth_flow(model, imgs=None, imgs0=None, imgs1=None, megasam_disps=None):
    """
    Compute depth and flow for image sequences.
    
    Args:
        model: Model containing depth_predictor and image_processor
        imgs: Full image sequence (optional)
        imgs0: First images in pairs (optional)
        imgs1: Second images in pairs (optional)
        megasam_disps: Pre-computed disparities from MegaSam (optional)
    """
    seq_imgs = []
    seq_depths = []
    seq_flow_occs_fwd = []
    seq_flow_occs_bwd = []

    if imgs is not None:
        imgs0 = imgs[:-1]
        imgs1 = imgs[1:]
    else:
        assert imgs0 is not None
        assert imgs1 is not None

    for (i, (img0, img1)) in tqdm(list(enumerate(zip(imgs0, imgs1)))):
        img_pair = torch.stack([img0, img1]).unsqueeze(0).cuda()

        images_ip_fwd, images_ip_bwd = model.image_processor(img_pair * 2 - 1, data={})

        # Use MegaSam disparities if available, otherwise use depth predictor
        if megasam_disps is not None:
            # Use pre-computed disparities from MegaSam
            disp = megasam_disps[i]  # Get disparity for current image
            
            # Align disparity shape with image shape if needed
            img_h, img_w = img0.shape[-2:]
            disp_h, disp_w = disp.shape[-2:]
            
            if disp_h != img_h or disp_w != img_w:
                # Resize disparity to match image dimensions
                if disp.dim() == 2:
                    disp = disp.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims for interpolation
                elif disp.dim() == 3:
                    disp = disp.unsqueeze(0)  # Add batch dim
                
                disp = torch.nn.functional.interpolate(
                    disp, 
                    size=(img_h, img_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Remove extra dimensions
                if disp.dim() == 4:
                    disp = disp.squeeze(0).squeeze(0)  # Remove batch and channel dims
                elif disp.dim() == 3:
                    disp = disp.squeeze(0)  # Remove batch dim
                
                if i == 0:  # Log only for first image to avoid spam
                    print(f"Resized MegaSam disparity from {disp_h}x{disp_w} to {img_h}x{img_w}")
            
            depth = 1.0 / disp.clamp_min(1e-6)  # Convert disparity to depth
            
            # Ensure depth is on the correct device and has the right shape
            if depth.device != torch.device('cuda'):
                depth = depth.cuda()
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)  # Add batch dimension
            
            if i == 0:  # Log only for first image to avoid spam
                print(f"Using MegaSam disparities for depth computation - disparity shape: {disp.shape}, depth shape: {depth.shape}")
        else:
            # Fall back to depth predictor
            depth = model.depth_predictor(img0.unsqueeze(0).cuda())
            depth = 1 / depth[0].clamp_min(1e-3)
            
            if i == 0:  # Log only for first image to avoid spam
                print(f"Using depth predictor for depth computation - depth shape: {depth.shape}")

        seq_imgs.append(img0.cpu())

        if imgs is not None:
            seq_flow_occs_fwd.append(images_ip_fwd[0, :(1 if i != len(imgs0)-1 else 2), 3:6].cpu())
            seq_flow_occs_bwd.append(images_ip_bwd[0, (1 if i != 0 else 0):, 3:6].cpu())
        else:
            seq_flow_occs_fwd.append(images_ip_fwd[0, :1, 3:6].cpu())
            seq_flow_occs_bwd.append(images_ip_bwd[0, 1:, 3:6].cpu())

        seq_depths.append(depth.cpu())

    if imgs is not None:
        # Handle the last image
        if megasam_disps is not None:
            # Use pre-computed disparities from MegaSam for the last image
            disp = megasam_disps[-1]  # Get disparity for last image
            
            # Align disparity shape with image shape if needed
            img_h, img_w = img1.shape[-2:]
            disp_h, disp_w = disp.shape[-2:]
            
            if disp_h != img_h or disp_w != img_w:
                # Resize disparity to match image dimensions
                if disp.dim() == 2:
                    disp = disp.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims for interpolation
                elif disp.dim() == 3:
                    disp = disp.unsqueeze(0)  # Add batch dim
                
                disp = torch.nn.functional.interpolate(
                    disp, 
                    size=(img_h, img_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Remove extra dimensions
                if disp.dim() == 4:
                    disp = disp.squeeze(0).squeeze(0)  # Remove batch and channel dims
                elif disp.dim() == 3:
                    disp = disp.squeeze(0)  # Remove batch dim
                
                print(f"Resized MegaSam disparity for last image from {disp_h}x{disp_w} to {img_h}x{img_w}")
            
            depth = 1.0 / disp.clamp_min(1e-6)  # Convert disparity to depth
            
            # Ensure depth is on the correct device and has the right shape
            if depth.device != torch.device('cuda'):
                depth = depth.cuda()
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)  # Add batch dimension
            
            print(f"Using MegaSam disparities for last image - disparity shape: {disp.shape}, depth shape: {depth.shape}")
        else:
            # Fall back to depth predictor for the last image
            depth = model.depth_predictor(img1.unsqueeze(0).cuda())
            depth = 1 / depth[0].clamp_min(1e-3)
            
            print(f"Using depth predictor for last image - depth shape: {depth.shape}")

        seq_imgs.append(img1.cpu())
        seq_depths.append(depth.cpu())

    seq_imgs = torch.stack(seq_imgs, dim=0)
    seq_depths = torch.cat(seq_depths, dim=0)
    seq_flow_occs_fwd = torch.cat(seq_flow_occs_fwd, dim=0)
    seq_flow_occs_bwd = torch.cat(seq_flow_occs_bwd, dim=0)

    return seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd



@torch.no_grad()
def load_precomputed_depths_for_ba(imgs, mono_depth_path, metric_depth_path, seq_name, square_crop_params=None):
    """
    Load precomputed depths specifically for BA refinement.
    Returns depths in metric format (not disparity) aligned to the image sequence.
    
    Args:
        imgs: Image sequence tensor
        mono_depth_path: Path to mono depth files  
        metric_depth_path: Path to metric depth files
        seq_name: Sequence name
        square_crop_params: Dict with crop parameters if square cropping was applied
    """
    
    n_frames = len(imgs)
    
    # Load pre-computed mono disparity and metric depth files
    mono_disp_paths = sorted(glob.glob(os.path.join(mono_depth_path, seq_name, "*.npy")))
    metric_depth_paths = sorted(glob.glob(os.path.join(metric_depth_path, seq_name, "*.npz")))
    
    if len(mono_disp_paths) == 0:
        raise FileNotFoundError(f"No mono depth files found in {os.path.join(mono_depth_path, seq_name)}")
    if len(metric_depth_paths) == 0:
        raise FileNotFoundError(f"No metric depth files found in {os.path.join(metric_depth_path, seq_name)}")
    
    # Ensure we have depth files for all frames we need
    mono_disp_paths = mono_disp_paths[:n_frames]
    metric_depth_paths = metric_depth_paths[:n_frames]
    
    # Calculate alignment parameters (scale and shift) following Sintel logic
    scales = []
    shifts = []
    mono_disp_list = []
    
    for mono_disp_file, metric_depth_file in zip(mono_disp_paths, metric_depth_paths):
        da_disp = np.float32(np.load(mono_disp_file))
        uni_data = np.load(metric_depth_file)
        metric_depth = uni_data["depth"]
        
        # Resize mono disparity to match metric depth resolution
        da_disp = cv2.resize(
            da_disp,
            (metric_depth.shape[1], metric_depth.shape[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        mono_disp_list.append(da_disp)
        
        gt_disp = 1.0 / (metric_depth + 1e-8)
        
        # Avoid some bug from UniDepth
        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2
        
        # Avoid cases where sky dominates entire video
        sky_ratio = np.sum(da_disp < 0.02) / (da_disp.shape[0] * da_disp.shape[1])
        
        if sky_ratio > 0.4:
            non_sky_mask = da_disp > 0.02
            gt_disp_ms = gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
            da_disp_ms = da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
        else:
            gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
            da_disp_ms = da_disp - np.median(da_disp) + 1e-8
            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp - scale * da_disp)
        
        scales.append(scale)
        shifts.append(shift)
    
    # Select median alignment parameters
    ss_product = np.array(scales) * np.array(shifts)
    med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))
    
    align_scale = scales[med_idx]
    align_shift = shifts[med_idx]
    normalize_scale = np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98) / 2.0
    
    # Process frames and compute aligned depths
    seq_precomputed_depths = []
    
    for i in range(n_frames):
        mono_disp = mono_disp_list[i]
        
        # Apply alignment to get depth in metric format
        depth = np.clip(
            1.0 / (1.0 / normalize_scale * (align_scale * mono_disp + align_shift)), 
            0.0, 1e4
        )
        depth[depth < 1e-2] = 0.0
        
        # Convert to tensor and match image dimensions
        depth = torch.as_tensor(depth).float()
        
        # Get target image dimensions
        img_h, img_w = imgs[i].shape[1], imgs[i].shape[2]
        
        # Resize depth to match image dimensions
        if depth.shape[0] != img_h or depth.shape[1] != img_w:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0), 
                size=(img_h, img_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
        
        # Apply square crop if it was used
        if square_crop_params is not None:
            h_start = square_crop_params['h_start']
            w_start = square_crop_params['w_start'] 
            h_end = square_crop_params['h_end']
            w_end = square_crop_params['w_end']
            depth = depth[h_start:h_end, w_start:w_end]
        
        seq_precomputed_depths.append(depth.unsqueeze(0).cpu())
    
    seq_precomputed_depths = torch.cat(seq_precomputed_depths, dim=0)
    return seq_precomputed_depths



@torch.no_grad()
def compute_depth_flow_from_precomputed(model, imgs=None, imgs0=None, imgs1=None, 
                                      mono_depth_path=None, metric_depth_path=None, 
                                      seq_name=None):
    """
    Load pre-computed depths from files and align them, similar to Sintel evaluation logic.
    Returns the same format as compute_depth_flow for seamless integration.
    """
    
    seq_imgs = []
    seq_depths = []
    seq_flow_occs_fwd = []
    seq_flow_occs_bwd = []

    if imgs is not None:
        imgs0 = imgs[:-1]
        imgs1 = imgs[1:]
        n_frames = len(imgs)
    else:
        assert imgs0 is not None
        assert imgs1 is not None
        n_frames = len(imgs0) + 1

    # Load pre-computed mono disparity and metric depth files
    mono_disp_paths = sorted(glob.glob(os.path.join(mono_depth_path, seq_name, "*.npy")))
    metric_depth_paths = sorted(glob.glob(os.path.join(metric_depth_path, seq_name, "*.npz")))
    
    if len(mono_disp_paths) == 0:
        raise FileNotFoundError(f"No mono depth files found in {os.path.join(mono_depth_path, seq_name)}")
    if len(metric_depth_paths) == 0:
        raise FileNotFoundError(f"No metric depth files found in {os.path.join(metric_depth_path, seq_name)}")
    
    # Ensure we have depth files for all frames we need
    mono_disp_paths = mono_disp_paths[:n_frames]
    metric_depth_paths = metric_depth_paths[:n_frames]
    
    # Calculate alignment parameters (scale and shift) following Sintel logic
    scales = []
    shifts = []
    mono_disp_list = []
    
    for mono_disp_file, metric_depth_file in zip(mono_disp_paths, metric_depth_paths):
        da_disp = np.float32(np.load(mono_disp_file))
        uni_data = np.load(metric_depth_file)
        metric_depth = uni_data["depth"]
        
        # Resize mono disparity to match metric depth resolution
        da_disp = cv2.resize(
            da_disp,
            (metric_depth.shape[1], metric_depth.shape[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        mono_disp_list.append(da_disp)
        
        gt_disp = 1.0 / (metric_depth + 1e-8)
        
        # Avoid some bug from UniDepth
        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2
        
        # Avoid cases where sky dominates entire video
        sky_ratio = np.sum(da_disp < 0.02) / (da_disp.shape[0] * da_disp.shape[1])
        
        if sky_ratio > 0.4:
            non_sky_mask = da_disp > 0.02
            gt_disp_ms = gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
            da_disp_ms = da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
        else:
            gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
            da_disp_ms = da_disp - np.median(da_disp) + 1e-8
            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp - scale * da_disp)
        
        scales.append(scale)
        shifts.append(shift)
    
    # Select median alignment parameters
    ss_product = np.array(scales) * np.array(shifts)
    med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))
    
    align_scale = scales[med_idx]
    align_shift = shifts[med_idx]
    normalize_scale = np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98) / 2.0
    
    # Process frames and compute aligned depths
    for (i, (img0, img1)) in tqdm(list(enumerate(zip(imgs0, imgs1)))):
        img_pair = torch.stack([img0, img1]).unsqueeze(0).cuda()

        # Still compute flow from image processor
        images_ip_fwd, images_ip_bwd = model.image_processor(img_pair * 2 - 1, data={})
        
        # Load and align pre-computed depth instead of predicting
        mono_disp = mono_disp_list[i]
        
        # Apply alignment: depth = 1 / (1/normalize_scale * (align_scale * mono_disp + align_shift))
        depth = np.clip(
            1.0 / (1.0 / normalize_scale * (align_scale * mono_disp + align_shift)), 
            0.0, 1e4
        )
        depth[depth < 1e-2] = 0.0
        
        # Convert to tensor and match original image resolution
        depth = torch.as_tensor(depth).float()
        
        # Resize depth to match image dimensions
        img_h, img_w = img0.shape[1], img0.shape[2]
        if depth.shape[0] != img_h or depth.shape[1] != img_w:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0), 
                size=(img_h, img_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
        
        seq_imgs.append(img0.cpu())

        if imgs is not None:
            seq_flow_occs_fwd.append(images_ip_fwd[0, :(1 if i != len(imgs0)-1 else 2), 3:6].cpu())
            seq_flow_occs_bwd.append(images_ip_bwd[0, (1 if i != 0 else 0):, 3:6].cpu())
        else:
            seq_flow_occs_fwd.append(images_ip_fwd[0, :1, 3:6].cpu())
            seq_flow_occs_bwd.append(images_ip_bwd[0, 1:, 3:6].cpu())

        seq_depths.append(depth.unsqueeze(0).cpu())

    if imgs is not None:
        # Handle the last frame
        mono_disp = mono_disp_list[-1]
        depth = np.clip(
            1.0 / (1.0 / normalize_scale * (align_scale * mono_disp + align_shift)), 
            0.0, 1e4
        )
        depth[depth < 1e-2] = 0.0
        depth = torch.as_tensor(depth).float()
        
        # Resize depth to match image dimensions
        img_h, img_w = img1.shape[1], img1.shape[2]
        if depth.shape[0] != img_h or depth.shape[1] != img_w:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0), 
                size=(img_h, img_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()

        seq_imgs.append(img1.cpu())
        seq_depths.append(depth.unsqueeze(0).cpu())

    seq_imgs = torch.stack(seq_imgs, dim=0)
    seq_depths = torch.cat(seq_depths, dim=0)
    seq_flow_occs_fwd = torch.cat(seq_flow_occs_fwd, dim=0)
    seq_flow_occs_bwd = torch.cat(seq_flow_occs_bwd, dim=0)

    return seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd



@torch.compile(mode="reduce-overhead", fullgraph=True)
def compute_loss(ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length, pixel_tracks, ba_indices, uncerts, w, h, loss_mask, max_uncert, rotation_representation="quaternion"):
    n, wc, gs, tl, _ = pixel_tracks.shape

    fl = (ba_param_focal_length * 2).exp()
    fl = fl.clamp(0.1, 10)

    # n, seq_len, 4, 4
    ba_poses_c2w = param_to_pose(ba_param_rot, ba_param_t)

    ba_poses_w2c = torch.inverse(ba_poses_c2w)

    ba_proj, ba_inv_proj = make_normalized_proj(fl, w/h)
    # ba_inv_proj  = torch.inverse(ba_proj)

    ba_proj = ba_proj.unsqueeze(0)
    ba_inv_proj = ba_inv_proj.unsqueeze(0)

    xy = pixel_tracks[:, :, :, 0]
    xyz = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    xyz = xyz.view(n, -1, 3).permute(0, 2, 1)

    xyz_cam = ba_inv_proj @ xyz
    xyz_cam = xyz_cam * (1 / ba_param_inv_depth.view(n, 1, -1).clamp_min(1e-4))
    xyz_cam[ba_param_inv_depth.view(n, 1, -1).expand(-1, 3, -1) < 1e-4] = 0

    xyzh_cam = torch.cat([xyz_cam, torch.ones_like(xyz_cam[:, :1])], dim=1)

    anchor_poses_c2w = get_corr_poses(ba_indices[:, :, :, :1], ba_poses_c2w)

    xyzh_world = anchor_poses_c2w @ xyzh_cam.reshape(n, 4, -1).permute(0, 2, 1).reshape(n, -1, 4, 1)

    ref_poses_w2c = get_corr_poses(ba_indices[:, :, :, 1:], ba_poses_w2c)

    xyzh_world_exp = xyzh_world.reshape(n, wc, gs, 1, 4).expand(n, wc, gs, tl-1, 4).reshape(n, -1, 4, 1)

    xyzh_cam = ref_poses_w2c @ xyzh_world_exp

    xyzh_cam = xyzh_cam.reshape(n, -1, 4).permute(0, 2, 1)

    xyz = xyzh_cam[:, :3, :] / xyzh_cam[:, 2:3, :]
    xyz[xyzh_cam[:, 2:3, :].view(n, 1, -1).expand(-1, 3, -1) < 1e-4] = 0

    xy = (ba_proj @ xyz)[:, :2]

    xy = xy.permute(0, 2, 1).view(n, wc, gs, tl-1, 2)

    dist = xy - pixel_tracks[:, :, :, 1:, :]

    # dist = dist.norm(dim=-1)

    dist = dist.abs().mean(dim=-1)

    # repr_loss = dist / (uncerts[:, :, :, 1:, 0] + 1e-4) ** 2
    # repr_loss = dist / (uncerts[:, :, :, 1:, 0] + 1e-4)
    # repr_loss = dist

    repr_loss = (max_uncert - uncerts[:, :, :, 1:, 0]).clamp_min(0) * dist

    loss_mask_filter = loss_mask

    if max_uncert > 0:
        loss_mask_filter = loss_mask & (uncerts[:, :, :, 1:, 0] < max_uncert)

    repr_loss[~loss_mask_filter] = 0

    
    curr_rel_poses = torch.inverse(ba_poses_c2w[:, :-1]) @ ba_poses_c2w[:, 1:]            
    curr_rel_rot, curr_rel_t = pose_to_param(curr_rel_poses, rotation_representation)
    curr_rel_t_abs = ba_poses_c2w[:, 1:, :3, 3] - ba_poses_c2w[:, :-1, :3, 3]
    # ref_rel_rot, ref_rel_t = pose_to_param(rel_poses, rotation_representation)

    # smoothness_loss_rot = (curr_rel_rot[:, 1:] - curr_rel_rot[:, :-1]).abs().mean()
    # smoothness_loss_t = (curr_rel_t[:, 1:] - curr_rel_t[:, :-1]).abs().mean()
    smoothness_loss_t = (curr_rel_t_abs[:, 1:] - curr_rel_t_abs[:, :-1]).abs().mean()

    smoothness_loss = smoothness_loss_t

    return repr_loss, smoothness_loss, ba_proj, xyzh_world



def compute_static_loss(adjusted_points, masks, w_static):
    """
    Compute static loss WITHIN each track independently using vectorized operations.
    
    Args:
        adjusted_points: Tensor of shape (B, T, 3) where B is the number of tracks, T is the track length
        masks: Boolean mask of shape (B, T, 1) indicating valid points
        w_static: Weight for static loss
        
    Returns:
        Static loss summed over all tracks, preserving track identity
    """
    # Safety check for NaN values
    if torch.isnan(adjusted_points).any():
        print("WARNING: NaN values detected in adjusted_points in compute_static_loss")
        return torch.tensor(1e-5, device=adjusted_points.device, requires_grad=True)
    
    B, T, _ = adjusted_points.shape
    
    # Reshape masks for easier handling
    masks_flat = masks.squeeze(-1)  # (B, T)
    
    # Clamp input values for numerical stability
    adjusted_points_clamped = torch.clamp(adjusted_points, -1e4, 1e4)
    
    # Create a tensor to accumulate all losses
    total_loss = torch.tensor(0.0, device=adjusted_points.device, requires_grad=True)
    
    # For each time step pair, compute distances between valid points
    # This matrix approach avoids loops but still respects track identity
    valid_counts = masks_flat.sum(dim=1)  # (B,)
    nonzero_tracks = valid_counts > 1  # Tracks with at least 2 valid points
    
    if not nonzero_tracks.any():
        return total_loss
        
    # Only process tracks with enough valid points
    points_valid = adjusted_points_clamped[nonzero_tracks]  # (B', T, 3)
    masks_valid = masks_flat[nonzero_tracks]  # (B', T)
    
    # Create a mask for valid pairs - both points must be valid
    # Shape: (B', T, T)
    valid_pairs = masks_valid.unsqueeze(2) & masks_valid.unsqueeze(1)
    
    # Compute pairwise differences for all time steps within each track
    # Shape: (B', T, T, 3)
    pairwise_diff = points_valid.unsqueeze(2) - points_valid.unsqueeze(1)
    pairwise_diff_clamped = torch.clamp(pairwise_diff, -1e2, 1e2)
    
    # Compute squared distances
    # Shape: (B', T, T)
    pairwise_dist = torch.sum(pairwise_diff_clamped ** 2, dim=-1)
    
    # Apply mask to only include valid pairs
    # Shape: (B', T, T)
    masked_dist = pairwise_dist * valid_pairs.float()
    
    # Sum distances for each track and normalize
    # Shape: (B',)
    track_pair_counts = valid_pairs.float().sum(dim=(1, 2))
    track_losses = masked_dist.sum(dim=(1, 2)) / torch.clamp(track_pair_counts, min=1.0)
    
    # Return weighted average of track losses
    return w_static * track_losses.mean()

def compute_dynamic_loss(adjusted_points, masks, ray_dir, w_dynamic, deltas=[1,3,5]):
    """
    Compute dynamic loss to enforce smooth motion using vectorized operations.
    
    Args:
        adjusted_points: Tensor of shape (B, T, 3) - 3D points for each track
        masks: Boolean mask of shape (B, T, 1) - valid points
        ray_dir: Tensor of shape (B, T, 3) - ray directions
        w_dynamic: Weight for dynamic loss
        deltas: List of frame offsets to compute acceleration
        
    Returns:
        Dynamic loss computed efficiently with matrix operations and properly normalized by sequence length
    """
    # Safety check for NaN values
    if torch.isnan(adjusted_points).any() or torch.isnan(ray_dir).any():
        print("WARNING: NaN values detected in inputs to compute_dynamic_loss")
        return torch.tensor(1e-5, device=adjusted_points.device, requires_grad=True)
    
    # Clamp input values for stability
    adjusted_points_clamped = torch.clamp(adjusted_points, -1e4, 1e4)
    ray_dir_clamped = torch.clamp(ray_dir, -1.0, 1.0)
    
    # Normalize ray direction
    ray_dir_norm = torch.norm(ray_dir_clamped, dim=-1, keepdim=True) + 1e-6
    ray_dir_normalized = ray_dir_clamped / ray_dir_norm
    
    # Initialize loss
    total_loss = torch.tensor(0.0, device=adjusted_points.device)
    masks_flat = masks.squeeze(-1)  # (B, T)
    B, T, _ = adjusted_points_clamped.shape
    
    # Following JAX implementation's normalization approach more closely
    # Track the total valid acceleration terms for proper normalization
    total_valid_terms = 0
    
    # Process each delta value without loops
    for delta in deltas:
        # Create padded versions using advanced indexing
        if T <= delta:
            continue  # Skip if sequence is too short for this delta
            
        # Extract points at t-delta, t, t+delta positions
        points_plus = adjusted_points_clamped[:, :T-delta]  # (B, T-delta, 3)
        points_center = adjusted_points_clamped[:, delta:T]  # (B, T-delta, 3)
        
        # Handle boundary conditions
        if 2*delta >= T:
            # If 2*delta exceeds sequence length, use reflection padding
            points_minus = torch.flip(adjusted_points_clamped[:, :delta], dims=[1])
        else:
            points_minus = adjusted_points_clamped[:, 2*delta:T+delta if T+delta < adjusted_points_clamped.shape[1] else None]
            
        # Ensure all tensors have the same size by trimming to minimum length
        min_len = min(points_plus.shape[1], points_center.shape[1], points_minus.shape[1])
        points_plus = points_plus[:, :min_len]  # (B, min_len, 3) 
        points_center = points_center[:, :min_len]  # (B, min_len, 3)
        points_minus = points_minus[:, :min_len]  # (B, min_len, 3)
        
        # Create corresponding masks
        masks_plus = masks_flat[:, :T-delta][:, :min_len]  # (B, min_len)
        masks_center = masks_flat[:, delta:T][:, :min_len]  # (B, min_len)
        if 2*delta < T:
            masks_minus = masks_flat[:, 2*delta:T+delta][:, :min_len]
        else:
            masks_minus = masks_flat[:, :delta][:, :min_len]
        
        # Combined mask for valid acceleration computation
        valid_mask = masks_plus & masks_center & masks_minus  # (B, min_len)
        
        # Calculate acceleration vectors
        accel = points_plus - 2 * points_center + points_minus  # (B, min_len, 3)
        accel_clamped = torch.clamp(accel, -1e2, 1e2)
        
        # Project acceleration onto ray direction
        ray_dir_subset = ray_dir_normalized[:, delta:T][:, :min_len]  # (B, min_len, 3)
        accel_dot = torch.sum(accel_clamped * ray_dir_subset, dim=-1)  # (B, min_len)
        
        # Apply mask and compute squared acceleration directly (closer to JAX implementation)
        masked_accel_dot = accel_dot * valid_mask.float()  # (B, min_len)
        squared_accel = masked_accel_dot ** 2  # (B, min_len)
        
        # Sum across all valid terms without per-track normalization
        # This matches JAX implementation's nansum approach
        delta_loss = squared_accel.sum()
        
        # Count valid terms for normalization
        num_valid = valid_mask.float().sum()
        if num_valid > 0:
            total_valid_terms += num_valid
            total_loss = total_loss + delta_loss
    
    # In case all deltas produced no valid points
    if total_valid_terms == 0:
        return torch.tensor(0.0, device=adjusted_points.device)
    
    # Normalize by total number of valid terms across all deltas
    # This ensures consistent normalization regardless of sequence length
    normalized_loss = total_loss / total_valid_terms
    
    # Scale by w_dynamic
    return w_dynamic * normalized_loss

def compute_dynamic_loss_xyz(adjusted_points, masks, w_dynamic, deltas=[1,3,5]):
    """
    Compute dynamic loss to enforce smooth motion using vectorized operations,
    penalizing the magnitude of the 3D acceleration vector.
    
    Args:
        adjusted_points: Tensor of shape (B, T, 3) - 3D points for each track
        masks: Boolean mask of shape (B, T, 1) - valid points
        w_dynamic: Weight for dynamic loss
        deltas: List of frame offsets to compute acceleration
        
    Returns:
        Dynamic loss computed efficiently with matrix operations and properly normalized by sequence length
    """
    # Safety check for NaN values
    if torch.isnan(adjusted_points).any():
        print("WARNING: NaN values detected in inputs to compute_dynamic_loss_xyz")
        return torch.tensor(1e-5, device=adjusted_points.device, requires_grad=True)
    
    # Clamp input values for stability
    adjusted_points_clamped = torch.clamp(adjusted_points, -1e4, 1e4)
    
    # Initialize loss
    total_loss = torch.tensor(0.0, device=adjusted_points.device)
    masks_flat = masks.squeeze(-1)  # (B, T)
    B, T, _ = adjusted_points_clamped.shape
    
    # Track the total valid acceleration terms for proper normalization
    total_valid_terms = 0
    
    # Process each delta value without loops
    for delta in deltas:
        # Create padded versions using advanced indexing
        if T <= 2 * delta: # Need at least 2*delta + 1 points for centered difference
            continue  # Skip if sequence is too short for this delta
            
        # Extract points at t-delta, t, t+delta positions
        # points_minus corresponds to adjusted_points_clamped[:, t-delta]
        # points_center corresponds to adjusted_points_clamped[:, t]
        # points_plus corresponds to adjusted_points_clamped[:, t+delta]
        
        # To align with accel = points_plus - 2 * points_center + points_minus logic
        # where accel is at time 't' (center point index)
        # points_minus_accel refers to point at t-delta
        # points_center_accel refers to point at t
        # points_plus_accel refers to point at t+delta

        points_minus_accel = adjusted_points_clamped[:, :-2*delta] # (B, T-2*delta, 3)
        points_center_accel = adjusted_points_clamped[:, delta:-delta] # (B, T-2*delta, 3)
        points_plus_accel = adjusted_points_clamped[:, 2*delta:] # (B, T-2*delta, 3)

        # Create corresponding masks
        masks_minus_accel = masks_flat[:, :-2*delta] 
        masks_center_accel = masks_flat[:, delta:-delta]
        masks_plus_accel = masks_flat[:, 2*delta:]
        
        # Combined mask for valid acceleration computation
        # All three points (t-delta, t, t+delta) must be valid
        valid_mask = masks_minus_accel & masks_center_accel & masks_plus_accel # (B, T-2*delta)
        
        # Calculate acceleration vectors
        # accel(t) = P(t+delta) - 2*P(t) + P(t-delta)
        accel = points_plus_accel - 2 * points_center_accel + points_minus_accel  # (B, T-2*delta, 3)
        accel_clamped = torch.clamp(accel, -1e2, 1e2) # Clamp for stability
        
        # Compute the squared L2 norm of the acceleration vector
        # squared_accel_norm = torch.sum(accel_clamped ** 2, dim=-1) # (B, T-2*delta)
        # Use L1 norm for less sensitivity to outliers, as in some other BA implementations
        l1_accel_norm = torch.sum(torch.abs(accel_clamped), dim=-1) # (B, T-2*delta)


        # Apply mask 
        masked_accel_norm = l1_accel_norm * valid_mask.float() # (B, T-2*delta)
        
        # Sum across all valid terms
        delta_loss = masked_accel_norm.sum()
        
        # Count valid terms for normalization
        num_valid = valid_mask.float().sum()
        if num_valid > 0:
            total_valid_terms += num_valid
            total_loss = total_loss + delta_loss
            
    if total_valid_terms == 0:
        return torch.tensor(0.0, device=adjusted_points.device)
        
    normalized_loss = total_loss / total_valid_terms
    
    return w_dynamic * normalized_loss


def dilate_zeros_torch(mask: torch.Tensor, window_size: int):
    """PyTorch implementation of zero dilation for 1D masks"""
    # Input mask: (N, T)
    # Add batch dimension and channel dimension for conv2d
    mask = mask.float().unsqueeze(1)  # (N, 1, T)
    
    # Create inversion and padding
    inv_mask = 1 - mask
    pad = (window_size//2, window_size - window_size//2 - 1)
    inv_mask_padded = F.pad(inv_mask, pad, mode="replicate")
    
    # Create max pooling kernel
    pool = torch.nn.MaxPool1d(window_size, stride=1, padding=0)
    max_filtered = pool(inv_mask_padded)
    
    # Crop back to original size
    if pad[1] > 0:
        max_filtered = max_filtered[..., :-pad[1]]
    
    # Invert back
    dilated_mask = 1 - max_filtered
    return (dilated_mask > 0.5).squeeze(1)  # (N, T)

def dilate_zeros(mask, window_size):
  """Dilate zeros in a 1D mask array.

  Args:
      mask (jnp.ndarray): 1D array of 0s and 1s.
      window_size (int): Size of the dilation window.

  Returns:
      jnp.ndarray: Dilated mask array.
  """
  # Invert the mask: zeros become ones, ones become zeros
  inv_mask = 1 - mask

  # Calculate padding to keep the output size the same
  pad_before = window_size // 2
  pad_after = window_size - pad_before - 1
  padding = [(pad_before, pad_after)]

  # Apply a maximum filter using reduce_window
  max_filtered = jax.lax.reduce_window(
      inv_mask,
      init_value=0.0,
      computation=jax.lax.max,
      window_dimensions=(window_size,),
      window_strides=(1,),
      padding=padding,
  )

  # Invert back to get the dilated mask
  dilated_mask = 1 - max_filtered

  return dilated_mask



def get_ray_directions(coords, proj_matrix):
    """
    Calculate ray directions from pixel coordinates and camera projection matrix.
    
    Args:
        coords: Tensor of pixel coordinates of shape (B, 2)
        proj_matrix: Camera projection matrix of shape (3, 3)
        
    Returns:
        Normalized ray directions of shape (B, 3)
    """
    # Create homogeneous coordinates
    homogeneous = torch.cat([coords, torch.ones_like(coords[:, :1])], dim=-1)  # (B, 3)
    
    # Get ray direction by multiplying with inverse projection matrix
    inv_proj = torch.inverse(proj_matrix)
    ray_dir = torch.matmul(inv_proj, homogeneous.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    
    # Normalize ray directions
    ray_dir = ray_dir / (torch.norm(ray_dir, dim=-1, keepdim=True) + 1e-8)
    
    return ray_dir

def get_3d_points(rays, depths, camera_to_world):
    """
    Compute 3D world points from ray directions, depths, and camera-to-world transform.
    
    Args:
        rays: Normalized ray directions in camera space of shape (B, 3)
        depths: Depth values of shape (B)
        camera_to_world: Camera-to-world transformation matrix of shape (4, 4)
        
    Returns:
        3D points in world space of shape (B, 3)
    """
    # Scale rays by depth
    points_camera = rays * depths.unsqueeze(-1)  # (B, 3)
    
    # Create homogeneous coordinates
    points_camera_homogeneous = torch.cat([points_camera, torch.ones_like(depths.unsqueeze(-1))], dim=-1)  # (B, 4)
    
    # Transform to world space
    points_world_homogeneous = torch.matmul(camera_to_world, points_camera_homogeneous.unsqueeze(-1)).squeeze(-1)  # (B, 4)
    
    # Convert back to 3D coordinates
    points_world = points_world_homogeneous[:, :3]  # (B, 3)
    
    return points_world

def param_to_pose_single(rot_param, t_param):
    """
    Convert rotation and translation parameters to a single 4x4 transformation matrix.
    
    Args:
        rot_param: Rotation parameters (quaternion or axis-angle)
        t_param: Translation parameters
        
    Returns:
        4x4 transformation matrix
    """
    # Use _quaternion_to_rotation_matrix to convert rotation parameters to matrix
    rot_matrix = _quaternion_to_rotation_matrix(rot_param)
    
    # Create 4x4 transformation matrix
    transform = torch.eye(4, device=rot_param.device)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = t_param
    
    return transform

def compute_reg_loss(sigma, initial_inv_depth_ba_window, lambda_reg):
    """
    Compute regularization loss for track depth adjustments using vectorized operations.
    
    Args:
        sigma: Tensor of shape (B, T, 1) - depth adjustments for each track
        initial_inv_depth_ba_window: Tensor of shape (B, T, 1) - inverse depths
        lambda_reg: Weight for regularization loss
        
    Returns:
        Regularization loss preserving track identity
    """
    # Convert inverse depth to depth with epsilon for numerical stability
    epsilon = 1e-6  # Small epsilon to prevent division by zero
    
    # Clamp values to prevent extreme values
    sigma_clamped = torch.clamp(sigma, -1e3, 1e3)
    initial_inv_depth_clamped = torch.clamp(initial_inv_depth_ba_window, 1e-5, 1e5)
    
    # Safely compute depths
    depth_sigma = 1 / (torch.abs(initial_inv_depth_clamped) + epsilon) + sigma_clamped
    initial_depth = 1 / (torch.abs(initial_inv_depth_clamped) + epsilon)
    
    # Use a more stable squared error
    diff = initial_depth - depth_sigma
    # Clip extremely large differences
    diff_clipped = torch.clamp(diff, -1e2, 1e2)
    
    # Use abs instead of square for very large values to prevent overflow
    large_values_mask = torch.abs(diff) > 10.0
    
    # Create mask for valid values (not NaN or Inf)
    valid_mask = ~torch.isnan(diff_clipped) & ~torch.isinf(diff_clipped)
    
    # Apply appropriate loss function based on magnitude
    loss_values = torch.where(
        large_values_mask,
        torch.abs(diff_clipped), 
        diff_clipped ** 2
    )
    
    # Apply valid mask
    masked_loss = loss_values * valid_mask.float()
    
    # Calculate per-track loss
    # First compute valid counts per track
    valid_counts = valid_mask.float().sum(dim=(1, 2))  # (B,)
    
    # Only include tracks with valid points
    nonzero_tracks = valid_counts > 0
    
    if not nonzero_tracks.any():
        return torch.tensor(0.0, device=sigma.device, requires_grad=True)
    
    # Compute sums per track and normalize
    track_sums = masked_loss.sum(dim=(1, 2))  # (B,)
    track_losses = track_sums[nonzero_tracks] / torch.clamp(valid_counts[nonzero_tracks], min=1.0)
    
    # Return average loss across all valid tracks
    return lambda_reg * track_losses.mean()



def verify_tracks_for_nan(
      tracks: np.ndarray, 
      visibles: np.ndarray, 
      depths: np.ndarray, 
      valid_depth_min: float, 
      valid_depth_max: float
  ) -> np.ndarray:
    """
    Verify tracks to filter out those that would result in NaN values in 3D coordinates.
    
    Args:
      tracks: 2D tracks array of shape (npt, nframe, 2)
      visibles: visibility mask of shape (npt, nframe)
      depths: depth maps of shape (nframe, imh, imw)
      valid_depth_min: minimum valid depth value
      valid_depth_max: maximum valid depth value
    
    Returns:
      np.ndarray: Boolean mask of shape (npt,) indicating which tracks are valid
    """
    npt, nframe, _ = tracks.shape
    _, imh, imw = depths.shape
    valid_track_mask = np.ones(npt, dtype=bool)
    
    for track_idx in range(npt):
      track_valid = True
      
      for frame_idx in range(nframe):
        # Only check frames where the track is supposed to be visible
        if not visibles[track_idx, frame_idx]:
          continue
          
        # Get the 2D coordinates for this track at this frame
        xy = tracks[track_idx, frame_idx]  # shape (2,)
        
        # Check if coordinates are within image bounds
        if xy[0] < 0 or xy[1] < 0 or xy[0] >= imw or xy[1] >= imh:
          continue
          
        # Get the depth value at this location
        x_query = int(np.clip(np.round(xy[0]), 0, imw - 1))
        y_query = int(np.clip(np.round(xy[1]), 0, imh - 1))
        depth_value = depths[frame_idx, y_query, x_query]
        
        # Check if depth value would cause NaN in 3D coordinates
        if (np.isnan(depth_value) or 
            np.isinf(depth_value) or 
            depth_value <= 0 or
            depth_value < valid_depth_min or 
            depth_value > valid_depth_max):
          track_valid = False
          break
      
      valid_track_mask[track_idx] = track_valid
    
    return valid_track_mask


def compute_motion_scores_per_point_frame(track3d, visible_list, rot_ba_window, t_ba_window, ba_proj, imw, imh, tracks_leave_trace=16):
    """
    Compute motion scores per point and per frame, similar to get_scene_motion_2d_displacement.
    
    Args:
        track3d: (npt, window_len, 3) tensor of 3D points
        visible_list: (npt, window_len) tensor of visibility masks
        rot_ba_window: List of rotation parameters for each frame
        t_ba_window: List of translation parameters for each frame
        ba_proj: Projection matrix
        imw: Image width
        imh: Image height
        tracks_leave_trace: Number of frames over which to compute displacement
    
    Returns:
        displacement: (npt, window_len) array of max 2D displacements
    """
    npt, window_len, _ = track3d.shape
    displacement = torch.zeros_like(visible_list, dtype=torch.float32)
    
    # Check if ba_proj is normalized
    is_normalized_proj = (ba_proj[0, 0] <= 5.0 and ba_proj[1, 1] <= 5.0 and
                          ba_proj[0, 2] >= -2.0 and ba_proj[0, 2] <= 2.0 and
                          ba_proj[1, 2] >= -2.0 and ba_proj[1, 2] <= 2.0)
    
    for t in range(window_len):
        s_start = max(0, t - tracks_leave_trace)
        s_end = t + 1  # Include current frame
        s_list = list(range(s_start, s_end))
        L = len(s_list)
        
        if L < 2:
            # Not enough frames to compute displacement
            continue
            
        # Extract positions and visibilities for relevant frames
        positions = track3d[:, s_list, :]  # Shape: (npt, L, 3)
        visibilities = visible_list[:, s_list]  # Shape: (npt, L)
        
        # Project all positions using the camera at frame t
        all_points_2d = []
        all_valid_masks = []
        
        for s_idx, s in enumerate(s_list):
            points_s = positions[:, s_idx, :]  # (npt, 3)
            points_2d_s, valid_mask_s = world_2_pix_np(
                points_s, ba_proj, rot_ba_window[t], t_ba_window[t],
                normalized=is_normalized_proj, imw=imw, imh=imh
            )
            all_points_2d.append(points_2d_s)
            all_valid_masks.append(valid_mask_s)
        
        # Stack all projected points
        points_2d = torch.stack(all_points_2d, dim=1)  # (npt, L, 2)
        valid_mask = torch.stack(all_valid_masks, dim=1)  # (npt, L)
        
        # Extract positions and masks at time t (last frame in the sequence)
        points_2d_t = points_2d[:, -1, :]  # Shape: (npt, 2)
        valid_mask_t = valid_mask[:, -1]   # Shape: (npt,)
        visibilities_t = visibilities[:, -1]  # Shape: (npt,)
        
        # Compute displacements to previous frames
        deltas = points_2d[:, :-1, :] - points_2d_t.unsqueeze(1)  # Shape: (npt, L-1, 2)
        
        # Convert to pixel space if needed
        if is_normalized_proj:
            pixel_deltas = torch.zeros_like(deltas)
            pixel_deltas[..., 0] = deltas[..., 0] * imw / 2
            pixel_deltas[..., 1] = deltas[..., 1] * imh / 2
            distances = torch.norm(pixel_deltas, dim=2)  # Shape: (npt, L-1)
        else:
            distances = torch.norm(deltas, dim=2)  # Shape: (npt, L-1)
        
        # Validity mask
        valid = (
            (valid_mask[:, :-1] > 0.5) & 
            (valid_mask_t.unsqueeze(1) > 0.5) & 
            (visibilities[:, :-1] > 0.5) & 
            (visibilities_t.unsqueeze(1) > 0.5)
        )
        
        # Apply validity mask
        distances = distances * valid.float()
        
        # Compute maximum displacement
        max_displacement, _ = torch.max(distances, dim=1)  # Shape: (npt,)
        displacement[:, t] = max_displacement
    
    return displacement 

@torch.no_grad()
def compute_alltracker_pixel_tracks(images, uncertainties, depths, ba_window=8, overlap=6, grid_size=16, device="cuda", image_processor=None):
    """
    Compute pixel tracks using AllTracker with windowed sequences, following the same logic as bundle adjustment.
    
    Args:
        images: Input image sequence (seq_len, c, h, w)
        uncertainties: Uncertainty maps (seq_len, 1, h, w)  
        depths: Depth maps (seq_len, 1, h, w)
        ba_window: Window size for sequence processing
        overlap: Overlap between consecutive windows
        grid_size: Grid size for track sampling
        device: Device to use
        image_processor: AllTracker image processor instance
    
    Returns:
        Dictionary containing extracted tracks for all windows and full sequence
    """
    
    if image_processor is None:
        raise ValueError("AllTracker image_processor must be provided")
    
    seq_len, c, h, w = images.shape
    
    # Normalize images to [-1, 1] range as expected by image processor
    if images.max() > 1.0:
        images = images / 255.0
    images = images * 2.0 - 1.0
    
    # Results storage
    windowed_tracks = []
    windowed_visibles = []
    windowed_indices = []
    windowed_depths = []
    windowed_rgbs = []
    windowed_uncerts = []
    windowed_initial_depths = []
    
    # Process windowed sequences (following BA logic)
    optimized_until = 1
    
    while optimized_until < seq_len:
        ba_window_start = max(optimized_until - overlap, 0)
        ba_window_end = min(ba_window_start + ba_window, seq_len)
        
        # Extract window sequence
        window_imgs = images[ba_window_start:ba_window_end]  # (window_len, c, h, w)
        window_len = window_imgs.shape[0]
        
        if window_len < 2:
            optimized_until = ba_window_end
            continue
            
        print(f"Processing AllTracker window: frames {ba_window_start} to {ba_window_end-1} (length: {window_len})")
        
        # Add batch dimension for AllTracker
        window_imgs_batch = window_imgs.unsqueeze(0).to(device)  # (1, window_len, c, h, w)
        
        # Process with AllTracker
        
        # extract tracks of the entire sequence
        if hasattr(image_processor, 'extract_alltracker_tracks'):
            tracks, visibles = image_processor.extract_alltracker_tracks(window_imgs_batch, grid_size)
            # tracks: (1, grid_size^2, window_len, 2)
            # visibles: (1, grid_size^2, window_len, 1)
            
            # Extract tracks in format compatible with existing pipeline
            pixel_tracks, track_visibles, track_indices, track_depths, track_rgbs, track_uncerts, initial_depths = convert_alltracker_tracks_to_format(
                tracks, visibles, window_imgs_batch, 
                uncertainties[ba_window_start:ba_window_end].unsqueeze(0),
                depths[ba_window_start:ba_window_end].unsqueeze(0),
                ba_window_start, device
            )
        else:
            # extract tracks in a pairwise way
            if hasattr(image_processor, 'process_sequence_with_alltracker'):
                flows_fwd, flows_bwd = image_processor.process_sequence_with_alltracker(window_imgs_batch)
            else:
                # Fallback to pairwise processing if sequence method not available
                flows_fwd_list = []
                flows_bwd_list = []
                
                for i in range(window_len - 1):
                    img0 = window_imgs_batch[:, i]    # (1, c, h, w)
                    img1 = window_imgs_batch[:, i+1]  # (1, c, h, w)
                    
                    flow_fwd, flow_bwd = image_processor.flow_alltracker(img0, img1)
                    flows_fwd_list.append(flow_fwd)
                    flows_bwd_list.append(flow_bwd)
                
                flows_fwd = torch.stack(flows_fwd_list, dim=1)  # (1, window_len-1, 2, h, w)
                flows_bwd = torch.stack(flows_bwd_list, dim=1)  # (1, window_len-1, 2, h, w)
            
            # Extract tracks from flows using the first frame as query
            pixel_tracks, track_visibles, track_indices, track_depths, track_rgbs, track_uncerts, initial_depths = extract_tracks_from_alltracker_flows(
                flows_fwd, flows_bwd, window_imgs_batch, 
                uncertainties[ba_window_start:ba_window_end].unsqueeze(0),
                depths[ba_window_start:ba_window_end].unsqueeze(0),
                grid_size, ba_window_start, device
            )
            
        # Store results
        windowed_tracks.append(pixel_tracks)
        windowed_visibles.append(track_visibles)
        windowed_indices.append(track_indices)
        windowed_depths.append(track_depths)
        windowed_rgbs.append(track_rgbs)
        windowed_uncerts.append(track_uncerts)
        windowed_initial_depths.append(initial_depths)
        
        optimized_until = ba_window_end
    
    # Process full-length sequence for global optimization
    print(f"Processing AllTracker full sequence: frames 0 to {seq_len-1}")
    
    # Full sequence processing
    full_imgs_batch = images.unsqueeze(0).to(device)  # (1, seq_len, c, h, w)
    
    if hasattr(image_processor, 'extract_alltracker_tracks'):
        full_tracks, full_visibles = image_processor.extract_alltracker_tracks(full_imgs_batch, grid_size)
        # full_tracks: (1, grid_size^2, seq_len, 2)
        # full_visibles: (1, grid_size^2, seq_len, 1)
        
        # Convert to expected format
        full_pixel_tracks, full_track_visibles, full_indices, full_depths, full_rgbs, full_uncerts, full_initial_depths = convert_alltracker_tracks_to_format(
            full_tracks, full_visibles, full_imgs_batch,
            uncertainties.unsqueeze(0), depths.unsqueeze(0),
            0, device, is_full_sequence=True
        )
    else:
        # Fallback to flow-based approach for full sequence
        if hasattr(image_processor, 'process_sequence_with_alltracker'):
            full_flows_fwd, full_flows_bwd = image_processor.process_sequence_with_alltracker(full_imgs_batch)
        else:
            # Fallback for full sequence
            full_flows_fwd_list = []
            full_flows_bwd_list = []
            
            for i in range(seq_len - 1):
                img0 = full_imgs_batch[:, i]    # (1, c, h, w)
                img1 = full_imgs_batch[:, i+1]  # (1, c, h, w)
                
                flow_fwd, flow_bwd = image_processor.flow_alltracker(img0, img1)
                full_flows_fwd_list.append(flow_fwd)
                full_flows_bwd_list.append(flow_bwd)
            
            full_flows_fwd = torch.stack(full_flows_fwd_list, dim=1)  # (1, seq_len-1, 2, h, w)
            full_flows_bwd = torch.stack(full_flows_bwd_list, dim=1)  # (1, seq_len-1, 2, h, w)
        
        # Extract full-length tracks
        full_pixel_tracks, full_track_visibles, full_indices, full_depths, full_rgbs, full_uncerts, full_initial_depths = extract_tracks_from_alltracker_flows(
            full_flows_fwd, full_flows_bwd, full_imgs_batch,
            uncertainties.unsqueeze(0), depths.unsqueeze(0),
            grid_size, 0, device, is_full_sequence=True
        )
    
    # Pad windowed results to consistent length before concatenation
    if windowed_tracks:
        # Find the maximum track length across all windows
        max_track_len = max(tracks.shape[3] for tracks in windowed_tracks)
        
        # Pad each window to max_track_len
        padded_tracks = []
        padded_visibles = []
        padded_indices = []
        padded_depths = []
        padded_rgbs = []
        padded_uncerts = []
        
        for i, (tracks, visibles, indices, depths, rgbs, uncerts) in enumerate(zip(
            windowed_tracks, windowed_visibles, windowed_indices, 
            windowed_depths, windowed_rgbs, windowed_uncerts
        )):
            n, wc, gs, tl, _ = tracks.shape
            
            if tl < max_track_len:
                # Need to pad this window
                pad_len = max_track_len - tl
                
                # Create padding tensors following compute_pixel_tracks logic
                # Use the last valid position for tracks
                last_tracks = tracks[:, :, :, -1:].expand(-1, -1, -1, pad_len, -1)
                
                # Invalid uncertainties (infinity)
                invalid_uncerts = torch.full(
                    (n, wc, gs, pad_len, 1), 
                    float("inf"), 
                    device=tracks.device, 
                    dtype=tracks.dtype
                )
                
                # Invalid indices (set to last frame of sequence)  
                invalid_indices = torch.full(
                    (n, wc, gs, pad_len, 1), 
                    seq_len - 1, 
                    device=tracks.device, 
                    dtype=indices.dtype
                )
                
                # Invalid depths (zeros)
                invalid_depths = torch.zeros(
                    (n, wc, gs, pad_len, 1), 
                    device=tracks.device, 
                    dtype=depths.dtype
                )
                
                # Invalid visibles (zeros)
                invalid_visibles = torch.zeros(
                    (n, wc, gs, pad_len, 1), 
                    device=tracks.device, 
                    dtype=visibles.dtype
                )
                
                # Invalid rgbs (use last valid RGB)
                last_rgbs = rgbs[:, :, :, -1:].expand(-1, -1, -1, pad_len, -1)
                
                # Concatenate valid and invalid parts
                padded_tracks.append(torch.cat([tracks, last_tracks], dim=3))
                padded_visibles.append(torch.cat([visibles, invalid_visibles], dim=3))
                padded_indices.append(torch.cat([indices, invalid_indices], dim=3))
                padded_depths.append(torch.cat([depths, invalid_depths], dim=3))
                padded_rgbs.append(torch.cat([rgbs, last_rgbs], dim=3))
                padded_uncerts.append(torch.cat([uncerts, invalid_uncerts], dim=3))
            else:
                # No padding needed
                padded_tracks.append(tracks)
                padded_visibles.append(visibles)
                padded_indices.append(indices)
                padded_depths.append(depths)
                padded_rgbs.append(rgbs)
                padded_uncerts.append(uncerts)
        
        # Now concatenate the padded results
        combined_tracks = torch.cat(padded_tracks, dim=1)         # (1, total_windows, grid_size^2, max_track_len, 2)
        combined_visibles = torch.cat(padded_visibles, dim=1)     # (1, total_windows, grid_size^2, max_track_len, 1)
        combined_indices = torch.cat(padded_indices, dim=1)       # (1, total_windows, grid_size^2, max_track_len, 1)
        combined_depths = torch.cat(padded_depths, dim=1)         # (1, total_windows, grid_size^2, max_track_len, 1)
        combined_rgbs = torch.cat(padded_rgbs, dim=1)             # (1, total_windows, grid_size^2, max_track_len, 3)
        combined_uncerts = torch.cat(padded_uncerts, dim=1)       # (1, total_windows, grid_size^2, max_track_len, 1)
        combined_initial_depths = torch.cat(windowed_initial_depths, dim=1)  # (1, total_windows, grid_size^2)
    else:
        # No windows processed
        combined_tracks = None
        combined_visibles = None
        combined_indices = None
        combined_depths = None
        combined_rgbs = None
        combined_uncerts = None
        combined_initial_depths = None
    
    return {
        'windowed': {
            'initial_depths': combined_initial_depths,
            'pixel_tracks': combined_tracks,
            'uncerts': combined_uncerts,
            'indices': combined_indices,
            'depths': combined_depths,
            'rgbs': combined_rgbs,
            'visibles': combined_visibles
        },
        'full_sequence': {
            'initial_depths': full_initial_depths,
            'pixel_tracks': full_pixel_tracks,
            'uncerts': full_uncerts,
            'indices': full_indices,
            'depths': full_depths,
            'rgbs': full_rgbs,
            'visibles': full_track_visibles
        }
    }


def convert_alltracker_tracks_to_format(tracks, visibles, images, uncertainties, depths, frame_offset, device, is_full_sequence=False):
    """
    Convert AllTracker direct tracks to the format expected by compute_pixel_tracks.
    
    Args:
        tracks: Track positions (1, grid_size^2, seq_len, 2)
        visibles: Track visibility (1, grid_size^2, seq_len, 1)
        images: Image sequence (1, seq_len, c, h, w)
        uncertainties: Uncertainty maps (1, seq_len, 1, h, w)
        depths: Depth maps (1, seq_len, 1, h, w)
        frame_offset: Offset for frame indices
        device: Device to use
        is_full_sequence: Whether this is processing the full sequence
        
    Returns:
        Tracks in the same format as compute_pixel_tracks
    """
    
    n, seq_len, c, h, w = images.shape
    n_tracks, track_len = tracks.shape[1], tracks.shape[2]
    
    # Initialize outputs
    track_uncerts = []
    track_indices = []
    track_depths = []
    track_rgbs = []
    track_visibles = []
    
    for t in range(track_len):
        # Get track positions at time t
        positions_t = tracks[:, :, t, :]  # (1, grid_size^2, 2)
        visibles_t = visibles[:, :, t, :]  # (1, grid_size^2, 1)
        
        # Reshape for grid_sample: (1, grid_size^2, 2) -> (1, 1, grid_size^2, 2)
        grid_t = positions_t.unsqueeze(1)
        
        # Sample uncertainties, depths, and rgbs at track positions
        grid_uncert = F.grid_sample(uncertainties[:, t, :1], grid_t, align_corners=False).squeeze(2).permute(0, 2, 1)  # (1, grid_size^2, 1)
        grid_depth = F.grid_sample(depths[:, t:t+1], grid_t, align_corners=False).squeeze(2).permute(0, 2, 1)  # (1, grid_size^2, 1)
        grid_rgb = F.grid_sample(images[:, t], grid_t, align_corners=False).squeeze(2).permute(0, 2, 1)  # (1, grid_size^2, 3)
        
        # Apply validity mask
        valid = visibles_t
        grid_uncert[~(valid.bool())] = float("inf")
        grid_depth[~(valid.bool())] = 0
        
        # Create frame indices
        frame_indices = torch.zeros_like(grid_uncert, dtype=torch.long) + frame_offset + t
        
        track_uncerts.append(grid_uncert)
        track_indices.append(frame_indices)
        track_depths.append(grid_depth)
        track_rgbs.append(grid_rgb)
        track_visibles.append(valid)
    
    # Get initial depths from first frame
    initial_positions = tracks[:, :, 0, :].unsqueeze(1)  # (1, 1, grid_size^2, 2)
    initial_depths = F.grid_sample(depths[:, 0:1], initial_positions, align_corners=False).squeeze(2).squeeze(1)  # (1, grid_size^2)
    
    # Stack results and reshape to match expected format
    if is_full_sequence:
        # For full sequence, we need different dimensions
        pixel_tracks = tracks.permute(0, 1, 2, 3).unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 2)
        
        track_uncerts = torch.stack(track_uncerts, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 1)
        track_indices = torch.stack(track_indices, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 1)
        track_depths_out = torch.stack(track_depths, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 1)
        track_rgbs_out = torch.stack(track_rgbs, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 3)
        track_visibles_out = torch.stack(track_visibles, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 1)
        
    else:
        # For windowed processing, match the expected format
        pixel_tracks = tracks.permute(0, 1, 2, 3).unsqueeze(1)  # (1, 1, grid_size^2, track_len, 2)
        
        track_uncerts = torch.stack(track_uncerts, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, track_len, 1)
        track_indices = torch.stack(track_indices, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, track_len, 1)
        track_depths_out = torch.stack(track_depths, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, track_len, 1)
        track_rgbs_out = torch.stack(track_rgbs, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, track_len, 3)
        track_visibles_out = torch.stack(track_visibles, dim=2).unsqueeze(1)  # (1, 1, grid_size^2, track_len, 1)
    
    return pixel_tracks, track_visibles_out, track_indices, track_depths_out, track_rgbs_out, track_uncerts, initial_depths.unsqueeze(1)


def extract_tracks_from_alltracker_flows(flows_fwd, flows_bwd, images, uncertainties, depths, grid_size, frame_offset, device, is_full_sequence=False):
    """
    Extract pixel tracks from AllTracker flow results.
    
    Args:
        flows_fwd: Forward flows (1, seq_len-1, 2, h, w)
        flows_bwd: Backward flows (1, seq_len-1, 2, h, w)  
        images: Image sequence (1, seq_len, c, h, w)
        uncertainties: Uncertainty maps (1, seq_len, 1, h, w)
        depths: Depth maps (1, seq_len, 1, h, w)
        grid_size: Grid size for track sampling
        frame_offset: Offset for frame indices
        device: Device to use
        is_full_sequence: Whether this is processing the full sequence
        
    Returns:
        Extracted tracks in the same format as compute_pixel_tracks
    """
    
    n, seq_len, c, h, w = images.shape
    
    # Create initial grid on first frame
    x = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(1, -1).expand(grid_size, -1)
    y = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(-1, 1).expand(-1, grid_size)
    grid = torch.stack([x, y], dim=0).view(1, 2, grid_size, grid_size).expand(n, 2, grid_size, grid_size)
    grid = grid.permute(0, 2, 3, 1)  # (1, grid_size, grid_size, 2)
    
    # Initialize tracks starting from first frame
    initial_depths = F.grid_sample(depths[:, 0], grid, align_corners=False).reshape(n, grid_size ** 2)
    
    tracks = [grid]
    indices = [torch.zeros_like(grid[..., 0:1], dtype=torch.long) + frame_offset]
    uncerts = [torch.zeros_like(grid[..., 0:1])]
    track_depths = [F.grid_sample(depths[:, 0], grid, align_corners=False).permute(0, 2, 3, 1)]
    rgbs = [F.grid_sample(images[:, 0], grid, align_corners=False).permute(0, 2, 3, 1)]
    visibles = [torch.ones_like(grid[..., 0:1])]
    
    # Track through sequence using flows
    current_grid = grid.clone()
    
    for t in range(seq_len - 1):
        # Get flow for this frame pair
        flow_t = flows_fwd[:, t]  # (1, 2, h, w)
        
        # Sample flow at current grid positions
        grid_flow = F.grid_sample(flow_t, current_grid, align_corners=False).permute(0, 2, 3, 1)
        
        # Update grid positions
        current_grid = current_grid + grid_flow
        
        # Check bounds and occlusion
        valid = (current_grid[..., :1].abs() < .99) & (current_grid[..., 1:2].abs() < .99)
        
        # Sample other quantities at new positions
        grid_uncert = F.grid_sample(uncertainties[:, t+1, :1], current_grid, align_corners=False).permute(0, 2, 3, 1)
        grid_depth = F.grid_sample(depths[:, t+1], current_grid, align_corners=False).permute(0, 2, 3, 1)
        grid_rgb = F.grid_sample(images[:, t+1], current_grid, align_corners=False).permute(0, 2, 3, 1)
        
        # Apply validity mask
        grid_uncert[~valid] = float("inf")
        grid_depth[~valid] = 0
        
        tracks.append(current_grid.clone())
        indices.append(torch.zeros_like(current_grid[..., 0:1], dtype=torch.long) + frame_offset + t + 1)
        uncerts.append(grid_uncert)
        track_depths.append(grid_depth)
        rgbs.append(grid_rgb)
        visibles.append(valid.float())
    
    # Stack results
    if is_full_sequence:
        # For full sequence, we need different dimensions
        track_len = seq_len
        stride = seq_len  # Only one group
        pixel_tracks = torch.stack(tracks, dim=1).reshape(n, track_len, grid_size ** 2, 2).permute(0, 2, 1, 3)
        pixel_tracks = pixel_tracks.unsqueeze(1)  # Add window dimension: (1, 1, grid_size^2, seq_len, 2)
        
        track_uncerts = torch.stack(uncerts, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        track_uncerts = track_uncerts.unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 1)
        
        track_indices = torch.stack(indices, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        track_indices = track_indices.unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 1)
        
        track_depths_out = torch.stack(track_depths, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        track_depths_out = track_depths_out.unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 1)
        
        track_rgbs = torch.stack(rgbs, dim=1).reshape(n, track_len, grid_size ** 2, 3).permute(0, 2, 1, 3)
        track_rgbs = track_rgbs.unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 3)
        
        track_visibles = torch.stack(visibles, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        track_visibles = track_visibles.unsqueeze(1)  # (1, 1, grid_size^2, seq_len, 1)
        
    else:
        # For windowed processing, match the expected format
        track_len = seq_len
        pixel_tracks = torch.stack(tracks, dim=1).reshape(n, track_len, grid_size ** 2, 2).permute(0, 2, 1, 3)
        pixel_tracks = pixel_tracks.unsqueeze(1)  # Add window dimension: (1, 1, grid_size^2, track_len, 2)
        
        track_uncerts = torch.stack(uncerts, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        track_uncerts = track_uncerts.unsqueeze(1)  # (1, 1, grid_size^2, track_len, 1)
        
        track_indices = torch.stack(indices, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        track_indices = track_indices.unsqueeze(1)  # (1, 1, grid_size^2, track_len, 1)
        
        track_depths_out = torch.stack(track_depths, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        track_depths_out = track_depths_out.unsqueeze(1)  # (1, 1, grid_size^2, track_len, 1)
        
        track_rgbs = torch.stack(rgbs, dim=1).reshape(n, track_len, grid_size ** 2, 3).permute(0, 2, 1, 3)
        track_rgbs = track_rgbs.unsqueeze(1)  # (1, 1, grid_size^2, track_len, 3)
        
        track_visibles = torch.stack(visibles, dim=1).reshape(n, track_len, grid_size ** 2, 1).permute(0, 2, 1, 3)
        track_visibles = track_visibles.unsqueeze(1)  # (1, 1, grid_size^2, track_len, 1)
    
    return pixel_tracks, track_visibles, track_indices, track_depths_out, track_rgbs, track_uncerts, initial_depths.unsqueeze(1)


def create_dummy_tracks(grid_size, track_len, frame_offset, h, w, device, is_full_sequence=False):
    """Create dummy track data when AllTracker processing fails."""
    
    n = 1
    
    # Create dummy grid
    x = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(1, -1).expand(grid_size, -1)
    y = torch.linspace(-1, 1, grid_size+2, device=device)[1:-1].view(-1, 1).expand(-1, grid_size)
    grid = torch.stack([x, y], dim=0).view(1, 2, grid_size, grid_size).expand(n, 2, grid_size, grid_size)
    grid = grid.permute(0, 2, 3, 1)
    
    # Create dummy data
    dummy_tracks = grid.unsqueeze(1).unsqueeze(3).expand(-1, 1, -1, track_len, -1)  # (1, 1, grid_size^2, track_len, 2)
    dummy_visibles = torch.zeros_like(dummy_tracks[..., :1])  # (1, 1, grid_size^2, track_len, 1)
    dummy_indices = torch.zeros_like(dummy_tracks[..., :1], dtype=torch.long)  # (1, 1, grid_size^2, track_len, 1)
    dummy_depths = torch.ones_like(dummy_tracks[..., :1])  # (1, 1, grid_size^2, track_len, 1)
    dummy_rgbs = torch.zeros(*dummy_tracks.shape[:-1], 3, device=device)  # (1, 1, grid_size^2, track_len, 3)
    dummy_uncerts = torch.full_like(dummy_tracks[..., :1], float('inf'))  # (1, 1, grid_size^2, track_len, 1)
    dummy_initial_depths = torch.ones(n, 1, grid_size ** 2, device=device)  # (1, 1, grid_size^2)
    
    # Set frame indices
    for t in range(track_len):
        dummy_indices[:, :, :, t, 0] = frame_offset + t
    
    return dummy_tracks, dummy_visibles, dummy_indices, dummy_depths, dummy_rgbs, dummy_uncerts, dummy_initial_depths