import uuid

import numpy as np
import torch
from tqdm import tqdm
import sys
import os
from PIL import Image
from functools import partial
import cv2
import matplotlib.pyplot as plt

sys.path.append("../..")
sys.path.append(".")


from anycam.utils.bundle_adjustment import *


try:
    import rerun as rr
except:
    rr = None


@torch.autocast(device_type="cuda", dtype=torch.float32)
@torch.enable_grad()
def ba_refinement_opt_tracks_global(config, initial_trajectory, proj, uncertainties, seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd, device="cuda", track3d=None, model=None):
    """
    Global track optimization version of BA refinement.
    
    Key differences from decoupled version:
    1. Removes track optimization from windowed BA (Phase 2)
    2. Adds full-length track optimization after final global BA
    3. Only runs when do_global=True
    """
    
    # Check if global optimization is enabled
    # do_global_tracks = config.get("do_global", False)
    # if not do_global_tracks:
    #     print("Global track optimization disabled (do_global=False). Using standard decoupled BA.")
    #     # Import and call the standard decoupled version
    #     from ba_refinement_opt_tracks_decoupled import ba_refinement_opt_tracks_decoupled
    #     return ba_refinement_opt_tracks_decoupled(config, initial_trajectory, proj, uncertainties, seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd, device, track3d)
    
    print("Starting BA refinement with global track optimization...")
    apply_reprojection_loss = config.get("apply_reprojection_loss", True)
    with_rerun = config.get("with_rerun", True)
    ba_window = config.get("ba_window", 8)
    overlap = config.get("overlap", 6)
    rotation_representation = config.get("rotation_representation", "quaternion")

    max_uncert = config.get("max_uncert", -1)
    use_best = config.get("use_best", False)

    lambda_smoothness = config.get("lambda_smoothness", 200)

    global_every_n = config.get("global_every_n", 2)

    n_steps_sliding = config.get("n_steps_sliding", 400)
    n_steps_global = config.get("n_steps_global", 100)
    n_steps_last_global = config.get("n_steps_last_global", 5000)
    n_steps_only_focal = config.get("n_steps_only_focal", 0)
    
    track_len = config.get("track_len", 8)
    stride = config.get("stride", 1)
    grid_size = config.get("grid_size", 16)
    long_tracks = config.get("long_tracks", False)

    optimize_relatives = config.get("optimize_relatives", False)
    dilated_window_size = config.get("dilated_window_size", 2)
    min_valid_length = config.get("min_valid_length", 2)
    lr = config.get("lr", 1e-4)
    
    # AllTracker sequence mode parameters
    use_alltracker_sequence_mode = config.get("use_alltracker_sequence_mode", True)  # Enable by default when flow_model is alltracker
    

    log_interval = config.get("log_interval", 200)
    rerun_offset = 10
    
    # print all parameters
    print(f"ba_window: {ba_window}")
    print(f"overlap: {overlap}")
    print(f"rotation_representation: {rotation_representation}")
    print(f"lr: {lr}")
    print(f"lambda_smoothness: {lambda_smoothness}")
    print(f"global_every_n: {global_every_n}")
    print(f"n_steps_sliding: {n_steps_sliding}")
    print(f"n_steps_global: {n_steps_global}")
    print(f"n_steps_last_global: {n_steps_last_global}")
    print(f"track_len: {track_len}")
    print(f"stride: {stride}")
    print(f"grid_size: {grid_size}")
    print(f"log_interval: {log_interval}")
    print(f"use_alltracker_sequence_mode: {use_alltracker_sequence_mode}")
    
    if with_rerun:
        rr.init("Global Track BA", recording_id=uuid.uuid4())
        rr.connect()

        rr.log("world", rr.Clear(recursive=True))
        rr.log("log", rr.Clear(recursive=True))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    seq_len = seq_flow_occs_fwd.shape[0]

    track_len = min(track_len, seq_len)

    # Check if we're using AllTracker and should use sequence-based tracking
    use_alltracker_sequence = False
    alltracker_model = None
    
    # Try to extract the flow model from the environment or config
    flow_model = config.get('flow_model', 'unimatch') # ["unimatch", "alltracker"] ### flow_model flag
    if flow_model == "alltracker" and use_alltracker_sequence_mode:
        print("Detected AllTracker flow model - using sequence-based tracking")
        use_alltracker_sequence = True
        alltracker_model = model.image_processor
    elif flow_model == "alltracker" and not use_alltracker_sequence_mode:
        print("AllTracker detected but sequence mode disabled in config - using standard flow-based tracking")
        use_alltracker_sequence = False

    # Extract windowed tracks for pose optimization
    if use_alltracker_sequence and alltracker_model is not None:
        print("Using AllTracker sequence-based track extraction...")
        
        # Use AllTracker for track extraction
        from anycam.utils.bundle_adjustment import compute_alltracker_pixel_tracks
        
        alltracker_results = compute_alltracker_pixel_tracks(
            seq_imgs.cuda(), 
            uncertainties.cuda(), 
            seq_depths.cuda(),
            ba_window=ba_window,
            overlap=overlap,
            grid_size=grid_size,
            device=device,
            image_processor=alltracker_model
        )
        
        # Extract windowed results
        windowed_data = alltracker_results['windowed']
        initial_depths_fwd = windowed_data['initial_depths'].cuda()
        pixel_tracks_fwd = windowed_data['pixel_tracks'].cuda()
        uncerts_fwd = windowed_data['uncerts'].cuda()
        indices_fwd = windowed_data['indices'].cuda()
        depths_fwd = windowed_data['depths'].cuda()
        rgbs_fwd = windowed_data['rgbs'].cuda()
        visibles_fwd = windowed_data['visibles'].cuda()
        
        # For AllTracker, we can use the same tracks for backward (as they're comprehensive)
        initial_depths_bwd = initial_depths_fwd
        pixel_tracks_bwd = pixel_tracks_fwd
        uncerts_bwd = uncerts_fwd
        indices_bwd = indices_fwd
        depths_bwd = depths_fwd
        rgbs_bwd = rgbs_fwd
        visibles_bwd = visibles_fwd
        
        # Store AllTracker data for later use
        alltracker_full_data = alltracker_results['full_sequence']
        
    else:
        print("Using standard flow-based track extraction...")
        # Standard flow-based approach
        initial_depths_fwd, pixel_tracks_fwd, uncerts_fwd, indices_fwd, depths_fwd, rgbs_fwd, visibles_fwd = compute_pixel_tracks(seq_flow_occs_fwd.cuda(), uncertainties.cuda(), seq_depths.cuda(), track_len=track_len, stride=stride, grid_size=grid_size, imgs=seq_imgs.cuda(), long_tracks=long_tracks, track3d=track3d)
        initial_depths_bwd, pixel_tracks_bwd, uncerts_bwd, indices_bwd, depths_bwd, rgbs_bwd, visibles_bwd = compute_pixel_tracks(seq_flow_occs_bwd.cuda(), uncertainties.cuda(), seq_depths.cuda(), track_len=track_len, stride=stride, grid_size=grid_size, is_backward=True, imgs=seq_imgs.cuda(), long_tracks=long_tracks, track3d=track3d)
        
        alltracker_full_data = None

    initial_depths = initial_depths_fwd
    pixel_tracks = pixel_tracks_fwd
    uncerts = uncerts_fwd
    indices = indices_fwd
    depths = depths_fwd
    rgbs = rgbs_fwd
    visibles = visibles_fwd.to(torch.int32)
    
    n, wc, gs = initial_depths.shape
    n, wc, gs, tl, c = pixel_tracks.shape
    n, wc, gs, tl, _ = uncerts.shape
    n, wc, gs, tl, _ = indices.shape
    seq_len, c, h, w = seq_imgs.shape

    ba_imgs = seq_imgs
    ba_indices = indices.cuda()
    ba_uncerts = uncerts.cuda()

    ba_proj = np.array(proj.squeeze())
    ba_proj[0, 0] = (ba_proj[0, 0] / w) * 2
    ba_proj[1, 1] = (ba_proj[1, 1] / h) * 2
    ba_proj[0, 2] = (ba_proj[0, 2] / w) * 2 - 1
    ba_proj[1, 2] = (ba_proj[1, 2] / h) * 2 - 1
    ba_proj = torch.tensor(ba_proj).unsqueeze(0).cuda()

    ba_poses_c2w = torch.tensor(np.array(initial_trajectory)).unsqueeze(0).cuda()
    rel_poses = torch.inverse(ba_poses_c2w[:, :-1]) @ ba_poses_c2w[:, 1:]

    ba_param_inv_depth = 1 / initial_depths
    ba_param_inv_depths = 1 / depths
    ba_param_rot, ba_param_t = pose_to_param(ba_poses_c2w, rotation_representation)
    
    ba_param_focal_length = (torch.tensor(proj[0, 0] / w * 2, device=device)).log() / 2
    # sigma_depth parameters for tracking optimized tracks
    
    
    if with_rerun:
        log_ba_imgs(ba_imgs, uncertainties=uncertainties, tracks=(ba_indices, ba_uncerts, pixel_tracks), timestep=-1)

        log_ba_state(
            param_to_pose(ba_param_rot, ba_param_t),
            imgs=ba_imgs,
            timestep=seq_len + rerun_offset,
            max_dist=10,
        )

    log_step = 1
    optimized_until = 1
    global_ba_step = 1
    last_global_done = False
    
    optimized_windows_data = []

    while optimized_until < seq_len or not last_global_done:
        if not apply_reprojection_loss:
            break
        ba_param_inv_depth.requires_grad = True
        ba_param_rot.requires_grad = True
        ba_param_t.requires_grad = True
        ba_param_focal_length.requires_grad = True
        ba_param_inv_depths.requires_grad = True
        # ba_param_sigma_depth.requires_grad = False
        
        do_last_global = optimized_until >= seq_len

        if do_last_global:
            last_global_done = True

        do_global = global_ba_step % global_every_n == 0 or do_last_global

        ba_window_start = max(optimized_until - overlap, 0)
        ba_window_end = min(ba_window_start + ba_window, seq_len)
        
        # Set up pose optimization parameters
        seq_ids = torch.arange(seq_len, device=device)
        
        if not do_global:
            ba_param_inv_depth_mask = (ba_indices[:, :, :, 0, 0] >= optimized_until) & (ba_indices[:, :, :, 0, 0] < ba_window_end)
            ba_param_pose_mask = ((seq_ids >= optimized_until) & (seq_ids < ba_window_end)).view(1, -1, 1)
            loss_mask = (ba_indices[:, :, :, 1:, 0] >= ba_window_start) & (ba_indices[:, :, :, 1:, 0] < ba_window_end)
        else:
            ba_param_inv_depth_mask = (ba_indices[:, :, :, 0, 0] >= 0) & (ba_indices[:, :, :, 0, 0] < optimized_until)
            ba_param_pose_mask = ((seq_ids >= 0) & (seq_ids < optimized_until)).view(1, -1, 1)
            loss_mask = (ba_indices[:, :, :, 1:, 0] >= 0) & (ba_indices[:, :, :, 1:, 0] < optimized_until)

        # Phase 1: Optimize poses (reprojection loss) - SAME AS DECOUPLED
        if not do_global:
            print(f"Phase 1: Optimizing poses from {ba_window_start} to {ba_window_end}")
        else:
            print(f"Phase 1: Optimizing poses from {0} to {optimized_until}")

        # Optimize poses with reprojection loss
        ba_poses = param_to_pose(ba_param_rot, ba_param_t).detach().clone()

        if optimize_relatives:
            ba_poses_ = [ba_poses[:, 0]]
            for i in range(1, seq_len):
                ba_poses_.append(torch.inverse(ba_poses[:, i-1]) @ ba_poses_[-1])
            ba_poses = torch.stack(ba_poses_, dim=1)

        ba_poses = ba_poses[:, :optimized_until]

        if optimized_until < seq_len:
            add_poses = []
            curr_pose = ba_poses[:, -1]
            for frame_idx in range(optimized_until, seq_len):
                curr_pose = curr_pose @ rel_poses[:, frame_idx - 1]
                add_poses.append(curr_pose)
            add_poses = torch.stack(add_poses, dim=1)
            ba_poses = torch.cat([ba_poses, add_poses], dim=1)

        ba_param_rot, ba_param_t = pose_to_param(ba_poses, rotation_representation)
        ba_param_rot = ba_param_rot.detach().clone()
        ba_param_t = ba_param_t.detach().clone()
        ba_param_rot.requires_grad = True
        ba_param_t.requires_grad = True
        ba_param_inv_depth.requires_grad = True
        ba_param_focal_length.requires_grad = True

        # Optimize poses only
        optimizer_poses = torch.optim.Adam([ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length], lr=lr)

        if not do_global:
            n_steps_poses = n_steps_sliding
        else:
            if do_last_global:
                n_steps_poses = n_steps_last_global
            else:
                n_steps_poses = n_steps_global

        print(f"Optimizing poses for {n_steps_poses} steps")
        pbar_poses = tqdm(range(n_steps_poses), desc="Pose optimization")

        for step in pbar_poses:
            optimizer_poses.zero_grad()

            # Detach relevant parameters
            ba_param_inv_depth_d = ba_param_inv_depth.clone()
            ba_param_inv_depth_d[~ba_param_inv_depth_mask].detach_()

            ba_param_rot_d = ba_param_rot.clone()            
            ba_param_rot_d[~ba_param_pose_mask.expand_as(ba_param_rot_d)] = ba_param_rot_d[~ba_param_pose_mask.expand_as(ba_param_rot_d)].detach()

            ba_param_t_d = ba_param_t.clone()
            ba_param_t_d[~ba_param_pose_mask.expand_as(ba_param_t_d)] = ba_param_t_d[~ba_param_pose_mask.expand_as(ba_param_t_d)].detach()

            repr_loss, smoothness_loss, ba_proj_updated, xyzh_world = compute_loss(ba_param_inv_depth_d, ba_param_rot_d, ba_param_t_d, ba_param_focal_length, pixel_tracks, ba_indices, ba_uncerts, w, h, loss_mask, max_uncert)

            if use_best:
                thresh = torch.quantile(repr_loss[repr_loss > 0], 0.9)
                repr_loss[repr_loss > thresh] = 0

            total_loss = repr_loss.mean() + smoothness_loss.mean() * (lambda_smoothness if do_global else 0)

            total_loss.backward()

            # Clip gradients
            for param in [ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length]:
                if param.grad is not None:
                    param.grad[torch.isnan(param.grad) | torch.isinf(param.grad)] = 0

            if do_global and step < n_steps_only_focal:
                ba_param_rot.grad *= 0
                ba_param_t.grad *= 0

            torch.nn.utils.clip_grad_norm_([ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length], 0.1)

            optimizer_poses.step()

            pbar_poses.set_postfix({"total_loss": total_loss.item(), "repr_loss": repr_loss.mean().item(), "smoothness_loss": smoothness_loss.mean().item()})

        # NOTE: NO Phase 2 track optimization in windows - this is the key difference!
        
        # Log visualization if enabled
        if with_rerun and log_step % log_interval == 0:
            ba_poses_c2w = param_to_pose(ba_param_rot, ba_param_t)
            
            log_ba_state(
                ba_poses_c2w,
                points=xyzh_world[:, ::1, :3, 0].permute(0, 2, 1) if 'xyzh_world' in locals() else None,
                point_colors=rgbs[:, :, :, :1, :].reshape(1, -1, 3)[:, ::1] if rgbs is not None else None,
                timestep=seq_len + rerun_offset * 2 + (log_step // log_interval),
                max_dist=10,
            )
            
        log_step += 1

        # Update optimization progress
        if not do_global:
            global_ba_step += 1
            optimized_until = ba_window_end
        else:
            last_global_done = True


    # PHASE 3: GLOBAL TRACK OPTIMIZATION - REMOVED
    print("Skipping Phase 3: Global track optimization...")

    scale_factor_s = 1.0

    # Initialize ba_param_sigma_depth with zeros matching the windowed structure
    # pixel_tracks comes from Phase 1/2 and has shape (n, wc, gs, tl, c)
    n, wc, gs, tl, c = pixel_tracks.shape
    ba_param_sigma_depth = torch.zeros((wc, gs, tl)).cuda()

    global_track_data = {
        'has_enough_tracks': False,
        'is_global': True,
        'num_tracks': 0,
        'track_len': seq_len,
    }

    # Final computations - same as decoupled
    ba_poses_c2w = param_to_pose(ba_param_rot, ba_param_t)
    ba_trajectory = ba_poses_c2w.detach()
    ba_proj = make_normalized_proj((ba_param_focal_length * 2).exp(), w/h)[0].cpu().detach()
    
    # Unnormalize projection matrix
    ba_proj[0, 0] = (ba_proj[0, 0] / 2) * w
    ba_proj[1, 1] = (ba_proj[1, 1] / 2) * h
    ba_proj[0, 2] = (ba_proj[0, 2] + 1) / 2 * w
    ba_proj[1, 2] = (ba_proj[1, 2] + 1) / 2 * h

    ba_extras = {
        "ba_param_inv_depth": ba_param_inv_depth.detach().cpu(),
        "ba_param_rot": ba_param_rot.detach().cpu(),
        "ba_param_t": ba_param_t.detach().cpu(),
        "ba_param_focal_length": ba_param_focal_length.detach().cpu(),
        "ba_param_sigma_depth": ba_param_sigma_depth.detach().cpu() if hasattr(ba_param_sigma_depth, 'detach') else ba_param_sigma_depth.cpu(),
        "ba_trajectory": ba_trajectory.cpu(),
        "indices": ba_indices.cpu(),
        "uncerts": ba_uncerts.cpu(),
        "initial_depths": initial_depths.cpu(),
        "pixel_tracks": pixel_tracks.cpu(),
        # Store empty window data and global track data
        "optimized_windows_data": optimized_windows_data,
        "global_track_data": global_track_data,
        "scale_factor_s": scale_factor_s if 'scale_factor_s' in locals() else 1.0,
    }

    # Log final visualization if rerun is enabled
    if with_rerun:
        final_viz_timestep = seq_len + rerun_offset * 4
        rr.set_time_sequence("final_result", final_viz_timestep)
        
        # Log final camera trajectory
        log_ba_state(
            ba_poses_c2w,
            timestep=final_viz_timestep,
            max_dist=10,
        )

    return ba_trajectory[0], ba_proj, ba_extras 

def visualize_pixel_tracks_video(seq_imgs, pixel_tracks, visibles, output_path="track_visualization.mp4", fps=10, track_colors=None, point_size=3):
    """
    Visualize pixel tracks overlaid on video frames.
    
    Args:
        seq_imgs: torch.Tensor of shape (seq_len, 3, H, W) - video sequence
        pixel_tracks: torch.Tensor of shape (n, wc, gs, tl, 2) - pixel tracks coordinates
        visibles: torch.Tensor of shape (n, wc, gs, tl, 1) or (n, wc, gs, tl) - visibility mask
        output_path: str - path to save the output video
        fps: int - frames per second for output video
        track_colors: optional array of colors for each track
        point_size: int - size of the track points
    """
    print(f"Visualizing pixel tracks and saving to {output_path}")
    
    # Convert tensors to numpy and ensure correct format
    if isinstance(seq_imgs, torch.Tensor):
        seq_imgs_np = seq_imgs.cpu().numpy()
    else:
        seq_imgs_np = seq_imgs
        
    if isinstance(pixel_tracks, torch.Tensor):
        pixel_tracks_np = pixel_tracks.cpu().numpy()
    else:
        pixel_tracks_np = pixel_tracks
        
    if isinstance(visibles, torch.Tensor):
        visibles_np = visibles.cpu().numpy()
    else:
        visibles_np = visibles
    
    # Handle visibility shape
    if len(visibles_np.shape) == 5:  # (n, wc, gs, tl, 1)
        visibles_np = visibles_np[..., 0]  # Remove last dimension
    
    seq_len, channels, height, width = seq_imgs_np.shape
    n, wc, gs, tl, _ = pixel_tracks_np.shape
    
    # Convert images from (C, H, W) to (H, W, C) and denormalize if needed
    frames = []
    for t in range(seq_len):
        img = seq_imgs_np[t].transpose(1, 2, 0)  # (H, W, C)
        
        # Denormalize if values are in [0, 1]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames.append(img_bgr)
    
    # Generate colors for tracks if not provided
    if track_colors is None:
        # Create a colormap for tracks
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, wc * gs)))
        track_colors = (colors[:, :3] * 255).astype(np.uint8)
        # Repeat colors if we have more tracks than colors
        if wc * gs > len(track_colors):
            track_colors = np.tile(track_colors, (wc * gs // len(track_colors) + 1, 1))[:wc * gs]
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_idx in range(min(seq_len, tl)):
        frame = frames[frame_idx].copy()
        
        # Draw tracks for this frame
        color_idx = 0
        for batch_idx in range(n):
            for window_idx in range(wc):
                for grid_idx in range(gs):
                    if frame_idx < tl:  # Make sure we don't exceed track length
                        # Get track coordinates for this frame
                        x, y = pixel_tracks_np[batch_idx, window_idx, grid_idx, frame_idx, :]
                        is_visible = visibles_np[batch_idx, window_idx, grid_idx, frame_idx]
                        
                        # Convert normalized coordinates to pixel coordinates if needed
                        if x <= 1.0 and y <= 1.0 and x >= -1.0 and y >= -1.0:
                            # Assume normalized coordinates [-1, 1]
                            pixel_x = int((x + 1) * width / 2)
                            pixel_y = int((y + 1) * height / 2)
                        else:
                            # Assume already in pixel coordinates
                            pixel_x = int(x)
                            pixel_y = int(y)
                        
                        # Check if coordinates are within frame bounds
                        if 0 <= pixel_x < width and 0 <= pixel_y < height:
                            # Get color for this track
                            color = track_colors[color_idx % len(track_colors)]
                            color_bgr = (int(color[2]), int(color[1]), int(color[0]))  # Convert RGB to BGR
                            
                            if is_visible:
                                # Draw solid circle for visible tracks
                                cv2.circle(frame, (pixel_x, pixel_y), point_size, color_bgr, -1)
                            else:
                                # Draw hollow circle for invisible tracks
                                cv2.circle(frame, (pixel_x, pixel_y), point_size, color_bgr, 1)
                    
                    color_idx += 1
        
        # Add frame number text
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        # Also save first and last frames as images for quick preview
        # if frame_idx == 0:
        #     cv2.imwrite(output_path.replace('.mp4', '_first_frame.jpg'), frame)
        # elif frame_idx == min(seq_len, tl) - 1:
        #     cv2.imwrite(output_path.replace('.mp4', '_last_frame.jpg'), frame)
    
    # Release video writer
    out.release()
    print(f"Track visualization saved to {output_path}")
    # print(f"Preview frames saved as {output_path.replace('.mp4', '_first_frame.jpg')} and {output_path.replace('.mp4', '_last_frame.jpg')}")


def visualize_global_tracks_video(seq_imgs, pixel_tracks, visibles, output_path="global_tracks_visualization.mp4", fps=10, max_tracks_to_show=100):
    """
    Specialized visualization function for global tracks with trajectory trails.
    
    Args:
        seq_imgs: torch.Tensor of shape (seq_len, 3, H, W) - video sequence
        pixel_tracks: torch.Tensor of shape (n, wc, gs, tl, 2) - pixel tracks coordinates
        visibles: torch.Tensor of shape (n, wc, gs, tl, 1) or similar - visibility mask
        output_path: str - path to save the output video
        fps: int - frames per second for output video
        max_tracks_to_show: int - maximum number of tracks to visualize (for performance)
    """
    print(f"Visualizing global tracks with trails and saving to {output_path}")
    
    # Convert tensors to numpy
    if isinstance(seq_imgs, torch.Tensor):
        seq_imgs_np = seq_imgs.cpu().numpy()
    else:
        seq_imgs_np = seq_imgs
        
    if isinstance(pixel_tracks, torch.Tensor):
        pixel_tracks_np = pixel_tracks.cpu().numpy()
    else:
        pixel_tracks_np = pixel_tracks
        
    if isinstance(visibles, torch.Tensor):
        visibles_np = visibles.cpu().numpy()
    else:
        visibles_np = visibles
    
    # Handle visibility shape
    if len(visibles_np.shape) == 5:
        visibles_np = visibles_np[..., 0]
    
    seq_len, channels, height, width = seq_imgs_np.shape
    n, wc, gs, tl, _ = pixel_tracks_np.shape
    
    # Limit number of tracks for performance
    total_tracks = wc * gs
    if total_tracks > max_tracks_to_show:
        print(f"Limiting visualization to {max_tracks_to_show} tracks out of {total_tracks} for performance")
        step = total_tracks // max_tracks_to_show
    else:
        step = 1
    
    # Generate distinct colors
    num_colors = min(total_tracks // step, max_tracks_to_show)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    track_colors = (colors[:, :3] * 255).astype(np.uint8)
    
    # Convert images and prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Store track history for trails
    track_history = {}
    
    for frame_idx in range(min(seq_len, tl)):
        # Get base frame
        img = seq_imgs_np[frame_idx].transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw track trails and current positions
        color_idx = 0
        for batch_idx in range(n):
            for window_idx in range(wc):
                for grid_idx in range(0, gs, step):  # Skip tracks based on step
                    track_id = f"{batch_idx}_{window_idx}_{grid_idx}"
                    
                    if frame_idx < tl:
                        x, y = pixel_tracks_np[batch_idx, window_idx, grid_idx, frame_idx, :]
                        is_visible = visibles_np[batch_idx, window_idx, grid_idx, frame_idx]
                        
                        # Convert coordinates
                        if x <= 1.0 and y <= 1.0 and x >= -1.0 and y >= -1.0:
                            pixel_x = int((x + 1) * width / 2)
                            pixel_y = int((y + 1) * height / 2)
                        else:
                            pixel_x = int(x)
                            pixel_y = int(y)
                        
                        if 0 <= pixel_x < width and 0 <= pixel_y < height:
                            # Update track history
                            if track_id not in track_history:
                                track_history[track_id] = []
                            
                            if is_visible:
                                track_history[track_id].append((pixel_x, pixel_y))
                            
                            # Limit history length
                            if len(track_history[track_id]) > 10:
                                track_history[track_id] = track_history[track_id][-10:]
                            
                            # Get color
                            color = track_colors[color_idx % len(track_colors)]
                            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
                            
                            # Draw trail
                            if len(track_history[track_id]) > 1:
                                for i in range(1, len(track_history[track_id])):
                                    pt1 = track_history[track_id][i-1]
                                    pt2 = track_history[track_id][i]
                                    # Fade the trail
                                    alpha = i / len(track_history[track_id])
                                    trail_color = tuple(int(c * alpha) for c in color_bgr)
                                    cv2.line(frame, pt1, pt2, trail_color, 1)
                            
                            # Draw current point
                            if is_visible:
                                cv2.circle(frame, (pixel_x, pixel_y), 4, color_bgr, -1)
                                cv2.circle(frame, (pixel_x, pixel_y), 5, (255, 255, 255), 1)
                            else:
                                cv2.circle(frame, (pixel_x, pixel_y), 4, color_bgr, 1)
                    
                    color_idx += 1
        
        # Add frame info
        cv2.putText(frame, f"Frame {frame_idx}/{min(seq_len, tl)-1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {num_colors}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Global tracks visualization with trails saved to {output_path}") 


def visualize_tracks_with_motion_scores(seq_imgs, pixel_tracks, visibles, motion_scores, output_path="tracks_with_motion_scores.mp4", fps=10, point_size=3, text_scale=0.4, show_top_n=50):
    """
    Visualize pixel tracks with motion score annotations.
    
    Args:
        seq_imgs: torch.Tensor of shape (seq_len, 3, H, W) - video sequence
        pixel_tracks: torch.Tensor - pixel tracks coordinates  
        visibles: torch.Tensor - visibility mask
        motion_scores: torch.Tensor of shape (num_tracks, seq_len) - motion scores per track per frame
        output_path: str - path to save the output video
        fps: int - frames per second
        point_size: int - size of track points
        text_scale: float - scale of text annotations
        show_top_n: int - show motion scores for only the top N tracks with highest motion to reduce clutter
    """
    print(f"Creating tracks with motion scores visualization: {output_path}")
    
    # Convert tensors to numpy
    if isinstance(seq_imgs, torch.Tensor):
        seq_imgs_np = seq_imgs.cpu().numpy()
    else:
        seq_imgs_np = seq_imgs
        
    if isinstance(pixel_tracks, torch.Tensor):
        pixel_tracks_np = pixel_tracks.cpu().numpy()
    else:
        pixel_tracks_np = pixel_tracks
        
    if isinstance(visibles, torch.Tensor):
        visibles_np = visibles.cpu().numpy()
    else:
        visibles_np = visibles
        
    if isinstance(motion_scores, torch.Tensor):
        motion_scores_np = motion_scores.detach().cpu().numpy()
    else:
        motion_scores_np = motion_scores
    
    # Handle visibility shape
    if len(visibles_np.shape) == 5:
        visibles_np = visibles_np[..., 0]
    
    seq_len, channels, height, width = seq_imgs_np.shape
    n, wc, gs, tl, _ = pixel_tracks_np.shape
    
    # Calculate average motion scores per track to identify top movers
    avg_motion_scores = np.mean(motion_scores_np, axis=1)
    top_indices = np.argsort(avg_motion_scores)[-show_top_n:] if len(avg_motion_scores) > show_top_n else np.arange(len(avg_motion_scores))
    
    # Create color map based on motion scores (normalized)
    max_motion = np.max(motion_scores_np) if motion_scores_np.size > 0 else 1.0
    min_motion = np.min(motion_scores_np) if motion_scores_np.size > 0 else 0.0
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_idx in range(min(seq_len, tl)):
        # Get base frame
        img = seq_imgs_np[frame_idx].transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Collect all visible tracks for this frame for smart positioning
        visible_tracks = []
        
        # First pass: collect all visible track positions and scores
        for batch_idx in range(n):
            for window_idx in range(wc):
                for grid_idx in range(gs):
                    if frame_idx < tl:
                        track_linear_idx = batch_idx * wc * gs + window_idx * gs + grid_idx
                        
                        # Skip if not in top tracks
                        if track_linear_idx not in top_indices:
                            continue
                            
                        x, y = pixel_tracks_np[batch_idx, window_idx, grid_idx, frame_idx, :]
                        is_visible = visibles_np[batch_idx, window_idx, grid_idx, frame_idx]
                        
                        # Convert coordinates
                        if x <= 1.0 and y <= 1.0 and x >= -1.0 and y >= -1.0:
                            pixel_x = int((x + 1) * width / 2)
                            pixel_y = int((y + 1) * height / 2)
                        else:
                            pixel_x = int(x)
                            pixel_y = int(y)
                        
                        if 0 <= pixel_x < width and 0 <= pixel_y < height and is_visible:
                            # Get motion score for this track and frame
                            if track_linear_idx < motion_scores_np.shape[0] and frame_idx < motion_scores_np.shape[1]:
                                motion_score = motion_scores_np[track_linear_idx, frame_idx]
                            else:
                                motion_score = 0.0
                            
                            visible_tracks.append({
                                'pos': (pixel_x, pixel_y),
                                'score': motion_score,
                                'track_idx': track_linear_idx
                            })
        
        # Second pass: draw tracks with smart text positioning
        text_positions = []  # Track text positions to avoid overlap
        
        for track_info in visible_tracks:
            pixel_x, pixel_y = track_info['pos']
            motion_score = track_info['score']
            
            # Color based on motion score (blue = low motion, red = high motion)
            if max_motion > min_motion:
                normalized_score = (motion_score - min_motion) / (max_motion - min_motion)
            else:
                normalized_score = 0.5
            
            # Color gradient: blue (low) -> green (medium) -> red (high)
            if normalized_score < 0.5:
                # Blue to green
                color = (int(255 * (1 - 2 * normalized_score)), int(255 * 2 * normalized_score), 255)
            else:
                # Green to red  
                color = (0, int(255 * (2 - 2 * normalized_score)), int(255 * (2 * normalized_score - 1)))
            
            # Draw track point
            cv2.circle(frame, (pixel_x, pixel_y), point_size, color, -1)
            cv2.circle(frame, (pixel_x, pixel_y), point_size + 1, (255, 255, 255), 1)  # White outline
            
            # Smart text positioning to avoid overlap
            text = f"{motion_score:.1f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)[0]
            
            # Try different positions around the point
            offset_positions = [
                (pixel_x + point_size + 2, pixel_y - 2),  # Right
                (pixel_x - text_size[0] - point_size - 2, pixel_y - 2),  # Left
                (pixel_x - text_size[0] // 2, pixel_y - point_size - 5),  # Top
                (pixel_x - text_size[0] // 2, pixel_y + point_size + text_size[1] + 2),  # Bottom
                (pixel_x + point_size + 2, pixel_y + text_size[1]),  # Bottom-right
                (pixel_x - text_size[0] - point_size - 2, pixel_y + text_size[1]),  # Bottom-left
            ]
            
            # Find position with minimal overlap
            best_pos = offset_positions[0]
            min_overlap = float('inf')
            
            for test_pos in offset_positions:
                test_x, test_y = test_pos
                
                # Check bounds
                if (test_x < 0 or test_y < 0 or 
                    test_x + text_size[0] >= width or test_y >= height):
                    continue
                
                # Calculate overlap with existing text
                overlap_count = 0
                for existing_pos, existing_size in text_positions:
                    if (abs(test_x - existing_pos[0]) < (text_size[0] + existing_size[0]) // 2 and
                        abs(test_y - existing_pos[1]) < (text_size[1] + existing_size[1]) // 2):
                        overlap_count += 1
                
                if overlap_count < min_overlap:
                    min_overlap = overlap_count
                    best_pos = test_pos
            
            # Draw text with background for better visibility
            text_x, text_y = best_pos
            if (0 <= text_x < width - text_size[0] and 
                text_size[1] <= text_y < height):
                
                # Draw background rectangle
                cv2.rectangle(frame, 
                             (text_x - 1, text_y - text_size[1] - 1),
                             (text_x + text_size[0] + 1, text_y + 2),
                             (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1)
                
                # Record text position
                text_positions.append(((text_x, text_y), text_size))
        
        # Add frame info and legend
        # cv2.putText(frame, f"Frame {frame_idx} - Motion Scores (Top {len(visible_tracks)} tracks)", 
        #            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add color legend
        legend_y = height - 60
        legend_width = 200
        legend_height = 15
        
        # Draw color bar
        for i in range(legend_width):
            norm_val = i / legend_width
            if norm_val < 0.5:
                legend_color = (int(255 * (1 - 2 * norm_val)), int(255 * 2 * norm_val), 255)
            else:
                legend_color = (0, int(255 * (2 - 2 * norm_val)), int(255 * (2 * norm_val - 1)))
            
            cv2.line(frame, (10 + i, legend_y), (10 + i, legend_y + legend_height), legend_color, 1)
        
        # Legend labels
        cv2.putText(frame, f"Low ({min_motion:.1f})", (10, legend_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"High ({max_motion:.1f})", (10 + legend_width - 40, legend_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Tracks with motion scores visualization saved to {output_path}") 

def visualize_tracks_with_ids(seq_imgs, pixel_tracks, visibles, filtered_indices, output_path="tracks_with_ids.mp4", fps=10, point_size=3, text_scale=0.4, show_max_tracks=200):
    """
    Visualize pixel tracks with their ID annotations (filtered indices).
    
    Args:
        seq_imgs: torch.Tensor of shape (seq_len, 3, H, W) - video sequence
        pixel_tracks: torch.Tensor - pixel tracks coordinates  
        visibles: torch.Tensor - visibility mask
        filtered_indices: torch.Tensor or np.ndarray - indices of filtered tracks
        output_path: str - path to save the output video
        fps: int - frames per second
        point_size: int - size of track points
        text_scale: float - scale of text annotations
        show_max_tracks: int - maximum number of tracks to show (for performance)
    """
    print(f"Creating tracks with IDs visualization: {output_path}")
    
    # Convert tensors to numpy
    if isinstance(seq_imgs, torch.Tensor):
        seq_imgs_np = seq_imgs.cpu().numpy()
    else:
        seq_imgs_np = seq_imgs
        
    if isinstance(pixel_tracks, torch.Tensor):
        pixel_tracks_np = pixel_tracks.cpu().numpy()
    else:
        pixel_tracks_np = pixel_tracks
        
    if isinstance(visibles, torch.Tensor):
        visibles_np = visibles.cpu().numpy()
    else:
        visibles_np = visibles
        
    if isinstance(filtered_indices, torch.Tensor):
        filtered_indices_np = filtered_indices.cpu().numpy()
    else:
        filtered_indices_np = filtered_indices
    
    # Handle visibility shape
    if len(visibles_np.shape) == 5:
        visibles_np = visibles_np[..., 0]
    
    seq_len, channels, height, width = seq_imgs_np.shape
    n, wc, gs, tl, _ = pixel_tracks_np.shape
    
    # Limit number of tracks for performance
    num_tracks = min(len(filtered_indices_np), show_max_tracks)
    display_indices = filtered_indices_np[:num_tracks]
    
    # Generate distinct colors for tracks
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    track_colors = (colors[:, :3] * 255).astype(np.uint8)
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_idx in range(min(seq_len, tl)):
        # Get base frame
        img = seq_imgs_np[frame_idx].transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Collect all visible tracks for this frame
        visible_tracks = []
        
        # First pass: collect all visible track positions and IDs
        for batch_idx in range(n):
            for window_idx in range(wc):
                for grid_idx in range(gs):
                    if frame_idx < tl:
                        track_linear_idx = batch_idx * wc * gs + window_idx * gs + grid_idx
                        
                        # Check if this track is in our display list
                        if track_linear_idx >= len(display_indices):
                            continue
                            
                        # track_id = display_indices[track_linear_idx]
                        track_id = track_linear_idx
                        x, y = pixel_tracks_np[batch_idx, window_idx, grid_idx, frame_idx, :]
                        is_visible = visibles_np[batch_idx, window_idx, grid_idx, frame_idx]
                        
                        # Convert coordinates
                        if x <= 1.0 and y <= 1.0 and x >= -1.0 and y >= -1.0:
                            pixel_x = int((x + 1) * width / 2)
                            pixel_y = int((y + 1) * height / 2)
                        else:
                            pixel_x = int(x)
                            pixel_y = int(y)
                        
                        if 0 <= pixel_x < width and 0 <= pixel_y < height and is_visible:
                            visible_tracks.append({
                                'pos': (pixel_x, pixel_y),
                                'id': track_id,
                                'track_idx': track_linear_idx
                            })
        
        # Second pass: draw tracks with smart text positioning
        text_positions = []  # Track text positions to avoid overlap
        
        for track_info in visible_tracks:
            pixel_x, pixel_y = track_info['pos']
            track_id = track_info['id']
            track_idx = track_info['track_idx']
            
            # Get color for this track (cycle through available colors)
            color_idx = track_idx % len(track_colors)
            color = track_colors[color_idx]
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))  # Convert RGB to BGR
            
            # Draw track point
            cv2.circle(frame, (pixel_x, pixel_y), point_size, color_bgr, -1)
            cv2.circle(frame, (pixel_x, pixel_y), point_size + 1, (255, 255, 255), 1)  # White outline
            
            # Smart text positioning to avoid overlap
            text = f"{track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)[0]
            
            # Try different positions around the point
            offset_positions = [
                (pixel_x + point_size + 2, pixel_y - 2),  # Right
                (pixel_x - text_size[0] - point_size - 2, pixel_y - 2),  # Left
                (pixel_x - text_size[0] // 2, pixel_y - point_size - 5),  # Top
                (pixel_x - text_size[0] // 2, pixel_y + point_size + text_size[1] + 2),  # Bottom
                (pixel_x + point_size + 2, pixel_y + text_size[1]),  # Bottom-right
                (pixel_x - text_size[0] - point_size - 2, pixel_y + text_size[1]),  # Bottom-left
            ]
            
            # Find position with minimal overlap
            best_pos = offset_positions[0]
            min_overlap = float('inf')
            
            for test_pos in offset_positions:
                test_x, test_y = test_pos
                
                # Check bounds
                if (test_x < 0 or test_y < 0 or 
                    test_x + text_size[0] >= width or test_y >= height):
                    continue
                
                # Calculate overlap with existing text
                overlap_count = 0
                for existing_pos, existing_size in text_positions:
                    if (abs(test_x - existing_pos[0]) < (text_size[0] + existing_size[0]) // 2 and
                        abs(test_y - existing_pos[1]) < (text_size[1] + existing_size[1]) // 2):
                        overlap_count += 1
                
                if overlap_count < min_overlap:
                    min_overlap = overlap_count
                    best_pos = test_pos
            
            # Draw text with background for better visibility
            text_x, text_y = best_pos
            if (0 <= text_x < width - text_size[0] and 
                text_size[1] <= text_y < height):
                
                # Draw background rectangle
                cv2.rectangle(frame, 
                             (text_x - 1, text_y - text_size[1] - 1),
                             (text_x + text_size[0] + 1, text_y + 2),
                             (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1)
                
                # Record text position
                text_positions.append(((text_x, text_y), text_size))
        
        # Add frame info
        cv2.putText(frame, f"Frame {frame_idx} - Track IDs ({len(visible_tracks)} visible)", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total filtered tracks: {len(filtered_indices_np)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Tracks with IDs visualization saved to {output_path}")