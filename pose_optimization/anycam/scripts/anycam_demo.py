import sys
import os
import uuid
import time

import cv2

sys.path.append(".")
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from moviepy import VideoFileClip
import imageio
import rerun as rr
# Import trajectory alignment utilities
try:
    from evo.core import trajectory
    from evo.core.trajectory import PosePath3D
    HAS_EVO = True
except ImportError:
    print("Warning: evo package not found. Trajectory alignment functionality will be limited.")
    HAS_EVO = False

# Import necessary functions from fit_video
try:
    from anycam.scripts.fit_video import compute_pixel_tracks, pix_2_world_np, world_2_pix_np, pose_to_param
    HAS_COMPUTE_PIXEL_TRACKS = True
except ImportError:
    print("Warning: compute_pixel_tracks function not found. Will use simplified track extraction.")
    HAS_COMPUTE_PIXEL_TRACKS = False


from anycam.loss import make_loss
from anycam.trainer import AnyCamWrapper
from anycam.common.geometry import get_grid_xy
from anycam.utils.geometry import se3_ensure_numerical_accuracy
from anycam.visualization.common import color_tensor

# Add necessary imports for semantic filtering
import jax
import jax.numpy as jnp
from einops import rearrange
from PIL import Image
# Add reference code path if needed
reference_path = "reference_stereo4d_code"
if reference_path not in sys.path and os.path.exists(reference_path):
    sys.path.append(reference_path)
# Import SemanticSegmentor for semantic filtering
try:
    from reference_stereo4d_code.segmentation import SemanticSegmentor
except ImportError:
    SemanticSegmentor = None
    print("Warning: SemanticSegmentor could not be imported. Semantic filtering will be disabled.")




def load_video(video_path):
    video = VideoFileClip(video_path)
    frames = [frame for frame in video.iter_frames()]
    frames = [frame.astype(np.float32) / 255.0 for frame in frames]
    fps = video.fps
    return frames, fps


def subsample_frames(frames, original_fps=None, target_fps=0):
    """
    Subsample frames to achieve target framerate
    
    Args:
        frames: List of frames
        original_fps: Original framerate of the video (if known)
        target_fps: Target framerate (0 or None means use all frames)
        
    Returns:
        List of subsampled frames
    """
    if not frames or target_fps <= 0 or not original_fps:
        return frames
        
    # Calculate the stride to achieve target fps
    stride = max(1, round(original_fps / target_fps))
    
    return frames[::stride]


def load_frames(image_path):
    frames = []

    for filename in tqdm(list(sorted(os.listdir(image_path)))):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_path, filename)
            frame = cv2.imread(file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

    return frames, None


def format_frames(frames, target_size=336):
    height, width = frames[0].shape[:2]

    if height < width:
        new_height = target_size
        new_width = int((target_size / height) * width)
    else:
        new_width = target_size
        new_height = int((target_size / width) * height)
    
    frames = [cv2.resize(frame, (new_width, new_height)) for frame in frames]

    return frames


def load_anycam(model_path, checkpoint=None, loaded_config=None):
    config = OmegaConf.load(model_path / "training_config.yaml")

    prefix = "training_checkpoint_"
    ckpts = Path(model_path).glob(f"{prefix}*.pt")

    model_conf = config["model"]
    model_conf['use_provided_flow'] = loaded_config['prediction']['use_provided_flow'] if loaded_config is not None else False
    model_conf['use_provided_masks'] = loaded_config['prediction']['use_provided_masks'] if loaded_config is not None else False
    model_conf['use_provided_depth'] = loaded_config['prediction']['use_provided_depth'] if loaded_config is not None else False
    model_conf["train_directions"] = "forward"
    model_conf['depth_predictor']['type'] = loaded_config['prediction']['depth_predictor'] if loaded_config is not None else 'unidepth'
    model_conf['flow_model'] = loaded_config['prediction']['flow_model'] if loaded_config is not None else 'unimatch'
    model_conf['mask_path'] = loaded_config['prediction']['mask_path'] if loaded_config is not None else None

    model = AnyCamWrapper(model_conf)

    criterion = [make_loss(cfg) for cfg in config.get("loss", [])][0]

    training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]

    if training_steps:
        if checkpoint is None:
            ckpt_path = f"{prefix}{max(training_steps)}.pt"
        else:
            ckpt_path = checkpoint

        ckpt_path = Path(model_path) / ckpt_path

        print(ckpt_path)

        cp = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(cp["model"], strict=False)

    return model, criterion


def process_video(model, criterion, frames, config, ba_refinement=True, model_name='anycam'):
    """
    Process a video by fitting the AnyCam model to the provided frames.
    
    Args:
        model: The AnyCam model
        criterion: The loss criterion
        frames: List of frames as numpy arrays with shape (H,W,3) and values in [0,1]
        config: Configuration dictionary for the fit_video function (required)
        ba_refinement: Whether to perform bundle adjustment refinement (default: True)
    
    Returns:
        trajectory: The estimated camera trajectory
        proj: The camera projection matrix
        extras_dict: Additional information from the fitting process
        ba_extras: Bundle adjustment extra information
    """
    from dotdict import dotdict
    from anycam.scripts.fit_video import fit_video
    from anycam.scripts.fit_video_megasam import fit_video_megasam
    from anycam.scripts.fit_video_droidslam import fit_video_droidslam
    from anycam.scripts.fit_video_vggt import fit_video_vggt
    # Ensure config is a dotdict
    if not isinstance(config, dotdict):
        config = dotdict(config)
    config.model_name = model_name
    
    # Ensure the BA refinement setting is applied to the config
    config.do_ba_refinement = ba_refinement
    
    # Set collect_optimized_tracks to True to collect optimized tracks
    if "ba_refinement" not in config:
        config.ba_refinement = dotdict({"collect_optimized_tracks": True})
    elif "collect_optimized_tracks" not in config.ba_refinement:
        config.ba_refinement.collect_optimized_tracks = True

    print(f"Processing {len(frames)} frames...")
    print(f"Bundle adjustment refinement: {'Enabled' if ba_refinement else 'Disabled'}")
    
    # Run fit_video function
    if model_name == 'anycam':
        trajectory, proj, extras_dict, ba_extras = fit_video(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            seq_name=config.seq_name
        )
    elif model_name == 'megasam':
        trajectory, proj, extras_dict, ba_extras = fit_video_megasam(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            recon_data_path=config['prediction']['recon_data_path'],
            seq_name=config.seq_name
        )
    elif model_name == 'droidslam':
        trajectory, proj, extras_dict, ba_extras = fit_video_droidslam(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            droidslam_data_path=config['prediction']['recon_data_path'],
            seq_name=config.seq_name
        )
    elif model_name == 'vggt':
        trajectory, proj, extras_dict, ba_extras = fit_video_vggt(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            vggt_data_path=config['prediction']['recon_data_path'],
            seq_name=config.seq_name
        )
    elif model_name == 'easi3r':
        from anycam.scripts.fit_video_easi3r import fit_video_easi3r
        trajectory, proj, extras_dict, ba_extras = fit_video_easi3r(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            easi3r_data_path=config['prediction']['recon_data_path'],
            seq_name=config.seq_name
        )
    elif model_name == 'monst3r':
        from anycam.scripts.fit_video_monst3r import fit_video_monst3r
        trajectory, proj, extras_dict, ba_extras = fit_video_monst3r(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            monst3r_data_path=config['prediction']['recon_data_path'],
            seq_name=config.seq_name
        )
    elif model_name == 'spatrackerv2':
        from anycam.scripts.fit_video_spatrackerv2 import fit_video_spatrackerv2
        trajectory, proj, extras_dict, ba_extras = fit_video_spatrackerv2(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            spatrackerv2_data_path=config['prediction']['recon_data_path'],
            seq_name=config.seq_name
        )
    elif model_name == 'cut3r':
        from anycam.scripts.fit_video_cut3r import fit_video_cut3r
        trajectory, proj, extras_dict, ba_extras = fit_video_cut3r(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            cut3r_data_path=config['prediction']['recon_data_path'],
            seq_name=config.seq_name
        )
    elif model_name == 'ttt3r':
        from anycam.scripts.fit_video_ttt3r import fit_video_ttt3r
        trajectory, proj, extras_dict, ba_extras = fit_video_ttt3r(
            config,
            model,
            criterion,
            frames,
            return_extras=True,
            ttt3r_data_path=config['prediction']['recon_data_path'],
            seq_name=config.seq_name
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}. Supported: 'anycam', 'megasam', 'vggt', 'spatrackerv2', 'cut3r', 'ttt3r'")
    print("Finished processing video")
    return trajectory, proj, extras_dict, ba_extras


def load_sintel_gt_poses(data_path, sequence, frame_ids):
    """
    Load ground-truth camera poses from Sintel dataset
    
    Args:
        data_path: Path to Sintel dataset
        sequence: Sequence name
        frame_ids: List of frame IDs to load poses for
        
    Returns:
        List of ground-truth camera poses as torch tensors
    """
    gt_poses = []
    TAG_FLOAT = 202021.25
    
    for id in frame_ids:
        cam_path = os.path.join(data_path, "camdata_left", sequence, f"frame_{id+1:04d}.cam")
        
        with open(cam_path, 'rb') as f:
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert check == TAG_FLOAT, f'Wrong tag in flow file (should be: {TAG_FLOAT}, is: {check})'
            
            # Read intrinsic matrix (skip)
            _ = np.fromfile(f, dtype='float64', count=9)
            
            # Read extrinsic matrix
            N = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
            
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = N
            pose = np.linalg.inv(pose).astype(np.float32)
            
            gt_poses.append(torch.tensor(pose))
    
    return gt_poses


def load_lightspeed_gt_poses(data_path, sequence, frame_ids):
    """
    Load ground-truth camera poses from a Lightspeed dataset poses.pkl
    (world-to-camera), convert to camera-to-world, and return as torch tensors.
    
    Args:
        data_path: Path to Lightspeed frames root or its parent
        sequence: Sequence name (key in poses.pkl)
        frame_ids: List of frame IDs to load poses for
    
    Returns:
        List of ground-truth camera poses (camera-to-world) as torch tensors
    """
    import pickle as pkl

    # Try common locations for poses.pkl next to frames or their parent
    candidates = [
        os.path.join(data_path, "poses.pkl"),
        os.path.join(os.path.dirname(data_path), "poses.pkl"),
    ]

    poses_by_seq = None
    for cand in candidates:
        if os.path.exists(cand):
            with open(cand, "rb") as f:
                poses_by_seq = pkl.load(f)
            break
    if poses_by_seq is None:
        raise FileNotFoundError(f"poses.pkl not found next to lightspeed frames. Tried: {candidates}")

    if sequence not in poses_by_seq:
        raise KeyError(f"Sequence {sequence} not found in poses.pkl")

    seq_poses = poses_by_seq[sequence]

    gt_poses = []
    num_available = len(seq_poses)
    for idx in frame_ids:
        # Clip index to available range to be robust to off-by-one differences
        idx_clip = max(min(idx, num_available - 1), 0)
        w2c = seq_poses[idx_clip].astype(np.float32)
        if w2c.shape == (3, 4):
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        c2w = np.linalg.inv(w2c).astype(np.float32)
        gt_poses.append(torch.tensor(c2w))

    return gt_poses


def align_trajectories(pred_traj, gt_traj):
    """
    Align predicted trajectory to ground truth trajectory with scale correction
    
    Args:
        pred_traj: List of predicted camera poses as torch tensors or numpy arrays
        gt_traj: List of ground truth camera poses as torch tensors or numpy arrays
        
    Returns:
        List of aligned predicted camera poses as torch tensors
    """
    if not HAS_EVO:
        print("Warning: evo package not available for proper trajectory alignment.")
        return pred_traj
        
    # Convert to numpy arrays if tensors
    pred_numpy = np.stack([pose.cpu().numpy() if isinstance(pose, torch.Tensor) else pose for pose in pred_traj]).astype(np.float64)
    gt_numpy = np.stack([pose.cpu().numpy() if isinstance(pose, torch.Tensor) else pose for pose in gt_traj]).astype(np.float64)
    
    # Create PosePath3D objects for alignment
    pred_path = PosePath3D(poses_se3=pred_numpy)
    gt_path = PosePath3D(poses_se3=gt_numpy)
    
    try:
        # Align predicted trajectory to ground truth with scale correction
        pred_path.align(gt_path, correct_scale=True)
        print("Successfully aligned trajectories with scale correction")
    except Exception as e:
        print(f"Warning: Could not align trajectories: {e}")
    
    # Convert aligned trajectory back to torch tensors
    aligned_traj = [torch.tensor(pose, dtype=torch.float32) for pose in pred_path.poses_se3]
    
    return aligned_traj


def plot_to_rerun(
        trajectory, 
        depths, 
        imgs, 
        proj,
        uncertainties=None, 
        radii=1.5, 
        uncertainty_thresh=-1, 
        max_depth=-1, 
        filter_depth_threshold=0.1,
        image_plane_distance=0.05,
        keyframes=None,
        rerun_mode="spawn",
        clear_rerun=False,
        gt_trajectory=None,
        initial_tracks=None,
        optimized_tracks=None,
        track_subsample=5,
        track_radii=2.0,
        initial_pixel_coords=None,
        initial_pixel_depths=None,
        initial_pixel_indices=None,
        point_trajectory_grid_size=16,
        structured_3d_tracks=None,
        track_optimization_data=None,
        show_2d_tracks=False,
        show_3d_tracks=True,  # New parameter for controlling 3D track visualization
        max_depth_filter=10000,  # New parameter for depth filtering
        multi_view_frame_indices=None
        ):
    
    h, w = imgs[0].shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Precomputed 2D tracks for visualization
    precomputed_2d_tracks = {}
    track_colors = {}
    
    # Additional structures for optimized tracks
    optimized_3d_tracks = {}
    
    # Initialize window-based track structures (always defined)
    window_2d_tracks = {}  # window_id -> {track_id -> track_points}
    window_track_colors = {}  # window_id -> {track_id -> color}
    
    # Window-based track data
    windowed_track_data = track_optimization_data.get('windowed_track_data', []) if track_optimization_data else []
    
    # Global track data
    global_track_data = track_optimization_data.get('global_track_data', None) if track_optimization_data else None
    
    if global_track_data is not None:
        print("Preparing global track visualization...")
        print(f"Global track data keys: {list(global_track_data.keys())}")
        print(f"Has enough tracks: {global_track_data.get('has_enough_tracks', False)}")
        
        # Only process if we have enough tracks
        if global_track_data.get('has_enough_tracks', False):
            # Use window_id = 0 for global tracks
            window_id = 0
            frame_range = [0, len(trajectory)]  # Entire sequence
            
            # Extract data from global_track_data - handle the correct tensor dimensions
            # pixel_coords has shape (1, 1, num_tracks, seq_len, 2)
            # pixel_depths has shape (1, 1, num_tracks, seq_len, 1)  
            # pixel_indices has shape (1, 1, num_tracks, seq_len, 2)
            # visibles (masks_full) has shape (num_tracks, seq_len, 1)
            pixel_coords = global_track_data['pixel_coords'][0, 0]  # (num_tracks, seq_len, 2)
            pixel_depths = global_track_data['pixel_depths'][0, 0, :, :, 0]  # (num_tracks, seq_len)
            pixel_indices = global_track_data['pixel_indices'][0, 0, :, :, 0]  # (num_tracks, seq_len)
            masks_filtered = global_track_data['visibles'][:, :, 0]  # (num_tracks, seq_len) - this is actually masks_full
            num_tracks = global_track_data['num_tracks']
            
            # Get filtering information if available (same as windowed processing)
            filtered_indices = global_track_data.get('filtered_indices', None)
            has_enough_tracks = global_track_data.get('has_enough_tracks', True)
            masks = global_track_data.get('masks', None)  # Original masks before filtering
            # masks_filtered = global_track_data.get('masks_filtered', None)  # Masks after filtering
            visible_list = global_track_data.get('visible_list', None)  # Original visibility
            visible_list_filtered = global_track_data.get('visible_list_filtered', None)  # Visibility after filtering
            
            print(f"Global tracks: {num_tracks} tracks, sequence length: {global_track_data['track_len']}")
            print(f"Global filtering info - filtered_indices: {filtered_indices is not None}, masks_filtered: {masks_filtered is not None}")
            
            if num_tracks == 0 or not has_enough_tracks:
                print("Global track data: num_tracks=0 or not has_enough_tracks - skipping")
            else:
                window_2d_tracks[window_id] = {}
                window_track_colors[window_id] = {}
                
                # Determine which tracks to process based on filtering (same logic as windowed)
                if filtered_indices is not None:
                    # Use only the filtered tracks that were actually optimized
                    # filtered_indices contains the original indices of tracks that passed filtering
                    # pixel_coords, pixel_depths, etc. are already filtered and contain only valid tracks
                    num_filtered_tracks = len(filtered_indices)
                    tracks_to_process = np.arange(num_filtered_tracks)  # Use indices into the filtered data
                    print(f"Global tracks: Using {num_filtered_tracks} filtered tracks out of {num_tracks} total tracks")
                else:
                    # Fall back to original track selection
                    tracks_to_process = np.arange(num_tracks)
                    if track_subsample > 1:
                        tracks_to_process = tracks_to_process[::track_subsample]
                
                # Apply subsampling to tracks if requested
                if track_subsample > 1:
                    tracks_to_process = tracks_to_process[::track_subsample]
                
                for i, track_filtered_idx in enumerate(tracks_to_process):
                    # track_filtered_idx is now an index into the filtered data
                    # Get the original track index for naming purposes
                    if filtered_indices is not None:
                        track_original_idx = int(filtered_indices[track_filtered_idx])
                        track_global_id = f"global_t{track_original_idx}"
                    else:
                        track_original_idx = track_filtered_idx
                        track_global_id = f"global_t{track_original_idx}"
                    
                    track_points_2d = []
                    
                    track_len_for_global = pixel_coords.shape[1]
                    for point_idx in range(track_len_for_global):
                        # Get point data using the filtered index
                        norm_x = pixel_coords[track_original_idx, point_idx, 0]
                        norm_y = pixel_coords[track_original_idx, point_idx, 1]
                        depth_val = pixel_depths[track_original_idx, point_idx]
                        frame_idx = int(pixel_indices[track_original_idx, point_idx])  # Global frame index
                        
                        # Simplified and consistent validity check (same logic as windowed)
                        # Use the filtered masks first (most accurate for the actual data being used)
                        point_valid = True
                        if masks_filtered is not None and i < len(masks_filtered):
                            # Use the filtered masks (after semantic filtering) - most accurate
                            if point_idx < masks_filtered.shape[1]:
                                point_valid = bool(masks_filtered[i, point_idx])
                        elif visible_list_filtered is not None and i < len(visible_list_filtered):
                            # Use filtered visibility as next best option
                            if point_idx < visible_list_filtered.shape[1]:
                                point_valid = bool(visible_list_filtered[i, point_idx])
                        elif masks is not None and track_filtered_idx < len(masks):
                            # Fall back to original masks using the filtered index
                            if point_idx < masks.shape[1]:
                                point_valid = bool(masks[track_filtered_idx, point_idx, 0])
                        elif visible_list is not None and track_filtered_idx < len(visible_list):
                            # Final fallback to original visibility using filtered index
                            if point_idx < visible_list.shape[1]:
                                point_valid = bool(visible_list[track_filtered_idx, point_idx])
                        else:
                            # Final fallback: use masks_full (visibles) from global_track_data
                            point_valid = bool(masks_full[track_filtered_idx, point_idx])
                        
                        # Additional validation for debugging (same as windowed)
                        if depth_val > 1e-5 and 0 <= frame_idx < len(trajectory) and point_valid and not np.isnan(depth_val):
                            # Convert normalized coordinates to pixel coordinates
                            pixel_x = (norm_x + 1) * w / 2 - 0.5
                            pixel_y = (norm_y + 1) * h / 2 - 0.5
                            
                            track_points_2d.append({
                                'frame': frame_idx,
                                'pixel_x': pixel_x,
                                'pixel_y': pixel_y,
                                'norm_x': norm_x,
                                'norm_y': norm_y,
                                'depth': depth_val,
                                'window_idx': window_id,
                                'track_local_idx': track_original_idx,
                                'point_idx': point_idx,
                                'is_valid': point_valid,
                                'frame_range': frame_range,
                                'is_global': True
                            })
                    
                    if len(track_points_2d) > 1:  # Only keep tracks with sufficient valid points for global tracks
                        window_2d_tracks[window_id][track_global_id] = track_points_2d
                        # Assign a consistent color to each global track (same pattern as windowed)
                        window_track_colors[window_id][track_global_id] = [
                            (window_id * 60 + track_original_idx * 40) % 255,
                            (180 + track_original_idx * 25) % 255,
                            255
                        ]
                
                print(f"Processed {len(window_2d_tracks[window_id])} valid global tracks for visualization")
        else:
            print("Global track data available but has_enough_tracks is False - skipping global track visualization")
    
    elif windowed_track_data:
        print("Preparing window-based track visualization...")
        
        for window_info in windowed_track_data:
            window_id = window_info['window_id']
            frame_range = window_info['frame_range']
            pixel_coords = window_info['pixel_coords']  # (num_tracks, track_len, 2)
            pixel_depths = window_info['pixel_depths']   # (num_tracks, track_len)
            pixel_indices = window_info['pixel_indices'] # (num_tracks, track_len) - global frame indices
            num_tracks = window_info['num_tracks']
            
            # Get filtering information if available
            filtered_indices = window_info.get('filtered_indices', None)
            has_enough_tracks = window_info.get('has_enough_tracks', True)
            masks = window_info.get('masks', None)  # Original masks before filtering
            masks_filtered = window_info.get('masks_filtered', None)  # Masks after filtering
            visible_list = window_info.get('visible_list', None)  # Original visibility
            visible_list_filtered = window_info.get('visible_list_filtered', None)  # Visibility after filtering
            
            if num_tracks == 0 or not has_enough_tracks:
                continue
                
            window_2d_tracks[window_id] = {}
            window_track_colors[window_id] = {}
            
            # Determine which tracks to process based on filtering
            if filtered_indices is not None:
                # Use only the filtered tracks that were actually optimized
                # filtered_indices contains the original indices of tracks that passed filtering
                # pixel_coords, pixel_depths, etc. are already filtered and contain only valid tracks
                num_filtered_tracks = len(filtered_indices)
                tracks_to_process = np.arange(num_filtered_tracks)  # Use indices into the filtered data
                print(f"Window {window_id}: Using {num_filtered_tracks} filtered tracks out of {num_tracks} total tracks")
            else:
                # Fall back to original track selection
                tracks_to_process = np.arange(num_tracks)
                if track_subsample > 1:
                    tracks_to_process = tracks_to_process[::track_subsample]
            
            # Apply subsampling to tracks if requested
            if track_subsample > 1:
                tracks_to_process = tracks_to_process[::track_subsample]
            
            for i, track_filtered_idx in enumerate(tracks_to_process):
                # track_filtered_idx is now an index into the filtered data
                # Get the original track index for naming purposes
                if filtered_indices is not None:
                    track_original_idx = int(filtered_indices[track_filtered_idx])
                    track_global_id = f"w{window_id}_t{track_original_idx}"
                else:
                    track_original_idx = track_filtered_idx
                    track_global_id = f"w{window_id}_t{track_original_idx}"
                
                track_points_2d = []
                
                track_len_for_window = pixel_coords.shape[1]
                for point_idx in range(track_len_for_window):
                    # Get point data using the filtered index
                    norm_x = pixel_coords[track_filtered_idx, point_idx, 0]
                    norm_y = pixel_coords[track_filtered_idx, point_idx, 1]
                    depth_val = pixel_depths[track_filtered_idx, point_idx]
                    frame_idx = int(pixel_indices[track_filtered_idx, point_idx])  # Global frame index
                    
                    # Simplified and consistent validity check
                    # Use the filtered masks first (most accurate for the actual data being used)
                    point_valid = True
                    if masks_filtered is not None and i < len(masks_filtered):
                        # Use the filtered masks (after semantic filtering) - most accurate
                        if point_idx < masks_filtered.shape[1]:
                            point_valid = bool(masks_filtered[i, point_idx, 0])
                    elif visible_list_filtered is not None and i < len(visible_list_filtered):
                        # Use filtered visibility as next best option
                        if point_idx < visible_list_filtered.shape[1]:
                            point_valid = bool(visible_list_filtered[i, point_idx])
                    elif masks is not None and track_filtered_idx < len(masks):
                        # Fall back to original masks using the filtered index
                        if point_idx < masks.shape[1]:
                            point_valid = bool(masks[track_filtered_idx, point_idx, 0])
                    elif visible_list is not None and track_filtered_idx < len(visible_list):
                        # Final fallback to original visibility using filtered index
                        if point_idx < visible_list.shape[1]:
                            point_valid = bool(visible_list[track_filtered_idx, point_idx])
                    
                    # Additional validation for debugging
                    if depth_val > 1e-5 and 0 <= frame_idx < len(trajectory) and point_valid:
                        # Convert normalized coordinates to pixel coordinates
                        pixel_x = (norm_x + 1) * w / 2 - 0.5
                        pixel_y = (norm_y + 1) * h / 2 - 0.5
                        
                        track_points_2d.append({
                            'frame': frame_idx,
                            'pixel_x': pixel_x,
                            'pixel_y': pixel_y,
                            'norm_x': norm_x,
                            'norm_y': norm_y,
                            'depth': depth_val,
                            'window_idx': window_id,
                            'track_local_idx': track_original_idx,
                            'point_idx': point_idx,
                            'is_valid': point_valid,
                            'frame_range': frame_range 
                        })
                
                if len(track_points_2d) > 1:  # Only keep tracks with multiple valid points
                    window_2d_tracks[window_id][track_global_id] = track_points_2d
                    # Assign a consistent color to each track (same for both unoptimized and optimized)
                    window_track_colors[window_id][track_global_id] = [
                        (window_id * 60 + track_original_idx * 40) % 255,
                        (180 + track_original_idx * 25) % 255,
                        255
                    ]
    
    elif initial_pixel_coords is not None and initial_pixel_depths is not None and initial_pixel_indices is not None:
        print("Precomputing 2D tracks for visualization...")
        
        for window_idx in range(len(initial_pixel_coords)):
            pixel_coords_window = initial_pixel_coords[window_idx]  # (num_tracks, track_len, 2)
            pixel_depths_window = initial_pixel_depths[window_idx]  # (num_tracks, track_len)
            pixel_indices_window = initial_pixel_indices[window_idx]  # (num_tracks, track_len)
            
            if pixel_coords_window.shape[0] == 0:
                continue
                
            num_tracks_in_window = pixel_coords_window.shape[0]
            track_len_for_window = pixel_coords_window.shape[1]
            
            # Subsample tracks if requested
            tracks_to_process = np.arange(num_tracks_in_window)
            if track_subsample > 1:
                tracks_to_process = tracks_to_process[::track_subsample]
            
            for track_local_idx in tracks_to_process:
                track_global_id = f"w{window_idx}_t{track_local_idx}"
                track_points_2d = []
                
                for point_idx in range(track_len_for_window):
                    # Convert normalized coordinates back to pixel coordinates
                    norm_x = pixel_coords_window[track_local_idx, point_idx, 0]
                    norm_y = pixel_coords_window[track_local_idx, point_idx, 1]
                    depth_val = pixel_depths_window[track_local_idx, point_idx]
                    frame_idx = int(pixel_indices_window[track_local_idx, point_idx])
                    
                    if depth_val > 1e-5 and 0 <= frame_idx < len(trajectory):
                        # Convert normalized coordinates to pixel coordinates
                        pixel_x = (norm_x + 1) * w / 2 - 0.5
                        pixel_y = (norm_y + 1) * h / 2 - 0.5
                        
                        track_points_2d.append({
                            'frame': frame_idx,
                            'pixel_x': pixel_x,
                            'pixel_y': pixel_y,
                            'norm_x': norm_x,
                            'norm_y': norm_y,
                            'depth': depth_val,
                            'window_idx': window_idx,
                            'track_local_idx': track_local_idx,
                            'point_idx': point_idx
                        })
                
                if len(track_points_2d) > 1:  # Only keep tracks with multiple points
                    precomputed_2d_tracks[track_global_id] = track_points_2d
                    # Assign a consistent color to each track (same for both unoptimized and optimized)
                    track_colors[track_global_id] = [
                        (window_idx * 50 + track_local_idx * 30) % 255,
                        (150 + track_local_idx * 20) % 255,
                        255
                    ]

    # Track 3D trajectory points for each track (unoptimized)
    track_3d_trajectories = {}
    
    # Track optimized 3D trajectory points for each track (with BA adjustments)
    optimized_track_3d_trajectories = {}
    
    def filter_depth(depth, threshold=0.1):
        _, h, w = depth.shape
        depth = depth.clone()[None, ...]
        
        # Apply depth filtering before other processing
        depth[depth > max_depth_filter] = 0
        
        median = torch.median(depth)
        depth_grad = torch.stack(torch.gradient(depth, dim=(-2, -1))).norm(dim=0)
        mask = depth_grad < median * threshold
        return mask
    
    def lift_image(img, depth, pose, proj):
        h_img_lift, w_img_lift = img.shape[:2]
        device_lift = depth.device

        # Check if depth dimensions match image dimensions
        depth_h_lift, depth_w_lift = depth.shape[-2:]
        if depth_h_lift != h_img_lift or depth_w_lift != w_img_lift:
            # Resize depth to match image dimensions
            current_depth_unsqueezed = depth.unsqueeze(0).unsqueeze(0) if depth.dim() == 2 else depth.unsqueeze(0)
            depth = F.interpolate(
                current_depth_unsqueezed, 
                size=(h_img_lift, w_img_lift), 
                mode='bilinear',
                align_corners=False
            )[0]
            if depth.shape[0] == 1 and depth.dim() == 3:
                depth = depth[0]

        # Filter out extremely large depths following compute_loss logic
        depth_filtered = depth.clone()
        depth_filtered[depth_filtered > max_depth_filter] = 0  # Set to 0 to exclude from visualization

        proj_tensor_lift = torch.tensor(proj, device=device_lift).float()

        # DON'T normalize intrinsics - use them directly like MegaSaM
        # Keep original focal lengths and principal points
        inv_proj_lift = torch.inverse(proj_tensor_lift[:3, :3])  # Use only 3x3 part

        # Create pixel grid in image coordinates (not normalized)
        y, x = torch.meshgrid(torch.arange(h_img_lift, device=device_lift), 
                             torch.arange(w_img_lift, device=device_lift), indexing='ij')
        pts_grid = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x.flatten())], dim=0)
        
        # Unproject to camera coordinates
        pts_lifted = inv_proj_lift @ pts_grid.float()
        pts_lifted = pts_lifted * depth_filtered.reshape(1, -1).to(device_lift)
        pts_lifted = torch.cat((pts_lifted, torch.ones(1, h_img_lift*w_img_lift, device=device_lift)), dim=0)
        
        # Transform to world coordinates
        pts_world = pose.to(pts_lifted.dtype) @ pts_lifted
        pts_world = pts_world[:3, :].T

        colors_rgb = torch.tensor(img.reshape(-1, 3)).to(device_lift)

        return pts_world, colors_rgb
    
    # Track statistics for depth filtering
    depth_filtered_count = 0
    total_lift_attempts = 0
    
    def lift_2d_point(norm_x, norm_y, depth_val, pose, proj_matrix):
        """Lift a single 2D point to 3D using unnormalized intrinsics like MegaSaM"""
        nonlocal depth_filtered_count, total_lift_attempts
        total_lift_attempts += 1
        
        # Filter out extremely large depths following compute_loss logic
        if depth_val > max_depth_filter:
            depth_filtered_count += 1
            return None
            
        device_lift = pose.device
        
        proj_tensor = torch.tensor(proj_matrix, device=device_lift).float()
        
        # Convert normalized coordinates back to pixel coordinates
        pixel_x = (norm_x + 1) * w / 2 - 0.5
        pixel_y = (norm_y + 1) * h / 2 - 0.5
        
        # Use unnormalized intrinsics
        inv_proj = torch.inverse(proj_tensor[:3, :3])
        
        # Create homogeneous point in pixel coordinates
        pt_pixel = torch.tensor([pixel_x, pixel_y, 1.0], device=device_lift, dtype=torch.float32)
        
        # Lift to camera coordinates
        pt_cam_unscaled = inv_proj @ pt_pixel
        pt_cam_scaled = pt_cam_unscaled * depth_val
        pt_cam_h = torch.cat((pt_cam_scaled, torch.tensor([1.0], device=device_lift, dtype=torch.float32)), dim=0)
        
        # Transform to world coordinates
        pt_world_h = pose.to(pt_cam_h.dtype) @ pt_cam_h
        pt_world = pt_world_h[:3]
        
        return pt_world
    
    def project_3d_to_2d(pt_3d_world, pose, proj_matrix):
        """Project a 3D world point to 2D pixel coordinates using camera pose and projection"""
        device_proj = pose.device
        
        # Convert to camera coordinates
        pose_inv = torch.inverse(pose)  # World to camera transform
        pt_3d_world_h = torch.cat([pt_3d_world, torch.ones(1, device=device_proj, dtype=pt_3d_world.dtype)])
        pt_3d_cam_h = pose_inv @ pt_3d_world_h
        pt_3d_cam = pt_3d_cam_h[:3]
        
        # Check if point is in front of camera
        if pt_3d_cam[2] <= 1e-6:  # Behind camera or too close
            return None, False
            
        # Project to 2D using camera intrinsics
        proj_tensor = torch.tensor(proj_matrix, device=device_proj, dtype=torch.float32)
        pt_2d_h = proj_tensor[:3, :3] @ pt_3d_cam
        pt_2d = pt_2d_h[:2] / pt_2d_h[2]  # Normalize by depth
        
        # Convert to pixel coordinates
        pixel_x = pt_2d[0].item()
        pixel_y = pt_2d[1].item()
        
        # Check if point is within image bounds
        if 0 <= pixel_x < w and 0 <= pixel_y < h:
            return [pixel_x, pixel_y], True
        else:
            return None, False
    
    imgs = np.array(imgs)

    # Initialize rerun with appropriate mode
    if rerun_mode == "spawn":
        rr.init("AnyCam Demo", recording_id=uuid.uuid4(), spawn=True)
    elif rerun_mode == "connect":
        rr.init("AnyCam Demo", recording_id=uuid.uuid4(), spawn=False)
        print(f"Connecting to existing Rerun server.")
        rr.connect()
    else:
        raise ValueError(f"Unsupported rerun mode: {rerun_mode}. Use 'spawn' or 'connect'.")
    
    # Clear previous data if requested
    if clear_rerun:
        print("Clearing previous rerun visualization data...")
        rr.log("world", rr.Clear(recursive=True))
        # Small delay to ensure clearing is processed
        time.sleep(0.1)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    # Create two separate scenes for unoptimized and optimized tracks
    rr.log("world/scene_unoptimized", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("world/scene_optimized", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    # Create blueprint with both scenes visible
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Vertical(
                rr.blueprint.Spatial3DView(
                    origin="/world/scene_unoptimized",
                    name="Unoptimized 3D",
                    background=[255, 255, 255],
                    line_grid=rr.blueprint.archetypes.LineGrid3D(visible=False),
                ),
                rr.blueprint.Spatial3DView(
                    origin="/world/scene_optimized",
                    name="Optimized 3D",
                    background=[255, 255, 255],
                    line_grid=rr.blueprint.archetypes.LineGrid3D(visible=False),
                ),
            ),
            rr.blueprint.Vertical(
                rr.blueprint.Spatial2DView(origin="/world/scene_unoptimized/active_cam/input", name="2D View"),
                rr.blueprint.Spatial2DView(origin="/world/scene_unoptimized/active_cam/uncertainty", name="Uncertainty"),
            ),
        ),
    )
    rr.send_blueprint(blueprint, make_active=True)

    if multi_view_frame_indices is not None:
        # Set a single time point for the static view
        rr.set_time_sequence("step", 0)

        # Log the full camera trajectory once
        for scene_name in ["scene_unoptimized", "scene_optimized"]:
            rr.log(f"world/{scene_name}/cam_traj", rr.LineStrips3D([pose[:3, 3].cpu().numpy().tolist() for pose in trajectory]), colors=[(0, 255, 0)])
            if gt_trajectory is not None:
                gt_points = [gt_pose[:3, 3].cpu().numpy().tolist() for gt_pose in gt_trajectory]
                rr.log(f"world/{scene_name}/gt_cam_traj", rr.LineStrips3D(gt_points, colors=[(255, 0, 0)]))

        # Loop through the selected frames
        for frame_idx in multi_view_frame_indices:
            if not (0 <= frame_idx < len(trajectory)):
                print(f"Warning: frame index {frame_idx} is out of bounds for trajectory/depths.")
                continue
            if not (0 <= frame_idx < len(imgs)):
                print(f"Warning: frame index {frame_idx} is out of bounds for images.")
                continue

            pose = trajectory[frame_idx]
            rot = pose[:3, :3].cpu().numpy()

            for scene_name in ["scene_unoptimized", "scene_optimized"]:
                # Log camera for this frame
                rr.log(f"world/{scene_name}/camera_{frame_idx}", rr.Pinhole(
                    resolution=[w, h],
                    focal_length=float(proj[0, 0]),
                    image_plane_distance=image_plane_distance,
                ))
                rr.log(f"world/{scene_name}/camera_{frame_idx}", rr.Transform3D(translation=pose[:3, 3].cpu(), mat3x3=rot))
                
                # Log input image for this camera
                rr.log(f"world/{scene_name}/camera_{frame_idx}/input", rr.Image((imgs[frame_idx] * 255).astype(np.uint8)).compress(jpeg_quality=95))

                # Log point cloud for this frame
                # In multi-view, depths and trajectory are aligned with keyframes
                kid = frame_idx
                
                # Check if depth map exists for this frame
                if 0 <= kid < len(depths):
                    current_img_np = imgs[frame_idx]
                    current_depth_map_tensor = depths[kid]
                    current_pose_cuda = trajectory[frame_idx].cuda()
                    current_depth_map_cuda = current_depth_map_tensor.cuda()
                    current_depth_map_cuda = current_depth_map_cuda.unsqueeze(dim=0) if len(current_depth_map_cuda.shape) == 2 else current_depth_map_cuda

                    pts, colors = lift_image(torch.tensor(current_img_np).cuda(), current_depth_map_cuda, current_pose_cuda, proj)
                    
                    mask = filter_depth(current_depth_map_cuda, threshold=filter_depth_threshold).view(-1)
                    if max_depth > 0:
                        depth_mask = current_depth_map_cuda.view(-1) < max_depth
                        mask = mask & depth_mask
                    
                    pts = pts[mask, :]
                    colors = colors[mask, :]

                    subsample_pts = 5
                    if len(pts) > subsample_pts:
                        pts = pts[subsample_pts//2::subsample_pts]
                        colors = colors[subsample_pts//2::subsample_pts]
                    
                    colors = (colors * 255).clamp(0, 255).to(torch.uint8)

                    rr.log(f"world/{scene_name}/points_{frame_idx}", 
                           rr.Points3D(pts[:, :3].cpu().numpy(), 
                                      colors=colors[:, :3].cpu().numpy(), 
                                      radii=rr.Radius.ui_points([radii])))
        return # End of function if multi-view

    for id_step_time in range(len(trajectory)):
        rr.set_time_sequence("step", id_step_time)

        pose = trajectory[id_step_time]
        rot = pose[:3, :3].cpu().numpy()

        # Log camera and other common elements to both scenes
        for scene_name in ["scene_unoptimized", "scene_optimized"]:
            rr.log(f"world/{scene_name}/active_cam", rr.Pinhole(
                resolution=[w, h],
                focal_length=float(proj[0, 0]),
                image_plane_distance=image_plane_distance, 
            ), static=True)
            rr.log(f"world/{scene_name}/active_cam", rr.Transform3D(translation=pose[:3, 3].cpu(), mat3x3=rot, axis_length=0.01))

            # Log predicted trajectory (green) in both scenes
            rr.log(f"world/{scene_name}/cam_traj", rr.LineStrips3D([pose[:3, 3].cpu().numpy().tolist() for pose in trajectory[:id_step_time+1]], 
                                                            colors=[(0, 255, 0)]), 
                   static=False)
            
            # Log ground-truth trajectory if provided (red) in both scenes
            if gt_trajectory is not None and id_step_time < len(gt_trajectory):
                gt_points = [gt_pose[:3, 3].cpu().numpy().tolist() for gt_pose in gt_trajectory[:id_step_time+1]]
                rr.log(f"world/{scene_name}/gt_cam_traj", rr.LineStrips3D(gt_points, 
                                                                  colors=[(255, 0, 0)]), 
                       static=False)

            # RESTORE the scene transform - this is what makes the lifted points move with the camera
            # This transform affects all child entities under world/scene, which is exactly what we want
            # for the lifted points to appear in the correct relative position to the camera
            rr.log(f"world/{scene_name}", rr.Transform3D(translation=pose[:3, 3].cpu(), mat3x3=rot, axis_length=0, from_parent=True))

        # Log input images FIRST to ensure camera frustums show the correct image
        rr.log("world/scene_unoptimized/active_cam/input", rr.Image((imgs[id_step_time] * 255).astype(np.uint8)).compress(jpeg_quality=95))
        rr.log("world/scene_optimized/active_cam/input", rr.Image((imgs[id_step_time] * 255).astype(np.uint8)).compress(jpeg_quality=95))
        
        # Log uncertainty image separately (this should NOT interfere with camera frustum)
        if uncertainties is not None:
            uncertainty_img = color_tensor((uncertainties[id_step_time] / uncertainty_thresh).clamp(0, 1), cmap="plasma", norm=False)[0]
            uncertainty_img = uncertainty_img.cpu().numpy()
            uncertainty_img = (uncertainty_img * 255).astype(np.uint8)
            rr.log("world/scene_unoptimized/active_cam/uncertainty", rr.Image(uncertainty_img).compress(jpeg_quality=95))

        # Window-based track visualization (including global tracks)
        if show_2d_tracks and (windowed_track_data or global_track_data is not None):
            # Find which windows are active for the current frame
            active_windows = []
            
            # Handle windowed track data
            if windowed_track_data:
                for window_info in windowed_track_data:
                    frame_range = window_info['frame_range']
                    if frame_range[0] <= id_step_time < frame_range[1]:
                        active_windows.append(window_info['window_id'])
            
            # Handle global track data (always active for all frames)
            if global_track_data is not None and global_track_data.get('has_enough_tracks', False):
                active_windows.append(0)  # Global tracks use window_id = 0
            
            # Debug: Print window information every 20 frames
            if id_step_time % 20 == 0:
                print(f"Frame {id_step_time}: Active windows: {active_windows}")
                print(f"Frame {id_step_time}: Available window track data: {list(window_2d_tracks.keys())}")
                for window_id in window_2d_tracks.keys():
                    print(f"  Window {window_id}: {len(window_2d_tracks[window_id])} tracks")
            
            # Add window timeline controls
            for window_id in window_2d_tracks.keys():
                rr.set_time_sequence("window", window_id)
                
                # Check if this window should be visualized at current frame
                if window_id in active_windows and window_id in window_2d_tracks:
                    current_window_tracks = window_2d_tracks[window_id]
                    current_window_colors = window_track_colors[window_id]
                    
                    # Debug: Print track information for global tracks
                    if window_id == 0 and global_track_data is not None and id_step_time % 20 == 0:
                        print(f"Frame {id_step_time}: Visualizing {len(current_window_tracks)} global tracks")
                    
                    # Visualize 2D tracks for this window (only in unoptimized scene to avoid duplication)
                    # Plot each track individually within its window group
                    for track_id, track_points in current_window_tracks.items():
                        # Get all points up to current frame for trail visualization
                        trail_points = []
                        current_point = None
                        
                        for point_data in track_points:
                            if point_data['frame'] <= id_step_time:
                                trail_points.append([point_data['pixel_x'], point_data['pixel_y']])
                                if point_data['frame'] == id_step_time:
                                    current_point = point_data
                    
                        # Debug: Print track connectivity info for debugging
                        if len(trail_points) > 1 and current_point is not None and id_step_time % 10 == 0:  # Print every 10 frames
                            last_trail_point = trail_points[-1]
                            current_2d_point = [current_point['pixel_x'], current_point['pixel_y']]
                            distance = ((last_trail_point[0] - current_2d_point[0])**2 + 
                                       (last_trail_point[1] - current_2d_point[1])**2)**0.5
                            print(f"Frame {id_step_time}, Track {track_id}: Trail-to-current distance = {distance:.2f} pixels")
                            if distance > 5.0:  # Alert if disconnect is > 5 pixels
                                print(f"  WARNING: Large disconnect detected for track {track_id}!")
                                print(f"  Last trail point: {last_trail_point}")
                                print(f"  Current point: {current_2d_point}")
                        
                        # Log trail for this specific track if it has multiple points
                        if len(trail_points) > 1:
                            track_type = "global" if window_id == 0 and global_track_data is not None else "windowed"
                            rr.log(f"world/scene_unoptimized/active_cam/input/{track_type}_window_{window_id}/track_trail_{track_id}", 
                                   rr.LineStrips2D([trail_points], 
                                                  colors=[current_window_colors[track_id]],
                                                  radii=[1.0]))
                        
                        # Log current point for this specific track if it exists
                        if current_point is not None:
                            track_type = "global" if window_id == 0 and global_track_data is not None else "windowed"
                            rr.log(f"world/scene_unoptimized/active_cam/input/{track_type}_window_{window_id}/track_point_{track_id}", 
                                   rr.Points2D([[current_point['pixel_x'], current_point['pixel_y']]], 
                                              colors=[current_window_colors[track_id]],
                                              radii=[track_radii]))
            
            # Reset timeline for frame sequence
            rr.set_time_sequence("step", id_step_time)

        # Legacy track visualization (for backward compatibility)
        elif show_2d_tracks and precomputed_2d_tracks:
            # Group tracks by window for consistent visualization
            window_grouped_tracks = {}
            window_grouped_colors = {}
            
            # Group precomputed tracks by window
            for track_id, track_points in precomputed_2d_tracks.items():
                if track_points:
                    # Extract window_idx from track_id (format: w{window_idx}_t{track_idx})
                    window_idx = int(track_id.split('_')[0][1:]) if track_id.startswith('w') else 0
                    
                    if window_idx not in window_grouped_tracks:
                        window_grouped_tracks[window_idx] = {}
                        window_grouped_colors[window_idx] = {}
                    
                    window_grouped_tracks[window_idx][track_id] = track_points
                    window_grouped_colors[window_idx][track_id] = track_colors[track_id]
            
            # Plot tracks window by window
            for window_idx, window_tracks in window_grouped_tracks.items():
                for track_id, track_points in window_tracks.items():
                    # Get all points up to current frame for trail visualization
                    trail_points = []
                    current_point = None
                    
                    for point_data in track_points:
                        if point_data['frame'] <= id_step_time:
                            trail_points.append([point_data['pixel_x'], point_data['pixel_y']])
                            if point_data['frame'] == id_step_time:
                                current_point = point_data
                
                    # Debug: Print track connectivity info for debugging (legacy path)
                    if len(trail_points) > 1 and current_point is not None and id_step_time % 10 == 0:  # Print every 10 frames
                        last_trail_point = trail_points[-1]
                        current_2d_point = [current_point['pixel_x'], current_point['pixel_y']]
                        distance = ((last_trail_point[0] - current_2d_point[0])**2 + 
                                   (last_trail_point[1] - current_2d_point[1])**2)**0.5
                        print(f"Frame {id_step_time}, Track {track_id} (legacy): Trail-to-current distance = {distance:.2f} pixels")
                        if distance > 5.0:  # Alert if disconnect is > 5 pixels
                            print(f"  WARNING: Large disconnect detected for track {track_id} (legacy)!")
                            print(f"  Last trail point: {last_trail_point}")
                            print(f"  Current point: {current_2d_point}")
                    
                    # Log trail for this specific track if it has multiple points
                    if len(trail_points) > 1:
                        rr.log(f"world/scene_unoptimized/active_cam/input/window_{window_idx}/track_trail_{track_id}", 
                               rr.LineStrips2D([trail_points], 
                                              colors=[window_grouped_colors[window_idx][track_id]],
                                              radii=[1.0]))
                    
                    # Log current point for this specific track if it exists
                    if current_point is not None:
                        rr.log(f"world/scene_unoptimized/active_cam/input/window_{window_idx}/track_point_{track_id}", 
                               rr.Points2D([[current_point['pixel_x'], current_point['pixel_y']]], 
                                          colors=[window_grouped_colors[window_idx][track_id]],
                                          radii=[track_radii]))

        # Lift 2D track points to 3D and accumulate trajectories
        if precomputed_2d_tracks:
            current_pose_cuda = pose.cuda()
            
            for track_id, track_points in precomputed_2d_tracks.items():
                # Find point for current frame
                current_point_3d = None
                current_point_3d_optimized = None
                
                for point_data in track_points:
                    if point_data['frame'] == id_step_time:
                        # Lift this 2D point to 3D using the same logic as lift_image
                        try:
                            pt_3d = lift_2d_point(
                                point_data['norm_x'], 
                                point_data['norm_y'],
                                point_data['depth'],
                                current_pose_cuda,
                                proj
                            )
                            # Skip if depth was filtered out (> 10,000)
                            if pt_3d is None:
                                continue
                                
                            current_point_3d = pt_3d.cpu().numpy()
                            
                            # Compute optimized 3D point if track_optimization_data is available
                            if track_optimization_data is not None:
                                ba_param_sigma_depth = track_optimization_data.get('ba_param_sigma_depth')
                                if ba_param_sigma_depth is not None:
                                    frame_idx = point_data['frame']
                                    
                                    # Get BA optimization delta for this specific track point
                                    # ba_param_sigma_depth has shape (wc, gs, tl) where wc=1, gs=group_size, tl=seq_len
                                    # We need to determine which group this track belongs to and extract the sigma value
                                            
                                    # Get track information from point_data
                                    window_idx = point_data.get('window_idx', 0)
                                    track_local_idx = point_data.get('track_local_idx', 0)
                                            
                                    # For windowed tracks, we need to map to the correct group
                                    # The group_idx corresponds to the window being processed
                                    group_idx = window_idx if window_idx < ba_param_sigma_depth.shape[1] else 0
                                            
                                    # Extract delta value: ba_param_sigma_depth[wc_idx, group_idx, frame_idx]
                                    if (frame_idx < ba_param_sigma_depth.shape[2] and 
                                        group_idx < ba_param_sigma_depth.shape[1]):
                                        delta_val = ba_param_sigma_depth[0, group_idx, frame_idx]
                                        if hasattr(delta_val, 'item'):
                                            delta_val = delta_val.item()
                                            
                                        # Compute original ray direction: from camera to 3D point
                                        camera_position = current_pose_cuda[:3, 3]  # Camera position in world coordinates
                                        original_ray_dir = pt_3d - camera_position  # Ray from camera to point
                                        original_ray_dir = original_ray_dir / (torch.norm(original_ray_dir) + 1e-8)  # Normalize
                                        
                                        # Apply BA optimization: adjusted_points_new = track3d + delta_new * original_ray_dir
                                        pt_3d_optimized = pt_3d + delta_val * original_ray_dir
                                        current_point_3d_optimized = pt_3d_optimized.cpu().numpy()
                            
                            break
                        except Exception as e:
                            print(f"Failed to lift point for track {track_id}: {e}")
                            continue
                
                # Add to unoptimized trajectory if we have a 3D point
                if current_point_3d is not None:
                    if track_id not in track_3d_trajectories:
                        track_3d_trajectories[track_id] = []
                    track_3d_trajectories[track_id].append(current_point_3d.tolist())
                    
                    # Log the unoptimized 3D trajectory so far (to unoptimized scene)
                    if show_3d_tracks:
                        if len(track_3d_trajectories[track_id]) > 1:
                            rr.log(f"world/scene_unoptimized/track_3d_{track_id}", 
                                   rr.LineStrips3D([track_3d_trajectories[track_id]], 
                                                  colors=[track_colors[track_id]]
                                                  ))
                        else:
                            # Single point
                            rr.log(f"world/scene_unoptimized/track_3d_{track_id}", 
                                   rr.Points3D([current_point_3d], 
                                              colors=[track_colors[track_id]],
                                              radii=rr.Radius.ui_points([radii])))
                
                # Add to optimized trajectory if we have an optimized 3D point
                if current_point_3d_optimized is not None:
                    if track_id not in optimized_track_3d_trajectories:
                        optimized_track_3d_trajectories[track_id] = []
                    optimized_track_3d_trajectories[track_id].append(current_point_3d_optimized.tolist())
                    
                    # Use the SAME color for optimized tracks (not brightened)
                    optimized_color = track_colors[track_id]
                    
                    # Log the optimized 3D trajectory so far (to optimized scene)
                    if show_3d_tracks:
                        if len(optimized_track_3d_trajectories[track_id]) > 1:
                            rr.log(f"world/scene_optimized/track_3d_{track_id}", 
                                   rr.LineStrips3D([optimized_track_3d_trajectories[track_id]], 
                                                  colors=[optimized_color]
                                                  ))
                        else:
                            # Single point
                            rr.log(f"world/scene_optimized/track_3d_{track_id}", 
                                   rr.Points3D([current_point_3d_optimized], 
                                              colors=[optimized_color],
                                              radii=rr.Radius.ui_points([radii])))

        # Window-based 3D track lifting and visualization
        if windowed_track_data:
            current_pose_cuda = pose.cuda()
            
            # Find which windows are active for the current frame and process their 3D tracks
            for window_info in windowed_track_data:
                window_id = window_info['window_id']
                frame_range = window_info['frame_range']
                
                # Only process windows that contain the current frame
                if frame_range[0] <= id_step_time < frame_range[1] and window_id in window_2d_tracks:
                    current_window_tracks = window_2d_tracks[window_id]
                    current_window_colors = window_track_colors[window_id]
                    
                    for track_id, track_points in current_window_tracks.items():
                        # Find point for current frame
                        current_point_3d = None
                        current_point_3d_optimized = None
                        
                        for point_data in track_points:
                            if point_data['frame'] == id_step_time:
                                # Lift this 2D point to 3D using the same logic as lift_image
                                try:
                                    pt_3d = lift_2d_point(
                                        point_data['norm_x'], 
                                        point_data['norm_y'],
                                        point_data['depth'],
                                        current_pose_cuda,
                                        proj
                                    )
                                    # Skip if depth was filtered out (> 10,000)
                                    if pt_3d is None:
                                        continue
                                        
                                    current_point_3d = pt_3d.cpu().numpy()
                                    
                                    # Compute optimized 3D point if track_optimization_data is available
                                    if track_optimization_data is not None:
                                        ba_param_sigma_depth = track_optimization_data.get('ba_param_sigma_depth')
                                        if ba_param_sigma_depth is not None:
                                            frame_idx = point_data['frame']
                                            point_idx = point_data['point_idx']
                                            # Get BA optimization delta for this specific track point
                                            # ba_param_sigma_depth has shape (wc, gs, tl) where wc=1, gs=group_size, tl=seq_len
                                            # We need to determine which group this track belongs to and extract the sigma value
                                            
                                            # Get track information from point_data
                                            window_idx = point_data.get('window_idx', 0)
                                            track_local_idx = point_data.get('track_local_idx', 0)
                                            
                                            # For windowed tracks, we need to map to the correct group
                                            # The group_idx corresponds to the window being processed
                                            # group_idx = window_idx if window_idx < ba_param_sigma_depth.shape[1] else 0
                                            group_idx = point_data['frame_range'][0]
                                            # Extract delta value: ba_param_sigma_depth[wc_idx, group_idx, frame_idx]
                                            if (point_idx < ba_param_sigma_depth.shape[2] and 
                                                group_idx < ba_param_sigma_depth.shape[1]):
                                                delta_val = ba_param_sigma_depth[group_idx, track_local_idx, point_idx]
                                                if hasattr(delta_val, 'item'):
                                                    delta_val = delta_val.item()
                                                
                                                # Compute original ray direction: from camera to 3D point
                                                camera_position = current_pose_cuda[:3, 3]  # Camera position in world coordinates
                                                original_ray_dir = pt_3d - camera_position  # Ray from camera to point
                                                original_ray_dir = original_ray_dir / (torch.norm(original_ray_dir) + 1e-8)  # Normalize
                                                
                                                # Apply BA optimization: adjusted_points_new = track3d + delta_new * original_ray_dir
                                                pt_3d_optimized = pt_3d + delta_val * original_ray_dir
                                                current_point_3d_optimized = pt_3d_optimized.cpu().numpy()
                                    
                                    break
                                except Exception as e:
                                    print(f"Failed to lift point for window track {track_id}: {e}")
                                    continue
                        
                        # Use window-specific track ID for 3D trajectories
                        window_track_3d_id = f"w{window_id}_{track_id}"
                        
                        # Add to unoptimized trajectory if we have a 3D point
                        if current_point_3d is not None:
                            if window_track_3d_id not in track_3d_trajectories:
                                track_3d_trajectories[window_track_3d_id] = []
                            track_3d_trajectories[window_track_3d_id].append(current_point_3d.tolist())
                            
                            # Log the unoptimized 3D trajectory so far (to unoptimized scene)
                            if show_3d_tracks:
                                if len(track_3d_trajectories[window_track_3d_id]) > 1:
                                    rr.log(f"world/scene_unoptimized/window_{window_id}/track_3d_{track_id}", 
                                           rr.LineStrips3D([track_3d_trajectories[window_track_3d_id]], 
                                                          colors=[current_window_colors[track_id]]
                                                          ))
                                else:
                                    # Single point
                                    rr.log(f"world/scene_unoptimized/window_{window_id}/track_3d_{track_id}", 
                                           rr.Points3D([current_point_3d], 
                                                      colors=[current_window_colors[track_id]],
                                                      radii=rr.Radius.ui_points([radii])))
                        
                        # Add to optimized trajectory if we have an optimized 3D point
                        if current_point_3d_optimized is not None:
                            window_track_3d_opt_id = f"w{window_id}_{track_id}_opt"
                            if window_track_3d_opt_id not in optimized_track_3d_trajectories:
                                optimized_track_3d_trajectories[window_track_3d_opt_id] = []
                            optimized_track_3d_trajectories[window_track_3d_opt_id].append(current_point_3d_optimized.tolist())
                            
                            # Use the SAME color for optimized tracks (not brightened)
                            optimized_color = current_window_colors[track_id]
                            
                            # Log the optimized 3D trajectory so far (to optimized scene)
                            if show_3d_tracks:
                                if len(optimized_track_3d_trajectories[window_track_3d_opt_id]) > 1:
                                    rr.log(f"world/scene_optimized/window_{window_id}/track_3d_{track_id}", 
                                           rr.LineStrips3D([optimized_track_3d_trajectories[window_track_3d_opt_id]], 
                                                          colors=[optimized_color]
                                                          ))
                                else:
                                    # Single point
                                    rr.log(f"world/scene_optimized/window_{window_id}/track_3d_{track_id}", 
                                           rr.Points3D([current_point_3d_optimized], 
                                                      colors=[optimized_color],
                                                      radii=rr.Radius.ui_points([radii])))

        # Global track 3D lifting and visualization
        if global_track_data is not None and 0 in window_2d_tracks:
            current_pose_cuda = pose.cuda()
            current_window_tracks = window_2d_tracks[0]  # Global tracks use window_id = 0
            current_window_colors = window_track_colors[0]
            
            # Track index counter for accessing tracks_3d_optimized by order
            optimized_track_counter = 0
            # Static BA-aware visualization flags and data
            use_static_ba_applied = bool(global_track_data.get('use_static_ba_applied', False))
            tracks_3d_optimized_global = global_track_data.get('tracks_3d_optimized', None)
            filtered_indices_global = global_track_data.get('filtered_indices', None)
            is_dynamic_mask_global = global_track_data.get('is_dynamic_mask', None)
            
            for track_id, track_points in current_window_tracks.items():
                # Find point for current frame
                current_point_3d = None
                current_point_3d_optimized = None
                current_point_3d_optimized_reference = None
                
                for point_data in track_points:
                    if point_data['frame'] == id_step_time:
                        # Lift this 2D point to 3D using the same logic as lift_image
                        try:
                            pt_3d = lift_2d_point(
                                point_data['norm_x'], 
                                point_data['norm_y'],
                                point_data['depth'],
                                current_pose_cuda,
                                proj
                            )
                            # Skip if depth was filtered out (> 10,000)
                            if pt_3d is None:
                                continue
                                
                            current_point_3d = pt_3d.cpu().numpy()
                            
                            # Compute optimized 3D point
                            # If static BA was applied, prefer global optimized 3D tracks directly
                            frame_idx = point_data['frame']
                            if use_static_ba_applied and tracks_3d_optimized_global is not None:
                                # Determine the row index in tracks_3d_optimized for this track
                                track_local_idx = point_data['track_local_idx']
                                if filtered_indices_global is not None:
                                    try:
                                        # Find index within filtered set
                                        row_candidates = np.where(filtered_indices_global == track_local_idx)[0]
                                        row_idx = int(row_candidates[0]) if len(row_candidates) > 0 else None
                                    except Exception:
                                        row_idx = None
                                else:
                                    row_idx = int(track_local_idx)
                                if (row_idx is not None and
                                    0 <= row_idx < tracks_3d_optimized_global.shape[0] and
                                    0 <= frame_idx < tracks_3d_optimized_global.shape[1]):
                                    current_point_3d_optimized = tracks_3d_optimized_global[row_idx, frame_idx]
                            else:
                                # Fallback to original per-frame depth-delta method
                                if track_optimization_data is not None:
                                    ba_param_sigma_depth = track_optimization_data.get('ba_param_sigma_depth')
                                    if ba_param_sigma_depth is not None:
                                        # For global tracks, use window_idx = 0 and track order index
                                        group_idx = 0  # Global tracks use the first window group
                                        track_local_idx = point_data['track_local_idx']
                                        # Extract delta value: ba_param_sigma_depth[group_idx, track_local_idx, frame_idx]
                                        if (frame_idx < ba_param_sigma_depth.shape[2] and 
                                            track_local_idx < ba_param_sigma_depth.shape[1] and
                                            group_idx < ba_param_sigma_depth.shape[0]):
                                            delta_val = ba_param_sigma_depth[group_idx, track_local_idx, frame_idx]
                                            if hasattr(delta_val, 'item'):
                                                delta_val = delta_val.item()
                                            # Compute original ray direction: from camera to 3D point
                                            camera_position = current_pose_cuda[:3, 3]
                                            original_ray_dir = pt_3d - camera_position
                                            original_ray_dir = original_ray_dir / (torch.norm(original_ray_dir) + 1e-8)
                                            # Apply BA optimization
                                            pt_3d_optimized = pt_3d + delta_val * original_ray_dir
                                            current_point_3d_optimized = pt_3d_optimized.cpu().numpy()
                            
                            # # Also get the reference optimized 3D point from global optimization results for comparison
                            # if track_optimization_data is not None and 'global_track_data' in track_optimization_data:
                            #     global_opt_data = track_optimization_data['global_track_data']
                            #     if global_opt_data.get('has_enough_tracks', False):
                            #         frame_idx = point_data['frame']
                                    
                            #         # Get optimized 3D coordinates directly from global optimization results
                            #         # Use optimized_track_counter since tracks_3d_optimized is already filtered by order
                            #         if 'tracks_3d_optimized' in global_opt_data:
                            #             tracks_3d_optimized = global_opt_data['tracks_3d_optimized']
                            #             if (optimized_track_counter < tracks_3d_optimized.shape[0] and 
                            #                 frame_idx < tracks_3d_optimized.shape[1]):
                            #                 current_point_3d_optimized_reference = tracks_3d_optimized[optimized_track_counter, frame_idx]
                                            
                            #                 # Compare the two methods for verification
                            #                 if current_point_3d_optimized is not None:
                            #                     diff = np.linalg.norm(current_point_3d_optimized - current_point_3d_optimized_reference)
                            #                     if diff > 1e-3:  # Threshold for significant difference
                            #                         print(f"Warning: Global track {track_id} frame {frame_idx}: ba_param_sigma_depth method differs from tracks_3d_optimized by {diff:.6f}")
                            #                     else:
                            #                         print(f"Global track {track_id} frame {frame_idx}: Methods agree (diff: {diff:.6f})")
                            
                            # break
                        except Exception as e:
                            print(f"Failed to lift point for global track {track_id}: {e}")
                            continue
                
                # Use global-specific track ID for 3D trajectories
                global_track_3d_id = f"global_{track_id}"
                
                # Add to unoptimized trajectory if we have a 3D point
                if current_point_3d is not None:
                    if global_track_3d_id not in track_3d_trajectories:
                        track_3d_trajectories[global_track_3d_id] = []
                    track_3d_trajectories[global_track_3d_id].append(current_point_3d.tolist())
                    
                    # Log the unoptimized 3D trajectory so far (to unoptimized scene)
                    if show_3d_tracks:
                        if len(track_3d_trajectories[global_track_3d_id]) > 1:
                            rr.log(f"world/scene_unoptimized/global_tracks/track_3d_{track_id}", 
                                   rr.LineStrips3D([track_3d_trajectories[global_track_3d_id]], 
                                                  colors=[current_window_colors[track_id]]
                                                  ))
                        else:
                            # Single point
                            rr.log(f"world/scene_unoptimized/global_tracks/track_3d_{track_id}", 
                                   rr.Points3D([current_point_3d], 
                                              colors=[current_window_colors[track_id]],
                                              radii=rr.Radius.ui_points([radii])))
                
                # Add to optimized trajectory if we have an optimized 3D point (using ba_param_sigma_depth result)
                if current_point_3d_optimized is not None:
                    global_track_3d_opt_id = f"global_{track_id}_opt"
                    if global_track_3d_opt_id not in optimized_track_3d_trajectories:
                        optimized_track_3d_trajectories[global_track_3d_opt_id] = []
                    optimized_track_3d_trajectories[global_track_3d_opt_id].append(current_point_3d_optimized.tolist())
                    
                    # Use the SAME color for optimized tracks (not brightened)
                    optimized_color = current_window_colors[track_id]
                    
                    # Log the optimized 3D trajectory so far (to optimized scene)
                    if show_3d_tracks:
                        if len(optimized_track_3d_trajectories[global_track_3d_opt_id]) > 1:
                            rr.log(f"world/scene_optimized/global_tracks/track_3d_{track_id}", 
                                   rr.LineStrips3D([optimized_track_3d_trajectories[global_track_3d_opt_id]], 
                                                  colors=[optimized_color]
                                                  ))
                        else:
                            # Single point
                            rr.log(f"world/scene_optimized/global_tracks/track_3d_{track_id}", 
                                   rr.Points3D([current_point_3d_optimized], 
                                              colors=[optimized_color],
                                              radii=rr.Radius.ui_points([radii])))
                
                # Increment the track counter for accessing tracks_3d_optimized by order
                optimized_track_counter += 1

        # Original point cloud visualization for keyframes (log to both scenes)
        if keyframes is None or id_step_time in keyframes:
            kid = keyframes.index(id_step_time) if keyframes is not None else id_step_time
            
            current_img_np = imgs[id_step_time]
            current_depth_map_tensor = depths[kid] if keyframes is not None else depths[id_step_time]
            current_depth_map_cuda = current_depth_map_tensor.cuda()
            current_pose_cuda = trajectory[id_step_time].cuda()
            current_depth_map_cuda = current_depth_map_cuda.unsqueeze(dim=0) if len(current_depth_map_cuda.shape) == 2 else current_depth_map_cuda
            # Lift dense point cloud
            pts, colors = lift_image(torch.tensor(current_img_np).cuda(), current_depth_map_cuda, current_pose_cuda, proj)
            mask = filter_depth(current_depth_map_cuda, threshold=filter_depth_threshold)
            mask = mask.view(-1)

            if max_depth > 0:
                depth_mask = current_depth_map_cuda.view(-1) < max_depth
                mask = mask & depth_mask

            pts = pts[mask, :]
            colors = colors[mask, :]

            # Subsample points for performance
            subsample_pts = 5  # Subsample factor
            pts = pts[subsample_pts//2::subsample_pts]
            colors = colors[subsample_pts//2::subsample_pts]
            colors = (colors * 255).clamp(0, 255).to(torch.uint8)

            # Log to both scenes
            for scene_name in ["scene_unoptimized", "scene_optimized"]:
                rr.log(f"world/{scene_name}/active_points", 
                       rr.Points3D(pts[:, :3].cpu().numpy(), 
                                  colors=colors[:, :3].cpu().numpy(), 
                                  radii=rr.Radius.ui_points([radii])))

        # NEW: 2D track visualization using 3D-to-2D projection (following reference implementation)
        # This replaces the original approach and ensures perfect trail connectivity
        current_pose_cuda = pose.cuda()
        
        # Visualize unoptimized tracks by projecting 3D trajectories to 2D
        if show_2d_tracks:
            for track_id, trajectory_3d in track_3d_trajectories.items():
                if len(trajectory_3d) == 0:
                    continue
                
                # Project all 3D points in the trajectory to the current camera view
                projected_trail_points = []
                current_projected_point = None
                
                for i, pt_3d_list in enumerate(trajectory_3d):
                    pt_3d_tensor = torch.tensor(pt_3d_list, device=current_pose_cuda.device, dtype=torch.float32)
                    projected_2d, is_valid = project_3d_to_2d(pt_3d_tensor, current_pose_cuda, proj)
                    
                    if is_valid and projected_2d is not None:
                        projected_trail_points.append(projected_2d)
                        # The last point in the trajectory corresponds to the current frame
                        if i == len(trajectory_3d) - 1:
                            current_projected_point = projected_2d
                
                # Get track color
                if track_id in track_colors:
                    color = track_colors[track_id]
                else:
                    # Generate a color for window-based tracks
                    if track_id.startswith('w'):
                        parts = track_id.split('_')
                        window_id = int(parts[0][1:])  # Extract window ID
                        track_idx = int(parts[2][1:])  # Extract track index
                        color = [
                            (window_id * 60 + track_idx * 40) % 255,
                            (180 + track_idx * 25) % 255,
                            255
                        ]
                    else:
                        color = [255, 255, 255]  # Default white
                
                # Extract window info for logging path
                if track_id.startswith('w'):
                    parts = track_id.split('_')
                    window_id = int(parts[0][1:])
                    log_path_prefix = f"world/scene_unoptimized/active_cam/input/window_{window_id}"
                else:
                    window_id = 0
                    log_path_prefix = f"world/scene_unoptimized/active_cam/input/window_{window_id}"
                
                # Log projected trail if we have multiple points
                if len(projected_trail_points) > 1:
                    # Debug: Check trail connectivity
                    if id_step_time % 10 == 0 and current_projected_point is not None:  # Print every 10 frames
                        last_trail_point = projected_trail_points[-1]
                        distance = ((last_trail_point[0] - current_projected_point[0])**2 + 
                                   (last_trail_point[1] - current_projected_point[1])**2)**0.5
                        print(f"Frame {id_step_time}, Track {track_id} (3D-proj): Trail-to-current distance = {distance:.2f} pixels")
                        if distance > 1.0:  # Should be 0 for perfect connectivity
                            print(f"  WARNING: Unexpected disconnect in 3D-projected track {track_id}!")
                    
                    rr.log(f"{log_path_prefix}/track_trail_{track_id}", 
                           rr.LineStrips2D([projected_trail_points], 
                                          colors=[color],
                                          radii=[1.0]))
                
                # Log current projected point if it exists
                if current_projected_point is not None:
                    rr.log(f"{log_path_prefix}/track_point_{track_id}", 
                           rr.Points2D([current_projected_point], 
                                      colors=[color],
                                      radii=[track_radii]))
        
            # Visualize optimized tracks by projecting 3D trajectories to 2D
            for track_id, trajectory_3d in optimized_track_3d_trajectories.items():
                if len(trajectory_3d) == 0:
                    continue
                    
                # Project all 3D points in the trajectory to the current camera view
                projected_trail_points = []
                current_projected_point = None
                
                for i, pt_3d_list in enumerate(trajectory_3d):
                    pt_3d_tensor = torch.tensor(pt_3d_list, device=current_pose_cuda.device, dtype=torch.float32)
                    projected_2d, is_valid = project_3d_to_2d(pt_3d_tensor, current_pose_cuda, proj)
                    
                    if is_valid and projected_2d is not None:
                        projected_trail_points.append(projected_2d)
                        # The last point in the trajectory corresponds to the current frame
                        if i == len(trajectory_3d) - 1:
                            current_projected_point = projected_2d
                
                # Get track color (use same color as unoptimized version)
                base_track_id = track_id.replace('_opt', '')  # Remove _opt suffix
                if base_track_id in track_colors:
                    color = track_colors[base_track_id]
                else:
                    # Generate a color for window-based tracks
                    if base_track_id.startswith('w'):
                        parts = base_track_id.split('_')
                        window_id = int(parts[0][1:])  # Extract window ID
                        track_idx = int(parts[2][1:])  # Extract track index
                        color = [
                            (window_id * 60 + track_idx * 40) % 255,
                            (180 + track_idx * 25) % 255,
                            255
                        ]
                    else:
                        color = [255, 255, 255]  # Default white
                
                # Extract window info for logging path
                if track_id.startswith('w'):
                    parts = track_id.split('_')
                    window_id = int(parts[0][1:])
                    log_path_prefix = f"world/scene_optimized/active_cam/input/window_{window_id}"
                else:
                    window_id = 0
                    log_path_prefix = f"world/scene_optimized/active_cam/input/window_{window_id}"
                
                # Log projected trail if we have multiple points
                if len(projected_trail_points) > 1:
                    # Debug: Check trail connectivity
                    if id_step_time % 10 == 0 and current_projected_point is not None:  # Print every 10 frames
                        last_trail_point = projected_trail_points[-1]
                        distance = ((last_trail_point[0] - current_projected_point[0])**2 + 
                                   (last_trail_point[1] - current_projected_point[1])**2)**0.5
                        print(f"Frame {id_step_time}, Track {track_id} (3D-proj): Trail-to-current distance = {distance:.2f} pixels")
                        if distance > 1.0:  # Should be 0 for perfect connectivity
                            print(f"  WARNING: Unexpected disconnect in 3D-projected track {track_id}!")
                    
                    rr.log(f"{log_path_prefix}/track_trail_{track_id}", 
                           rr.LineStrips2D([projected_trail_points], 
                                          colors=[color],
                                          radii=[1.0]))
                
                # Log current projected point if it exists
                if current_projected_point is not None:
                    rr.log(f"{log_path_prefix}/track_point_{track_id}", 
                           rr.Points2D([current_projected_point], 
                                      colors=[color],
                                      radii=[track_radii]))

        # OLD 2D track visualization (DISABLED - replaced by 3D-to-2D projection above)
        # Window-based track visualization 
        if False and windowed_track_data:  # Disabled
            pass  # Placeholder for disabled code

    # Log depth filtering statistics
    if total_lift_attempts > 0:
        filter_percentage = (depth_filtered_count / total_lift_attempts) * 100
        print(f"Depth filtering statistics: {depth_filtered_count}/{total_lift_attempts} points filtered ({filter_percentage:.1f}%) for depth > {max_depth_filter}")
    else:
        print("No 3D lifting attempts made during visualization")


def save_uncertainty_video(uncertainties, video_path, fps=24, show_colorbar=False):
    """
    Save the per-frame uncertainty visualization to a video using the same
    colorization logic as plot_to_rerun, and optionally append a color bar.

    Args:
        uncertainties: Tensor or list with time as the first dimension.
        video_path: Output mp4 path.
        fps: Frames per second for the output video.
        show_colorbar: If True, appends a color bar to the right of each frame.
    """
    # Normalize shape to (T, 1, H, W) or (T, H, W)
    if isinstance(uncertainties, list):
        uncertainties = torch.stack(uncertainties)

    unc = uncertainties if isinstance(uncertainties, torch.Tensor) else torch.tensor(uncertainties)

    seq_len = unc.shape[0]

    max_val = unc.max()

    # Write RGB frames using imageio (ffmpeg backend)
    with imageio.get_writer(str(video_path), fps=float(fps), codec="libx264") as writer:
        for t in range(seq_len):
            # Colorize frame
            frame_unc = unc[t]
            if isinstance(frame_unc, torch.Tensor) and frame_unc.dim() == 3 and frame_unc.shape[0] == 1:
                frame_unc = frame_unc[0]
            img = color_tensor((frame_unc / (max_val + 1e-8)).clamp(0, 1), cmap="custom_uncertainty", norm=False)
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)  # H, W, 3

            frame_to_write = img_np
            if show_colorbar:
                H, W, _ = img_np.shape

                # Create vertical color bar (top = 1, bottom = 0)
                bar_width = 16
                pad_width = 8
                label_width = 56

                grad = torch.linspace(1.0, 0.0, steps=H, device=unc.device if isinstance(unc, torch.Tensor) else None)
                grad = grad.unsqueeze(1).repeat(1, bar_width)  # (H, bar_width)
                bar_rgb = color_tensor(grad, cmap="custom_uncertainty", norm=False)  # (H, bar_width, 3)
                bar_np = (bar_rgb.cpu().numpy() * 255).astype(np.uint8)  # (H, bar_width, 3)

                # Padding between image and bar
                pad_np = np.full((H, pad_width, 3), 255, dtype=np.uint8)

                # Label area to the right of the bar
                label_np = np.full((H, label_width, 3), 255, dtype=np.uint8)
                max_text = f"{float(max_val):.3g}"
                min_text = "0"

                # Draw labels (top for max, bottom for min)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                text_color = (0, 0, 0)  # black

                # Compute text sizes
                (max_w, max_h), _ = cv2.getTextSize(max_text, font, font_scale, thickness)
                (min_w, min_h), _ = cv2.getTextSize(min_text, font, font_scale, thickness)

                # Put max near top
                cv2.putText(label_np, max_text, (4, max_h + 2), font, font_scale, text_color, thickness, cv2.LINE_AA)
                # Put min near bottom
                cv2.putText(label_np, min_text, (4, H - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)

                # Concatenate: image | pad | bar | label
                frame_to_write = np.concatenate([img_np, pad_np, bar_np, label_np], axis=1)

            writer.append_data(frame_to_write)


def extract_tracks_from_window(
    flow_occs_fwd, 
    uncertainties, 
    depths, 
    proj, 
    trajectory, 
    grid_size=16, 
    track_len=8, 
    stride=1, 
    window_start=0, 
    window_end=None,
    imgs=None, 
    device="cuda",
    rotation_representation="quaternion",
    pixel_tracks_full=None,  # Pre-computed full sequence tracks
    indices_full=None,       # Pre-computed full sequence indices
    depths_full=None,        # Pre-computed full sequence depths
    initial_depths_full=None # Pre-computed initial depths
):
    """
    Extracts 2D and 3D tracks from a single window of data.
    Now uses pre-computed full-sequence tracks and extracts the relevant window.
    Returns: initial_3d_tracks, pixel_coords_2d, depths_at_2d, frame_indices_for_2d
    for this specific window.
    """
    current_device = torch.device(device)
    window_range = range(window_start, window_end)
    actual_window_len = len(window_range)

    if actual_window_len == 0:
        return None, None, None, None

    if pixel_tracks_full is None or indices_full is None or depths_full is None:
        print("Warning: Pre-computed tracks not provided. Cannot extract windowed tracks.")
        return None, None, None, None

    # Extract tracks that start within this window
    n, wc, gs, tl, c = pixel_tracks_full.shape
    
    # Find tracks that start in this window
    window_mask = (indices_full[0, :, :, 0, 0] >= window_start) & (indices_full[0, :, :, 0, 0] < window_end)
    
    if not window_mask.any():
        return None, None, None, None
    
    # Get the window indices
    window_group_indices, window_point_indices = torch.where(window_mask)
    
    if len(window_group_indices) == 0:
        return None, None, None, None
    
    # Select one group from this window (e.g., the first group that has tracks in this window)
    selected_group = window_group_indices[0].item()
    
    # Extract tracks from the selected group
    pixel_tracks_window = pixel_tracks_full[0, selected_group].cpu().numpy()  # (gs, tl, 2)
    depths_window = depths_full[0, selected_group, :, :, 0].cpu().numpy()      # (gs, tl)
    indices_window = indices_full[0, selected_group, :, :, 0].cpu().numpy()    # (gs, tl)
    
    # Filter tracks that actually start in this window
    track_starts_in_window = (indices_window[:, 0] >= window_start) & (indices_window[:, 0] < window_end)
    
    if not track_starts_in_window.any():
        return None, None, None, None
    
    # Keep only tracks that start in this window
    pixel_tracks_window = pixel_tracks_window[track_starts_in_window]
    depths_window = depths_window[track_starts_in_window] 
    indices_window = indices_window[track_starts_in_window]

    num_tracks_in_window = pixel_tracks_window.shape[0]
    track_length = pixel_tracks_window.shape[1]
    
    # Get image dimensions
    h, w = flow_occs_fwd.shape[-2:]
    
    # Initialize 3D tracks
    lifted_3d_tracks_for_window = np.zeros((num_tracks_in_window, track_length, 3), dtype=np.float32)
    validity_mask_3d = np.zeros((num_tracks_in_window, track_length), dtype=bool)

    if proj is not None and trajectory is not None and num_tracks_in_window > 0:
        proj_tensor = proj if isinstance(proj, torch.Tensor) else torch.tensor(proj, dtype=torch.float32)
        proj_tensor = proj_tensor.to(current_device)

        ba_proj_norm = proj_tensor.clone()
        ba_proj_norm[0, 0] = (ba_proj_norm[0, 0] / w) * 2
        ba_proj_norm[1, 1] = (ba_proj_norm[1, 1] / h) * 2
        ba_proj_norm[0, 2] = (ba_proj_norm[0, 2] / w) * 2 - 1
        ba_proj_norm[1, 2] = (ba_proj_norm[1, 2] / h) * 2 - 1
        
        trajectory_tensors = [(p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)).to(current_device) for p in trajectory]
        all_poses_c2w_stacked = torch.stack(trajectory_tensors)
        all_ba_param_rot, all_ba_param_t = pose_to_param(all_poses_c2w_stacked.unsqueeze(0), rotation_representation)

        # for track_idx in range(num_tracks_in_window):
        for point_in_track_idx in range(track_length):
            xy_norm_tensor = torch.tensor(pixel_tracks_window[:, point_in_track_idx, :], device=current_device, dtype=torch.float32)
            depth_val = depths_window[:, point_in_track_idx]
            global_frame_idx = indices_window[:, point_in_track_idx]

            if not (0 <= global_frame_idx < len(trajectory) and depth_val > 1e-5):
                continue
            inv_depth_val = 1.0 / depth_val

            current_pose_rot = all_ba_param_rot[0, global_frame_idx]
            current_pose_t = all_ba_param_t[0, global_frame_idx]

            xyz_world, valid_mask_pix = pix_2_world_np(
                xy=xy_norm_tensor.unsqueeze(0),
                depth=torch.tensor([inv_depth_val], device=current_device, dtype=torch.float32),
                valid_depth_min=0.0, valid_depth_max=1000.0,
                ba_proj=ba_proj_norm,
                ba_param_rot=current_pose_rot,
                ba_param_t=current_pose_t,
                normalized=True, imh=h, imw=w
            )
            if valid_mask_pix[0]:
                lifted_3d_tracks_for_window[:, point_in_track_idx, :] = xyz_world[0].cpu().numpy()
                validity_mask_3d[:, point_in_track_idx] = True
        
        flat_3d_points_this_window = lifted_3d_tracks_for_window[validity_mask_3d]
        if flat_3d_points_this_window.ndim == 1 and flat_3d_points_this_window.size == 0:
             flat_3d_points_this_window = np.array([])
        elif flat_3d_points_this_window.ndim != 2 or flat_3d_points_this_window.shape[1] != 3:
             flat_3d_points_this_window = np.array([])

        return flat_3d_points_this_window, pixel_tracks_window, depths_window, indices_window

    return None, None, None, None


def extract_tracks_from_windows(
    seq_flow_occs_fwd, 
    uncertainties, 
    seq_depths, 
    proj, 
    trajectory, 
    ba_window=8, 
    overlap=6, 
    grid_size=16, 
    track_len=8, 
    stride_cpt=1, 
    imgs=None,
    device="cuda"
):
    """
    Extract 3D tracks from multiple sliding windows.
    Now follows ba_refinement_opt_tracks_multistep approach:
    1. Call compute_pixel_tracks once on full sequence
    2. Extract relevant windows from the results
    """
    seq_len_total = seq_flow_occs_fwd.shape[0]
    
    all_initial_3d_tracks_lifted_flat = [] 
    all_pixel_coords_2d = []               
    all_depths_at_2d_coords = []           
    all_frame_indices_for_2d_coords = []   
    
    current_device = torch.device(device)
    # Ensure inputs are tensors on the correct device
    _seq_flow = (seq_flow_occs_fwd if isinstance(seq_flow_occs_fwd, torch.Tensor) 
                 else torch.tensor(seq_flow_occs_fwd, dtype=torch.float32)).to(current_device)
    _uncertainties = (uncertainties if isinstance(uncertainties, torch.Tensor) 
                      else torch.tensor(uncertainties, dtype=torch.float32)).to(current_device)
    _seq_depths = (seq_depths if isinstance(seq_depths, torch.Tensor) 
                   else torch.tensor(seq_depths, dtype=torch.float32)).to(current_device)
    _imgs = None
    if imgs is not None:
        _imgs = (imgs if isinstance(imgs, torch.Tensor) 
                 else torch.tensor(imgs, dtype=torch.float32)).to(current_device)

    if not HAS_COMPUTE_PIXEL_TRACKS:
        print("Warning: compute_pixel_tracks not available. Cannot extract windowed tracks.")
        return all_initial_3d_tracks_lifted_flat, all_pixel_coords_2d, all_depths_at_2d_coords, all_frame_indices_for_2d_coords

    # Step 1: Call compute_pixel_tracks on the FULL sequence, just like ba_refinement_opt_tracks_multistep
    print("Computing pixel tracks on full sequence...")
    initial_depths_fwd, pixel_tracks_fwd, uncerts_fwd, indices_fwd, depths_fwd, rgbs_fwd, visibles_fwd = compute_pixel_tracks(
        _seq_flow.cuda(), 
        _uncertainties, 
        _seq_depths.cuda(), 
        track_len=track_len, 
        stride=stride_cpt, 
        grid_size=grid_size, 
        imgs=_imgs.cuda() if _imgs is not None else None, 
        long_tracks=False
    )
    
    if pixel_tracks_fwd.shape[1] == 0:
        print("Warning: No tracks generated by compute_pixel_tracks")
        return all_initial_3d_tracks_lifted_flat, all_pixel_coords_2d, all_depths_at_2d_coords, all_frame_indices_for_2d_coords

    # Step 2: Extract windows from the full sequence tracks
    optimized_until = stride_cpt  # Start from stride_cpt instead of 1
    
    while optimized_until < seq_len_total:
        ba_window_start = max(optimized_until - overlap, 0)
        ba_window_end = min(ba_window_start + ba_window, seq_len_total)
        
        if ba_window_start >= ba_window_end:
            break

        print(f"Extracting windowed tracks from frame {ba_window_start} to {ba_window_end}")
        
        # Use the new approach with pre-computed tracks
        initial_3d_w, pixel_coords_2d_w, depths_2d_w, frame_indices_2d_w = extract_tracks_from_window(
            _seq_flow, 
            _uncertainties, 
            _seq_depths,    
            proj,           
            trajectory,     
            grid_size=grid_size,
            track_len=track_len, 
            stride=stride_cpt, 
            window_start=ba_window_start, 
            window_end=ba_window_end,
            imgs=_imgs, 
            device=device,
            # Pass the pre-computed full sequence tracks
            pixel_tracks_full=pixel_tracks_fwd,
            indices_full=indices_fwd,
            depths_full=depths_fwd,
            initial_depths_full=initial_depths_fwd
        )
        
        if initial_3d_w is not None and initial_3d_w.size > 0:
            all_initial_3d_tracks_lifted_flat.append(initial_3d_w)
        if pixel_coords_2d_w is not None and pixel_coords_2d_w.size > 0:
            all_pixel_coords_2d.append(pixel_coords_2d_w)
        if depths_2d_w is not None and depths_2d_w.size > 0:
            all_depths_at_2d_coords.append(depths_2d_w)
        if frame_indices_2d_w is not None and frame_indices_2d_w.size > 0:
            all_frame_indices_for_2d_coords.append(frame_indices_2d_w)
        
        optimized_until = ba_window_end
        if optimized_until <= ba_window_start: 
            break
            
    return all_initial_3d_tracks_lifted_flat, all_pixel_coords_2d, all_depths_at_2d_coords, all_frame_indices_for_2d_coords


def extract_full_length_tracks_data(
    seq_flow_occs_fwd,  # (SeqLen, C, H, W)
    uncertainties_tensor,      # (SeqLen, 1, H, W)
    seq_depths_tensor,  # (SeqLen, 1, H, W) - per-frame depth maps
    proj_matrix,        # (4, 4) or (3, 4) numpy or tensor
    full_trajectory_list,    # List of (4,4) pose tensors/arrays, length SeqLen
    grid_size,
    imgs_tensor_seq=None,   # (SeqLen, C, H, W)
    device_str="cuda",
    rotation_representation="quaternion",
    apply_semantic_filtering=False,  # New parameter for semantic filtering
    semantic_segmentor=None,  # Optional pre-initialized semantic segmentor
    drift_threshold=5,  # Threshold for detecting drifting points
    min_tracks_threshold=5  # Minimum tracks needed after filtering
):
    """
    Extracts one set of full-length 2D tracks and lifts them to 3D.
    Calls compute_pixel_tracks once to get tracks starting from a grid at frame 0
    and traced for the entire sequence length.
    Returns data wrapped in lists (to mimic a single "window") for plot_to_rerun.
    
    Args:
        apply_semantic_filtering: Whether to apply semantic filtering to remove sky, boundary, and drifting points
        semantic_segmentor: Pre-initialized semantic segmentor (optional)
        drift_threshold: Threshold for detecting drifting static points
        min_tracks_threshold: Minimum number of tracks needed after filtering
    """
    current_device = torch.device(device_str)
    # Ensure inputs are tensors on the correct device and have batch dimension for CPT
    if not isinstance(seq_flow_occs_fwd, torch.Tensor):
        seq_flow_occs_fwd = torch.tensor(seq_flow_occs_fwd, dtype=torch.float32)
    if not isinstance(uncertainties_tensor, torch.Tensor):
        uncertainties_tensor = torch.tensor(uncertainties_tensor, dtype=torch.float32)
    if not isinstance(seq_depths_tensor, torch.Tensor):
        seq_depths_tensor = torch.tensor(seq_depths_tensor, dtype=torch.float32)
    
    _seq_flow_occs_fwd = seq_flow_occs_fwd.to(current_device) # Add batch dim N=1
    _uncertainties = uncertainties_tensor.to(current_device)
    _seq_depths_frames = seq_depths_tensor.to(current_device)
    
    if imgs_tensor_seq is not None:
        if not isinstance(imgs_tensor_seq, torch.Tensor):
            imgs_tensor_seq = torch.tensor(imgs_tensor_seq, dtype=torch.float32)
        _imgs_tensor_batched = imgs_tensor_seq.to(current_device)
    else:
        _imgs_tensor_batched = None

    seq_len, _, h, w = seq_flow_occs_fwd.shape # Get shape from original non-batched tensor

    (initial_depths_at_start_raw,
     pixel_tracks_2d_raw,
     uncerts_raw, frame_indices_raw,
     depths_at_2d_points_raw,
     rgbs_raw, visibles_raw) = compute_pixel_tracks(
        _seq_flow_occs_fwd,
        _uncertainties,
        _seq_depths_frames,
        track_len=seq_len,
        stride=seq_len, 
        grid_size=grid_size,
        imgs=_imgs_tensor_batched,
        long_tracks=True 
    )
    
    if pixel_tracks_2d_raw.shape[1] == 0:
        print("Warning: compute_pixel_tracks returned no track sets for full_length_tracks.")
        empty_np_list = [np.array([])]
        return empty_np_list, empty_np_list, empty_np_list, empty_np_list, empty_np_list

    _pixel_tracks_2d_np = pixel_tracks_2d_raw[0, 0].cpu().numpy() 
    _depths_at_2d_points_np = depths_at_2d_points_raw[0, 0, :, :, 0].cpu().numpy()
    _frame_indices_for_tracks_np = frame_indices_raw[0, 0, :, :, 0].cpu().numpy()
    _visibles_raw_np = visibles_raw[0, 0, :, :, 0].cpu().numpy()  # Extract visibility data
    
    # Apply semantic filtering if requested
    filtered_indices = None
    if apply_semantic_filtering and _imgs_tensor_batched is not None:
        # Initialize semantic segmentor if not provided
        if semantic_segmentor is None and SemanticSegmentor is not None:
            print("Initializing semantic segmentation model...")
            semantic_segmentor = SemanticSegmentor()
        
        if semantic_segmentor is not None:
            print("Applying semantic segmentation filtering to full-length tracks...")
            
            # Get images for segmentation
            imgs_for_seg = _imgs_tensor_batched.cpu().numpy()
            
            # Transpose images for segmentation model (from NCHW to NHWC)
            transposed_imgs = np.transpose(imgs_for_seg, (0, 2, 3, 1))
            
            # Run semantic segmentation
            seg_maps = []
            for fid in tqdm(range(len(transposed_imgs))):
                original_im = Image.fromarray((transposed_imgs[fid] * 255).astype(np.uint8))
                seg_map = semantic_segmentor.model.run(original_im)
                seg_maps.append(seg_map)
            seg_maps = np.stack(seg_maps, axis=0)
            
            # Create track2d dict directly from compute_pixel_tracks outputs
            # Extract 2D tracks and visibility information
            track2d = {
                'tracks': _pixel_tracks_2d_np,  # Shape: (num_tracks, seq_len, 2)
                'visibles': _visibles_raw_np    # Shape: (num_tracks, seq_len)
            }
            
            # Get sky masks
            sky_masks = semantic_segmentor.get_sky_track_mask(
                seg_maps, track2d['tracks']
            )
            sky_masks = (sky_masks & (track2d['visibles'] > 0)).any(axis=1)
            print(f"Percentage of sky points: {sky_masks.mean()}")
            
            # Get static masks
            static_masks = semantic_segmentor.get_static_track_mask(
                seg_maps, track2d['tracks']
            )
            print(f"Percentage of static points: {static_masks.mean()}")
            
            # Detect boundary points
            mean_visible = (static_masks * track2d['visibles']).sum(axis=1) / (track2d['visibles'].sum(axis=1) + 1e-8)
            boundary_masks = (mean_visible > 0) & (mean_visible < 1)
            print(f"Percentage of boundary points: {boundary_masks.mean()}")
            
            # Detect static points that drift
            displacement = np.zeros(track2d['tracks'].shape[0], dtype=bool)
            
            # Detect drift using 2D displacements
            for i in range(1, track2d['tracks'].shape[1]):
                disp = np.linalg.norm(track2d['tracks'][:, i] - track2d['tracks'][:, i-1], axis=1)
                valid = static_masks[:, i] & static_masks[:, i-1] & (track2d['visibles'][:, i] > 0) & (track2d['visibles'][:, i-1] > 0)
                displacement = displacement | ((disp > drift_threshold) & valid)
            
            drift_masks = displacement
            print(f"Percentage of drifting points: {drift_masks.mean()}")
            
            # Create filtered_remaining_mask
            filtered_remaining_mask = ~(sky_masks | boundary_masks | drift_masks)
            print(f"Remaining tracks percentage: {filtered_remaining_mask.mean()}")
            print(f"Remaining tracks count: {filtered_remaining_mask.sum()} of {len(filtered_remaining_mask)}")
            
            # Apply filter if we have enough tracks
            if filtered_remaining_mask.sum() >= min_tracks_threshold:
                filtered_indices = np.where(filtered_remaining_mask)[0]
                _pixel_tracks_2d_np = _pixel_tracks_2d_np[filtered_indices]
                _depths_at_2d_points_np = _depths_at_2d_points_np[filtered_indices]
                _frame_indices_for_tracks_np = _frame_indices_for_tracks_np[filtered_indices]
                print(f"Applied filtering: kept {len(filtered_indices)} of {len(filtered_remaining_mask)} tracks")
            else:
                print(f"Warning: Only {filtered_remaining_mask.sum()} tracks remained after filtering, which is below the minimum threshold of {min_tracks_threshold}.")
                print("Proceeding without filtering.")
        else:
            print("SemanticSegmentor not available. Proceeding without semantic filtering.")
    
    num_total_tracks = _pixel_tracks_2d_np.shape[0]
    lifted_3d_tracks_all = np.zeros((num_total_tracks, seq_len, 3)) # Pre-allocate
    validity_mask_for_3d_tracks = np.zeros((num_total_tracks, seq_len), dtype=bool)

    if proj_matrix is not None and full_trajectory_list is not None and num_total_tracks > 0:
        proj_tensor = proj_matrix if isinstance(proj_matrix, torch.Tensor) else torch.tensor(proj_matrix, dtype=torch.float32)
        proj_tensor = proj_tensor.to(current_device)
        
        ba_proj_unnorm = proj_tensor.clone()
        ba_proj_norm = proj_tensor.clone()
        ba_proj_norm[0, 0] = (ba_proj_norm[0, 0] / w) * 2
        ba_proj_norm[1, 1] = (ba_proj_norm[1, 1] / h) * 2
        ba_proj_norm[0, 2] = (ba_proj_norm[0, 2] / w) * 2 - 1
        ba_proj_norm[1, 2] = (ba_proj_norm[1, 2] / h) * 2 - 1
        
        trajectory_tensors = [(p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)).to(current_device) for p in full_trajectory_list]
        ba_poses_c2w_stacked = torch.stack(trajectory_tensors)
        ba_param_rot, ba_param_t = pose_to_param(ba_poses_c2w_stacked.unsqueeze(0), rotation_representation)

        # Process points by frame in batches rather than one-by-one
        for frame_idx in range(seq_len):
            # Find all points for this frame
            track_indices, point_indices = np.where(_frame_indices_for_tracks_np == frame_idx)
            
            if len(track_indices) == 0:
                continue
            
            # Get depths and filter invalid ones
            depths = _depths_at_2d_points_np[track_indices, point_indices]
            valid_mask = depths > 1e-5
            
            if not np.any(valid_mask):
                continue
            
            # Filter to valid points
            track_indices = track_indices[valid_mask]
            point_indices = point_indices[valid_mask]
            depths = depths[valid_mask]
            
            # Get xy coordinates and prepare tensors
            xy_coords = _pixel_tracks_2d_np[track_indices, point_indices]
            xy_tensor = torch.tensor(xy_coords, device=current_device, dtype=torch.float32)
            inv_depth_tensor = torch.tensor(1.0 / depths, device=current_device, dtype=torch.float32)
            
            # Get pose for this frame
            current_pose_rot = ba_param_rot[0, frame_idx]
            current_pose_t = ba_param_t[0, frame_idx]
            
            # Process all points for this frame in a single batch
            xyz_world, valid_points = pix_2_world_np(
                xy=xy_tensor,
                depth=inv_depth_tensor,
                valid_depth_min=0.0, valid_depth_max=20.0,
                ba_proj=ba_proj_norm, 
                ba_param_rot=current_pose_rot,
                ba_param_t=current_pose_t,
                normalized=True, imh=h, imw=w
            )
            
            # Store valid results
            valid_indices = torch.where(valid_points)[0].cpu()
            for i in valid_indices:
                track_idx = track_indices[i]
                point_idx = point_indices[i]
                lifted_3d_tracks_all[track_idx, point_idx, :] = xyz_world[i].cpu().numpy()
                validity_mask_for_3d_tracks[track_idx, point_idx] = True
    
    # For the `initial_tracks` (Points3D fallback) parameter of plot_to_rerun,
    # we provide a flat list of all valid 3D points from all tracks.
    flat_3d_points_for_pointcloud = lifted_3d_tracks_all[validity_mask_for_3d_tracks]
    if flat_3d_points_for_pointcloud.ndim == 1 and flat_3d_points_for_pointcloud.size == 0:
        flat_3d_points_for_pointcloud = np.array([]) # Ensure it is empty Nx3 if no points
    elif flat_3d_points_for_pointcloud.ndim == 1 and flat_3d_points_for_pointcloud.size > 0:
         # This case should ideally not happen if reshaping is correct or no points are found.
         # If it's a flat array of coordinates, try to reshape. Otherwise, empty it.
        if flat_3d_points_for_pointcloud.size % 3 == 0:
            flat_3d_points_for_pointcloud = flat_3d_points_for_pointcloud.reshape(-1,3)
        else:
            flat_3d_points_for_pointcloud = np.array([])
    elif flat_3d_points_for_pointcloud.ndim == 2 and flat_3d_points_for_pointcloud.shape[1] != 3:
        flat_3d_points_for_pointcloud = np.array([]) # Ensure Nx3

    # Return the structured 3D tracks along with other data
    # The 2D data is returned as is, for plot_to_rerun to lift per track for LineStrips3D.
    if _pixel_tracks_2d_np.size > 0:
        return ([flat_3d_points_for_pointcloud], 
                [_pixel_tracks_2d_np], 
                [_depths_at_2d_points_np],
                [_frame_indices_for_tracks_np],
                [lifted_3d_tracks_all])  # Add structured 3D tracks
    else:
        empty_np_list = [np.array([])]
        return empty_np_list, empty_np_list, empty_np_list, empty_np_list, empty_np_list


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


@hydra.main(version_base=None, config_name=None)
def main(cfg: DictConfig):
    """
    AnyCam demo script for processing videos and extracting 3D information.
    
    Config parameters:
    - input_path: Path to video or directory of images
    - output_path: Path to save outputs
    - model_path: Path to model (optional)
    - model_name: Model to use - 'anycam', 'megasam', or 'vggt' (default: 'anycam')
    - checkpoint: Specific checkpoint to use (optional)
    - visualize: Whether to visualize results with rerun (boolean)
    - rerun_mode: Mode to use for rerun visualization ('spawn' or 'connect', default: 'spawn')
    - rerun_address: Address to connect to when using rerun_mode=connect (default: localhost:8787)
    - clear_rerun: Whether to clear previous rerun visualizations (default: False)
    - export_colmap: Whether to export to COLMAP format (boolean)
    - image_size: Target image size for processing (default: 336)
    - ba_refinement: Whether to perform bundle adjustment refinement (default: True)
    - fps: Target frames per second (default: 0, use all frames)
    - load_gt: Whether to load ground-truth trajectory (default: False)
    - gt_data_path: Path to Sintel dataset for ground-truth (required if load_gt is True)
    - align_trajectories: Whether to align predicted trajectory to ground-truth (default: False)
    - visualize_tracks: Whether to visualize 3D tracks (default: False)
    - track_grid_size: Grid size for track sampling (default: 16)
    - track_length: Length of tracks to extract (default: 8)
    - track_stride: Stride for track extraction (default: 1)
    - track_ba_window: Window size for track extraction, should match BA window (default: 8)
    - track_overlap: Overlap between windows for track extraction (default: 6)
    - apply_semantic_filtering: Whether to apply semantic filtering to tracks (default: True)
    - drift_threshold: Threshold for detecting drifting points (default: 5)
    - min_tracks_threshold: Minimum tracks needed after filtering (default: 5)
    
    Processing config parameters (can be overridden via Hydra config):
    - seq_name: Sequence name for processing (default: 'temple_3')
    - model_seq_len: Model sequence length (default: 100)
    - shift: Processing shift parameter (default: 99)
    - square_crop: Whether to use square cropping (default: False)
    - return_all_uncerts: Whether to return all uncertainties (default: False)
    - use_precomputed_depths: Whether to use precomputed depths (default: True)
    - mono_depth_path: Path to mono depth files (default: preset path)
    - metric_depth_path: Path to metric depth files (default: preset path)
    - ba_with_rerun: Whether to use rerun during BA (default: False)
    - max_uncert: Maximum uncertainty threshold for BA (default: 0.05)
    - lambda_smoothness: Smoothness regularization parameter (default: 0.1)
    - long_tracks: Whether to use long tracks in BA (default: True)
    - n_steps_last_global: Number of steps for last global optimization (default: 5000)
    - global_every_n: Frequency of global optimization (default: 100)
    - collect_optimized_tracks: Whether to collect optimized tracks (default: True)
    - ba_refinement_level: Bundle adjustment refinement level (default: 2)
    
    - vis: Visualization parameters subconfig with the following options:
        - subsample_pts: Point sampling rate (default: 1)
        - radii: Point radius for visualization (default: 1.5)
        - uncertainty_thresh: Threshold for uncertainty visualization (default: 0.05)
        - max_depth: Maximum depth value to consider (default: -1, no limit)
        - filter_depth_threshold: Threshold for depth filtering (default: 0.1)
        - image_plane_distance: Distance of image plane in visualization (default: 0.05)
        - track_subsample: Subsample factor for tracks visualization (default: 5)
        - track_radii: Point radius for track visualization (default: 2.0)
        - show_3d_tracks: Whether to show 3D tracks in visualization (default: True)
        - multi_view_frames: List of frame indices to display in multi-view mode (default: None)
    """
    from dotdict import dotdict
    model_name = cfg.get("model_name", "anycam")
    input_path = cfg.get("input_path", None)
    output_path = cfg.get("output_path", None)
    model_path = cfg.get("model_path", None)
    checkpoint = cfg.get("checkpoint", None)
    visualize = cfg.get("visualize", False)
    rerun_mode = cfg.get("rerun_mode", "spawn")
    clear_rerun = cfg.get("clear_rerun", False)
    export_colmap = cfg.get("export_colmap", False)
    image_size = cfg.get("image_size", 336)
    ba_refinement = cfg.get("ba_refinement", True)
    target_fps = cfg.get("fps", 0)  # 0 means use all frames
    
    # Ground truth parameters
    load_gt = cfg.get("load_gt", False)
    gt_data_path = cfg.get("gt_data_path", None)
    do_align_trajectories = cfg.get("align_trajectories", False)
    
    # Track visualization parameters
    visualize_tracks = cfg.get("visualize_tracks", False)
    track_extraction_mode = cfg.get("track_extraction_mode", "full_length") # New config
    # Parameters for windowed track extraction (if mode is "windowed")
    track_ba_window = cfg.get("track_ba_window", 8) 
    track_overlap = cfg.get("track_overlap", 6)
    track_stride_cpt = cfg.get("track_stride_cpt", 1) # Stride for CPT within each window
    # Common parameters for both modes
    # track_grid_size = cfg.get("track_grid_size", 16)
    track_length = cfg.get("track_length", 8) # For windowed, it's CPT track_len; for full, CPT uses seq_len
    ba_refinement_cfg = cfg.get("fit_video").get("ba_refinement")
    prediction_cfg = cfg.get("fit_video").get("prediction")
    # Semantic filtering parameters
    apply_semantic_filtering = ba_refinement_cfg.get("apply_semantic_filtering", True)
    drift_threshold = ba_refinement_cfg.get("drift_threshold", 5)
    min_tracks_threshold = ba_refinement_cfg.get("min_tracks_threshold", 5)
    track_grid_size = ba_refinement_cfg.get("grid_size", 16)
    motion_threshold = ba_refinement_cfg.get("motion_threshold", 30)
    dilated_window_size = ba_refinement_cfg.get("dilated_window_size", 5)
    
    # Automatically set track_extraction_mode based on ba_type
    ba_type = ba_refinement_cfg.get("ba_type", "pose_only")
    if ba_type == "global":
        track_extraction_mode = "global"
        print(f"Automatically setting track_extraction_mode to 'global' because ba_type is 'global'")
    elif ba_type in ["decoupled", "multi_step", "single_step"] and track_extraction_mode == "full_length":
        track_extraction_mode = "windowed"
        print(f"Automatically setting track_extraction_mode to 'windowed' because ba_type is '{ba_type}'")
    
    print(f"Using track_extraction_mode: {track_extraction_mode}")
    
    # Initialize semantic segmentor if semantic filtering is enabled
    semantic_segmentor = None
    
    if input_path is None:
        print("Error: input_path is required")
        return
        
    if model_path is None:
        print("Using default model path")
        model_path = Path(__file__).parent.parent.parent / "outputs"
    else:
        model_path = Path(model_path)
    
    # Load input data
    fps = None
    if os.path.isdir(input_path):
        print(f"Loading frames from directory: {input_path}")
        frames, _ = load_frames(input_path)
    else:
        print(f"Loading video from: {input_path}")
        frames, fps = load_video(input_path)
    
    if not frames:
        print("Error: No frames loaded")
        return
        
    print(f"Loaded {len(frames)} frames")
    
    # Subsample frames if target_fps is specified
    if target_fps > 0 and fps:
        frames = subsample_frames(frames, original_fps=fps, target_fps=target_fps)
        print(f"Subsampled frames to {len(frames)} frames at {target_fps} fps")
    
    # Format frames for processing
    frames = format_frames(frames, target_size=image_size)
    print(f"Resized frames to {frames[0].shape[:2]}")

    # Create default configuration for process_video
    default_config = {
        "with_rerun": False,
        "do_ba_refinement": ba_refinement,
        'seq_name': cfg.get('seq_name', 'temple_3'),
        "prediction": {
            "model_seq_len": prediction_cfg.get("model_seq_len", 100),
            "shift": prediction_cfg.get("shift", 99),
            "square_crop": prediction_cfg.get("square_crop", False),
            "return_all_uncerts": prediction_cfg.get("return_all_uncerts", False),
            "use_provided_depth": prediction_cfg.get("use_provided_depth", False),
            "use_provided_masks": prediction_cfg.get("use_provided_masks", False),
            "use_provided_flow": prediction_cfg.get("use_provided_flow", False),
            "flow_model": prediction_cfg.get("flow_model", "unimatch"),
            "depth_predictor": prediction_cfg.get("depth_predictor", "unidepth"),
            "mask_path": prediction_cfg.get("mask_path", None),
            "mono_depth_path": prediction_cfg.get("mono_depth_path", "/home/zhuoyuanwu/mega-sam/Depth-Anything/video_visualization"),
            "metric_depth_path": prediction_cfg.get("metric_depth_path", "/home/zhuoyuanwu/mega-sam/UniDepth/outputs"),
            "recon_data_path": prediction_cfg.get("recon_data_path", None),
            "uncertainty_type": prediction_cfg.get("uncertainty_type", "pose")
        },
        "ba_refinement": {
            "with_rerun": ba_refinement_cfg.get("with_rerun", False),
            "max_uncert": ba_refinement_cfg.get("max_uncert", 0.05),
            "lambda_smoothness": ba_refinement_cfg.get("lambda_smoothness", 0.1),
            "long_tracks": ba_refinement_cfg.get("long_tracks", True),
            "n_steps_last_global": ba_refinement_cfg.get("n_steps_last_global", 5000),
            "global_every_n": ba_refinement_cfg.get("global_every_n", 100),
            "collect_optimized_tracks": ba_refinement_cfg.get("collect_optimized_tracks", True),
            "apply_semantic_filtering": apply_semantic_filtering,
            "drift_threshold": drift_threshold,
            "min_tracks_threshold": min_tracks_threshold,
            "w_track3d": ba_refinement_cfg.get("w_track3d", 1e-6),
            "lambda_reg_track3d": ba_refinement_cfg.get("lambda_reg_track3d", 100),
            "w_static_track3d": ba_refinement_cfg.get("w_static_track3d", 100),
            "w_dynamic_track3d": ba_refinement_cfg.get("w_dynamic_track3d", 100),
            "grid_size": track_grid_size,
            "apply_min_length_filter": ba_refinement_cfg.get("apply_min_length_filter", False),
            "min_valid_length": ba_refinement_cfg.get("min_valid_length", 5),
            "motion_threshold": motion_threshold,
            "ba_type": ba_refinement_cfg.get("ba_type", "pose_only"),
            "dilated_window_size": dilated_window_size,
            "visualize_tracks": ba_refinement_cfg.get("visualize_tracks", False),
            "apply_reprojection_loss": ba_refinement_cfg.get("apply_reprojection_loss", True),
            "visualize_uncertainty": ba_refinement_cfg.get("visualize_uncertainty", False),
            "generate_depth": ba_refinement_cfg.get("generate_depth", False),
            "lr": ba_refinement_cfg.get("lr", 1e-4),
            "tracks_only": ba_refinement_cfg.get("tracks_only", False),
            "tracks_opt_with_uncert": ba_refinement_cfg.get("tracks_opt_with_uncert", False),
            "lr_track3d_global": ba_refinement_cfg.get("lr_track3d_global", 0.05),
            "n_steps_track3d_global": ba_refinement_cfg.get("n_steps_track3d_global", 200),
            "refined_loss": ba_refinement_cfg.get("refined_loss", False),
            "use_static_ba": ba_refinement_cfg.get("use_static_ba", False),
        },
        "ba_refinement_level": cfg.get("fit_video").get("ba_refinement_level", 2),
        "dataset": {
            "image_size": [image_size, None]
        }
    }
    
    # Load model
    if model_name == 'anycam':
        print(f"Loading model from {model_path}")
        model, criterion = load_anycam(model_path, checkpoint, default_config)
    elif model_name == 'megasam':
        from anycam.scripts.fit_video_megasam import load_megasam
        print(f"Loading model from {model_path}")
        model, criterion = load_megasam(model_path, checkpoint, default_config)
        default_config['model_name'] = 'megasam'
    elif model_name == 'droidslam':
        from anycam.scripts.fit_video_droidslam import load_droidslam
        print(f"Loading model from {model_path}")
        model, criterion = load_droidslam(model_path, checkpoint, default_config)
        default_config['model_name'] = 'droidslam'
    elif model_name == 'vggt':
        from anycam.scripts.fit_video_vggt import load_vggt
        print(f"Loading model from {model_path}")
        model, criterion = load_vggt(model_path, checkpoint, default_config)
    elif model_name == 'easi3r':
        from anycam.scripts.fit_video_easi3r import load_easi3r
        print(f"Loading model from {model_path}")
        model, criterion = load_easi3r(model_path, checkpoint, default_config)
        default_config['model_name'] = 'easi3r'
    elif model_name == 'spatrackerv2':
        from anycam.scripts.fit_video_spatrackerv2 import load_spatrackerv2
        print(f"Loading model from {model_path}")
        model, criterion = load_spatrackerv2(model_path, checkpoint, default_config)
    elif model_name == 'cut3r':
        from anycam.scripts.fit_video_cut3r import load_cut3r
        print(f"Loading model from {model_path}")
        model, criterion = load_cut3r(model_path, checkpoint, default_config)
        default_config['model_name'] = 'cut3r'
    elif model_name == 'monst3r':
        from anycam.scripts.fit_video_monst3r import load_monst3r
        print(f"Loading model from {model_path}")
        model, criterion = load_monst3r(model_path, checkpoint, default_config)
        default_config['model_name'] = 'monst3r'
    elif model_name == 'ttt3r':
        from anycam.scripts.fit_video_ttt3r import load_ttt3r
        print(f"Loading model from {model_path}")
        model, criterion = load_ttt3r(model_path, checkpoint, default_config)
        default_config['model_name'] = 'ttt3r'
    else:
        raise ValueError(f"Invalid model name: {model_name}. Supported: 'anycam', 'megasam', 'vggt', 'spatrackerv2', 'cut3r'")
    model = model.cuda().eval()
    
    # Convert to dotdict for easier access
    process_config = dotdict(default_config)
    # Process frames
    trajectory, proj, extras_dict, ba_extras = process_video(
        model, 
        criterion, 
        frames, 
        config=process_config,
        ba_refinement=ba_refinement,
        model_name=model_name
    )

    # trajectory = [se3_ensure_numerical_accuracy(torch.tensor(pose)) for pose in trajectory] # TODO: Remove this
    
    # Extract depth and uncertainty information
    best_candidate = extras_dict["best_candidate"]
    depths = extras_dict["seq_depths"]
    ba_refinement_level = process_config.get("ba_refinement_level", 0) + 1
    if not ba_refinement:
        read_frames = frames
        frames_tensor = extras_dict["images"]
        frames_np = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()
        keyframes = [i for i in range(len(trajectory))]
        if isinstance(extras_dict["uncertainties"], list):
            uncertainties = torch.stack(extras_dict["uncertainties"])[:, 0, best_candidate, :1, :, :]
            # For non-BA case, uncertainties has length seq_len-1, so we need to add one more
            uncertainties = torch.cat((uncertainties, uncertainties[-1:]), dim=0)
        else:
            uncertainties = extras_dict["uncertainties"]
    else:
        # keyframes = [i * 3 for i in range(len(trajectory) // 3)]
        trajectory = extras_dict["keyframe_trajectory"]
        keyframes = [i for i in range(len(trajectory))] # Since we have keyframe trajectory, we can use all frames
        uncertainties = extras_dict["ba_uncertainties"]
        read_frames = frames
        frames_np = np.array(frames)
        frames_np = frames_np[::ba_refinement_level]
        # For BA case, ba_uncertainties already has the correct length, no concatenation needed

    print(f"Processed video: {len(trajectory)} poses, {len(depths)} depth maps")
    # Extract 3D tracks if requested
    initial_tracks = None
    optimized_tracks = None
    structured_3d_tracks = None  # Initialize here to avoid undefined variable errors
    
    if visualize and visualize_tracks:
        print("Extracting 3D tracks for visualization...")
        
        # Extract flows from extras_dict
        seq_flow_occs_fwd = extras_dict.get("seq_flow_occs_fwd", None)
        
        if seq_flow_occs_fwd is not None:
            print("Extracting full-length 2D and 3D tracks...")
            
            # Ensure all necessary data are tensors on CUDA for track extraction
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            _seq_flow_occs_fwd = seq_flow_occs_fwd.to(current_device) 
            _uncertainties = uncertainties.to(current_device)
            if _uncertainties.dim() == 6: 
                _uncertainties = _uncertainties[:, 0, best_candidate, :, :, :] 
            elif _uncertainties.dim() == 5: 
                 _uncertainties = _uncertainties[:, best_candidate, :, :, :]
            if _uncertainties.dim() == 3: 
                _uncertainties = _uncertainties.unsqueeze(1) 
            
            _depths = depths.to(current_device) 
            if _depths.dim() == 3: 
                _depths = _depths.unsqueeze(1) 

            _proj = torch.tensor(proj, dtype=torch.float32).to(current_device)

            if ba_refinement: 
                _imgs_for_track_extraction = extras_dict["images"].to(current_device)
            else: 
                _imgs_for_track_extraction = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2).to(current_device)

            if track_extraction_mode == "full_length":
                print("Extracting full-length 2D and 3D tracks...")
                initial_tracks, initial_pixel_coords, initial_pixel_depths, initial_pixel_indices, structured_3d_tracks = extract_full_length_tracks_data(
                    _seq_flow_occs_fwd, 
                    _uncertainties,    
                    _depths,           
                    _proj, 
                    trajectory,          
                    grid_size=track_grid_size,
                    imgs_tensor_seq=_imgs_for_track_extraction,
                    device_str=str(current_device),
                    apply_semantic_filtering=True,
                    semantic_segmentor=semantic_segmentor,
                    drift_threshold=drift_threshold,
                    min_tracks_threshold=min_tracks_threshold
                )
                if initial_tracks and initial_tracks[0].size > 0:
                    print(f"Extracted {initial_tracks[0].shape[0]} 3D points from full-length tracks (for Points3D fallback).")
                if initial_pixel_coords and initial_pixel_coords[0].size > 0:
                    print(f"Extracted {initial_pixel_coords[0].shape[0]} full-length 2D pixel tracks for LineStrips3D lifting.")
                if structured_3d_tracks and structured_3d_tracks[0].size > 0:
                    print(f"Extracted structured 3D tracks with shape {structured_3d_tracks[0].shape}")
            
            elif track_extraction_mode == "windowed":
                print("Using pre-computed windowed tracking data from BA optimization...")
                
                # Get optimized windows data from extras_dict if available  
                ba_optimized_windows_data = extras_dict.get("ba_optimized_windows_data", None)
                ba_param_sigma_depth = extras_dict.get("ba_param_sigma_depth", None)
                
                if ba_optimized_windows_data is not None and len(ba_optimized_windows_data) > 0:
                    print(f"Loading {len(ba_optimized_windows_data)} optimized windows from BA optimization results...")
                    
                    # Create structured windowed track data for plot_to_rerun
                    windowed_track_data = []
                    
                    for window_data in ba_optimized_windows_data:
                        if not window_data.get("has_enough_tracks", True):
                            print(f"Skipping window {window_data.get('window_id', 'unknown')} - insufficient tracks")
                            continue
                            
                        # Get the actually optimized tracks for this window
                        tracks2d_window = window_data["tracks2d_ba_window"]  # (1, 1, gs, tl, 2)
                        inv_depth_window = window_data["inv_depth_ba_window"]  # (1, 1, gs, tl, 1)
                        visible_window = window_data["visible_ba_window"]  # (1, 1, gs, tl, 1)
                        filtered_indices = window_data["filtered_indices"]
                        depth_sigma_indices = window_data["depth_sigma_indices"]  # Window-relative indices
                        frame_range = window_data["frame_range"]  # (frame_range_start, frame_range_end)
                        window_id = window_data["window_id"]
                        
                        # Get filtering and mask information for valid track visualization
                        has_enough_tracks = window_data.get("has_enough_tracks", True)
                        masks = window_data.get("masks", None)  # Original masks (npt, window_len, 1)
                        masks_filtered = window_data.get("masks_filtered", None)  # Filtered masks
                        visible_list = window_data.get("visible_list", None)  # Original visibility (npt, window_len)
                        visible_list_filtered = window_data.get("visible_list_filtered", None)  # Filtered visibility
                        
                        # Convert window-relative indices to global frame indices
                        frame_range_start, frame_range_end = frame_range
                        global_frame_indices = depth_sigma_indices + frame_range_start  # Convert to global indices
                        
                        # Squeeze to remove batch and group dimensions: (gs, tl, 2)
                        window_pixel_coords = tracks2d_window.squeeze(0).squeeze(0)  # (gs, tl, 2)
                        window_pixel_depths = 1.0 / inv_depth_window.squeeze(0).squeeze(0).squeeze(-1)  # (gs, tl)
                        
                        # Use global frame indices instead of window-relative ones
                        window_pixel_indices = global_frame_indices.unsqueeze(0).expand(window_pixel_coords.shape[0], -1).float()  # (gs, tl)
                        
                        # Apply filtering if it was used during optimization
                        if filtered_indices is not None:
                            window_pixel_coords = window_pixel_coords[filtered_indices]  # (filtered_gs, tl, 2)
                            window_pixel_depths = window_pixel_depths[filtered_indices]  # (filtered_gs, tl)
                            window_pixel_indices = window_pixel_indices[filtered_indices]  # (filtered_gs, tl)
                        
                        # Create structured window info for plot_to_rerun
                        window_info = {
                            'window_id': window_id,
                            'frame_range': frame_range,
                            'pixel_coords': window_pixel_coords.cpu().numpy(),  # (num_tracks, track_len, 2)
                            'pixel_depths': window_pixel_depths.cpu().numpy(),  # (num_tracks, track_len)
                            'pixel_indices': window_pixel_indices.cpu().numpy(),  # (num_tracks, track_len) - global indices
                            'num_tracks': window_pixel_coords.shape[0],
                            'track_len': window_pixel_coords.shape[1] if window_pixel_coords.shape[0] > 0 else 0,
                            # Add filtering information for visualization
                            'filtered_indices': filtered_indices,
                            'has_enough_tracks': has_enough_tracks,
                            'masks': masks,  # Original masks before filtering
                            'masks_filtered': masks_filtered,  # Masks after filtering
                            'visible_list': visible_list,  # Original visibility before filtering
                            'visible_list_filtered': visible_list_filtered,  # Visibility after filtering
                            'optimized_tracks': window_data.get("optimized_tracks", None),
                        }
                        windowed_track_data.append(window_info)
                    
                    # For backward compatibility, also create the legacy arrays
                    initial_pixel_coords = []
                    initial_pixel_depths = []
                    initial_pixel_indices = []
                    
                    for window_info in windowed_track_data:
                        initial_pixel_coords.append(window_info['pixel_coords'])
                        initial_pixel_depths.append(window_info['pixel_depths'])
                        initial_pixel_indices.append(window_info['pixel_indices'])
                    
                    structured_3d_tracks = None  # Not computed in windowed mode
                    initial_tracks = None  # Will be computed during visualization if needed
                    
                    num_windows_processed = len(windowed_track_data)
                    total_window_tracks = sum(window_info['num_tracks'] for window_info in windowed_track_data)
                    print(f"Loaded {total_window_tracks} optimized 2D pixel tracks across {num_windows_processed} windows.")
                    
                    # Store BA optimization data for track lifting, including structured data
                    track_optimization_data = {
                        "ba_param_sigma_depth": ba_param_sigma_depth,
                        "optimized_windows_data": ba_optimized_windows_data,
                        "windowed_track_data": windowed_track_data,  # Add structured data
                    }
                    
                else:
                    print("BA optimized windows data not found in extras_dict, falling back to extract_tracks_from_windows...")
                    # Fallback to original implementation
                    initial_tracks, initial_pixel_coords, initial_pixel_depths, initial_pixel_indices = extract_tracks_from_windows(
                        _seq_flow_occs_fwd,    # Full sequence tensor
                        _uncertainties,        # Full sequence tensor
                        _depths,               # Full sequence tensor
                        _proj,                 # Proj matrix (tensor)
                        trajectory,            # Full trajectory (list of tensors)
                        ba_window=track_ba_window,
                        overlap=track_overlap,
                        grid_size=track_grid_size,
                        track_len=track_length, # Max track length within a window for CPT
                        stride_cpt=track_stride_cpt, # Stride for CPT call within window
                        imgs=_imgs_for_track_extraction, # Full sequence images tensor
                        device=str(current_device) # Device string
                    )
                    structured_3d_tracks = None
                    track_optimization_data = None
                    num_windows_processed = len(initial_pixel_coords) if initial_pixel_coords else 0
                    total_window_tracks = sum(len(w_tracks) for w_tracks in initial_pixel_coords) if initial_pixel_coords else 0
                    print(f"Extracted {total_window_tracks} windowed 2D pixel tracks across {num_windows_processed} windows.")
            
            elif track_extraction_mode == "global":
                print("Using pre-computed global tracking data from BA optimization...")
                
                # Get global track data from extras_dict if available
                ba_global_track_data = extras_dict.get("ba_global_track_data", None)
                ba_param_sigma_depth = extras_dict.get("ba_param_sigma_depth", None)
                
                if ba_global_track_data is not None and ba_global_track_data.get("has_enough_tracks", False):
                    print(f"Loading global track data from BA optimization results...")
                    print(f"Global tracks: {ba_global_track_data.get('num_tracks', 0)} tracks, sequence length: {ba_global_track_data.get('track_len', 0)}")
                    
                    # Store global track optimization data for plot_to_rerun
                    track_optimization_data = {
                        "ba_param_sigma_depth": ba_param_sigma_depth,
                        "global_track_data": ba_global_track_data,
                    }
                    
                    # For backward compatibility, set these to None since we use track_optimization_data
                    initial_pixel_coords = None
                    initial_pixel_depths = None
                    initial_pixel_indices = None
                    structured_3d_tracks = None
                    initial_tracks = None
                    
                    print(f"Loaded {ba_global_track_data.get('num_tracks', 0)} global optimized tracks.")
                    
                else:
                    print("BA global track data not found or insufficient tracks, falling back to full_length extraction...")
                    # Fallback to full_length extraction
                    initial_tracks, initial_pixel_coords, initial_pixel_depths, initial_pixel_indices, structured_3d_tracks = extract_full_length_tracks_data(
                        _seq_flow_occs_fwd, 
                        _uncertainties,    
                        _depths,           
                        _proj, 
                        trajectory,          
                        grid_size=track_grid_size,
                        imgs_tensor_seq=_imgs_for_track_extraction,
                        device_str=str(current_device),
                        apply_semantic_filtering=True,
                        semantic_segmentor=semantic_segmentor,
                        drift_threshold=drift_threshold,
                        min_tracks_threshold=min_tracks_threshold
                    )
                    track_optimization_data = None
                    if initial_tracks and initial_tracks[0].size > 0:
                        print(f"Extracted {initial_tracks[0].shape[0]} 3D points from full-length tracks (fallback).")
            
            else:
                print(f"Warning: Unknown track_extraction_mode: {track_extraction_mode}. No tracks extracted.")
                initial_tracks, initial_pixel_coords, initial_pixel_depths, initial_pixel_indices, structured_3d_tracks = [None]*5

        else:
            print("Warning: Flow occlusions not found in extras_dict, cannot extract tracks")
            structured_3d_tracks = None  # Initialize when tracks cannot be extracted
            track_optimization_data = None  # Initialize when tracks cannot be extracted
        
        keyframe_depths = depths # Assuming 'depths' from process_video are suitable for keyframes
    else:
        keyframe_depths = depths # Fallback or if not BA refinement
        track_optimization_data = None  # Initialize when BA refinement is not used

    # Load ground truth trajectory if requested (supports 'sintel' and 'lightspeed')
    gt_trajectory = None
    if load_gt and gt_data_path and cfg.seq_name:
        print(f"Loading ground-truth trajectory from {gt_data_path}, sequence {cfg.seq_name}")

        # Create frame IDs for the ground truth poses
        frame_ids = list(range(len(trajectory)))

        # Determine dataset type from path
        supported_datasets = ["sintel", "lightspeed"]
        dataset_type = None
        try:
            lower_path = str(gt_data_path).lower()
            for ds in supported_datasets:
                if ds in lower_path:
                    dataset_type = ds
                    break
        except Exception:
            dataset_type = None

        try:
            if dataset_type == "sintel":
                gt_trajectory = load_sintel_gt_poses(gt_data_path, cfg.seq_name, frame_ids)
            elif dataset_type == "lightspeed":
                gt_trajectory = load_lightspeed_gt_poses(gt_data_path, cfg.seq_name, frame_ids)
            else:
                print(f"Unsupported dataset for gt poses. Supported: {supported_datasets}. Got path: {gt_data_path}")
                gt_trajectory = None

            if gt_trajectory is not None:
                print(f"Loaded {len(gt_trajectory)} ground-truth poses")

                # Align trajectories if requested and gt is available
                if do_align_trajectories and HAS_EVO:
                    print("Aligning predicted trajectory to ground-truth...")
                    aligned_trajectory = align_trajectories(trajectory, gt_trajectory)
                    trajectory = aligned_trajectory
                    print("Trajectory alignment complete")
        except Exception as e:
            print(f"Error loading ground-truth trajectory: {e}")
            gt_trajectory = None
    else:
        trajectory = torch.tensor(trajectory).cuda()
    
    if export_colmap:
        if output_path is None:
            print("Warning: output_path not specified, using temporary directory")
        
        from anycam.utils.colmap_io import export_to_colmap
        
        print("Exporting results to COLMAP format...")
        colmap_path = export_to_colmap(
            trajectory=trajectory,
            proj=proj,
            imgs=read_frames,
            out_dir=output_path
        )
        print(f"Exported COLMAP reconstruction to {colmap_path}")
    
    # Save trajectory and projection matrix if output_path is specified
    if output_path and not export_colmap:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to {output_path}")
        
        # Save trajectory as numpy array
        trajectory_np = np.stack([pose.cpu().numpy() for pose in trajectory])
        np.save(output_path / "trajectory.npy", trajectory_np)
        
        # Save projection matrix
        np.save(output_path / "projection.npy", proj.cpu().numpy())
        
        # Save depths if available
        if depths:
            depths_np = np.stack([depth.cpu().numpy() for depth in depths])
            np.save(output_path / "depths.npy", depths_np)
        
        # Save uncertainties if available
        if uncertainties is not None:
            uncertainties_np = np.stack([uncert.cpu().numpy() for uncert in uncertainties])
            np.save(output_path / "uncertainties.npy", uncertainties_np)
            
        # Save ground truth trajectory if available
        if gt_trajectory is not None:
            gt_trajectory_np = np.stack([pose.cpu().numpy() for pose in gt_trajectory])
            np.save(output_path / "gt_trajectory.npy", gt_trajectory_np)
            
        # Save initial and optimized tracks if available
        if initial_tracks:
            np.save(output_path / "initial_tracks.npy", np.array(initial_tracks, dtype=object))
        
        if optimized_tracks:
            np.save(output_path / "optimized_tracks.npy", np.array(optimized_tracks, dtype=object))
            
        print("Saved all results successfully")

    vis_config = cfg.get("vis", {})

    # Save uncertainty visualization video if requested
    if ba_refinement_cfg.get("visualize_uncertainty", False) and (uncertainties is not None):
        seq_name_str = cfg.get("seq_name", "sequence")
        model_name_str = default_config.get("model_name", model_name)
        video_filename = f"{model_name_str}_{seq_name_str}_uncert.mp4"
        # Prefer output_path if provided, otherwise current directory
        video_dir = Path(output_path) if output_path else Path(".")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / video_filename
        fps_out = target_fps if (target_fps and target_fps > 0) else (fps if fps else 24)
        if ba_refinement_cfg.get("uncertainty_type", "pose") == "depth":
            uncertainties = 1 - uncertainties
        save_uncertainty_video(
            uncertainties=uncertainties,
            video_path=video_path,
            fps=fps_out,
            show_colorbar=False
        )
    # Visualization or export
    if visualize:
        # Get visualization parameters from vis subconfig
        
        
        print(f"Visualizing results with rerun (mode: {rerun_mode})...")
        plot_to_rerun(
            trajectory=trajectory,
            depths=keyframe_depths, # Use keyframe_depths
            imgs=frames_np,
            proj=proj,
            uncertainties=uncertainties,
            radii=vis_config.get("radii", 1.5),
            uncertainty_thresh=vis_config.get("uncertainty_thresh", 0.05),
            max_depth=vis_config.get("max_depth", -1),
            filter_depth_threshold=vis_config.get("filter_depth_threshold", 0.1),
            image_plane_distance=vis_config.get("image_plane_distance", 0.05),
            keyframes=keyframes,
            rerun_mode=rerun_mode,
            clear_rerun=clear_rerun,
            gt_trajectory=gt_trajectory,
            initial_tracks=initial_tracks, # These are the 3D lifted points from extract_tracks_from_window
            optimized_tracks=optimized_tracks,
            track_subsample=vis_config.get("track_subsample", 5),
            track_radii=vis_config.get("track_radii", 2.0),
            # Pass the new 2D track data for initial tracks visualization
            initial_pixel_coords=initial_pixel_coords if visualize_tracks else None,
            initial_pixel_depths=initial_pixel_depths if visualize_tracks else None,
            initial_pixel_indices=initial_pixel_indices if visualize_tracks else None,
            point_trajectory_grid_size=track_grid_size, # Pass track_grid_size here
            structured_3d_tracks=structured_3d_tracks, # Pass structured 3D tracks here
            track_optimization_data=track_optimization_data,  # Pass optimization data for track lifting
            show_2d_tracks=False,
            show_3d_tracks=vis_config.get("show_3d_tracks", True),
            max_depth_filter=20,  # New parameter for depth filtering
            multi_view_frame_indices=vis_config.get("multi_view_frames", None)
        )
        
    print("Done")


if __name__ == "__main__":
    main()