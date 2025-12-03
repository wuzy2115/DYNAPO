import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import uuid
import cv2
from pathlib import Path
from omegaconf import OmegaConf
import logging
from minipytorch3d.rotation_conversions import quaternion_to_matrix

from anycam.models.vggt_wrapper import VGGTWrapper
from anycam.utils.bundle_adjustment import compute_depth_flow
from anycam.scripts.ba_refinement_opt_tracks_global import ba_refinement_opt_tracks_global
from anycam.utils.geometry import average_pose
from dotdict import dotdict
from anycam.loss import make_loss

import rerun as rr

logger = logging.getLogger(__name__)


def load_vggt(model_path, checkpoint=None, loaded_config=None):
    """
    Load VGGTWrapper model from AnyCam checkpoint.
    
    Args:
        model_path: Path to the model directory containing training_config.yaml
        checkpoint: Specific checkpoint name (optional, defaults to latest)
        loaded_config: Additional configuration overrides
        
    Returns:
        model: VGGTWrapper instance
        criterion: Loss criterion for compatibility
    """
    model_path = Path(model_path)
    config = OmegaConf.load(model_path / "training_config.yaml")

    prefix = "training_checkpoint_"
    ckpts = list(model_path.glob(f"{prefix}*.pt"))

    model_conf = config["model"]
    model_conf['use_provided_flow'] = loaded_config['prediction']['use_provided_flow'] if loaded_config is not None else False
    model_conf['use_provided_masks'] = loaded_config['prediction']['use_provided_masks'] if loaded_config is not None else False
    model_conf['use_provided_depth'] = loaded_config['prediction']['use_provided_depth'] if loaded_config is not None else False
    model_conf["train_directions"] = "forward"
    model_conf['depth_predictor']['type'] = loaded_config['prediction']['depth_predictor'] if loaded_config is not None else 'unidepth'
    model_conf['flow_model'] = loaded_config['prediction']['flow_model'] if loaded_config is not None else 'unimatch'
    model_conf['mask_path'] = loaded_config['prediction']['mask_path'] if loaded_config is not None else None
    
    # Create VGGTWrapper
    model = VGGTWrapper(model_conf)

    criterion = [make_loss(cfg) for cfg in config.get("loss", [])][0]

    training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]

    if training_steps:
        if checkpoint is None:
            ckpt_path = f"{prefix}{max(training_steps)}.pt"
        else:
            ckpt_path = checkpoint

        ckpt_path = model_path / ckpt_path

        print(f"Loading checkpoint: {ckpt_path}")

        cp = torch.load(ckpt_path, map_location="cpu")

        # Filter out pose_predictor related weights since VGGTWrapper doesn't have pose_predictor
        filtered_state_dict = {}
        for key, value in cp["model"].items():
            if not key.startswith("pose_predictor"):
                filtered_state_dict[key] = value
            else:
                print(f"Skipping pose_predictor weight: {key}")

        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (expected for VGGT): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print(f"Successfully loaded {len(filtered_state_dict)} parameters")

    return model, criterion


def load_images(input_path):
    """Load images from input path"""
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if input_path.is_file():
        # Single video file
        import cv2
        cap = cv2.VideoCapture(str(input_path))
        imgs = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame.transpose(2, 0, 1)).float() / 255.0
            imgs.append(frame)
        cap.release()
        return torch.stack(imgs)
    else:
        # Directory with images
        image_files = sorted(list(input_path.glob("*.jpg"))) + sorted(list(input_path.glob("*.png")))
        imgs = []
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img.transpose(2, 0, 1)).float() / 255.0
            imgs.append(img)
        return torch.stack(imgs)


def make_dataset(config, imgs, device):

    target_size = config.get("image_size", None)
    center_crop = config.get("center_crop", False)

    class ImageDataset(Dataset):
        def __init__(self, imgs, target_size=None, center_crop=False):
            self.imgs = torch.tensor(np.array(imgs), device=device).permute(0, 3, 1, 2)

            bh, bw = self.imgs.shape[-2:]

            if target_size is not None:
                h, w = target_size

                if h is None:
                    h = int(bh / bw * w)
                elif w is None:
                    w = int(bw / bh * h)

                self.imgs = F.interpolate(self.imgs, (h, w), mode="bilinear", align_corners=False)
            else:
                h, w = self.imgs.shape[-2:]

            self.scale_factor = h / bh

            if center_crop:
                h, w = self.imgs.shape[-2:]

                h_ = min(h, w)
                w_ = min(h, w)

                h_start = (h - h_) // 2
                w_start = (w - w_) // 2

                self.imgs = self.imgs[:, :, h_start:h_start+h_, w_start:w_start+w_]

        def __len__(self):
            return len(self.imgs)
        
        def __getitem__(self, idx):
            img = self.imgs[idx]
            return img

    return ImageDataset(imgs, target_size=target_size, center_crop=center_crop)


def fit_video_vggt_wrapper(config, model, criterion, imgs, device, vggt_data_path=None, seq_name=None, gt_proj=None):
    """Wrapper function to maintain compatibility with original interface"""
    return fit_video_vggt(config=config, model=model, criterion=criterion, imgs=imgs, device=device, vggt_data_path=vggt_data_path, seq_name=seq_name, gt_proj=gt_proj, return_extras=False)


@torch.autocast(device_type="cuda", enabled=True)
@torch.no_grad()
def fit_video_vggt(config, model, criterion, imgs, device="cuda", 
                   return_extras=False, gt_proj=None, vggt_data_path=None, seq_name=None):
    """
    Main function for video fitting using VGGT predictions.
    
    Args:
        config: Configuration dict
        model: VGGTWrapper instance 
        criterion: Loss criterion (not used but kept for compatibility)
        imgs: Input images
        device: Device to run on
        vggt_data_path: Path to VGGT prediction files (COLMAP format)
        seq_name: Sequence name for loading VGGT data
        return_extras: Whether to return extra information
        gt_proj: Ground truth projection (optional)
    """
    print(seq_name)
    print(config)

    if vggt_data_path is None:
        raise ValueError("vggt_data_path must be provided")
    if seq_name is None:
        raise ValueError("seq_name must be provided")

    dataset_config = config.get("dataset", {})
    do_ba_refinement = config.get("do_ba_refinement", False)
    ba_refinement_level = config.get("ba_refinement_level", 0) + 1
    ba_refinement_config = config.get("ba_refinement", {})
    save_data_path = config.get('prediction', {}).get("save_data_path", None)
    if save_data_path is not None and seq_name is not None:
        save_data_path = Path(save_data_path) / config.get('dataset_type') / seq_name
        save_data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"dataset_config: {dataset_config}")
    print(f"do_ba_refinement: {do_ba_refinement}")
    print(f"ba_refinement_level: {ba_refinement_level}")
    print(f"ba_refinement_config: {ba_refinement_config}")

    # Create dataset
    dataset = make_dataset(dataset_config, imgs, device="cpu")

    if config.get("with_rerun", False):
        rr.init("VGGT Prediction", recording_id=uuid.uuid4())
        rr.connect()

        for i, img in enumerate(dataset.imgs):
            rr.set_time_sequence("timestep", i)
            rr.log(f"world/img", rr.Image((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).compress(jpeg_quality=95))

    logger.info("Loading VGGT predictions")
    
    # Load VGGT predictions from COLMAP format
    vggt_image_size = None  # Initialize to avoid undefined variable issues
    vggt_data = model.load_vggt_predictions(vggt_data_path, seq_name)
    logger.info(f"Loaded VGGT data: poses {vggt_data['poses'].shape}, intrinsics {vggt_data['intrinsics'].shape}")
    
    # Get VGGT's calibrated image size
    vggt_image_size = vggt_data.get('image_size', None)
    if vggt_image_size is not None:
        vggt_width, vggt_height = vggt_image_size
        logger.info(f"VGGT intrinsics calibrated for: {vggt_width}x{vggt_height}")
    else:
        logger.warning("VGGT image size not available, assuming intrinsics match dataset images")
            

    # Predict depths if needed for BA
    generate_depth = ba_refinement_config.get("generate_depth", False)
    if generate_depth:
        output = model.generate_depths(dataset.imgs, visualize=False, viz_output_path=f"depths_{seq_name}.mp4")
        vggt_data['disps'] = 1 / output['depth']
        vggt_data['mask'] = output['mask']
        vggt_data['intrinsics'] = output['intrinsics']
        vggt_data['disps'][~vggt_data['mask']] = 0  # set invalid disps to 0
    
    # Preprocess images for depth and flow
    logger.info("Preprocessing images")
    c, h, w = dataset.imgs.shape[1:]
    seq_imgs = dataset.imgs
    
    # Rescale intrinsics if dataset image size differs from VGGT's calibrated size
    # This is similar to rename_colmap_recons_and_rescale_camera in demo_colmap.py
    if vggt_image_size is not None:
        vggt_width, vggt_height = vggt_image_size
        dataset_width, dataset_height = w, h
        
        if vggt_width != dataset_width or vggt_height != dataset_height:
            logger.info(f"Rescaling intrinsics from VGGT size {vggt_width}x{vggt_height} to dataset size {dataset_width}x{dataset_height}")
            
            # Calculate resize ratio (matching demo_colmap.py approach)
            # VGGT uses max(real_image_size) / img_size for scaling
            vggt_max_dim = max(vggt_width, vggt_height)
            dataset_max_dim = max(dataset_width, dataset_height)
            resize_ratio = dataset_max_dim / vggt_max_dim
            
            # Rescale intrinsics
            rescaled_intrinsics = vggt_data['intrinsics'].clone()
            rescaled_intrinsics[0, 0] *= resize_ratio  # fx
            rescaled_intrinsics[1, 1] *= resize_ratio  # fy
            rescaled_intrinsics[0, 2] = dataset_width / 2   # cx = center
            rescaled_intrinsics[1, 2] = dataset_height / 2  # cy = center
            
            vggt_data['intrinsics'] = rescaled_intrinsics
            logger.info(f"Rescaled intrinsics by factor {resize_ratio:.4f}")
            logger.info(f"New intrinsics: fx={rescaled_intrinsics[0,0]:.2f}, fy={rescaled_intrinsics[1,1]:.2f}, cx={rescaled_intrinsics[0,2]:.2f}, cy={rescaled_intrinsics[1,2]:.2f}")

    # Compute depth and flow using the model
    seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(
        model, seq_imgs, megasam_disps=vggt_data.get('disps')
    )
    
    print(f"After compute_depth_flow:")
    print(f"  seq_imgs shape: {seq_imgs.shape}")
    print(f"  seq_depths shape: {seq_depths.shape}")
    print(f"  seq_flow_occs_fwd shape: {seq_flow_occs_fwd.shape}")
    print(f"  seq_flow_occs_bwd shape: {seq_flow_occs_bwd.shape}")
    
    # Convert VGGT poses to trajectory
    vggt_poses = vggt_data['poses'].cpu().numpy()
    print(f"VGGT poses shape: {vggt_poses.shape}")
    
    # VGGT already provides camera-to-world poses (from COLMAP conversion in VGGTWrapper)
    best_trajectory = []
    for i in range(len(vggt_poses)):
        pose_matrix = torch.tensor(vggt_poses[i], dtype=torch.float32)
        best_trajectory.append(pose_matrix.numpy())
        
        # Debug: Print first few poses
        if i < 3:
            print(f"VGGT pose {i} (camera-to-world):")
            print(pose_matrix.numpy())
            print()
    
    if len(best_trajectory) != len(dataset.imgs):
        best_trajectory = best_trajectory[:len(dataset.imgs)]
    
    print(f"Converted trajectory: {len(best_trajectory)} poses")
    
    # Extract projection matrix from VGGT intrinsics
    vggt_intrinsics = vggt_data['intrinsics'].cpu().numpy()
    proj = vggt_intrinsics.copy()
    
    # Use motion probabilities as uncertainties (default to ones for VGGT)
    motion_prob = vggt_data['motion_prob'].cpu().numpy()

    if len(motion_prob) != len(dataset.imgs):
        motion_prob = motion_prob[:len(dataset.imgs)]
    
    print(f"Motion prob shape: {motion_prob.shape}")
    
    # Use motion_prob directly as uncertainties
    ba_uncertainties = motion_prob
    
    # Ensure proper shape for BA: should be (n-1, 1, H, W) for n images
    if ba_uncertainties.ndim == 1:
        # If it's just (n,), expand to full image dimensions
        n_imgs = len(ba_uncertainties)
        ba_uncertainties = np.repeat(ba_uncertainties[:, None, None, None], h * w, axis=(2, 3))
        ba_uncertainties = ba_uncertainties.reshape(n_imgs, 1, h, w)
    elif ba_uncertainties.ndim == 3:
        ba_uncertainties = ba_uncertainties[:, None, :, :]  # Add channel dimension
    
    print(f"BA uncertainties shape after processing: {ba_uncertainties.shape}")
    
    # Convert to tensor and upsample to match image dimensions
    ba_uncertainties_tensor = torch.tensor(ba_uncertainties, dtype=torch.float32)
    
    # Get target dimensions from seq_imgs
    target_h, target_w = seq_imgs.shape[2], seq_imgs.shape[3]
    current_h, current_w = ba_uncertainties_tensor.shape[2], ba_uncertainties_tensor.shape[3]
    
    if current_h != target_h or current_w != target_w:
        print(f"Resizing uncertainties from {current_h}x{current_w} to {target_h}x{target_w}")
        ba_uncertainties_tensor = F.interpolate(
            ba_uncertainties_tensor, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
    
    print(f"Final ba_uncertainties_tensor shape: {ba_uncertainties_tensor.shape}")
    
    # Validate shapes before BA
    print(f"Shape validation before BA:")
    print(f"  seq_imgs: {seq_imgs.shape}")
    print(f"  seq_depths: {seq_depths.shape}")
    print(f"  seq_flow_occs_fwd: {seq_flow_occs_fwd.shape}")
    print(f"  seq_flow_occs_bwd: {seq_flow_occs_bwd.shape}")
    print(f"  ba_uncertainties_tensor: {ba_uncertainties_tensor.shape}")
    print(f"  best_trajectory: {len(best_trajectory)} poses")

    logger.info(f"Loaded trajectory: {len(best_trajectory)} poses")
    logger.info(f"Loaded uncertainties: {ba_uncertainties_tensor.shape}")

    # Bundle Adjustment Refinement
    if do_ba_refinement:
        logger.info("Starting BA refinement with VGGT initial trajectory")
        
        if ba_refinement_level > 1:
            # Subsample images and recompute flows
            imgs0 = dataset.imgs[:-1:ba_refinement_level]
            imgs1 = dataset.imgs[1::ba_refinement_level]
            
            # Subsample VGGT disparities to match BA refinement level
            vggt_disps_ba = None
            if vggt_data.get('disps') is not None:
                vggt_disps_ba = vggt_data['disps'][:-1:ba_refinement_level]
            
            seq_imgs_ba, seq_depths_ba, seq_flow_occs_fwd_ba, seq_flow_occs_bwd_ba = compute_depth_flow(
                model, imgs0=imgs0, imgs1=imgs1, megasam_disps=vggt_disps_ba
            )
            
            # Subsample trajectory and uncertainties to match the new sequence length
            trajectory_ba = best_trajectory[::ba_refinement_level][:len(seq_imgs_ba)]
            
            # For uncertainties, we need to subsample to match the flow length (n-1 for n images)
            ba_uncertainties_ba = ba_uncertainties_tensor[::ba_refinement_level][:len(seq_imgs_ba)]
        else:
            seq_imgs_ba = seq_imgs
            seq_depths_ba = seq_depths
            seq_flow_occs_fwd_ba = seq_flow_occs_fwd
            seq_flow_occs_bwd_ba = seq_flow_occs_bwd
            ba_uncertainties_ba = ba_uncertainties_tensor
            trajectory_ba = best_trajectory

        print(f"BA input shapes:")
        print(f"  seq_imgs_ba: {seq_imgs_ba.shape}")
        print(f"  seq_depths_ba: {seq_depths_ba.shape}")
        print(f"  seq_flow_occs_fwd_ba: {seq_flow_occs_fwd_ba.shape}")
        print(f"  seq_flow_occs_bwd_ba: {seq_flow_occs_bwd_ba.shape}")
        print(f"  ba_uncertainties_ba: {ba_uncertainties_ba.shape}")
        print(f"  trajectory_ba: {len(trajectory_ba)} poses")

        # Run BA refinement
        best_trajectory_ba, proj_ba, ba_extras = ba_refinement_opt_tracks_global(
            ba_refinement_config, 
            trajectory_ba, 
            proj, 
            ba_uncertainties_ba,
            seq_imgs_ba, 
            seq_depths_ba, 
            seq_flow_occs_fwd_ba, 
            seq_flow_occs_bwd_ba, 
            device=device,
            model=model
        )
        
        logger.info(f"BA refinement completed: {len(best_trajectory_ba)} refined poses")
        
        # Interpolate poses back to original frame rate
        interpolated_poses = []
        l = len(best_trajectory_ba)
        keyframe_trajectory = best_trajectory_ba
        
        for i in range(len(dataset.imgs)):
            if i % ba_refinement_level == 0 and (i // ba_refinement_level) < l:
                interpolated_poses.append(torch.tensor(best_trajectory_ba[i // ba_refinement_level]))
            else:
                if i // ba_refinement_level + 1 < l:
                    prev_pose = best_trajectory_ba[i // ba_refinement_level]
                    next_pose = best_trajectory_ba[(i // ba_refinement_level) + 1]

                    t = (i % ba_refinement_level) / ba_refinement_level
                    interpolated_pose = average_pose(torch.stack([torch.tensor(prev_pose), torch.tensor(next_pose)]), weight=t)
                    interpolated_poses.append(interpolated_pose)
                else:
                    # Extrapolate using previous motion
                    last1 = interpolated_poses[-2]
                    last0 = interpolated_poses[-1]
                    rel = torch.inverse(last1.to(torch.float32)) @ last0
                    next = last0 @ rel
                    interpolated_poses.append(torch.tensor(next))

        interpolated_poses = torch.stack(interpolated_poses).cpu()
        best_trajectory = interpolated_poses
        proj = proj_ba

    else:
        ba_extras = None
        # Convert best_trajectory to tensor format
        best_trajectory = torch.stack([torch.tensor(pose) for pose in best_trajectory])

    # Rescale intrinsics to match the resolution used in compute_depth_flow.
    # Disparities were resized inside compute_depth_flow to match seq_imgs,
    # so scale K from disparity resolution to seq_imgs resolution (per-axis).
    target_h, target_w = int(seq_imgs.shape[2]), int(seq_imgs.shape[3])
    if vggt_data.get('disps') is not None:
        disp0 = vggt_data['disps'][0]
        # Support shapes (H,W), (1,H,W) or (C,H,W)
        if hasattr(disp0, 'shape'):
            depth_h, depth_w = int(disp0.shape[-2]), int(disp0.shape[-1])
            if depth_h > 0 and depth_w > 0:
                scale_x = target_w / depth_w
                scale_y = target_h / depth_h
                # Scale fx, fy, cx, cy accordingly
                proj[0, 0] = proj[0, 0] * scale_x
                proj[1, 1] = proj[1, 1] * scale_y
                proj[0, 2] = proj[0, 2] * scale_x
                proj[1, 2] = proj[1, 2] * scale_y
                print(f"Rescaled intrinsics using disparity->image scales: sx={scale_x:.6f}, sy={scale_y:.6f}")

    if save_data_path is not None:
        # Save poses as poses.npy (N, 4, 4)
        if isinstance(best_trajectory, torch.Tensor):
            poses_np = best_trajectory.detach().cpu().numpy().astype(np.float32)
        else:
            poses_np = np.asarray(best_trajectory, dtype=np.float32)
        np.save(save_data_path / "poses.npy", poses_np)
        print(f"Saved poses to {save_data_path / 'poses.npy'} with shape {poses_np.shape}")

        # Save intrinsics as intrinsics.npy with a leading frame dimension (1,3,3)
        if isinstance(proj, torch.Tensor):
            intr_np = proj.detach().cpu().numpy().astype(np.float32)
        else:
            intr_np = np.asarray(proj, dtype=np.float32)
        intr_np_batched = intr_np[None, ...]
        np.save(save_data_path / "intrinsics.npy", intr_np_batched)
        print(f"Saved intrinsics to {save_data_path / 'intrinsics.npy'} with shape {intr_np_batched.shape}")

        # Save disparities as disps.npy (N, H, W) computed from depths
        if isinstance(seq_depths, torch.Tensor):
            disps_t = 1.0 / torch.clamp(seq_depths, min=1e-6)
            disps_np = disps_t.detach().cpu().numpy().astype(np.float32)
        else:
            disps_np = (1.0 / np.clip(seq_depths, 1e-6, None)).astype(np.float32)
        np.save(save_data_path / "disps.npy", disps_np)
        print(f"Saved disparities to {save_data_path / 'disps.npy'} with shape {disps_np.shape}")

        # Save motion probabilities as motion_prob.npy (N, H, W)
        if isinstance(ba_uncertainties_tensor, torch.Tensor):
            motion_prob_np = ba_uncertainties_tensor.squeeze(1).detach().cpu().numpy().astype(np.float32)
        else:
            motion_prob_np = np.asarray(ba_uncertainties_tensor, dtype=np.float32)
            if motion_prob_np.ndim == 4 and motion_prob_np.shape[1] == 1:
                motion_prob_np = motion_prob_np[:, 0]
        np.save(save_data_path / "motion_prob.npy", motion_prob_np)
        print(f"Saved motion probabilities to {save_data_path / 'motion_prob.npy'} with shape {motion_prob_np.shape}")

        # Save images for convenience (N,3,H,W)
        if isinstance(seq_imgs, torch.Tensor):
            images_np = seq_imgs.detach().cpu().numpy().astype(np.float32)
        else:
            images_np = np.asarray(seq_imgs, dtype=np.float32)
        np.save(save_data_path / "images.npy", images_np)
        print(f"Saved images to {save_data_path / 'images.npy'} with shape {images_np.shape}")

    if not return_extras:
        return best_trajectory, proj
    else:
        extras_dict = {
            "ba_uncertainties": ba_uncertainties_tensor if do_ba_refinement else None,
            "candidate_trajectories": [best_trajectory],  # Single trajectory from VGGT
            "pred_labels": torch.tensor([1.0]),  # Single prediction
            "flow_labels": torch.tensor([1.0]),
            "images": seq_imgs,
            "seq_depths": seq_depths,
            "seq_flow_occs_fwd": seq_flow_occs_fwd,
            "seq_flow_occs_bwd": seq_flow_occs_bwd,
            "uncertainties": ba_uncertainties_tensor,
            "best_candidate": 0,  # Only one candidate from VGGT
            "focal_length_candidates": torch.tensor([[proj[0, 0]]]),  # Single focal length
            "ba_optimized_windows_data": ba_extras.get("optimized_windows_data") if ba_extras is not None else None,
            "ba_param_sigma_depth": ba_extras.get("ba_param_sigma_depth") if ba_extras is not None else None,
            "ba_global_track_data": ba_extras.get("global_track_data") if ba_extras is not None else None,
            "keyframe_trajectory": keyframe_trajectory if do_ba_refinement else best_trajectory,
            "vggt_data": vggt_data,  # Include original VGGT data
        }
        return best_trajectory, proj, extras_dict, ba_extras

