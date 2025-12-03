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

from anycam.models.cut3r_wrapper import Cut3RWrapper
from anycam.utils.bundle_adjustment import compute_depth_flow
from anycam.scripts.ba_refinement_opt_tracks_global import ba_refinement_opt_tracks_global
from anycam.utils.geometry import average_pose
from dotdict import dotdict
from anycam.loss import make_loss

import rerun as rr

logger = logging.getLogger(__name__)


def load_cut3r(model_path, checkpoint=None, loaded_config=None):
    """
    Load Cut3RWrapper model from AnyCam checkpoint.

    Args:
        model_path: Path to the model directory containing training_config.yaml
        checkpoint: Specific checkpoint name (optional, defaults to latest)
        loaded_config: Additional configuration overrides
        
    Returns:
        model: Cut3RWrapper instance
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

    # Create Cut3RWrapper
    model = Cut3RWrapper(model_conf)

    criterion = [make_loss(cfg) for cfg in config.get("loss", [])][0]

    # Optionally load checkpoint to share weights for depth predictor, etc.
    training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]
    if training_steps:
        if checkpoint is None:
            ckpt_path = f"{prefix}{max(training_steps)}.pt"
        else:
            ckpt_path = checkpoint
        ckpt_path = model_path / ckpt_path
        print(f"Loading checkpoint: {ckpt_path}")
        cp = torch.load(ckpt_path, map_location="cpu")
        # Load all non pose_predictor weights if present, mirroring VGGT flow
        filtered_state_dict = {}
        for key, value in cp["model"].items():
            if not key.startswith("pose_predictor"):
                filtered_state_dict[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys (expected for Cut3RWrapper): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        print(f"Successfully loaded {len(filtered_state_dict)} parameters")

    return model, criterion


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


def fit_video_cut3r_wrapper(config, model, criterion, imgs, device, cut3r_data_path=None, seq_name=None, gt_proj=None):
    """Wrapper function to maintain compatibility with original interface"""
    return fit_video_cut3r(config=config, model=model, criterion=criterion, imgs=imgs, device=device, cut3r_data_path=cut3r_data_path, seq_name=seq_name, gt_proj=gt_proj, return_extras=False)


@torch.autocast(device_type="cuda", enabled=True)
@torch.no_grad()
def fit_video_cut3r(config, model, criterion, imgs, device="cuda", 
                   return_extras=False, gt_proj=None, cut3r_data_path=None, seq_name=None):
    """
    Main function for video fitting using CUT3R predictions.
    """
    print(seq_name)
    print(config)

    if cut3r_data_path is None:
        raise ValueError("cut3r_data_path must be provided")
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
        rr.init("CUT3R Prediction", recording_id=uuid.uuid4())
        rr.connect()

        for i, img in enumerate(dataset.imgs):
            rr.set_time_sequence("timestep", i)
            rr.log(f"world/img", rr.Image((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).compress(jpeg_quality=95))

    logger.info("Loading CUT3R predictions")
    
    # Load CUT3R predictions from output folder
    cut3r_image_size = None
    cut3r_data = model.load_cut3r_predictions(cut3r_data_path, seq_name)
    logger.info(f"Loaded CUT3R data: poses {cut3r_data['poses'].shape}, intrinsics {cut3r_data['intrinsics'].shape}")
    
    cut3r_image_size = cut3r_data.get('image_size', None)
    if cut3r_image_size is not None:
        cut3r_width, cut3r_height = cut3r_image_size
        logger.info(f"CUT3R intrinsics calibrated for: {cut3r_width}x{cut3r_height}")
    else:
        logger.warning("CUT3R image size not available, assuming intrinsics match dataset images")
            

    # Preprocess images for depth and flow
    logger.info("Preprocessing images")
    c, h, w = dataset.imgs.shape[1:]
    seq_imgs = dataset.imgs
    
    # Rescale intrinsics if dataset image size differs from CUT3R's calibrated size
    if cut3r_image_size is not None and cut3r_width > 0 and cut3r_height > 0:
        dataset_width, dataset_height = w, h
        if cut3r_width != dataset_width or cut3r_height != dataset_height:
            logger.info(f"Rescaling intrinsics from CUT3R size {cut3r_width}x{cut3r_height} to dataset size {dataset_width}x{dataset_height}")
            cut3r_max_dim = max(cut3r_width, cut3r_height)
            dataset_max_dim = max(dataset_width, dataset_height)
            resize_ratio = dataset_max_dim / cut3r_max_dim
            rescaled_intrinsics = cut3r_data['intrinsics'].clone()
            rescaled_intrinsics[0, 0] *= resize_ratio  # fx
            rescaled_intrinsics[1, 1] *= resize_ratio  # fy
            rescaled_intrinsics[0, 2] = dataset_width / 2   # cx
            rescaled_intrinsics[1, 2] = dataset_height / 2  # cy
            cut3r_data['intrinsics'] = rescaled_intrinsics
            logger.info(f"Rescaled intrinsics by factor {resize_ratio:.4f}")
            logger.info(f"New intrinsics: fx={rescaled_intrinsics[0,0]:.2f}, fy={rescaled_intrinsics[1,1]:.2f}, cx={rescaled_intrinsics[0,2]:.2f}, cy={rescaled_intrinsics[1,2]:.2f}")

    # Compute depth and flow using the model; use CUT3R disparities if available
    seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(
        model, seq_imgs, megasam_disps=cut3r_data.get('disps')
    )
    
    # Convert CUT3R poses to trajectory
    cut3r_poses = cut3r_data['poses'].cpu().numpy()
    best_trajectory = []
    for i in range(len(cut3r_poses)):
        pose_matrix = torch.tensor(cut3r_poses[i], dtype=torch.float32)
        best_trajectory.append(pose_matrix.numpy())
        if i < 3:
            print(f"CUT3R pose {i} (camera-to-world):")
            print(pose_matrix.numpy())
            print()
    if len(best_trajectory) != len(dataset.imgs):
        best_trajectory = best_trajectory[:len(dataset.imgs)]
    
    # Projection matrix from CUT3R intrinsics
    cut3r_intrinsics = cut3r_data['intrinsics'].cpu().numpy()
    proj = cut3r_intrinsics.copy()
    
    # Use confidence maps as uncertainties if available
    motion_prob = cut3r_data['motion_prob']
    motion_prob = motion_prob.cpu().numpy() if motion_prob is not None else None
    
    if len(motion_prob) != len(dataset.imgs):
        motion_prob = motion_prob[:len(dataset.imgs)]

    # Prepare BA uncertainties tensor
    if motion_prob is None:
        # Default to ones of size (n, 1, h, w)
        n_imgs = len(dataset.imgs)
        ba_uncertainties = np.ones((n_imgs, 1, h, w), dtype=np.float32)
    else:
        # motion_prob expected (n, h, w)
        if motion_prob.ndim == 3:
            ba_uncertainties = motion_prob[:, None, :, :]
        elif motion_prob.ndim == 2:
            ba_uncertainties = motion_prob[None, None, :, :]
        else:
            ba_uncertainties = motion_prob

    ba_uncertainties_tensor = torch.tensor(ba_uncertainties, dtype=torch.float32)
    
    # Resize uncertainties to match image size if needed
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

    logger.info(f"Loaded trajectory: {len(best_trajectory)} poses")
    logger.info(f"Loaded uncertainties: {ba_uncertainties_tensor.shape}")

    # Bundle Adjustment Refinement
    if do_ba_refinement:
        logger.info("Starting BA refinement with CUT3R initial trajectory")
        
        if ba_refinement_level > 1:
            imgs0 = dataset.imgs[:-1:ba_refinement_level]
            imgs1 = dataset.imgs[1::ba_refinement_level]
            cut3r_disps_ba = None
            if cut3r_data.get('disps') is not None:
                cut3r_disps_ba = cut3r_data['disps'][:-1:ba_refinement_level]
            seq_imgs_ba, seq_depths_ba, seq_flow_occs_fwd_ba, seq_flow_occs_bwd_ba = compute_depth_flow(
                model, imgs0=imgs0, imgs1=imgs1, megasam_disps=cut3r_disps_ba
            )
            trajectory_ba = best_trajectory[::ba_refinement_level][:len(seq_imgs_ba)]
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
        best_trajectory = torch.stack([torch.tensor(pose) for pose in best_trajectory])

    # Rescale intrinsics to match the resolution used in compute_depth_flow.
    # Disparities were resized inside compute_depth_flow to match seq_imgs,
    # so scale K from disparity resolution to seq_imgs resolution (per-axis).
    target_h, target_w = int(seq_imgs.shape[2]), int(seq_imgs.shape[3])
    if cut3r_data.get('disps') is not None:
        disp0 = cut3r_data['disps'][0]
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
            "candidate_trajectories": [best_trajectory],
            "pred_labels": torch.tensor([1.0]),
            "flow_labels": torch.tensor([1.0]),
            "images": seq_imgs,
            "seq_depths": seq_depths,
            "seq_flow_occs_fwd": seq_flow_occs_fwd,
            "seq_flow_occs_bwd": seq_flow_occs_bwd,
            "uncertainties": ba_uncertainties_tensor,
            "best_candidate": 0,
            "focal_length_candidates": torch.tensor([[proj[0, 0]]]),
            "ba_optimized_windows_data": ba_extras.get("optimized_windows_data") if ba_extras is not None else None,
            "ba_param_sigma_depth": ba_extras.get("ba_param_sigma_depth") if ba_extras is not None else None,
            "ba_global_track_data": ba_extras.get("global_track_data") if ba_extras is not None else None,
            "keyframe_trajectory": keyframe_trajectory if do_ba_refinement else best_trajectory,
            "cut3r_data": cut3r_data,
        }
        return best_trajectory, proj, extras_dict, ba_extras


