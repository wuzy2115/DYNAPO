import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset

from anycam.models.spatrackerv2_wrapper import SpaTrackerV2Wrapper
from anycam.utils.bundle_adjustment import compute_depth_flow
from anycam.scripts.ba_refinement_opt_tracks_global import ba_refinement_opt_tracks_global
from anycam.utils.geometry import average_pose
from anycam.loss import make_loss

logger = logging.getLogger(__name__)


def load_spatrackerv2(model_path, checkpoint=None, loaded_config=None):
    """
    Load SpaTrackerV2Wrapper (no checkpoint usage; kept for API compatibility).
    """
    model_path = Path(model_path)
    config = OmegaConf.load(model_path / "training_config.yaml")

    prefix = "training_checkpoint_"
    ckpts = list(model_path.glob(f"{prefix}*.pt"))

    model_conf = config["model"]
    model_conf['use_provided_flow'] = loaded_config['prediction']['use_provided_flow'] if loaded_config is not None else False
    model_conf['use_provided_masks'] = loaded_config['prediction']['use_provided_masks'] if loaded_config is not None else False
    model_conf['use_provided_depth'] = loaded_config['prediction']['use_provided_depth'] if loaded_config is not None else True
    model_conf["train_directions"] = "forward"
    model_conf['depth_predictor']['type'] = loaded_config['prediction']['depth_predictor'] if loaded_config is not None else 'unidepth'
    model_conf['flow_model'] = loaded_config['prediction']['flow_model'] if loaded_config is not None else 'unimatch'
    model_conf['mask_path'] = loaded_config['prediction']['mask_path'] if loaded_config is not None else None

    model = SpaTrackerV2Wrapper(model_conf)

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

        # Filter out pose_predictor related weights since MegaSamWrapper doesn't have pose_predictor
        filtered_state_dict = {}
        for key, value in cp["model"].items():
            if not key.startswith("pose_predictor"):
                filtered_state_dict[key] = value
            else:
                print(f"Skipping pose_predictor weight: {key}")

        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (expected for MegaSam): {missing_keys}")
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



def fit_video_spatrackerv2_wrapper(config, model, criterion, imgs, device, spatrackerv2_data_path=None, seq_name=None, gt_proj=None):
    return fit_video_spatrackerv2(config=config, model=model, criterion=criterion, imgs=imgs, device=device, spatrackerv2_data_path=spatrackerv2_data_path, seq_name=seq_name, gt_proj=gt_proj, return_extras=False)


@torch.autocast(device_type="cuda", enabled=True)
@torch.no_grad()
def fit_video_spatrackerv2(config, model, criterion, imgs, device="cuda",
                           return_extras=False, gt_proj=None, spatrackerv2_data_path=None, seq_name=None):
    print(seq_name)
    print(config)
    if spatrackerv2_data_path is None:
        raise ValueError("spatrackerv2_data_path must be provided")
    if seq_name is None:
        raise ValueError("seq_name must be provided")

    do_ba_refinement = config.get("do_ba_refinement", False)
    ba_refinement_level = config.get("ba_refinement_level", 0) + 1
    ba_refinement_config = config.get("ba_refinement", {})
    dataset_config = config.get("dataset", {})
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
        rr.init("MegaSam Prediction", recording_id=uuid.uuid4())
        rr.connect()

        for i, img in enumerate(dataset.imgs):
            rr.set_time_sequence("timestep", i)
            rr.log(f"world/img", rr.Image((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).compress(jpeg_quality=95))

    logger.info("Loading SpaTrackerV2 predictions")
    st_data = model.load_spatrackerv2_predictions(spatrackerv2_data_path, seq_name)

    # Dataset resizing handled similar to VGGT path via consumer scripts
    logger.info("Preprocessing images")
    c, h, w = dataset.imgs.shape[1:]
    seq_imgs = dataset.imgs

    # Compute depth and flow using the AnyCam utilities
    seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(
        model, seq_imgs, megasam_disps=st_data.get('disps', None)
    )

    # Trajectory from SpaTrackerV2 c2w poses
    best_trajectory = [torch.tensor(pose, dtype=torch.float32) for pose in st_data['poses'].cpu().numpy()]
    if len(best_trajectory) != len(seq_imgs):
        best_trajectory = best_trajectory[:len(seq_imgs)]

    # Intrinsics
    proj = st_data['intrinsics'].cpu().numpy()

    # Motion probability equivalent
    motion_prob = st_data.get('motion_prob', None)
    if len(motion_prob) != len(dataset.imgs):
        motion_prob = motion_prob[:len(dataset.imgs)]

    # Ensure uncertainties shape matches (n,1,H,W) expected downstream
    ba_uncertainties_tensor = motion_prob.float()
    if ba_uncertainties_tensor.ndim == 1:
        ba_uncertainties_tensor = ba_uncertainties_tensor[:, None, None, None].expand(-1, 1, h, w).contiguous()
    elif ba_uncertainties_tensor.ndim == 3:
        ba_uncertainties_tensor = ba_uncertainties_tensor[:, None, :, :]

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
    
    if do_ba_refinement:
        if ba_refinement_level > 1:
            imgs0 = seq_imgs[:-1:ba_refinement_level]
            imgs1 = seq_imgs[1::ba_refinement_level]
            # Subsample disparities if provided
            st_disps_ba = None
            if disps is not None:
                st_disps_ba = disps[:-1:ba_refinement_level]
            seq_imgs_ba, seq_depths_ba, seq_flow_occs_fwd_ba, seq_flow_occs_bwd_ba = compute_depth_flow(
                model, imgs0=imgs0, imgs1=imgs1, megasam_disps=st_disps_ba
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
            model=model,
        )

        # Interpolate back to original frame rate
        interpolated_poses = []
        l = len(best_trajectory_ba)
        keyframe_trajectory = best_trajectory_ba
        for i in range(len(seq_imgs)):
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
                    nxt = last0 @ rel
                    interpolated_poses.append(torch.tensor(nxt))

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
    if st_data.get('disps') is not None:
        disp0 = st_data['disps'][0]
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
            "focal_length_candidates": torch.tensor([[proj[0, 0]]]) if isinstance(proj, np.ndarray) else torch.tensor([[proj[0,0]]]),
            "ba_optimized_windows_data": ba_extras.get("optimized_windows_data") if ba_extras is not None else None,
            "ba_param_sigma_depth": ba_extras.get("ba_param_sigma_depth") if ba_extras is not None else None,
            "ba_global_track_data": ba_extras.get("global_track_data") if ba_extras is not None else None,
            "keyframe_trajectory": best_trajectory,
            "spatrackerv2_data": st_data,
        }
        return best_trajectory, proj, extras_dict, ba_extras


