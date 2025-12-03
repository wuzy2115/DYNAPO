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

from anycam.models.droidslam_wrapper import DroidSlamWrapper
from anycam.utils.bundle_adjustment import compute_depth_flow
from anycam.scripts.ba_refinement_opt_tracks_global import ba_refinement_opt_tracks_global
from anycam.utils.geometry import average_pose
from dotdict import dotdict
from anycam.loss import make_loss

import rerun as rr

logger = logging.getLogger(__name__)


def load_images(input_path):
    if isinstance(input_path, str):
        input_path = Path(input_path)

    if input_path.is_file():
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


def load_droidslam(model_path, checkpoint=None, loaded_config=None):
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

    model = DroidSlamWrapper(model_conf)

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
        # Load non-strict (droid wrapper may not match anycam model exactly)
        missing_keys, unexpected_keys = model.load_state_dict(cp.get("model", {}), strict=False)
        if missing_keys:
            print(f"Missing keys (expected for DroidSLAM wrapper): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    return model, criterion


def fit_video_droidslam_wrapper(config, model, criterion, imgs, device, droidslam_data_path, seq_name, gt_proj=None):
    return fit_video_droidslam(config=config, model=model, criterion=criterion, imgs=imgs, device=device, droidslam_data_path=droidslam_data_path, seq_name=seq_name, gt_proj=gt_proj, return_extras=False)


@torch.autocast(device_type="cuda", enabled=True)
@torch.no_grad()
def fit_video_droidslam(config, model, criterion, imgs, device="cuda", droidslam_data_path=None, seq_name=None,
                        return_extras=False, gt_proj=None):
    if droidslam_data_path is None:
        raise ValueError("droidslam_data_path must be provided")
    if seq_name is None:
        raise ValueError("seq_name must be provided")

    dataset_config = config.get("dataset", {})
    do_ba_refinement = config.get("do_ba_refinement", False)
    ba_refinement_level = config.get("ba_refinement_level", 0) + 1
    ba_refinement_config = config.get("ba_refinement", {})

    dataset = make_dataset(dataset_config, imgs, device="cpu")

    logger.info("Loading DroidSLAM predictions")
    try:
        droid_data = model.load_droidslam_predictions(droidslam_data_path, seq_name)
        logger.info(f"Loaded DroidSLAM data: poses {droid_data['poses'].shape}, intrinsics {droid_data['intrinsics'].shape}")
    except Exception as e:
        logger.error(f"Failed to load DroidSLAM predictions: {e}")
        raise

    generate_depth = ba_refinement_config.get("generate_depth", False)
    if generate_depth:
        output = model.generate_depths(dataset.imgs, visualize=False, viz_output_path=f"depths_{seq_name}.mp4")
        droid_data['disps'] = 1 / output['depth']
        droid_data['disps'][~output['mask']] = 0
        droid_data['intrinsics'] = output['intrinsics']

    c, h, w = dataset.imgs.shape[1:]
    seq_imgs = dataset.imgs

    seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(
        model, seq_imgs, megasam_disps=droid_data.get('disps')
    )

    print(f"After compute_depth_flow:")
    print(f"  seq_imgs shape: {seq_imgs.shape}")
    print(f"  seq_depths shape: {seq_depths.shape}")
    print(f"  seq_flow_occs_fwd shape: {seq_flow_occs_fwd.shape}")
    print(f"  seq_flow_occs_bwd shape: {seq_flow_occs_bwd.shape}")

    # DroidSLAM poses are already camera-to-world in most saved outputs we support
    droid_poses_c2w = droid_data['poses'].cpu().numpy()
    if len(droid_poses_c2w) != len(dataset.imgs):
        droid_poses_c2w = droid_poses_c2w[:len(dataset.imgs)]

    best_trajectory = [torch.tensor(pose, dtype=torch.float32).numpy() for pose in droid_poses_c2w]

    intrinsics = droid_data['intrinsics'].cpu().numpy()
    if intrinsics.ndim == 3:
        proj = intrinsics[0].copy()
    else:
        proj = intrinsics.copy()

    motion_prob = droid_data['motion_prob'].cpu().numpy()
    if len(motion_prob) != len(dataset.imgs):
        motion_prob = motion_prob[:len(dataset.imgs)]
    if motion_prob.ndim == 3:
        ba_uncertainties = motion_prob[:, None, :, :]
    else:
        ba_uncertainties = motion_prob

    ba_uncertainties_tensor = torch.tensor(ba_uncertainties, dtype=torch.float32)
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
            imgs0 = dataset.imgs[:-1:ba_refinement_level]
            imgs1 = dataset.imgs[1::ba_refinement_level]
            disps_ba = None
            if droid_data.get('disps') is not None:
                disps_ba = droid_data['disps'][:-1:ba_refinement_level]
            seq_imgs_ba, seq_depths_ba, seq_flow_occs_fwd_ba, seq_flow_occs_bwd_ba = compute_depth_flow(
                model, imgs0=imgs0, imgs1=imgs1, megasam_disps=disps_ba
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
            model=model
        )

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
                    nxt = last0 @ rel
                    interpolated_poses.append(torch.tensor(nxt))

        interpolated_poses = torch.stack(interpolated_poses).cpu()
        best_trajectory = interpolated_poses
        proj = proj_ba
    else:
        ba_extras = None
        best_trajectory = torch.stack([torch.tensor(pose) for pose in best_trajectory])

    proj = proj / dataset.scale_factor

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
            "droidslam_data": droid_data,
        }
        return best_trajectory, proj, extras_dict, ba_extras


