import logging
import uuid

import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append("../..")
sys.path.append(".")

from anycam.visualization.common import color_tensor
from anycam.trainer import induce_flow_dist, make_proj_from_focal_length
from anycam.utils.geometry import average_pose
from anycam.utils.bundle_adjustment import *
from anycam.loss.metric import rotation_angle
from anycam.scripts.ba_refinement_opt_tracks_global import ba_refinement_opt_tracks_global
try:
    import rerun as rr
except:
    rr = None


logger = logging.getLogger(__name__)

def load_images(input_path):
    img_paths = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")))

    imgs = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        imgs.append(img)

    imgs = np.stack(imgs)

    return imgs


def visualize_uncertainty_video(uncertainties, seq_imgs, max_uncert=-1, save_path=None, fps=10):
    """
    Create a video visualization of uncertainty masks over time.
    
    Args:
        uncertainties: Tensor of shape (n, 1, H, W) or (n, H, W) - uncertainty values
        seq_imgs: Tensor of shape (n, 3, H, W) - original images
        max_uncert: Maximum uncertainty threshold (-1 to use 90th percentile)
        save_path: Path to save the video (should end with .mp4)
        fps: Frames per second for the output video
    """
    # Ensure uncertainties has the right shape
    if uncertainties.dim() == 3:
        uncertainties = uncertainties.unsqueeze(1)  # Add channel dim
    
    n_frames = uncertainties.shape[0]
    
    # Determine max_uncert threshold if not provided
    if max_uncert <= 0:
        max_uncert = torch.quantile(uncertainties, 0.9).item()
        print(f"Using 90th percentile as max_uncert threshold: {max_uncert:.4f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Uncertainty Analysis for Bundle Adjustment', fontsize=16)
    
    # Initialize empty plots
    ax_img = axes[0, 0]
    ax_uncert = axes[0, 1]
    ax_overlay = axes[1, 0]
    ax_hist = axes[1, 1]
    
    # Set up the plots
    ax_img.set_title('Original Image')
    ax_img.axis('off')
    
    ax_uncert.set_title('Uncertainty Map\n(High=Uncertain, Low=Confident)')
    ax_uncert.axis('off')
    
    ax_overlay.set_title('BA Mask Overlay (Red=Ignored, Green=Used)')
    ax_overlay.axis('off')
    
    ax_hist.set_title('Uncertainty Distribution')
    ax_hist.set_xlabel('Uncertainty Value')
    ax_hist.set_ylabel('Pixel Count')
    
    # Create custom colormap for uncertainty
    colors = ['green', 'yellow', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('uncertainty', colors, N=n_bins)
    
    # Initialize plot elements
    im_img = ax_img.imshow(np.zeros((100, 100, 3)))
    im_uncert = ax_uncert.imshow(np.zeros((100, 100)), cmap=cmap, vmin=0, vmax=max_uncert)
    im_overlay = ax_overlay.imshow(np.zeros((100, 100, 3)))
    
    # Add colorbar for uncertainty
    cbar = plt.colorbar(im_uncert, ax=ax_uncert, shrink=0.8)
    cbar.set_label('Uncertainty Value')
    
    # Text elements for statistics
    stats_text = ax_uncert.text(0.02, 0.98, '', transform=ax_uncert.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        # Get image and uncertainty data
        img = seq_imgs[frame].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        uncert_map = uncertainties[frame, 0].cpu().numpy()
        
        # Update image
        im_img.set_array(img)
        
        # Update uncertainty map
        im_uncert.set_array(uncert_map)
        
        # Create overlay based on BA masking logic
        # Regions with uncertainty > max_uncert are ignored (red)
        # Regions with uncertainty <= max_uncert are used (green)
        overlay = img.copy()
        
        # Dynamic/ignored regions (red overlay) - high uncertainty
        ignored_mask = uncert_map > max_uncert
        overlay[ignored_mask] = overlay[ignored_mask] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
        
        # Static/used regions (green overlay) - low uncertainty
        used_mask = uncert_map <= max_uncert
        overlay[used_mask] = overlay[used_mask] * 0.7 + np.array([0.0, 1.0, 0.0]) * 0.3
        
        im_overlay.set_array(overlay)
        
        # Update histogram
        ax_hist.clear()
        ax_hist.hist(uncert_map.flatten(), bins=50, alpha=0.7, color='blue', density=True)
        ax_hist.axvline(x=max_uncert, color='red', linestyle='--', alpha=0.7, label=f'Max uncert threshold: {max_uncert:.3f}')
        ax_hist.set_xlabel('Uncertainty Value')
        ax_hist.set_ylabel('Density')
        ax_hist.set_xlim(0, max(uncert_map.max(), max_uncert * 1.2))
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Update statistics
        ignored_ratio = (uncert_map > max_uncert).mean()
        used_ratio = (uncert_map <= max_uncert).mean()
        mean_uncert = uncert_map.mean()
        
        stats_str = f'Frame {frame+1}/{n_frames}\n'
        stats_str += f'Mean: {mean_uncert:.4f}\n'
        stats_str += f'Ignored: {ignored_ratio:.1%}\n'
        stats_str += f'Used: {used_ratio:.1%}'
        
        stats_text.set_text(stats_str)
        
        return [im_img, im_uncert, im_overlay, stats_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False, repeat=True)
    
    # Save video if path provided
    if save_path:
        print(f"Saving uncertainty video to: {save_path}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='AnyCam'), bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Video saved successfully!")
    
    # Show the animation
    plt.tight_layout()
    plt.show()
    
    return anim


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


def fit_video_wrapper(config, model, criterion, imgs, device, gt_proj=None, seq_name=None):
    pyramid = config.get("pyramid", 1)

    if type(pyramid) == int:
        pyramid = [pyramid]

    candidates = []

    # Use gt focal or not
    if not config.get("use_gt_focal", False):
        gt_proj = None

    print(f"Using GT focal: {config.get('use_gt_focal', False)} {gt_proj is not None}")

    for level in pyramid:
        imgs_level = imgs[::(level + 1)]

        level_poses, level_proj = fit_video(config, model, criterion, imgs_level, device=device, gt_proj=gt_proj, seq_name=seq_name)

        interpolated_poses = []
        interpolated_proj = []

        for i in range(len(imgs)):
            if i % (level + 1) == 0:
                interpolated_poses.append(torch.tensor(level_poses[i // (level + 1)]))
            else:
                prev_pose = level_poses[i // (level + 1)]
                if i // (level + 1) + 1 < len(level_poses):
                    next_pose = level_poses[(i // (level + 1)) + 1]

                    t = (i % (level + 1)) / (level + 1)

                    interpolated_pose = average_pose(torch.stack([torch.tensor(prev_pose), torch.tensor(next_pose)]), weight=t)

                    interpolated_poses.append(interpolated_pose)
                else:
                    interpolated_poses.append(torch.tensor(prev_pose))

            interpolated_proj.append(torch.tensor(level_proj))

        interpolated_poses = torch.stack(interpolated_poses).cpu()
        interpolated_proj = torch.stack(interpolated_proj).cpu()

        candidates.append(interpolated_poses)

    final_poses = None
    final_projs = interpolated_proj.mean(dim=0)

    for i, poses in enumerate(candidates):
        if final_poses is None:
            final_poses = poses
        else:
            final_poses = average_pose(torch.stack([final_poses, poses]), weight=1/(i+1))

    return final_poses, final_projs


@torch.autocast(device_type="cuda", enabled=True)
@torch.no_grad()
def fit_video(config, model, criterion, imgs, device="cuda", return_extras=False, gt_proj=None, seq_name=None):
    print(seq_name)
    print(config)

    dataset_config = config.get("dataset", {})

    do_ba_refinement = config.get("do_ba_refinement", False)

    ba_refinement_level = config.get("ba_refinement_level", 0) + 1
    ba_refinement_config = config.get("ba_refinement", {})

    prediction_config = config.get("prediction", {})
    save_data_path = prediction_config.get("save_data_path", None)
    if save_data_path is not None and seq_name is not None:
        save_data_path = Path(save_data_path) / config.get('dataset_type') / seq_name
        save_data_path.mkdir(parents=True, exist_ok=True)

    # Add configuration for precomputed depths
    use_precomputed_depths = prediction_config.get("use_precomputed_depths", False)
    mono_depth_path = prediction_config.get("mono_depth_path", None)
    metric_depth_path = prediction_config.get("metric_depth_path", None)

    model_seq_len = prediction_config.get("model_seq_len", 64)
    shift = prediction_config.get("shift", 63)
    square_crop = prediction_config.get("square_crop", False)
    return_all_uncerts = prediction_config.get("return_all_uncerts", False)

    proj_strategy = prediction_config.get("proj_strategy", "weighted")
    proj_label_source = prediction_config.get("proj_label", "prediction")

    ba_type = ba_refinement_config.get("ba_type", "pose_only")

    print(f"dataset_config: {dataset_config}")
    print(f"do_ba_refinement: {do_ba_refinement}")
    print(f"ba_refinement_level: {ba_refinement_level}")
    print(f"ba_refinement_config: {ba_refinement_config}")
    print(f"prediction_config: {prediction_config}")
    print(f"use_precomputed_depths: {use_precomputed_depths}")
    print(f"mono_depth_path: {mono_depth_path}")
    print(f"metric_depth_path: {metric_depth_path}")
    print(f"model_seq_len: {model_seq_len}")
    print(f"shift: {shift}")
    print(f"proj_strategy: {proj_strategy}")
    print(f"proj_label_source: {proj_label_source}")


    dataset = make_dataset(dataset_config, imgs, device="cpu")

    if config.with_rerun:
        rr.init("Prediction", recording_id=uuid.uuid4())
        rr.connect()

        for i, img in enumerate(dataset.imgs):
            rr.set_time_sequence("timestep", i)
            rr.log(f"world/img", rr.Image((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).compress(jpeg_quality=95))

    # Preprocess all images

    logger.info("Preprocessing images")

    dont_compute = False

    if model.pose_predictor.backbone_type == "croco":
        logger.info("Ignore flow and depth for CroCo")
        dont_compute = True



    c, h, w = dataset.imgs.shape[1:]

    if square_crop:
        sq = min(h, w)
        h_, w_ = sq, sq
    else:
        h_, w_ = h, w

    seq_imgs = dataset.imgs

    if square_crop:
        seq_imgs = seq_imgs[:, :, (h-sq)//2:(h-sq)//2+sq, (w-sq)//2:(w-sq)//2+sq]

    seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(model, seq_imgs)
    
    def prepare_batch(batch_ids_ids):
        batch_size, frame_count = batch_ids_ids.shape

        batch_ids_ids = batch_ids_ids.cpu()

        imgs = seq_imgs[batch_ids_ids.view(-1), :, :, :].reshape(batch_size, frame_count, c, h_, w_)
        depths = seq_depths[batch_ids_ids.view(-1)].reshape(batch_size, frame_count, 1, h_, w_)
        flow_occ_fwd = seq_flow_occs_fwd[batch_ids_ids.view(-1)].reshape(batch_size, frame_count, 3, h_, w_)
        flow_occ_bwd = seq_flow_occs_bwd[batch_ids_ids.view(-1)].reshape(batch_size, frame_count, 3, h_, w_)

        # Concatenate forward and backward
        imgs = torch.cat([imgs, imgs.flip(1)], dim=0).cuda()
        depths = torch.cat([depths, depths.flip(1)], dim=0).cuda()
        flow_occs = torch.cat([flow_occ_fwd, flow_occ_bwd.flip(1)], dim=0).cuda()

        return imgs, depths, flow_occs
    
    # device = "cuda"
    
    candidate_trajectories = [torch.eye(4, device=device).view(1, 4, 4).expand(model.pose_predictor.focal_num_candidates, -1, -1)]
    sub_trajectories = []
    
    proj_labels = []

    uncertainties = []
    angle_sum = 0

    pose_predictor = model.pose_predictor
    pose_predictor.eval()

    for i in tqdm(range(0, len(dataset)-1, shift)):
        batch_ids = torch.arange(i, i+1, device=device).view(-1, 1)
        seq_len_ = min(model_seq_len, len(dataset)-i)
        ids = torch.arange(0, seq_len_, device=device).view(1, -1)
        batch_ids_ids = batch_ids + ids

        imgs, depths, flow_occs, = prepare_batch(batch_ids_ids)

        flow_occs[:, -1, :2] = 0
        flow_occs[:, -1, 2] = 1

        pose_result = pose_predictor(
            images=imgs,
            depths=depths,
            flow_occs=flow_occs,
        )

        if proj_label_source == "prediction":
            proj_label = pose_result["focal_length_probs"][:, 0]
        elif proj_label_source == "loss":
            if model.try_focal_length_candidates:
                proj_candidates = make_proj_from_focal_length(pose_result["focal_length_candidates"], w/h)
            else:
                proj_candidates = make_proj_from_focal_length(pose_result["focal_length"].unsqueeze(-1))

            induced_flow, dist = induce_flow_dist(depths.unsqueeze(2), proj_candidates, pose_result["poses"].clone())

            pose_result["flow_occs_in"] = flow_occs
            pose_result["aligned_depths"] = depths.unsqueeze(2)
            pose_result["induced_flow"] = induced_flow
            pose_result["dist"] = dist

            _, losses = criterion({"pose_result": pose_result}, return_extra_data=True)
            proj_label = losses["flow_soft_label"]
        else:
            raise ValueError(f"Unknown proj_label_source: {proj_label_source}")

        num_candidates = proj_label.shape[-1]

        fwd_poses = pose_result["poses"][0, :-1]

        bwd_poses = pose_result["poses"][1, :-1]
        bwd_poses = torch.inverse(bwd_poses.flip(0))

        poses = average_pose(torch.stack([fwd_poses, bwd_poses]))

        with torch.autocast(device_type="cuda", enabled=False):
            r_angles = rotation_angle(poses[:, 16, :3, :3].to(torch.float32), torch.eye(3, device=device).view(1, 3, 3).expand(poses.shape[0], -1, -1))
            mean_angle = torch.mean(r_angles).item()
            mean_angle = 1
            angle_sum = angle_sum + mean_angle

        proj_labels.append(proj_label[0] * mean_angle)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            sub_trajectory = [torch.eye(4, device=device).view(1, 4, 4).expand(num_candidates, -1, -1)]

            for j in range(seq_len_-1):
                sub_trajectory.append(sub_trajectory[-1] @ torch.inverse(poses[j, :]))

            extra_poses = candidate_trajectories[i:]

            candidate_trajectories = candidate_trajectories[:i+1]
            uncertainties = uncertainties[:i]

            last_pose = candidate_trajectories[-1]

            sub_trajectory = [last_pose @ pose for pose in sub_trajectory]

            for k, pose in enumerate(poses):
                if k+1 < len(extra_poses):
                    cand_rel_pose = torch.inverse(extra_poses[k+1]) @ extra_poses[k]

                    pose = average_pose(torch.stack([pose, cand_rel_pose]))

                candidate_trajectories.append(candidate_trajectories[-1] @ torch.inverse(pose))
                uncertainties.append(pose_result["uncert"][:, k])

        sub_trajectories.append(sub_trajectory)

        if config.with_rerun:

            cmap = plt.get_cmap('hsv')
            cmap_cycle = 16

            for k in range(0, seq_len_):
                rr.set_time_sequence("timestep", i+k)
                
                img = (imgs[0, k].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                uncert = pose_result["uncert"][0, k:k+1, 16, 0, :, :].detach()
                uncert = color_tensor(uncert, cmap="plasma", norm=True)[0].cpu().numpy()
                uncert = (uncert * 255).astype(np.uint8)

                rr.log("world/img", rr.Image(img).compress(jpeg_quality=95))
                rr.log("world/uncert", rr.Image(uncert).compress(jpeg_quality=95))

            for i in range(num_candidates):
                rr.log(f"world/cand_{i:04d}/traj", rr.LineStrips3D([[p[i][:3, 3].cpu().numpy().tolist() for p in candidate_trajectories]], colors=[[0, 255, 0]]))

                sub_colors = [(np.array(cmap((k % cmap_cycle) / cmap_cycle)) * 255).astype(np.uint8).tolist() for k in range(len(sub_trajectories))]
                st_plot = [[p[i][:3, 3].cpu().numpy().tolist() for p in st] for st in sub_trajectories]

                rr.log(f"world/cand_{i:04d}/sub_traj", rr.LineStrips3D(st_plot, colors=sub_colors))

                rr.log(f"world/cand_{i:04d}/cam/pinhole", rr.Pinhole(
                    resolution=[w, h],
                    focal_length=w,
                ))

                rr.log(f"world/cand_{i:04d}/cam", rr.Transform3D(translation=candidate_trajectories[-1][i, :3, 3].cpu().numpy(), mat3x3=candidate_trajectories[-1][i, :3, :3].cpu().numpy()))

    # logger.warning(f"Using new pred labels")
    proj_labels = torch.stack(proj_labels).sum(dim=0) / angle_sum

    best_candidate = proj_labels.argmax()

    print(f"Best candidate: {best_candidate.item()}")

    if proj_strategy == "best":
        proj = make_proj_from_focal_length(pose_result["focal_length_candidates"][0:1, best_candidate:best_candidate+1], w_/h_)[0]
    elif proj_strategy == "mean":
        avg_label = (torch.arange(0, proj_labels.shape[-1], device=device) * proj_labels).sum(-1)
        print(avg_label)
        lower = torch.floor(avg_label)
        upper = torch.ceil(avg_label)
        interp = avg_label - lower
        fl = pose_result["focal_length_candidates"][0:1, lower.long()] * (1 - interp) + pose_result["focal_length_candidates"][0:1, upper.long()] * interp
        proj = make_proj_from_focal_length(fl.unsqueeze(1), w_/h_)[0]
    elif proj_strategy == "weighted":
        proj = make_proj_from_focal_length(pose_result["focal_length_candidates"][0:1, :], w_/h_)[0]
        proj = proj * proj_labels.view(-1, 1, 1)
        proj = proj.sum(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unknown proj_strategy: {proj_strategy}")

    if gt_proj is not None:
        # Pick candidate that is closest to GT
        gt_normalized_focal = gt_proj[0, 0] / w_ * 2
        candidates = pose_result["focal_length_candidates"][0]
        candidate_distances = torch.abs(candidates - gt_normalized_focal)
        gt_best_candidate = candidate_distances.argmin()

        print(f"Replacing best candidate with GT candidate {best_candidate.item()} -> {gt_best_candidate.item()}")
        print(f"GT focal: {gt_normalized_focal.item()} vs Candidate focal: {proj[0, 0, 0].item()}")

        best_candidate = gt_best_candidate

    proj[:, 0, 0] = proj[:, 0, 0] * w_ / w
    proj[:, 1, 1] = proj[:, 1, 1] * h_ / h

    proj[:, 0, 0] = (proj[:, 0, 0] * 0.5) * w
    proj[:, 1, 1] = (proj[:, 1, 1] * 0.5) * h
    proj[:, 0, 2] = (proj[:, 0, 2] * 0.5 + 0.5) * w
    proj[:, 1, 2] = (proj[:, 1, 2] * 0.5 + 0.5) * h

    proj = proj[0].cpu().numpy()

    best_trajectory = [p[best_candidate].cpu().numpy() for p in candidate_trajectories]
    keyframe_trajectory = best_trajectory

    if do_ba_refinement:
        if ba_refinement_level > 1 or square_crop:

            if square_crop:
                print("Recomputing uncertainties.")

                ba_uncertainties = []

                print("Reverting back to original resolution.")
                h_, w_ = h, w

                if return_all_uncerts:
                    uncert_level = 1
                else:
                    uncert_level = ba_refinement_level


                imgs0 = dataset.imgs[:-1:uncert_level]
                imgs1 = dataset.imgs[1::uncert_level]

                n_new = len(imgs0)

                seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(model, imgs0=imgs0, imgs1=imgs1)

                seq_imgs = torch.cat([seq_imgs, seq_imgs[-1:]], dim=0)
                seq_depths = torch.cat([seq_depths, seq_depths[-1:]], dim=0)
                seq_flow_occs_fwd = torch.cat([seq_flow_occs_fwd, seq_flow_occs_fwd[-1:]], dim=0)
                seq_flow_occs_bwd = torch.cat([seq_flow_occs_bwd, seq_flow_occs_bwd[-1:]], dim=0)

                for i in tqdm(range(0, n_new, shift)):
                    batch_ids = torch.arange(i, i+1, device=device).view(-1, 1)
                    seq_len_ = min(model_seq_len, n_new+1-i)
                    ids = torch.arange(0, seq_len_, device=device).view(1, -1)
                    batch_ids_ids = batch_ids + ids

                    imgs, depths, flow_occs, = prepare_batch(batch_ids_ids)

                    imgs = imgs[:1]
                    depths = depths[:1]
                    flow_occs = flow_occs[:1]

                    flow_occs[:, -1, :2] = 0
                    flow_occs[:, -1, 2] = 1

                    pose_result = pose_predictor(
                        images=imgs,
                        depths=depths,
                        flow_occs=flow_occs,
                    )
                    ba_uncertainties = ba_uncertainties[:i]

                    for k in range(pose_result["poses"].shape[1]):
                        ba_uncertainties.append(pose_result["uncert"][:, k])

                seq_imgs = seq_imgs[:-1]

                ba_uncertainties = ba_uncertainties[:-1]

                if return_all_uncerts:
                    uncertainties = ba_uncertainties
                    ba_uncertainties = uncertainties[::ba_refinement_level]
                    seq_imgs = seq_imgs[::ba_refinement_level]

                print("Recomputing uncertainties done.", len(ba_uncertainties))

                seq_imgs = dataset.imgs[::ba_refinement_level][:len(ba_uncertainties)]
                # print(len(seq_imgs), len(dataset.imgs[::ba_refinement_level]))
            else:
                seq_imgs = dataset.imgs[::ba_refinement_level]
                ba_uncertainties = uncertainties[::ba_refinement_level]


            c, h, w = dataset.imgs.shape[1:]

            # Switch between original depth prediction and precomputed depth loading  
            if use_precomputed_depths and mono_depth_path is not None and metric_depth_path is not None and seq_name is not None:
                # Use precomputed depths for uncertainty recomputation
                seq_depths = load_precomputed_depths_for_ba(
                    dataset.imgs, mono_depth_path, metric_depth_path, seq_name, square_crop_params=None
                )
                seq_depths = seq_depths[::ba_refinement_level][:len(ba_uncertainties)]
                _, _, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(model, seq_imgs)
            else:
                seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(model, seq_imgs)

        else:
            seq_imgs = dataset.imgs
            ba_uncertainties = uncertainties
 
        if len(ba_uncertainties) != len(seq_imgs):
            seq_imgs = seq_imgs[:len(ba_uncertainties)]

        if len(ba_uncertainties[0].shape) == 5:
            ba_uncertainties = torch.stack(ba_uncertainties)
            ba_uncertainties = ba_uncertainties[:, 0, best_candidate, :1]

        mask_tensor = model._load_masks_from_path(seq_name)
        if mask_tensor is not None:
            ba_uncertainties = mask_tensor

        if ba_uncertainties.ndim == 3:
            ba_uncertainties = ba_uncertainties[:, None, :, :]

        # Get target dimensions from seq_imgs
        target_h, target_w = seq_imgs.shape[2], seq_imgs.shape[3]
        current_h, current_w = ba_uncertainties.shape[-2], ba_uncertainties.shape[-1]
        
        if current_h != target_h or current_w != target_w:
            print(f"Resizing uncertainties from {current_h}x{current_w} to {target_h}x{target_w}")
            ba_uncertainties = F.interpolate(
                ba_uncertainties, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )

        best_trajectory, proj, ba_extras = ba_refinement_opt_tracks_global(
            ba_refinement_config, 
            best_trajectory[::ba_refinement_level][:len(seq_imgs)], 
            proj, 
            # uncertainties[:len(ba_imgs)-1], 
            ba_uncertainties,
            seq_imgs, 
            seq_depths, 
            seq_flow_occs_fwd, 
            seq_flow_occs_bwd, 
            device=device
        )
        
        interpolated_poses = []

        l = len(best_trajectory)
        keyframe_trajectory = best_trajectory
        for i in range(len(dataset.imgs)):
            if i % ba_refinement_level == 0 and (i // ba_refinement_level )< l:
                interpolated_poses.append(torch.tensor(best_trajectory[i // ba_refinement_level]))
            else:
                if i // ba_refinement_level + 1 < l:
                    prev_pose = best_trajectory[i // ba_refinement_level]
                    
                    next_pose = best_trajectory[(i // ba_refinement_level) + 1]

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
        
    else:
        ba_extras = None
        ba_uncertainties = None

    proj = proj / dataset.scale_factor

    # Save outputs if requested (compatible with load_megasam_predictions convention)
    if save_data_path is not None:
        # poses.npy: (N, 4, 4)
        if isinstance(best_trajectory, torch.Tensor):
            poses_np = best_trajectory.detach().cpu().numpy().astype(np.float32)
        else:
            poses_np = np.asarray(best_trajectory, dtype=np.float32)
        np.save(save_data_path / "poses.npy", poses_np)
        print(f"Saved poses to {save_data_path / 'poses.npy'} with shape {poses_np.shape}")

        # intrinsics.npy: (1, 3, 3)
        intr_np = np.asarray(proj, dtype=np.float32)
        intr_np_batched = intr_np[None, ...]
        np.save(save_data_path / "intrinsics.npy", intr_np_batched)
        print(f"Saved intrinsics to {save_data_path / 'intrinsics.npy'} with shape {intr_np_batched.shape}")

        # disps.npy: (N, H, W) from seq_depths (N,1,H,W) or (N,H,W)
        if isinstance(seq_depths, torch.Tensor):
            depths_np = seq_depths.detach().cpu().numpy().astype(np.float32)
        else:
            depths_np = np.asarray(seq_depths, dtype=np.float32)
        if depths_np.ndim == 4 and depths_np.shape[1] == 1:
            depths_np = depths_np[:, 0]
        disps_np = (1.0 / np.clip(depths_np, 1e-6, None)).astype(np.float32)
        np.save(save_data_path / "disps.npy", disps_np)
        print(f"Saved disparities to {save_data_path / 'disps.npy'} with shape {disps_np.shape}")

        # motion_prob.npy: (N, H, W) from ba_uncertainties if available; else ones
        if 'ba_uncertainties' in locals() and ba_uncertainties is not None:
            if isinstance(ba_uncertainties, torch.Tensor):
                mp_np = ba_uncertainties.detach().cpu().numpy().astype(np.float32)
            else:
                mp_np = np.asarray(ba_uncertainties, dtype=np.float32)
            if mp_np.ndim == 4 and mp_np.shape[1] == 1:
                mp_np = mp_np[:, 0]
            elif mp_np.ndim == 5 and mp_np.shape[2] == 1:
                # Handle (N,*,1,H,W) shapes conservatively by selecting the spatial map
                mp_np = mp_np[:, 0, 0]
        else:
            n, _, h_img, w_img = seq_imgs.shape
            mp_np = np.ones((n, h_img, w_img), dtype=np.float32)
        np.save(save_data_path / "motion_prob.npy", mp_np)
        print(f"Saved motion probabilities to {save_data_path / 'motion_prob.npy'} with shape {mp_np.shape}")

        # images.npy: (N, 3, H, W)
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
            "ba_uncertainties": ba_uncertainties, 
            "candidate_trajectories": candidate_trajectories, 
            "pred_labels": proj_labels, 
            "flow_labels": proj_labels,
            "images": seq_imgs, 
            "seq_depths": seq_depths, 
            "seq_flow_occs_fwd": seq_flow_occs_fwd, 
            "seq_flow_occs_bwd": seq_flow_occs_bwd, 
            "uncertainties": uncertainties,
            "best_candidate": best_candidate,
            "focal_length_candidates": pose_result["focal_length_candidates"],
            # Add optimized tracks data for track visualization  
            "ba_optimized_windows_data": ba_extras.get("optimized_windows_data") if ba_extras is not None else None,
            "ba_param_sigma_depth": ba_extras.get("ba_param_sigma_depth") if ba_extras is not None else None,
            # Add global track data for global BA
            "ba_global_track_data": ba_extras.get("global_track_data") if ba_extras is not None else None,
            "keyframe_trajectory": keyframe_trajectory,
        }
        return best_trajectory, proj, extras_dict, ba_extras


@torch.autocast(device_type="cuda", dtype=torch.float32)
@torch.enable_grad()
def ba_refinement(config, initial_trajectory, proj, uncertainties, seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd, device="cuda", track3d=None):
    with_rerun = config.get("with_rerun", True)
    ba_window = config.get("ba_window", 8) # 8
    overlap = config.get("overlap", 6) # 4
    rotation_representation = config.get("rotation_representation", "quaternion")

    max_uncert = config.get("max_uncert", -1)
    use_best = config.get("use_best", False)

    lambda_smoothness = config.get("lambda_smoothness", 200) # 2000 

    global_every_n = config.get("global_every_n", 2)

    n_steps_sliding = config.get("n_steps_sliding", 400) # 500 # 250
    n_steps_global = config.get("n_steps_global", 1000) # 1000 # 100
    n_steps_last_global = config.get("n_steps_last_global", 5000) # 5000
    n_steps_only_focal = config.get("n_steps_only_focal", 0) # 1000

    all_reg_to_zero = config.get("all_reg_to_zero", True)

    track_len = config.get("track_len", 8) # 8
    stride = config.get("stride", 1)
    grid_size = config.get("grid_size", 16) # 16
    long_tracks = config.get("long_tracks", False) # False

    optimize_relatives = config.get("optimize_relatives", False) # False

    lr = config.get("lr", 1e-4)

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
    print(f"all_reg_to_zero: {all_reg_to_zero}")
    print(f"track_len: {track_len}")
    print(f"stride: {stride}")
    print(f"grid_size: {grid_size}")
    print(f"log_interval: {log_interval}")

    if with_rerun:
        rr.init("Sliding Window BA", recording_id=uuid.uuid4())
        rr.connect()

        rr.log("world", rr.Clear(recursive=True))
        rr.log("log", rr.Clear(recursive=True))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    seq_len = seq_flow_occs_fwd.shape[0]

    track_len = min(track_len, seq_len)

    initial_depths_fwd, pixel_tracks_fwd, uncerts_fwd, indices_fwd, depths_fwd, rgbs_fwd, _ = compute_pixel_tracks(seq_flow_occs_fwd.cuda(), uncertainties, seq_depths.cuda(), track_len=track_len, stride=stride, grid_size=grid_size, imgs=seq_imgs.cuda(), long_tracks=long_tracks, track3d=track3d)
    initial_depths_bwd, pixel_tracks_bwd, uncerts_bwd, indices_bwd, depths_bwd, rgbs_bwd, _ = compute_pixel_tracks(seq_flow_occs_bwd.cuda(), uncertainties, seq_depths.cuda(), track_len=track_len, stride=stride, grid_size=grid_size, is_backward=True, imgs=seq_imgs.cuda(), long_tracks=long_tracks, track3d=track3d)
    # initial_depths_fwd, pixel_tracks_fwd, uncerts_fwd, indices_fwd, depths_fwd, rgbs_fwd = compute_pixel_tracks_full_frames(seq_flow_occs_fwd.cuda(), uncertainties, seq_depths.cuda(), track_len=track_len, stride=stride, imgs=seq_imgs.cuda(), long_tracks=long_tracks)
    # initial_depths_bwd, pixel_tracks_bwd, uncerts_bwd, indices_bwd, depths_bwd, rgbs_bwd = compute_pixel_tracks_full_frames(seq_flow_occs_bwd.cuda(), uncertainties, seq_depths.cuda(), track_len=track_len, stride=stride, is_backward=True, imgs=seq_imgs.cuda(), long_tracks=long_tracks)
    initial_depths = torch.cat([initial_depths_fwd, initial_depths_bwd], dim=1)
    pixel_tracks = torch.cat([pixel_tracks_fwd, pixel_tracks_bwd], dim=1)
    uncerts = torch.cat([uncerts_fwd, uncerts_bwd], dim=1)
    indices = torch.cat([indices_fwd, indices_bwd], dim=1)
    depths = torch.cat([depths_fwd, depths_bwd], dim=1) 
    rgbs = torch.cat([rgbs_fwd, rgbs_bwd], dim=1)

    initial_depths = initial_depths_fwd
    pixel_tracks = pixel_tracks_fwd
    uncerts = uncerts_fwd
    indices = indices_fwd
    depths = depths_fwd
    rgbs = rgbs_fwd

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

    ba_param_rot, ba_param_t = pose_to_param(ba_poses_c2w, rotation_representation)

    ba_param_focal_length = (torch.tensor(proj[0, 0] / w * 2, device=device)).log() / 2

    ba_param_inv_depth.requires_grad = True
    ba_param_rot.requires_grad = True
    ba_param_t.requires_grad = True
    ba_param_focal_length.requires_grad = True

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

    while optimized_until < seq_len or not last_global_done:
        do_last_global = optimized_until >= seq_len

        if do_last_global:
            last_global_done = True

        do_global = global_ba_step % global_every_n == 0 or do_last_global

        ba_window_start = max(optimized_until - overlap, 0)
        ba_window_end = min(ba_window_start + ba_window, seq_len)

        seq_ids = torch.arange(seq_len, device=device)

        if not do_global:
            print(f"Optimizing from {ba_window_start} to {ba_window_end}")


            ba_param_inv_depth_mask = (ba_indices[:, :, :, 0, 0] >= optimized_until) & (ba_indices[:, :, :, 0, 0] < ba_window_end)
            ba_param_pose_mask = ((seq_ids >= optimized_until) & (seq_ids < ba_window_end)).view(1, -1, 1)
            loss_mask = (ba_indices[:, :, :, 1:, 0] >= ba_window_start) & (ba_indices[:, :, :, 1:, 0] < ba_window_end)

        else:

            print(f"Optimizing globally util {optimized_until}")

            ba_param_inv_depth_mask = (ba_indices[:, :, :, 0, 0] >= 0) & (ba_indices[:, :, :, 0, 0] < optimized_until)
            ba_param_pose_mask = ((seq_ids >= 0) & (seq_ids < optimized_until)).view(1, -1, 1)
            loss_mask = (ba_indices[:, :, :, 1:, 0] >= 0) & (ba_indices[:, :, :, 1:, 0] < optimized_until)

        if all_reg_to_zero and do_last_global:
            lambda_depth = 0
            lambda_pose = 0


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

        optimizer = torch.optim.Adam([ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length], lr=lr)

        if not do_global:
            n_steps_ = n_steps_sliding
        else:
            if do_last_global:
                n_steps_ = n_steps_last_global
            else:
                n_steps_ = n_steps_global

        pbar = tqdm(range(n_steps_))

        for step in pbar:
            optimizer.zero_grad()

            # Detach relevant parameters:
            ba_param_inv_depth_d = ba_param_inv_depth.clone()
            ba_param_inv_depth_d[~ba_param_inv_depth_mask].detach_()

            ba_param_rot_d = ba_param_rot.clone()            
            ba_param_rot_d[~ba_param_pose_mask.expand_as(ba_param_rot_d)] = ba_param_rot_d[~ba_param_pose_mask.expand_as(ba_param_rot_d)].detach()
            # ba_param_rot_d[~ba_param_pose_mask.expand_as(ba_param_rot_d)].detach_()

            ba_param_t_d = ba_param_t.clone()
            ba_param_t_d[~ba_param_pose_mask.expand_as(ba_param_t_d)] = ba_param_t_d[~ba_param_pose_mask.expand_as(ba_param_t_d)].detach()
            # ba_param_t_d[~ba_param_pose_mask.expand_as(ba_param_t_d)].detach_()

            repr_loss, smoothness_loss, ba_proj, xyzh_world = compute_loss(ba_param_inv_depth_d, ba_param_rot_d, ba_param_t_d, ba_param_focal_length, pixel_tracks, ba_indices, ba_uncerts, w, h, loss_mask, max_uncert)

            if use_best:
                thresh = torch.quantile(repr_loss[repr_loss > 0], 0.9)
                repr_loss[repr_loss > thresh] = 0

            total_loss = repr_loss.mean() + smoothness_loss.mean() * (lambda_smoothness if do_global else 0)

            total_loss.backward()

            # clip gradients

            for param in [ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length]:
                param.grad[torch.isnan(param.grad) | torch.isinf(param.grad)] = 0

            
            if do_global and step < n_steps_only_focal:
                ba_param_rot.grad *= 0
                ba_param_t.grad *= 0

            torch.nn.utils.clip_grad_norm_([ba_param_inv_depth, ba_param_rot, ba_param_t, ba_param_focal_length], .1)


            optimizer.step()

            # pbar.set_postfix({"total_loss": total_loss.item(), "loss": loss.item(), "pose_loss": pose_loss.item(), "depth_loss": depth_loss.item(), "depth_repr_loss": depth_loss_repr.mean().item(),"smoothness_loss": smoothness_loss.item(), "fx": ba_proj[0, 0, 0].item(), "fy": ba_proj[0, 1, 1].item(), "uncert_mean": pt_mean.item(), "uncert_std": pt_std.item()})
            pbar.set_postfix({"l": total_loss.item(), "l_s": smoothness_loss.item(), "fx": ba_proj[0, 0, 0].item(), "fy": ba_proj[0, 1, 1].item()})

            log_step += 1

            if log_step % log_interval == 0 and with_rerun:
                ba_poses_c2w = param_to_pose(ba_param_rot, ba_param_t)

                if optimize_relatives:
                    ba_poses_c2w_ = [ba_poses[:, 0]]
                    for i in range(1, seq_len):
                        ba_poses_c2w_.append(torch.inverse(ba_poses_c2w[:, i-1]) @ ba_poses_c2w_[-1])

                    ba_poses_c2w = torch.stack(ba_poses_c2w_, dim=1)

                log_ba_imgs(ba_imgs, timestep=seq_len + rerun_offset * 2 + (log_step // log_interval), frame_idx=ba_window_end-1)
                
                log_ba_state(
                    ba_poses_c2w,
                    points=xyzh_world[:, ::1, :3, 0].permute(0, 2, 1),
                    point_colors=rgbs[:, :, :, :1, :].reshape(1, -1, 3)[:, ::1],
                    timestep=seq_len + rerun_offset * 2  + (log_step // log_interval),
                    max_dist=10,
                )

        if not do_global:
            optimized_until = ba_window_end

        global_ba_step += 1

    ba_poses_c2w = param_to_pose(ba_param_rot, ba_param_t)

    ba_trajectory = ba_poses_c2w.detach()
    ba_proj = make_normalized_proj((ba_param_focal_length * 2).exp(), w/h)[0].cpu().detach()

    # Unnormalize proj
    ba_proj[0, 0] = (ba_proj[0, 0] / 2) * w
    ba_proj[1, 1] = (ba_proj[1, 1] / 2) * h
    ba_proj[0, 2] = (ba_proj[0, 2] + 1) / 2 * w
    ba_proj[1, 2] = (ba_proj[1, 2] + 1) / 2 * h

    ba_extras = {
        "ba_param_inv_depth": ba_param_inv_depth.detach().cpu(),
        "ba_param_rot": ba_param_rot.detach().cpu(),
        "ba_param_t": ba_param_t.detach().cpu(),
        "ba_param_focal_length": ba_param_focal_length.detach().cpu(),
        "ba_trajectory": ba_trajectory.cpu(),
        "indices": ba_indices.cpu(),
        "uncerts": ba_uncerts.cpu(),
        "initial_depths": initial_depths.cpu(),
        "pixel_tracks": pixel_tracks.cpu(),
    }

    return ba_trajectory[0], ba_proj, ba_extras
