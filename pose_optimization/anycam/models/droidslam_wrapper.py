import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from copy import deepcopy
import logging
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, List

from anycam.trainer import normalize_proj
from anycam.utils.bundle_adjustment import compute_depth_flow
from anycam.common.image_processor import make_image_processor
from anycam.models import make_depth_predictor, make_depth_aligner
from dotdict import dotdict

logger = logging.getLogger(__name__)


class DroidSlamWrapper(nn.Module):
    """
    Wrapper class for integrating DroidSLAM predictions into AnyCam pipeline.
    This class mirrors MegaSamWrapper's interfaces but only contains DroidSLAM-specific utilities.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.use_provided_depth = config.get("use_provided_depth", False)
        self.use_provided_flow = config.get("use_provided_flow", False)
        self.use_provided_proj = config.get("use_provided_proj", False)
        self.use_provided_masks = config.get("use_provided_masks", True)
        self.mask_path = config.get("mask_path", '/data/zhuoyuan/DAVIS/Annotations/480p')
        self.mask_visualization_output = config.get("mask_visualization_output", None)
        self.mask_visualization_fps = config.get("mask_visualization_fps", 10)

        self.flow_model = config.get("flow_model", "unimatch")

        if not self.use_provided_depth:
            # Initialize depth predictor and aligner (same as AnyCamWrapper)
            self.depth_predictor = make_depth_predictor(config["depth_predictor"])
            self.depth_aligner = make_depth_aligner(config["depth_aligner"])

            # Freeze depth predictor parameters
            for param in self.depth_predictor.parameters():
                param.requires_grad = False

        # Initialize image processor for flow/occlusion computation
        self.image_processor = make_image_processor(
            {"type": "flow_occlusion"},
            flow_model=self.flow_model,
            use_provided_flow=self.use_provided_flow,
            pair_mode="sequential"
        )

        self.renderer = dotdict({"net": None})
        self.renderer.net = None

        self.z_near = config.get("z_near", 0.1)
        self.z_far = config.get("z_far", 10)

        self._counter = 0

    def _load_masks_from_path(self, seq_name: str) -> Optional[torch.Tensor]:
        if not self.use_provided_masks:
            return None

        if self.mask_path is None:
            raise ValueError("mask_path must be provided when use_provided_masks is True")

        mask_dir = Path(self.mask_path) / seq_name
        mask_files: List[Path] = []
        mask_tensor: Optional[torch.Tensor] = None

        if mask_dir.exists():
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                mask_files.extend(list(mask_dir.glob(f'*{ext}')))
                mask_files.extend(list(mask_dir.glob(f'*{ext.upper()}')))

        if not mask_files:
            video_path = Path(self.mask_path) / f"{seq_name}/segmentation_mask.mp4"
            if video_path is not None and video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    loaded: List[np.ndarray] = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        mask_bin = (np.any(frame > 127, axis=2)).astype(np.float32)
                        loaded.append(mask_bin)
                    cap.release()
                    if loaded:
                        masks_np = np.stack(loaded, axis=0)
                        mask_tensor = torch.tensor(masks_np, dtype=torch.float32)
                        self._save_mask_visualization(masks_np, seq_name, Path(video_path))
                        logger.info(
                            f"Loaded {mask_tensor.shape[0]} segmentation masks from video {video_path} with shape: {mask_tensor.shape}"
                        )
                else:
                    logger.error(f"Cannot open mask video file: {video_path}")
            else:
                logger.warning(f"No mask images or videos found in {mask_dir}")
        else:
            mask_files.sort(key=lambda x: x.name)
            from PIL import Image
            loaded_masks: list[np.ndarray] = []
            for mask_file in mask_files:
                try:
                    mask_img = Image.open(mask_file)
                    if mask_img.mode != 'L':
                        mask_img = mask_img.convert('L')
                    mask_array = np.array(mask_img, dtype=np.float32)
                    if mask_array.max() > 1.0:
                        mask_array = mask_array / 255.0
                    loaded_masks.append(mask_array)
                except Exception as exc:
                    logger.error(f"Failed to load mask image {mask_file}: {exc}")
            if loaded_masks:
                masks_np = np.stack(loaded_masks, axis=0)
                mask_tensor = torch.tensor(masks_np, dtype=torch.float32)
                self._save_mask_visualization(masks_np, seq_name, mask_dir)
                logger.info(f"Loaded {len(loaded_masks)} segmentation masks from {mask_dir} with shape: {mask_tensor.shape}")
            else:
                logger.warning(f"No valid mask images could be loaded from {mask_dir}")

        return mask_tensor

    def visualize_depths(self, images, depths, output_path, fps=30, colormap='viridis'):
        if images.dim() == 4:
            bs, c, h, w = images.shape
        else:
            raise ValueError(f"Expected images to be 4D tensor, got shape {images.shape}")

        if depths.dim() == 4:
            depths = depths.squeeze(1)
        elif depths.dim() != 3:
            raise ValueError(f"Expected depths to be 3D or 4D tensor, got shape {depths.shape}")

        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
        else:
            images_np = images

        if isinstance(depths, torch.Tensor):
            depths_np = depths.detach().cpu().numpy()
        else:
            depths_np = depths

        if images_np.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)

        images_np = np.transpose(images_np, (0, 2, 3, 1))
        images_bgr = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np])

        depths_finite = depths_np.copy()
        is_finite = np.isfinite(depths_finite)

        if not np.any(is_finite):
            logger.warning("No finite depth values found, using fallback normalization")
            depths_normalized = np.zeros_like(depths_finite)
        else:
            finite_depths = depths_finite[is_finite]
            depth_min = finite_depths.min()
            depth_max = finite_depths.max()
            depths_normalized = np.zeros_like(depths_finite)
            depths_normalized[is_finite] = (finite_depths - depth_min) / (depth_max - depth_min + 1e-8)
            depths_normalized[~is_finite] = 0.0
            logger.info(f"Depth range (finite values): [{depth_min:.3f}, {depth_max:.3f}]")
            logger.info(f"Finite depth pixels: {np.sum(is_finite)}/{depths_finite.size} ({100*np.sum(is_finite)/depths_finite.size:.1f}%)")

        cmap = cm.get_cmap(colormap)
        depths_colored = []
        for depth_frame in depths_normalized:
            depth_colored = cmap(depth_frame)[:, :, :3]
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            depths_colored.append(depth_colored)
        depths_colored = np.array(depths_colored)

        h, w = images_bgr.shape[1:3]
        combined_width = w * 2
        combined_height = h

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

        for i in range(len(images_bgr)):
            img_frame = images_bgr[i]
            depth_frame = depths_colored[i]
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            combined_frame[:, :w] = img_frame
            combined_frame[:, w:] = depth_frame
            cv2.putText(combined_frame, 'Original', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, 'Depth', (w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(combined_frame)

        out.release()
        logger.info(f"Depth visualization video saved to: {output_path}")

    def generate_depths(self, images, chunk_size=16, visualize=False, viz_output_path=None, viz_fps=30, viz_colormap='viridis'):
        bs, _, h, w = images.shape
        if bs <= chunk_size:
            with torch.no_grad():
                output = self.depth_predictor.infer(images)
            return output

        all_outputs = []
        with torch.no_grad():
            for i in range(0, bs, chunk_size):
                end_idx = min(i + chunk_size, bs)
                chunk = images[i:end_idx]
                chunk_output = self.depth_predictor.infer(chunk)
                all_outputs.append(chunk_output)

        if not all_outputs:
            raise ValueError("No outputs to concatenate")

        output_keys = all_outputs[0].keys()
        final_output = {}
        for key in output_keys:
            values = [chunk_output[key] for chunk_output in all_outputs]
            if isinstance(values[0], torch.Tensor):
                final_output[key] = torch.cat(values, dim=0)
            else:
                final_output[key] = values

        if 'intrinsics' in final_output:
            final_output['intrinsics'] = final_output['intrinsics'].mean(dim=0)

        if visualize:
            if viz_output_path is None:
                viz_output_path = "depths_droidslam.mp4"
            if 'depth' in final_output:
                depths = final_output['depth']
                self.visualize_depths(
                    images=images,
                    depths=depths,
                    output_path=viz_output_path,
                    fps=viz_fps,
                    colormap=viz_colormap
                )
        return final_output

    def load_droidslam_predictions(self, droidslam_data_path, seq_name):
        data_file = Path(droidslam_data_path) / seq_name / f"{seq_name}.npy"

        if data_file is None:
            raise FileNotFoundError(f"Could not find DroidSLAM data for sequence {seq_name} in {droidslam_data_path}")

        raw_data = np.load(data_file, allow_pickle=True)
        if isinstance(raw_data, np.lib.npyio.NpzFile):
            data_dict = {k: raw_data[k] for k in raw_data.files}
        elif isinstance(raw_data, np.ndarray) and raw_data.shape == ():
            data_dict = raw_data.item()
        elif isinstance(raw_data, dict):
            data_dict = raw_data
        else:
            data_dict = raw_data.item()

        if not isinstance(data_dict, dict):
            raise ValueError(f"Unexpected DroidSLAM data format in {data_file}")

        poses_np = data_dict.get('traj_c2w')
        if poses_np is None:
            poses_np = data_dict.get('traj_c2w_mat')
        if poses_np is None:
            poses_np = data_dict.get('map_c2w')
        if poses_np is None:
            poses_np = data_dict.get('map_c2w_mat')
        if poses_np is None:
            raise KeyError("DroidSLAM data does not contain trajectory poses")

        poses_np = np.asarray(poses_np)

        if poses_np.ndim == 3 and poses_np.shape[-2:] == (3, 4):
            bottom_row = np.tile(np.array([[0, 0, 0, 1]], dtype=poses_np.dtype), (poses_np.shape[0], 1, 1))
            poses_np = np.concatenate([poses_np, bottom_row], axis=1)
        elif poses_np.ndim == 3 and poses_np.shape[-2:] == (4, 4):
            pass
        elif poses_np.ndim == 2 and poses_np.shape[-1] in (7, 8):
            from lietorch import SE3
            poses_np = SE3(torch.tensor(poses_np)).matrix().numpy()

        poses = torch.tensor(poses_np, dtype=torch.float32)

        intrinsics_np = data_dict.get('intrinsics') * 8 # droid-slam downsample 8x for inference
        if intrinsics_np is None:
            raise KeyError("DroidSLAM data does not contain 'intrinsics'")

        intrinsics_np = np.asarray(intrinsics_np)
        if intrinsics_np.shape == (4,):
            fx, fy, cx, cy = intrinsics_np
            intrinsics_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics = torch.tensor(intrinsics_np, dtype=torch.float32)

        masks_np = data_dict.get('masks')
        if masks_np is not None:
            masks_np = np.asarray(masks_np).astype(np.float32)
            motion_prob = torch.tensor(masks_np, dtype=torch.float32)
        else:
            img_shape = data_dict.get('img_shape')
            if img_shape is not None and len(img_shape) == 2:
                motion_prob = torch.ones((poses.shape[0], img_shape[0], img_shape[1]), dtype=torch.float32)
            else:
                motion_prob = torch.ones((poses.shape[0], 1, 1), dtype=torch.float32)

        disps_np = data_dict.get('disps') or data_dict.get('disps_up')
        disps = torch.tensor(disps_np, dtype=torch.float32) if disps_np is not None else None

        mask_tensor = self._load_masks_from_path(seq_name)
        if mask_tensor is not None:
            motion_prob = mask_tensor

        return {
            'poses': poses,
            'intrinsics': intrinsics,
            'motion_prob': motion_prob,
            'disps': disps
        }

    def forward(self, data, droidslam_data=None, **kwargs):
        if droidslam_data is None:
            raise ValueError("DroidSLAM data must be provided for forward pass")

        data = dict(data)

        images = data["imgs"]
        n, f, c, h, w = images.shape
        device = images.device

        droid_intrinsics = droidslam_data['intrinsics'].to(device)
        gt_projs = normalize_proj(droid_intrinsics.unsqueeze(0), h, w)

        if droidslam_data.get('disps') is not None:
            droid_disps = droidslam_data['disps'][:f].to(device)
            img_h, img_w = images.shape[-2:]
            disp_h, disp_w = droid_disps.shape[-2:]
            if disp_h != img_h or disp_w != img_w:
                if droid_disps.dim() == 3:
                    droid_disps = torch.nn.functional.interpolate(
                        droid_disps.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False
                    ).squeeze(1)
            depths = 1.0 / droid_disps.clamp_min(1e-6)
            depths = depths.unsqueeze(0).unsqueeze(2)
        elif self.use_provided_depth:
            depths = data["depths"]
        else:
            depth_in = images.view(n * f, c, h, w)
            with torch.no_grad():
                depths, depth_features = self.depth_predictor(depth_in, return_features=True)
            depths = depths[0]
            depths = 1 / depths.clamp_min(1e-3).view(n, -1, 1, *depths.shape[-2:])

        data["pred_depths"] = depths * .1
        data["pred_depths_list"] = [depths]

        images_ip_fwd, images_ip_bwd = self.image_processor(images * 2 - 1, data=data)
        flow_occ_fwd = images_ip_fwd[:, :, 3:6]
        flow_occ_bwd = images_ip_bwd[:, :, 3:6]

        droid_poses = droidslam_data['poses'][:f].to(device).unsqueeze(0)
        droid_uncertainties = droidslam_data['motion_prob'][:f].to(device).unsqueeze(0)

        data["images_ip"] = images_ip_fwd
        data["valid"] = images_ip_fwd[:, :, 5:6] > 0.5
        data["proc_poses"] = droid_poses
        data["proc_projs"] = gt_projs
        data["uncertainties"] = droid_uncertainties.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        data["weights_proc"] = data["uncertainties"]
        data["scaled_depths"] = [depths]

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        if self.training:
            self._counter += 1

        return data

    def _save_mask_visualization(self, masks_np: np.ndarray, seq_name: str, source_path: Path):
        if self.mask_visualization_output is None:
            return
        masks_to_save = masks_np.copy()
        if masks_to_save.max() > 1.0:
            masks_to_save = np.clip(masks_to_save, 0, 255)
        masks_to_save = (masks_to_save > 0).astype(np.uint8) * 255
        masks_to_save = np.expand_dims(masks_to_save, axis=-1)
        masks_to_save = np.repeat(masks_to_save, 3, axis=-1)
        try:
            output_path = Path(self.mask_visualization_output)
            if output_path.is_dir() or str(output_path).endswith(('/', '\\')):
                output_path = output_path / f"{seq_name}_masks.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {'source': str(source_path), 'sequence': seq_name}
            imageio.mimwrite(
                output_path,
                masks_to_save,
                fps=self.mask_visualization_fps,
                codec='libx264',
                quality=8,
                macro_block_size=None,
                metadata=metadata
            )
            logger.info(f"Mask visualization video saved to {output_path}")
        except Exception as e:
            logger.error(
                f"Failed to save mask visualization video to {self.mask_visualization_output}: {e}"
            )

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


