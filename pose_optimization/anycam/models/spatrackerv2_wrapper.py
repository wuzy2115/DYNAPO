import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import cv2
from typing import Optional, List
from PIL import Image

from anycam.common.image_processor import make_image_processor
from anycam.models import make_depth_predictor, make_depth_aligner
from dotdict import dotdict

logger = logging.getLogger(__name__)


class SpaTrackerV2Wrapper(nn.Module):
    """
    Wrapper to integrate SpaTrackerV2 predictions saved on disk into the AnyCam pipeline.
    This class only loads precomputed SpaTrackerV2 results; it does not run the model.
    """

    def __init__(self, config) -> None:
        super().__init__()

        # SpaTrackerV2 provides depth via its outputs by default
        self.use_provided_depth = config.get("use_provided_depth", True)
        self.use_provided_flow = config.get("use_provided_flow", False)
        self.use_provided_proj = config.get("use_provided_proj", False)
        self.use_provided_masks = config.get("use_provided_masks", False)
        self.mask_path = config.get("mask_path", '/data/zhuoyuan/DAVIS/Annotations/480p')

        self.flow_model = config.get("flow_model", "unimatch")

        if not self.use_provided_depth:
            self.depth_predictor = make_depth_predictor(config["depth_predictor"])  # pragma: no cover
            self.depth_aligner = make_depth_aligner(config["depth_aligner"])       # pragma: no cover
            for param in self.depth_predictor.parameters():
                param.requires_grad = False

        # Image processor for optical flow / occlusion
        self.image_processor = make_image_processor(
            {"type": "flow_occlusion"},
            flow_model=self.flow_model,
            use_provided_flow=self.use_provided_flow,
            pair_mode="sequential",
        )

        self.renderer = dotdict({"net": None})
        self.renderer.net = None

        self.z_near = config.get("z_near", 0.1)
        self.z_far = config.get("z_far", 10)

        self._counter = 0

    def forward(self, x):  # pragma: no cover
        raise NotImplementedError(
            "SpaTrackerV2Wrapper does not support forward pass. Use load_spatrackerv2_predictions instead."
        )

    def _find_result_npz(self, root: Path, seq_name: str) -> Path:
        """Locate SpaTrackerV2 result npz file under given root and sequence name.
        Expected locations (checked in order):
          - root/seq_name/result.npz
          - root/seq_name.npz
          - root/seq_name/**/result.npz (recursive)
          - root/result.npz (if root already points to a sequence directory)
        """
        candidates = [
            root / seq_name / "result.npz",
            root / f"{seq_name}.npz",
            root / "result.npz",
        ]
        for c in candidates:
            if c.exists():
                return c
        # fallback: recursive search
        try:
            matches = list((root / seq_name).rglob("result.npz"))
            if matches:
                return matches[0]
        except Exception:
            pass
        raise FileNotFoundError(f"SpaTrackerV2 result.npz not found under {root} for sequence {seq_name}")

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
                        logger.info(
                            f"Loaded {mask_tensor.shape[0]} segmentation masks from video {video_path} with shape: {mask_tensor.shape}"
                        )
                else:
                    logger.error(f"Cannot open mask video file: {video_path}")
            else:
                logger.warning(f"No mask images or videos found in {mask_dir}")
        else:
            mask_files.sort(key=lambda x: x.name)
            loaded_masks: List[np.ndarray] = []
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
                logger.info(f"Loaded {len(loaded_masks)} segmentation masks from {mask_dir} with shape: {mask_tensor.shape}")
            else:
                logger.warning(f"No valid mask images could be loaded from {mask_dir}")

        return mask_tensor

    def load_spatrackerv2_predictions(self, spatrackerv2_data_path, seq_name):
        """
        Load SpaTrackerV2 predictions from a saved npz file.

        Returns dict containing:
          - poses: (N, 4, 4) camera-to-world matrices
          - intrinsics: (3, 3) camera intrinsics (first frame)
          - motion_prob: (N,) per-frame dynamic probability (mean across tracks)
          - disps: (N, H, W) per-frame disparity maps (from depths)
          - image_size: (width, height)
          - images: (N, 3, H, W) float tensor in [0, 1] if available
        """
        data_root = Path(spatrackerv2_data_path)
        npz_path = self._find_result_npz(data_root, seq_name)

        logger.info(f"Loading SpaTrackerV2 predictions from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        # Required fields
        if "intrinsics" not in data or "extrinsics" not in data or "depths" not in data:
            raise ValueError(
                f"SpaTrackerV2 npz missing required keys. Found: {list(data.keys())}"
            )

        intrs = data["intrinsics"]  # shape: (N, 3, 3) or (3, 3)
        extrs = data["extrinsics"]  # shape: (N, 4, 4) camera-to-world
        depths = data["depths"]      # shape: (N, H, W)

        # Optional fields
        dynamic_scores = data["dynamic_scores"] if "dynamic_scores" in data else None
        video = data["video"] if "video" in data else None  # (N, C, H, W) in [0,1]

        # Camera-to-world poses
        if extrs.ndim == 3:
            # c2w_list = [np.linalg.inv(extrs[i]).astype(np.float32) for i in range(extrs.shape[0])]
            c2w_list = [extrs[i].astype(np.float32) for i in range(extrs.shape[0])]
        else:
            raise ValueError("Expected extrinsics with shape (N, 4, 4)")

        poses = torch.tensor(np.stack(c2w_list, axis=0), dtype=torch.float32)

        # Intrinsics: if per-frame provided, use the first (assume shared camera)
        if intrs.ndim == 3:
            intrinsics = torch.tensor(intrs[0], dtype=torch.float32)
        elif intrs.ndim == 2:
            intrinsics = torch.tensor(intrs, dtype=torch.float32)
        else:
            raise ValueError("Invalid intrinsics shape; expected (N,3,3) or (3,3)")

        # Disparity from depth
        depths_safe = np.clip(depths, 1e-6, None)
        disps = torch.tensor(1.0 / depths_safe, dtype=torch.float32)

        # Image size
        if video is not None and video.ndim == 4:
            _, c, h, w = video.shape
            image_size = (w, h)
            images = torch.tensor(video, dtype=torch.float32)
        else:
            # Fallback to depths shape
            n, h, w = depths.shape
            image_size = (w, h)
            images = None

        mask_tensor = self._load_masks_from_path(seq_name)
        if mask_tensor is not None:
            if mask_tensor.shape[0] == poses.shape[0]:
                motion_prob = mask_tensor
            else:
                logger.warning(
                    "Mask tensor length %s does not match number of poses %s for sequence %s",
                    mask_tensor.shape[0], poses.shape[0], seq_name
                )

        result = {
            "poses": poses,
            "intrinsics": intrinsics,
            "motion_prob": motion_prob,
            "disps": disps,
            "image_size": image_size,
        }
        if images is not None:
            result["images"] = images

        return result


