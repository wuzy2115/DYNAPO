import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import cv2
import imageio.v2 as imageio

from anycam.common.image_processor import make_image_processor
from anycam.models import make_depth_predictor, make_depth_aligner
from dotdict import dotdict

logger = logging.getLogger(__name__)


class TTT3RWrapper(nn.Module):
    """
    Wrapper class for integrating TTT3R predictions into AnyCam pipeline.
    This class loads precomputed TTT3R results written by ref_TTT3R/demo.py.
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
            # Initialize depth predictor and aligner (same as MegaSam/VGGT wrappers)
            self.depth_predictor = make_depth_predictor(config["depth_predictor"]) 
            self.depth_aligner = make_depth_aligner(config["depth_aligner"]) 

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


    def load_ttt3r_predictions(self, ttt3r_data_path, seq_name):
        """
        Load TTT3R predictions from the output directory structure created by ref_TTT3R/demo.py.

        Layout expected:
          <ttt3r_data_path>/<seq_name>/
            - camera/000000.npz (pose=4x4 c2w, intrinsics=3x3)
            - depth/000000.npy  (H, W) metric depth
            - conf/000000.npy   (H, W) confidence
            - color/000000.png  (H, W, 3) RGB

        Returns dict with:
          - poses: (N, 4, 4) torch.float32 camera-to-world
          - intrinsics: (3, 3) torch.float32 (first frame; others assumed shared)
          - motion_prob: (N, H, W) or (N, 1, H, W) torch.float32 if available; else None
          - disps: (N-optional, H, W) disparities (1/depth) torch.float32 if depths available; else None
          - image_size: (width, height)
        """
        root = Path(ttt3r_data_path) / seq_name
        camera_dir = root / "camera"
        depth_dir = root / "depth"
        conf_dir = root / "conf"
        color_dir = root / "color"

        if not camera_dir.exists():
            raise FileNotFoundError(f"TTT3R camera directory not found: {camera_dir}")

        # Determine image size from color if available
        image_width = None
        image_height = None
        if color_dir.exists():
            color_files = sorted(list(color_dir.glob("*.png")) + list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.jpeg")))
            if color_files:
                img0 = cv2.imread(str(color_files[0]))
                if img0 is None:
                    raise ValueError(f"Failed to read color image: {color_files[0]}")
                image_height, image_width = img0.shape[:2]

        # Load camera parameters and poses
        cam_files = sorted(list(camera_dir.glob("*.npz")))
        if len(cam_files) == 0:
            raise ValueError(f"No camera npz files found in {camera_dir}")

        poses_list = []
        intrinsics_list = []
        for f in cam_files:
            data = np.load(f)
            pose = data["pose"].astype(np.float32)  # camera-to-world
            intrins = data["intrinsics"].astype(np.float32)  # 3x3
            poses_list.append(pose)
            intrinsics_list.append(intrins)

        poses = torch.tensor(np.stack(poses_list), dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics_list[0], dtype=torch.float32)

        # Load depths and convert to disparities if available
        disps = None
        if depth_dir.exists():
            depth_files = sorted(list(depth_dir.glob("*.npy")))
            if depth_files:
                depths = []
                for f in depth_files:
                    d = np.load(f)
                    depths.append(d)
                depths_np = np.stack(depths).astype(np.float32)
                # numeric guard to avoid division by zero
                disps_np = 1.0 / np.clip(depths_np, 1e-6, None)
                disps = torch.tensor(disps_np, dtype=torch.float32)

                # If image size unknown, infer from depths
                if image_width is None or image_height is None:
                    image_height, image_width = depths_np.shape[-2:]

        # Load confidences as motion_prob if available
        motion_prob = None
        if conf_dir.exists():
            conf_files = sorted(list(conf_dir.glob("*.npy")))
            if conf_files:
                confs = []
                for f in conf_files:
                    c = np.load(f).astype(np.float32)
                    confs.append(c)
                confs_np = np.stack(confs)
                motion_prob = torch.tensor(confs_np, dtype=torch.float32)

                if image_width is None or image_height is None:
                    image_height, image_width = confs_np.shape[-2:]

        # Optionally load segmentation masks
        if self.use_provided_masks:
            if self.mask_path is None:
                raise ValueError("mask_path must be provided when use_provided_masks is True")
            
            mask_dir = Path(self.mask_path) / seq_name
            mask_files = []
            masks = None
            if mask_dir.exists():
                # Find all mask image files and sort them
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                    mask_files.extend(list(mask_dir.glob(f'*{ext}')))
                    mask_files.extend(list(mask_dir.glob(f'*{ext.upper()}')))
            
            if not mask_files:
                video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg']
                video_path = Path(self.mask_path) / f"{seq_name}/segmentation_mask.mp4"
                
                if not video_path.exists():
                    logger.warning(f"No mask images or videos found in {mask_dir}")
                else:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        logger.error(f"Cannot open mask video file: {video_path}")
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        loaded = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            mask_bin = (np.any(frame > 127, axis=2)).astype(np.float32)
                            loaded.append(mask_bin)
                        cap.release()
                        if len(loaded) == 0:
                            logger.warning(f"No frames decoded from mask video: {video_path}")
                        else:
                            masks_np = np.stack(loaded, axis=0)
                            masks = torch.tensor(masks_np, dtype=torch.float32)
                            self._save_mask_visualization(masks_np, seq_name, Path(video_path))
                            logger.info(
                                f"Loaded {masks.shape[0]} segmentation masks from video {video_path} with shape: {masks.shape}"
                            )
            else:
                mask_files.sort(key=lambda x: x.name)
                from PIL import Image
                loaded_masks = []
                for mask_file in mask_files:
                    try:
                        mask_img = Image.open(mask_file)
                        if mask_img.mode != 'L':
                            mask_img = mask_img.convert('L')
                        mask_array = np.array(mask_img, dtype=np.float32)
                        if mask_array.max() > 1.0:
                            mask_array = mask_array / 255.0
                        loaded_masks.append(mask_array)
                    except Exception as e:
                        logger.error(f"Failed to load mask image {mask_file}: {e}")
                        continue
                if loaded_masks:
                    masks_np = np.stack(loaded_masks, axis=0)
                    masks = torch.tensor(masks_np, dtype=torch.float32)
                    self._save_mask_visualization(masks_np, seq_name, mask_dir)
                    logger.info(f"Loaded {len(loaded_masks)} segmentation masks from {mask_dir} with shape: {masks.shape}")
                else:
                    logger.warning(f"No valid mask images could be loaded from {mask_dir}")
            if 'masks' in locals() and masks is not None:
                motion_prob = masks

        if image_width is None or image_height is None:
            logger.warning("TTT3R image size not available; defaulting to (0, 0)")
            image_width, image_height = 0, 0

        logger.info(f"Loaded TTT3R data: poses {poses.shape}, intrinsics {intrinsics.shape}")
        return {
            'poses': poses,
            'intrinsics': intrinsics,
            'motion_prob': motion_prob,
            'disps': disps,
            'image_size': (image_width, image_height),
        }


    def generate_depths(self, images, chunk_size=16, visualize=False, viz_output_path=None, viz_fps=30, viz_colormap='viridis'):
        if self.use_provided_depth:
            raise ValueError("generate_depths called but use_provided_depth is True")
        device = images.device
        bs, c, h, w = images.shape
        all_depths = []
        all_masks = []
        all_intrinsics = []
        for i in range(0, bs, chunk_size):
            chunk_imgs = images[i:i+chunk_size]
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=True):
                    output = self.depth_predictor(chunk_imgs)
            depth = output['depth']
            mask = output['mask']
            intrinsics = output['intrinsics']
            all_depths.append(depth)
            all_masks.append(mask)
            all_intrinsics.append(intrinsics)
        final_depth = torch.cat(all_depths, dim=0)
        final_mask = torch.cat(all_masks, dim=0)
        final_intrinsics = torch.cat(all_intrinsics, dim=0)
        final_output = {
            'depth': final_depth,
            'mask': final_mask,
            'intrinsics': final_intrinsics
        }
        if visualize and viz_output_path is not None:
            from anycam.models.megasam_wrapper import MegaSamWrapper
            MegaSamWrapper.visualize_depths(None, images, final_depth, viz_output_path, viz_fps, viz_colormap)
        return final_output


    def forward(self, x):
        raise NotImplementedError("TTT3RWrapper does not support forward pass. Use load_ttt3r_predictions instead.")

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



