import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import pycolmap
import cv2
import imageio.v2 as imageio

from anycam.common.image_processor import make_image_processor
from anycam.models import make_depth_predictor, make_depth_aligner
from dotdict import dotdict

logger = logging.getLogger(__name__)


class VGGTWrapper(nn.Module):
    """
    Wrapper class for integrating VGGT pose predictions into AnyCam pipeline.
    This class loads precomputed VGGT results from COLMAP format.
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
            # Initialize depth predictor and aligner (same as MegaSamWrapper)
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


    def load_vggt_predictions(self, vggt_data_path, seq_name):
        """
        Load VGGT predictions from COLMAP format.
        
        Args:
            vggt_data_path: Path to the directory containing VGGT results
            seq_name: Name of the sequence
            
        Returns:
            dict containing:
                - poses: Camera poses (N, 4, 4) - camera-to-world matrices
                - intrinsics: Camera intrinsics (3, 3)
                - motion_prob: Motion probabilities/uncertainties (N,) - default to ones
                - image_size: (width, height) tuple for the camera calibration
        """
        # VGGT saves results in COLMAP format at: vggt_data_path/seq_name/sparse/
        data_dir = Path(vggt_data_path) / seq_name / "sparse"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"VGGT data directory not found: {data_dir}")
        
        logger.info(f"Loading VGGT predictions from COLMAP format: {data_dir}")
        
        # Read COLMAP reconstruction
        try:
            reconstruction = pycolmap.Reconstruction(str(data_dir))
        except Exception as e:
            raise RuntimeError(f"Failed to read COLMAP reconstruction from {data_dir}: {e}")
        
        cameras = reconstruction.cameras
        images = reconstruction.images
        
        if len(cameras) == 0:
            raise ValueError(f"No cameras found in COLMAP reconstruction: {data_dir}")
        if len(images) == 0:
            raise ValueError(f"No images found in COLMAP reconstruction: {data_dir}")
        
        logger.info(f"Loaded {len(images)} images and {len(cameras)} cameras from COLMAP")
        
        poses = []
        projs = []
        
        # Sort images by name to maintain frame order
        sorted_images = sorted(images.values(), key=lambda im: im.name)
        
        # Get the first camera to extract image dimensions and check if shared camera
        first_camera = cameras[sorted_images[0].camera_id]
        vggt_image_width = first_camera.width
        vggt_image_height = first_camera.height
        
        logger.info(f"VGGT camera calibration is for image size: {vggt_image_width}x{vggt_image_height}")
        
        for image in sorted_images:
            camera = cameras[image.camera_id]
            
            cam_from_world = image.cam_from_world
            cam_to_world = cam_from_world
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = cam_to_world.rotation.matrix()
            pose[:3, 3] = np.asarray(cam_to_world.translation)
            
            # Extract camera intrinsics
            cam_params = camera.params
            
            proj = np.eye(3)
            
            # Handle different COLMAP camera models
            if camera.model.name == "SIMPLE_PINHOLE":
                # params: [f, cx, cy]
                if len(cam_params) == 3:
                    proj[0, 0] = cam_params[0]  # fx
                    proj[1, 1] = cam_params[0]  # fy (shared focal length)
                    proj[0, 2] = cam_params[1]  # cx
                    proj[1, 2] = cam_params[2]  # cy
            elif camera.model.name == "PINHOLE":
                # params: [fx, fy, cx, cy]
                if len(cam_params) == 4:
                    proj[0, 0] = cam_params[0]  # fx
                    proj[1, 1] = cam_params[1]  # fy
                    proj[0, 2] = cam_params[2]  # cx
                    proj[1, 2] = cam_params[3]  # cy
            elif camera.model.name in ["SIMPLE_RADIAL", "RADIAL"]:
                # params: [f, cx, cy, k] or [fx, fy, cx, cy, k1, k2]
                # We'll just use the focal length and principal point, ignoring distortion
                if len(cam_params) >= 3:
                    proj[0, 0] = cam_params[0]  # fx
                    proj[1, 1] = cam_params[0] if len(cam_params) <= 4 else cam_params[1]  # fy
                    proj[0, 2] = cam_params[1] if len(cam_params) <= 4 else cam_params[2]  # cx
                    proj[1, 2] = cam_params[2] if len(cam_params) <= 4 else cam_params[3]  # cy
                logger.warning(f"Camera model {camera.model.name} has radial distortion, which is ignored")
            else:
                logger.warning(f"Unknown camera model: {camera.model.name}, attempting to parse parameters")
                if len(cam_params) >= 4:
                    proj[0, 0] = cam_params[0]
                    proj[1, 1] = cam_params[1]
                    proj[0, 2] = cam_params[2]
                    proj[1, 2] = cam_params[3]
            
            poses.append(pose)
            projs.append(proj)
        
        # Convert to tensors
        poses = torch.tensor(np.stack(poses), dtype=torch.float32)
        
        # Use the first camera's intrinsics (or average if shared_camera)
        # For simplicity, we'll use the first one
        intrinsics = torch.tensor(projs[0], dtype=torch.float32)
        
        # VGGT doesn't provide motion probabilities, so we default to ones (all static)
        motion_prob = None
        
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
                    # Read binary masks from video
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
                            # Convert frame to binary mask: any non-zero pixel is foreground.
                            # Expectation: mask video uses 0 (bg) and 255 (fg) or colored overlay on black.
                            mask_bin = (np.any(frame > 127, axis=2)).astype(np.float32)
                            loaded.append(mask_bin)
                        cap.release()
                        if len(loaded) == 0:
                            logger.warning(f"No frames decoded from mask video: {video_path}")
                        else:
                            masks_np = np.stack(loaded, axis=0)  # (T, H, W) float32 in {0.0,1.0}
                            masks = torch.tensor(masks_np, dtype=torch.float32)
                            self._save_mask_visualization(masks_np, seq_name, Path(video_path))
                            logger.info(
                                f"Loaded {masks.shape[0]} segmentation masks from video {video_path} with shape: {masks.shape}"
                            )
            else:
                # Sort mask files by filename
                mask_files.sort(key=lambda x: x.name)
                
                # Load mask images
                from PIL import Image
                loaded_masks = []
                for mask_file in mask_files:
                    try:
                        mask_img = Image.open(mask_file)
                        # Convert to grayscale if needed
                        if mask_img.mode != 'L':
                            mask_img = mask_img.convert('L')
                        mask_array = np.array(mask_img, dtype=np.float32)
                        # Normalize to 0-1 range if needed (assuming masks are 0-255)
                        if mask_array.max() > 1.0:
                            mask_array = mask_array / 255.0
                        loaded_masks.append(mask_array)
                    except Exception as e:
                        logger.error(f"Failed to load mask image {mask_file}: {e}")
                        continue
                
                if loaded_masks:
                    masks_np = np.stack(loaded_masks, axis=0)  # Shape: (N, H, W)
                    masks = torch.tensor(masks_np, dtype=torch.float32)
                    self._save_mask_visualization(masks_np, seq_name, mask_dir)
                    logger.info(f"Loaded {len(loaded_masks)} segmentation masks from {mask_dir} with shape: {masks.shape}")
                else:
                    logger.warning(f"No valid mask images could be loaded from {mask_dir}")
            if 'masks' in locals() and masks is not None:
                motion_prob = masks
        
        # Optionally load disparities/depths saved alongside COLMAP outputs
        # Expected layout (mirrors MegaSam outputs):
        #   <vggt_data_path>/<seq_name>/disps.npy OR depths.npy
        #   <vggt_data_path>/<seq_name>/uncert.npy (optional)
        seq_root = Path(vggt_data_path) / seq_name
        disps = None
        depths_np = None
        
        disps_path = seq_root / 'disps.npy'
        depths_path = seq_root / 'depths.npy'
        uncert_path = seq_root / 'uncert.npy'

        if disps_path.exists():
            disps_np = np.load(disps_path)
            disps = torch.tensor(disps_np, dtype=torch.float32)
        elif depths_path.exists():
            depths_np = np.load(depths_path)
            # Convert depth to disparity with numeric guard
            disps_np = 1.0 / np.clip(depths_np, 1e-6, None)
            disps = torch.tensor(disps_np, dtype=torch.float32)

        # Load optional uncertainties to motion_prob if masks not provided
        if motion_prob is None and uncert_path.exists():
            uncert_np = np.load(uncert_path)
            motion_prob = torch.tensor(uncert_np, dtype=torch.float32)

        logger.info(f"Loaded VGGT data: poses {poses.shape}, intrinsics {intrinsics.shape}")
        logger.info(f"Note: Intrinsics are calibrated for {vggt_image_width}x{vggt_image_height} images")
        logger.info(f"If your images have different dimensions, intrinsics will be rescaled automatically in fit_video_vggt")

        return {
            'poses': poses,
            'intrinsics': intrinsics,
            'motion_prob': motion_prob,
            'disps': disps,
            'image_size': (vggt_image_width, vggt_image_height),
        }


    def generate_depths(self, images, chunk_size=16, visualize=False, viz_output_path=None, viz_fps=30, viz_colormap='viridis'):
        """
        Generate depth maps for input images using the depth predictor.
        
        Args:
            images: Input images tensor of shape (bs, 3, h, w) in range [0, 1]
            chunk_size: Number of images to process at once
            visualize: Whether to create a visualization video
            viz_output_path: Path to save visualization video
            viz_fps: Frames per second for visualization
            viz_colormap: Colormap for depth visualization
            
        Returns:
            dict containing:
                - depth: (bs, h, w) depth maps
                - mask: (bs, h, w) validity masks
                - intrinsics: (bs, 3, 3) camera intrinsics
        """
        if self.use_provided_depth:
            raise ValueError("generate_depths called but use_provided_depth is True")
        
        device = images.device
        bs, c, h, w = images.shape
        
        all_depths = []
        all_masks = []
        all_intrinsics = []
        
        # Process in chunks to avoid OOM
        for i in range(0, bs, chunk_size):
            chunk_imgs = images[i:i+chunk_size]
            
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=True):
                    output = self.depth_predictor(chunk_imgs)
            
            # Extract depth, mask, and intrinsics
            depth = output['depth']  # (chunk_bs, h, w)
            mask = output['mask']  # (chunk_bs, h, w)
            intrinsics = output['intrinsics']  # (chunk_bs, 3, 3)
            
            all_depths.append(depth)
            all_masks.append(mask)
            all_intrinsics.append(intrinsics)
        
        # Concatenate all chunks
        final_depth = torch.cat(all_depths, dim=0)
        final_mask = torch.cat(all_masks, dim=0)
        final_intrinsics = torch.cat(all_intrinsics, dim=0)
        
        final_output = {
            'depth': final_depth,
            'mask': final_mask,
            'intrinsics': final_intrinsics
        }
        
        # Optionally visualize
        if visualize and viz_output_path is not None:
            from anycam.models.megasam_wrapper import MegaSamWrapper
            # Reuse the visualization method from MegaSamWrapper
            MegaSamWrapper.visualize_depths(None, images, final_depth, viz_output_path, viz_fps, viz_colormap)
        
        return final_output


    def forward(self, x):
        """
        Forward pass - not used for VGGT since we load precomputed results.
        """
        raise NotImplementedError("VGGTWrapper does not support forward pass. Use load_vggt_predictions instead.")

    def _save_mask_visualization(self, masks_np: np.ndarray, seq_name: str, source_path: Path):
        """
        Save mask visualization as a video using imageio.

        Args:
            masks_np: numpy array of shape (T, H, W) with mask values in {0, 1}
        """
        if self.mask_visualization_output is None:
            return

        masks_to_save = masks_np.copy()

        if masks_to_save.max() > 1.0:
            masks_to_save = np.clip(masks_to_save, 0, 255)
        masks_to_save = (masks_to_save > 0).astype(np.uint8) * 255

        # Expand masks to 3 channels for visualization
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

