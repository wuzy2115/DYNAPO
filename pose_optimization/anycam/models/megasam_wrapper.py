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

class MegaSamWrapper(nn.Module):
    """
    Wrapper class for integrating MegaSam pose predictions into AnyCam pipeline.
    This class replaces the pose_predictor with precomputed MegaSam results.
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


    def unnormalize_intrinsics(self, normalized_intrinsics, h, w):
        """
        Convert normalized camera intrinsics back to unnormalized (pixel-space) intrinsics.
        
        Args:
            normalized_intrinsics: Normalized intrinsics matrix (3, 3)
            h: Image height in pixels
            w: Image width in pixels
            
        Returns:
            Unnormalized intrinsics matrix (3, 3)
        """
        unnormalized = normalized_intrinsics.clone()
        
        unnormalized[0, 0] = normalized_intrinsics[0, 0] * w  # fx
        unnormalized[1, 1] = normalized_intrinsics[1, 1] * h  # fy
        unnormalized[0, 2] = normalized_intrinsics[0, 2] * w  # cx
        unnormalized[1, 2] = normalized_intrinsics[1, 2] * h  # cy

        return unnormalized

    def visualize_depths(self, images, depths, output_path, fps=30, colormap='viridis'):
        """
        Create a side-by-side video showing original images and predicted depths.
        
        Args:
            images: Input images tensor of shape (bs, 3, h, w) in range [0, 1]
            depths: Predicted depths tensor of shape (bs, h, w) or (bs, 1, h, w)
            output_path: Path to save the output video
            fps: Frames per second for the output video
            colormap: Matplotlib colormap name for depth visualization
        """
        # Ensure images are in the right format and range
        if images.dim() == 4:
            bs, c, h, w = images.shape
        else:
            raise ValueError(f"Expected images to be 4D tensor, got shape {images.shape}")
        
        # Ensure depths are in the right format
        if depths.dim() == 4:
            depths = depths.squeeze(1)  # Remove channel dimension if present
        elif depths.dim() != 3:
            raise ValueError(f"Expected depths to be 3D or 4D tensor, got shape {depths.shape}")
        
        # Convert tensors to numpy arrays
        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
        else:
            images_np = images
            
        if isinstance(depths, torch.Tensor):
            depths_np = depths.detach().cpu().numpy()
        else:
            depths_np = depths
        
        # Ensure images are in [0, 255] range for OpenCV
        if images_np.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)
        
        # Convert from RGB to BGR for OpenCV
        images_np = np.transpose(images_np, (0, 2, 3, 1))  # (bs, h, w, c)
        images_bgr = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np])
        
        # Normalize depths for visualization
        # Handle infinite values (like sky regions) by filtering them out
        depths_finite = depths_np.copy()
        is_finite = np.isfinite(depths_finite)
        
        if not np.any(is_finite):
            logger.warning("No finite depth values found, using fallback normalization")
            depths_normalized = np.zeros_like(depths_finite)
        else:
            # Get finite depth range
            finite_depths = depths_finite[is_finite]
            depth_min = finite_depths.min()
            depth_max = finite_depths.max()
            
            # Normalize only finite values
            depths_normalized = np.zeros_like(depths_finite)
            depths_normalized[is_finite] = (finite_depths - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Set infinite values to 0 (or you could set them to 1 for a different visualization)
            depths_normalized[~is_finite] = 0.0
            
            logger.info(f"Depth range (finite values): [{depth_min:.3f}, {depth_max:.3f}]")
            logger.info(f"Finite depth pixels: {np.sum(is_finite)}/{depths_finite.size} ({100*np.sum(is_finite)/depths_finite.size:.1f}%)")
        
        # Apply colormap to depths
        cmap = cm.get_cmap(colormap)
        depths_colored = []
        for depth_frame in depths_normalized:
            # Apply colormap and convert to BGR
            depth_colored = cmap(depth_frame)[:, :, :3]  # Remove alpha channel
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            depths_colored.append(depth_colored)
        depths_colored = np.array(depths_colored)
        
        # Create side-by-side video
        h, w = images_bgr.shape[1:3]
        combined_width = w * 2
        combined_height = h
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
        
        # Create frames
        for i in range(len(images_bgr)):
            # Get current frame
            img_frame = images_bgr[i]
            depth_frame = depths_colored[i]
            
            # Create side-by-side frame
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            combined_frame[:, :w] = img_frame
            combined_frame[:, w:] = depth_frame
            
            # Add text labels
            cv2.putText(combined_frame, 'Original', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, 'Depth', (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add depth range info
            if np.any(is_finite):
                depth_info = f"Depth: [{depth_min:.3f}, {depth_max:.3f}] (finite)"
                inf_count = np.sum(~is_finite)
                if inf_count > 0:
                    depth_info += f" | {inf_count} inf pixels"
            else:
                depth_info = "Depth: No finite values"
            cv2.putText(combined_frame, depth_info, (10, combined_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(combined_frame)
        
        out.release()
        logger.info(f"Depth visualization video saved to: {output_path}")
        logger.info(f"Video info: {len(images_bgr)} frames, {combined_width}x{combined_height}, {fps} fps")
        if np.any(is_finite):
            logger.info(f"Finite depth range: [{depth_min:.3f}, {depth_max:.3f}]")
            logger.info(f"Finite depth pixels: {np.sum(is_finite)}/{depths_finite.size} ({100*np.sum(is_finite)/depths_finite.size:.1f}%)")
        else:
            logger.warning("No finite depth values found in the sequence")

    def generate_depths(self, images, chunk_size=16, visualize=False, viz_output_path=None, viz_fps=30, viz_colormap='viridis'):
        """
        Generate depths from images using depth predictor.
        
        Args:
            images: Input images of shape (bs, 3, h, w)
            chunk_size: Maximum batch size for each inference chunk to avoid OOM
            visualize: Whether to create a visualization video
            viz_output_path: Path for the visualization video (required if visualize=True)
            viz_fps: Frames per second for visualization video
            viz_colormap: Colormap for depth visualization
        """
        bs, _, h, w = images.shape
        
        # If batch size is smaller than chunk size, process all at once
        if bs <= chunk_size:
            with torch.no_grad():
                output = self.depth_predictor.infer(images)
            return output
        
        # Process in chunks
        all_outputs = []
        
        with torch.no_grad():
            for i in range(0, bs, chunk_size):
                # Get the current chunk
                end_idx = min(i + chunk_size, bs)
                chunk = images[i:end_idx]
                
                # Process the chunk
                chunk_output = self.depth_predictor.infer(chunk)
                all_outputs.append(chunk_output)
        
        # Concatenate all outputs
        # The output should be a dictionary with keys that have batch dimensions
        if not all_outputs:
            raise ValueError("No outputs to concatenate")
        
        # Get the keys from the first output
        output_keys = all_outputs[0].keys()
        
        # Initialize the final output dictionary
        final_output = {}
        
        # Concatenate each key across all chunks
        for key in output_keys:
            # Collect all values for this key
            values = [chunk_output[key] for chunk_output in all_outputs]
            
            # Concatenate along batch dimension (dimension 0)
            if isinstance(values[0], torch.Tensor):
                final_output[key] = torch.cat(values, dim=0)
            else:
                # Handle non-tensor values (like lists)
                final_output[key] = values
        
        # Average intrinsics and provide both normalized and unnormalized versions
        if 'intrinsics' in final_output:
            # Average the normalized intrinsics
            final_output['intrinsics'] = final_output['intrinsics'].mean(dim=0)  # (3, 3)
            
            # Also provide unnormalized version
            final_output['intrinsics'] = self.unnormalize_intrinsics(
                final_output['intrinsics'], h, w
            )
        
        # Create visualization if requested
        if visualize:
            if viz_output_path is None:
                viz_output_path = "depths_mogev2.mp4"
            
            # Extract depths from output
            if 'depth' in final_output:
                depths = final_output['depth']
                self.visualize_depths(
                    images=images,
                    depths=depths,
                    output_path=viz_output_path,
                    fps=viz_fps,
                    colormap=viz_colormap
                )
            else:
                logger.warning("No 'depth' key found in output, skipping visualization")
        
        return final_output
        """
        `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
        The maps are in the same size as the input image. 
        {
            "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
            "depth": (H, W),        # depth map
            "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
            "mask": (H, W),         # a binary mask for valid pixels. 
            "intrinsics": (3, 3),   # normalized camera intrinsics
        }
        """

    def load_megasam_predictions(self, recon_data_path, seq_name):
        """
        Load MegaSam predictions from saved files.
        
        Args:
            recon_data_path: Path to the directory containing MegaSam results
            seq_name: Name of the sequence
            
        Returns:
            dict containing:
                - poses: Camera poses (N, 4, 4)
                - intrinsics: Camera intrinsics (3, 3)
                - motion_prob: Motion probabilities/uncertainties (N,)
                - disps: Disparities (N, H, W) - optional
                - masks: Segmentation masks (N, H, W) - optional, 0=static, 1=dynamic
        """
        data_dir = Path(recon_data_path) / seq_name
        
        if not data_dir.exists():
            raise FileNotFoundError(f"MegaSam data directory not found: {data_dir}")
        
        # Load poses
        poses_path = data_dir / "poses.npy"
        if poses_path.exists():
            poses = np.load(poses_path)  # Shape should be (N, 4, 4)
            poses = torch.tensor(poses, dtype=torch.float32)
        else:
            raise FileNotFoundError(f"Poses file not found: {poses_path}")
        
        # Load intrinsics
        intrinsics_path = data_dir / "intrinsics.npy" 
        if intrinsics_path.exists():
            intrinsics = np.load(intrinsics_path)[0]  # Take first frame intrinsics
            # Convert from [fx, fy, cx, cy] format to 3x3 matrix
            if intrinsics.shape == (4,):
                fx, fy, cx, cy = intrinsics
                K = np.array([[fx, 0, cx],
                              [0, fy, cy], 
                              [0, 0, 1]], dtype=np.float32)
            else:
                K = intrinsics.astype(np.float32)
            intrinsics = torch.tensor(K, dtype=torch.float32)
        else:
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
        
        # Load motion probabilities (uncertainties)
        motion_prob_path = data_dir / "motion_prob.npy"
        if motion_prob_path.exists():
            motion_prob = np.load(motion_prob_path)
            motion_prob = torch.tensor(motion_prob, dtype=torch.float32)
            # The uncertainty in MegaSAM has an inverse meaning for our BA.
            # We use 1 - motion_prob and clamp it to [0, 1].
            motion_prob = 1.0 - motion_prob
            motion_prob = torch.clamp(motion_prob, 0.0, 1.0)
        else:
            logger.warning(f"Motion prob file not found: {motion_prob_path}, using default uncertainties")
            motion_prob = torch.ones(len(poses), dtype=torch.float32)
        
        # Optionally load disparities
        disps_path = data_dir / "disps.npy"
        disps = None
        if disps_path.exists():
            disps = np.load(disps_path)
            disps = torch.tensor(disps, dtype=torch.float32)

        mask_tensor = self._load_masks_from_path(seq_name)
        if mask_tensor is not None:
            if mask_tensor.shape[0] == motion_prob.shape[0]:
                motion_prob = mask_tensor
            else:
                logger.warning(
                    "Mask tensor length %s does not match motion probability length %s for sequence %s",
                    mask_tensor.shape[0], motion_prob.shape[0], seq_name
                )
        
        return {
            'poses': poses,
            'intrinsics': intrinsics, 
            'motion_prob': motion_prob,
            'disps': disps
        }

    def forward(self, data, megasam_data=None, **kwargs):
        """
        Forward pass using preloaded MegaSam data instead of pose_predictor.
        
        Args:
            data: Input data dict containing images
            megasam_data: Preloaded MegaSam predictions dict
        """
        if megasam_data is None:
            raise ValueError("MegaSam data must be provided for forward pass")
        
        data = dict(data)

        images = data["imgs"]  # B, n_frames, c, h, w
        n, f, c, h, w = images.shape
        device = images.device

        # Use MegaSam intrinsics instead of gt_projs
        megasam_intrinsics = megasam_data['intrinsics'].to(device)
        # Normalize projection matrices
        gt_projs = normalize_proj(megasam_intrinsics.unsqueeze(0), h, w)

        # Get depth from MegaSam disparities if available, otherwise use depth predictor
        if megasam_data.get('disps') is not None:
            # Use MegaSam disparities
            megasam_disps = megasam_data['disps'][:f].to(device)  # (n_frames, H, W)
            
            # Align disparity shape with image shape if needed
            img_h, img_w = images.shape[-2:]
            disp_h, disp_w = megasam_disps.shape[-2:]
            
            if disp_h != img_h or disp_w != img_w:
                # Resize disparities to match image dimensions
                if megasam_disps.dim() == 3:
                    megasam_disps = megasam_disps.unsqueeze(1)  # Add channel dim for interpolation
                
                megasam_disps = torch.nn.functional.interpolate(
                    megasam_disps, 
                    size=(img_h, img_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Remove channel dimension if it was added
                if megasam_disps.dim() == 4:
                    megasam_disps = megasam_disps.squeeze(1)
                
                logger.info(f"Resized MegaSam disparities from {disp_h}x{disp_w} to {img_h}x{img_w}")
            
            # Convert disparities to depths (disparity is typically inverse of depth)
            # Clamp disparities to avoid division by zero
            depths = 1.0 / megasam_disps.clamp_min(1e-6)
            
            # Add batch and channel dimensions to match expected format
            depths = depths.unsqueeze(0).unsqueeze(2)  # (1, n_frames, 1, H, W)
            
            logger.info(f"Using MegaSam disparities, converted to depths with shape: {depths.shape}")
            
        elif self.use_provided_depth:
            depths = data["depths"]
        else:
            # Fall back to depth predictor
            depth_in = images.view(n * f, c, h, w)

            with torch.no_grad():
                depths, depth_features = self.depth_predictor(depth_in, return_features=True)
            depths = depths[0]

            depths = 1 / depths.clamp_min(1e-3).view(n, -1, 1, *depths.shape[-2:])
            depth_features = depth_features.view(n, -1, *depth_features.shape[1:])

        data["pred_depths"] = depths * .1
        data["pred_depths_list"] = [depths]

        # Compute flow and occlusion (same as AnyCamWrapper)
        images_ip_fwd, images_ip_bwd = self.image_processor(images * 2 - 1, data=data)

        flow_occ_fwd = images_ip_fwd[:, :, 3:6]
        flow_occ_bwd = images_ip_bwd[:, :, 3:6]

        # Use MegaSam poses and uncertainties instead of pose_predictor results
        megasam_poses = megasam_data['poses'][:f].to(device).unsqueeze(0)  # (1, n_frames, 4, 4)
        megasam_uncertainties = megasam_data['motion_prob'][:f].to(device).unsqueeze(0)  # (1, n_frames)
        
        # Prepare data for compatibility with existing pipeline
        data["images_ip"] = images_ip_fwd
        data["valid"] = images_ip_fwd[:, :, 5:6] > 0.5
        data["proc_poses"] = megasam_poses
        data["proc_projs"] = gt_projs
        data["uncertainties"] = megasam_uncertainties.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Match expected shape
        data["weights_proc"] = data["uncertainties"]
        data["scaled_depths"] = [depths]

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        if self.training:
            self._counter += 1
            
        return data

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

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result 