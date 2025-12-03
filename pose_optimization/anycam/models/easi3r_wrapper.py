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


def _quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz

    xx2 = qx * x2
    yy2 = qy * y2
    zz2 = qz * z2
    xy2 = qx * y2
    xz2 = qx * z2
    yz2 = qy * z2
    wx2 = qw * x2
    wy2 = qw * y2
    wz2 = qw * z2

    rot = np.empty((3, 3), dtype=np.float32)
    rot[0, 0] = 1.0 - (yy2 + zz2)
    rot[0, 1] = xy2 - wz2
    rot[0, 2] = xz2 + wy2
    rot[1, 0] = xy2 + wz2
    rot[1, 1] = 1.0 - (xx2 + zz2)
    rot[1, 2] = yz2 - wx2
    rot[2, 0] = xz2 - wy2
    rot[2, 1] = yz2 + wx2
    rot[2, 2] = 1.0 - (xx2 + yy2)
    return rot


def _read_tum_trajectory(tum_path: Path) -> np.ndarray:
    poses = []
    with open(tum_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(x) for x in line.split()]
            # Accept either 7 (tx ty tz qw qx qy qz) or 8 columns (t tx ty tz qw qx qy qz)
            if len(parts) == 7:
                tx, ty, tz, qw, qx, qy, qz = parts
            elif len(parts) >= 8:
                _, tx, ty, tz, qw, qx, qy, qz = parts[:8]
            else:
                continue
            rot = _quaternion_to_rotation_matrix(qx, qy, qz, qw)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot
            pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
            poses.append(pose)
    if len(poses) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)
    return np.stack(poses, axis=0).astype(np.float32)


def _read_intrinsics_txt(k_path: Path) -> np.ndarray:
    with open(k_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    values = []
    for ln in lines:
        values.extend([float(x) for x in ln.split()])
    if len(values) >= 9:
        vals = values[:9]
        k = np.array(vals, dtype=np.float32).reshape(3, 3)
        return k
    if len(values) >= 4:
        fx, fy, cx, cy = values[:4]
        k = np.eye(3, dtype=np.float32)
        k[0, 0] = fx
        k[1, 1] = fy
        k[0, 2] = cx
        k[1, 2] = cy
        return k
    # Fallback identity
    return np.eye(3, dtype=np.float32)


def _load_depth_stack(root_dir: Path) -> np.ndarray | None:
    # Prefer .npy if available
    npy_candidates = [
        root_dir / "depths.npy",
        root_dir / "depth.npy",
        root_dir / "pred_depths.npy",
    ]
    for p in npy_candidates:
        if p.exists():
            arr = np.load(p)
            return arr.astype(np.float32)

    # Look for a directory of depth images
    dir_candidates = [
        root_dir / "depth",
        root_dir / "depths",
        root_dir / "depth_maps",
        root_dir / "Depth",
    ]
    image_exts = [".png", ".exr", ".pfm", ".jpg", ".jpeg"]
    for d in dir_candidates:
        if d.exists() and d.is_dir():
            files = sorted([p for p in d.iterdir() if p.suffix.lower() in image_exts])
            if len(files) == 0:
                continue
            frames = []
            for fp in files:
                if fp.suffix.lower() == ".exr":
                    img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if img.ndim == 3:
                        img = img[..., 0]
                    frames.append(img.astype(np.float32))
                elif fp.suffix.lower() == ".pfm":
                    with open(fp, "rb") as f:
                        header = f.readline().decode("ascii").strip()
                        dims = f.readline().decode("ascii").strip()
                        scale = f.readline().decode("ascii").strip()
                        w, h = map(int, dims.split())
                        data = np.fromfile(f, "<f")
                        img = np.reshape(data, (h, w))
                    frames.append(img.astype(np.float32))
                else:
                    img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    frames.append(img.astype(np.float32))
            if len(frames) > 0:
                return np.stack(frames, axis=0).astype(np.float32)

    # Look for per-frame numpy depth files like frame_*.npy or depth_*.npy
    npy_frames = sorted((root_dir).glob("frame_*.npy"))
    if len(npy_frames) == 0:
        npy_frames = sorted((root_dir).glob("depth_*.npy"))
    if len(npy_frames) > 0:
        frames = []
        # Sort by numeric index if possible
        def _frame_index(p: Path) -> int:
            stem = p.stem
            if "frame_" in stem:
                num = stem.split("frame_")[-1]
            elif "depth_" in stem:
                num = stem.split("depth_")[-1]
            else:
                num = stem
            try:
                return int(num)
            except:
                return 0
        npy_frames = sorted(npy_frames, key=_frame_index)
        for fp in npy_frames:
            arr = np.load(fp)
            frames.append(arr.astype(np.float32))
        if len(frames) > 0:
            return np.stack(frames, axis=0).astype(np.float32)
    return None


def _load_conf_stack(root_dir: Path) -> np.ndarray | None:
    npy_candidates = [
        root_dir / "conf.npy",
        root_dir / "confs.npy",
        root_dir / "confidence.npy",
    ]
    for p in npy_candidates:
        if p.exists():
            arr = np.load(p)
            return arr.astype(np.float32)

    dir_candidates = [
        root_dir / "conf",
        root_dir / "confs",
        root_dir / "confidence",
        root_dir / "confidence_maps",
    ]
    image_exts = [".png", ".jpg", ".jpeg", ".exr"]
    for d in dir_candidates:
        if d.exists() and d.is_dir():
            files = sorted([p for p in d.iterdir() if p.suffix.lower() in image_exts])
            if len(files) == 0:
                continue
            frames = []
            for fp in files:
                img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frames.append(img.astype(np.float32))
            if len(frames) > 0:
                return np.stack(frames, axis=0).astype(np.float32)

    # Look for per-frame numpy confidence files like conf_*.npy
    npy_frames = sorted((root_dir).glob("conf_*.npy"))
    if len(npy_frames) > 0:
        frames = []
        def _frame_index(p: Path) -> int:
            stem = p.stem
            if "conf_" in stem:
                num = stem.split("conf_")[-1]
            else:
                num = stem
            try:
                return int(num)
            except:
                return 0
        npy_frames = sorted(npy_frames, key=_frame_index)
        for fp in npy_frames:
            arr = np.load(fp)
            frames.append(arr.astype(np.float32))
        if len(frames) > 0:
            return np.stack(frames, axis=0).astype(np.float32)
    return None


class Easi3RWrapper(nn.Module):
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
            self.depth_predictor = make_depth_predictor(config["depth_predictor"])
            self.depth_aligner = make_depth_aligner(config["depth_aligner"])
            for param in self.depth_predictor.parameters():
                param.requires_grad = False

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

    def load_easi3r_predictions(self, easi3r_data_path, seq_name):
        base_dir = Path(easi3r_data_path)
        candidates = [
            base_dir / seq_name / seq_name,
            base_dir / seq_name,
            base_dir,
        ]
        root_dir = None
        for cand in candidates:
            if (cand / "pred_traj.txt").exists() and (cand / "pred_intrinsics.txt").exists():
                root_dir = cand
                break
        if root_dir is None:
            logger.error(f"Easi3R data directory not found for {seq_name} under {easi3r_data_path}")
            raise FileNotFoundError(f"Easi3R outputs not found: {easi3r_data_path}/{seq_name}")

        poses_path = root_dir / "pred_traj.txt"
        intrinsics_path = root_dir / "pred_intrinsics.txt"

        poses_np = _read_tum_trajectory(poses_path)
        intrinsics_np = _read_intrinsics_txt(intrinsics_path)

        # Optional depths and confidences
        depths_np = _load_depth_stack(root_dir)
        disps_tensor = None
        if depths_np is not None:
            depths_np = depths_np.astype(np.float32)
            depths_np = np.clip(depths_np, 1e-6, None)
            disps_np = 1.0 / depths_np
            disps_tensor = torch.tensor(disps_np, dtype=torch.float32)

        motion_prob = None
        conf_np = _load_conf_stack(root_dir)
        conf_tensor = None
        if conf_np is not None:
            # Normalize to [0,1] if likely 0-255
            if conf_np.max() > 1.0:
                conf_np = conf_np / 255.0
            conf_tensor = torch.tensor(conf_np.astype(np.float32), dtype=torch.float32)

        # Do not miss the condition self.use_provided_masks, directly copy and paste the snippet.
        if self.use_provided_masks:
            if self.mask_path is None:
                raise ValueError("mask_path must be provided when use_provided_masks is True")

            mask_dir = Path(self.mask_path) / seq_name
            mask_files = []
            masks = None
            if mask_dir.exists():
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
        else:
            # Load dynamic masks produced by the optimizer if available
            # File pattern: dynamic_mask_{i}.png (values 0/255)
            def _sorted_dynamic_mask_files(folder: Path):
                files = list(folder.glob('dynamic_mask_*.png'))
                files += list(folder.glob('dynamic_mask_*.PNG'))
                def extract_idx(p: Path):
                    name = p.stem  # dynamic_mask_{i}
                    try:
                        return int(name.split('_')[-1])
                    except Exception:
                        return name
                return sorted(files, key=extract_idx)

            masks = None
            candidate_dirs = [root_dir, root_dir / 'dynamic_masks']
            for cand in candidate_dirs:
                if not cand.exists():
                    continue
                dm_files = _sorted_dynamic_mask_files(cand)
                if not dm_files:
                    continue
                loaded = []
                for p in dm_files:
                    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mask_bin = (img > 127).astype(np.float32)
                    loaded.append(mask_bin)
                if loaded:
                    masks_np = np.stack(loaded, axis=0)
                    masks = torch.tensor(masks_np, dtype=torch.float32)
                    self._save_mask_visualization(masks_np, seq_name, cand)
                    logger.info(
                        f"Loaded {masks.shape[0]} dynamic masks from {cand} with shape: {masks.shape}"
                    )
                    break
            if 'masks' in locals() and masks is not None:
                motion_prob = masks

        # Fallback to confidence maps only if no masks are available
        if motion_prob is None and conf_tensor is not None:
            motion_prob = conf_tensor

        image_size = None
        if disps_tensor is not None and disps_tensor.ndim == 3:
            image_size = (disps_tensor.shape[2], disps_tensor.shape[1])

        return {
            'poses': torch.tensor(poses_np, dtype=torch.float32),
            'intrinsics': torch.tensor(intrinsics_np, dtype=torch.float32),
            'motion_prob': motion_prob,
            'disps': disps_tensor,
            'image_size': image_size,
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
        raise NotImplementedError("Easi3RWrapper does not support forward pass. Use load_easi3r_predictions instead.")

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


