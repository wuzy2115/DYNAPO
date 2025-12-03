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


class MonST3RWrapper(nn.Module):
    """
    Wrapper class for integrating MonST3R predictions into AnyCam pipeline.
    This class loads precomputed MonST3R results saved by ref_monst3r/demo.py.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.use_provided_depth = config.get("use_provided_depth", True)
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
            pair_mode="sequential",
        )

        self.renderer = dotdict({"net": None})
        self.renderer.net = None

        self.z_near = config.get("z_near", 0.1)
        self.z_far = config.get("z_far", 10)

        self._counter = 0

    def _read_intrinsics_txt(self, intrinsics_path: Path):
        with open(intrinsics_path, 'r') as f:
            tokens = f.read().strip().replace('\n', ' ').split()
        vals = [float(x) for x in tokens if x.replace('.', '', 1).replace('-', '', 1).isdigit() or True]
        # Robust to files where content is appended repeatedly: take the last valid block
        # Prefer 3x3 blocks if present, else (fx, fy, cx, cy), else (f, cx, cy)
        K = np.eye(3, dtype=np.float32)
        if len(vals) >= 9 and (len(vals) % 9 == 0 or len(vals) > 9):
            block = vals[-9:]
            K = np.array(block, dtype=np.float32).reshape(3, 3)
            return torch.tensor(K, dtype=torch.float32)
        if len(vals) >= 4 and (len(vals) % 4 == 0 or len(vals) > 4):
            fx, fy, cx, cy = vals[-4:]
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            return torch.tensor(K, dtype=torch.float32)
        if len(vals) >= 3 and (len(vals) % 3 == 0 or len(vals) > 3):
            f, cx, cy = vals[-3:]
            K[0, 0] = f
            K[1, 1] = f
            K[0, 2] = cx
            K[1, 2] = cy
            return torch.tensor(K, dtype=torch.float32)
        raise ValueError(f"Unsupported intrinsics format in {intrinsics_path}: length={len(vals)} content head={vals[:10]}")

    def _quat_to_rot(self, qx, qy, qz, qw):
        x, y, z, w = qx, qy, qz, qw
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        R = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ], dtype=np.float32)
        return R

    def _read_poses_tum(self, traj_path: Path):
        poses = []
        with open(traj_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                vals = [float(x) for x in parts]
                # Supported patterns:
                #  - t tx ty tz qw qx qy qz   (w-first quaternion, with timestamp)
                #  - t qx qy qz qw tx ty tz   (q-first quaternion, with timestamp)
                #  - tx ty tz qw qx qy qz     (w-first quaternion, no timestamp)
                #  - qx qy qz qw tx ty tz     (q-first quaternion, no timestamp)
                if len(vals) >= 8:
                    if len(vals) > 8:
                        tokens = vals[-8:]
                    else:
                        tokens = vals
                    # Build two candidates and choose the one whose quaternion norm is closest to 1
                    # Candidate A: [t, tx ty tz, qw qx qy qz]
                    tx_a, ty_a, tz_a = tokens[1], tokens[2], tokens[3]
                    qw_a, qx_a, qy_a, qz_a = tokens[4], tokens[5], tokens[6], tokens[7]
                    norm_a = qw_a*qw_a + qx_a*qx_a + qy_a*qy_a + qz_a*qz_a
                    # Candidate B: [t, qx qy qz qw, tx ty tz]
                    qx_b, qy_b, qz_b, qw_b = tokens[1], tokens[2], tokens[3], tokens[4]
                    tx_b, ty_b, tz_b = tokens[5], tokens[6], tokens[7]
                    norm_b = qw_b*qw_b + qx_b*qx_b + qy_b*qy_b + qz_b*qz_b
                    if abs(norm_a - 1.0) <= abs(norm_b - 1.0):
                        tx, ty, tz = tx_a, ty_a, tz_a
                        qx, qy, qz, qw = qx_a, qy_a, qz_a, qw_a
                    else:
                        tx, ty, tz = tx_b, ty_b, tz_b
                        qx, qy, qz, qw = qx_b, qy_b, qz_b, qw_b
                elif len(vals) == 7:
                    # No timestamp; test both orders
                    # Candidate A: [tx ty tz, qw qx qy qz]
                    tx_a, ty_a, tz_a = vals[0], vals[1], vals[2]
                    qw_a, qx_a, qy_a, qz_a = vals[3], vals[4], vals[5], vals[6]
                    norm_a = qw_a*qw_a + qx_a*qx_a + qy_a*qy_a + qz_a*qz_a
                    # Candidate B: [qx qy qz qw, tx ty tz]
                    qx_b, qy_b, qz_b, qw_b = vals[0], vals[1], vals[2], vals[3]
                    tx_b, ty_b, tz_b = vals[4], vals[5], vals[6]
                    norm_b = qw_b*qw_b + qx_b*qx_b + qy_b*qy_b + qz_b*qz_b
                    if abs(norm_a - 1.0) <= abs(norm_b - 1.0):
                        tx, ty, tz = tx_a, ty_a, tz_a
                        qx, qy, qz, qw = qx_a, qy_a, qz_a, qw_a
                    else:
                        tx, ty, tz = tx_b, ty_b, tz_b
                        qx, qy, qz, qw = qx_b, qy_b, qz_b, qw_b
                else:
                    raise ValueError(f"Unsupported pose format in {traj_path}: {vals}")
                R = self._quat_to_rot(qx, qy, qz, qw)
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
                poses.append(T)
        if len(poses) == 0:
            raise ValueError(f"No poses parsed from {traj_path}")
        return torch.tensor(np.stack(poses), dtype=torch.float32)

    def _load_depth_sequence(self, folder: Path):
        # MonST3R notebooks read predicted depths from per-frame npy files like frame_XXXX.npy
        # Try that first, then fall back to other common patterns.
        def sorted_by_frame_index(paths):
            def key(p):
                stem = p.stem
                # Expect frame_0001
                parts = stem.split('_')
                try:
                    return int(parts[-1])
                except Exception:
                    return stem
            return sorted(paths, key=key)

        patterns_priority = [
            "frame_*.npy",  # primary MonST3R save format
            "frame*.npy",
            "depth*.npy",
            "depth_*.npy",
            "frame_*.png",
            "depth*.png",
            "depth_*.png",
        ]

        for pat in patterns_priority:
            files = list(folder.glob(pat))
            if not files:
                continue
            files = sorted_by_frame_index(files)
            loaded = []
            for p in files:
                if p.suffix.lower() == '.npy':
                    arr = np.load(p)
                else:
                    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    arr = img.astype(np.float32)
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[..., 0]
                loaded.append(arr)
            if loaded:
                depths_np = np.stack(loaded, axis=0)
                return torch.tensor(depths_np, dtype=torch.float32)

        # Also support a stacked depths.npy file (N,H,W)
        depths_npy = folder / 'depths.npy'
        if depths_npy.exists():
            depths_np = np.load(depths_npy)
            return torch.tensor(depths_np, dtype=torch.float32)

        return None

    def _load_conf_sequence(self, folder: Path):
        confs_npy = folder / 'confs.npy'
        if confs_npy.exists():
            confs_np = np.load(confs_npy)
            return torch.tensor(confs_np, dtype=torch.float32)
        candidates = []
        for pat in ["conf*.npy", "conf_*.npy", "confidence*.npy", "confidence_*.npy", "conf*.png", "conf_*.png"]:
            candidates.extend(sorted([p for p in folder.glob(pat)]))
        if candidates:
            loaded = []
            for p in candidates:
                if p.suffix.lower() == '.npy':
                    arr = np.load(p)
                else:
                    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    arr = img.astype(np.float32)
                loaded.append(arr)
            confs_np = np.stack(loaded, axis=0)
            return torch.tensor(confs_np, dtype=torch.float32)
        return None

    def load_monst3r_predictions(self, monst3r_data_path, seq_name):
        """
        Load MonST3R predictions from outputs saved by ref_monst3r/demo.py

        Expected files inside <monst3r_data_path>/<seq_name>/:
          - pred_traj.txt            (poses in TUM format)
          - pred_intrinsics.txt      (fx fy cx cy OR 3x3)
          - depths.npy or depth_*.{npy,png}
          - confs.npy or conf_*.{npy,png}
        """
        data_dir = Path(monst3r_data_path) / seq_name / 'NULL'
        if not data_dir.exists():
            raise FileNotFoundError(f"MonST3R data directory not found: {Path(monst3r_data_path) / seq_name}")

        traj_path = data_dir / 'pred_traj.txt'
        intr_path = data_dir / 'pred_intrinsics.txt'
        if not traj_path.exists():
            raise FileNotFoundError(f"pred_traj.txt not found in {data_dir}")
        if not intr_path.exists():
            raise FileNotFoundError(f"pred_intrinsics.txt not found in {data_dir}")

        poses = self._read_poses_tum(traj_path)
        intrinsics = self._read_intrinsics_txt(intr_path)

        depths = self._load_depth_sequence(data_dir)
        confs = self._load_conf_sequence(data_dir)

        # Convert depth to disparity if available
        disps = None
        image_size = None
        if depths is not None and depths.numel() > 0:
            depths_np = depths.cpu().numpy()
            depths_np = np.clip(depths_np, 1e-6, None)
            disps = torch.tensor(1.0 / depths_np, dtype=torch.float32)
            h, w = depths.shape[-2], depths.shape[-1]
            image_size = (w, h)

        motion_prob = None
        # Masks handling (copy-paste as requested)
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
                    mask_img = Image.open(mask_file)
                    if mask_img.mode != 'L':
                        mask_img = mask_img.convert('L')
                    mask_array = np.array(mask_img, dtype=np.float32)
                    if mask_array.max() > 1.0:
                        mask_array = mask_array / 255.0
                    loaded_masks.append(mask_array)
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
            # See ref_monst3r/dust3r/cloud_opt/base_opt.py: save_dynamic_masks
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
            candidate_dirs = [data_dir, data_dir / 'dynamic_masks']
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

        if motion_prob is None and confs is not None:
            motion_prob = confs

        result = {
            'poses': poses,
            'intrinsics': intrinsics,
            'motion_prob': motion_prob,
            'disps': disps,
            'image_size': image_size,
        }
        return result

    def forward(self, x):
        raise NotImplementedError("MonST3RWrapper does not support forward pass. Use load_monst3r_predictions instead.")

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


