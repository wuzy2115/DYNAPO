import os
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from anycam.datasets.common import (
    get_flow_selector,
    get_target_size_and_crop,
    process_img,
    process_proj,
)


class LightspeedDataset(Dataset):
    NAME = "Lightspeed"

    def __init__(
        self,
        data_path: str,
        split_path: Optional[str],
        image_size: Optional[tuple] = None,
        frame_count: int = 2,
        dilation: int = 1,
        return_depth: bool = False,
        return_flow: bool = False,
        preprocessed_path: Optional[str] = None,
        flow_selector=None,
        index_selector=None,
        sequence_sampler=None,
    ):
        # Depth/flow not available for Lightspeed, flags remain for interface compatibility
        self.data_path = data_path
        self.split_path = split_path
        self.image_size = image_size

        self.return_depth = False
        self.return_flow = False
        self.preprocessed_path = preprocessed_path

        self.frame_count = frame_count
        self.dilation = dilation

        self._left_offset = ((self.frame_count - 1) // 2) * self.dilation

        self._sequences = self._get_sequences(self.data_path)

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path)
        else:
            self._datapoints = self._full_split(self._sequences, self._left_offset, (self.frame_count - 1) * dilation, sequence_sampler)

        if flow_selector is None:
            self.flow_selector = get_flow_selector(self.frame_count)
        else:
            self.flow_selector = flow_selector

        self.index_selector = index_selector

        # Load poses once; expected file format matches visualize_lightspeed_pose.py
        self._poses_by_seq = self._load_all_poses(self.data_path)

        self.length = len(self._datapoints)

    @staticmethod
    def _get_sequences(data_path: str):
        sequences = {}
        # Expect per-sequence folder with images at data_path/<seq>/*.png or *.jpg
        for seq in os.listdir(data_path):
            seq_dir = os.path.join(data_path, seq, 'images')
            if os.path.isdir(seq_dir):
                # Count frames by counting PNG/JPG files
                files = [f for f in os.listdir(seq_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                if len(files) > 0:
                    sequences[seq] = len(files)
        return sequences

    @staticmethod
    def _full_split(sequences: dict, left_offset: int = 0, sub_seq_len: int = 2, sequence_sampler=None):
        datapoints = []
        for seq, seq_len in sequences.items():
            if sequence_sampler is not None:
                datapoints.extend(sequence_sampler(seq, seq_len, left_offset, sub_seq_len))
            else:
                if seq_len < sub_seq_len:
                    continue
                for i in range(seq_len - 1):  # -1 because we need at least two frames
                    datapoints.append((seq, i))
        return datapoints

    @staticmethod
    def _load_split(split_path: str):
        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(l):
            segments = l.split(" ")
            seq = segments[0]
            idx = int(segments[1])
            return seq, idx

        return list(map(split_line, lines))

    @staticmethod
    def _load_all_poses(data_path: str):
        # visualize_lightspeed_pose.py loads poses from <data_path_parent>/poses.pkl keyed by sequence
        # Here assume a sibling poses.pkl next to frames root or inside data_path
        # Try common locations in order
        candidates = [
            os.path.join(data_path, "poses.pkl"),
            os.path.join(os.path.dirname(data_path), "poses.pkl"),
        ]
        poses_by_seq = None
        for cand in candidates:
            if os.path.exists(cand):
                import pickle as pkl
                with open(cand, "rb") as f:
                    poses_by_seq = pkl.load(f)
                break
        if poses_by_seq is None:
            raise FileNotFoundError(f"poses.pkl not found next to lightspeed frames. Tried: {candidates}")
        return poses_by_seq

    def __len__(self):
        return len(self._datapoints)

    def _index_to_seq_ids(self, index):
        if index >= self.length:
            raise IndexError()

        sequence, idx = self._datapoints[index]
        seq_len = self._sequences[sequence]

        if self.index_selector is not None:
            ids = self.index_selector(idx, self.frame_count, self.dilation, self._left_offset)
        else:
            ids = [idx] + [i for i in range(idx - self._left_offset, idx - self._left_offset + self.frame_count * self.dilation, self.dilation) if i != idx]

        ids = [max(min(i, seq_len - 1), 0) for i in ids]

        return sequence, ids

    def _resolve_image_path(self, seq: str, fid: int):
        # Assume files are consecutively named; try common patterns
        seq_dir = os.path.join(self.data_path, seq, 'images')
        # Try frame_0001.png style
        candidates = [
            os.path.join(seq_dir, f"frame_{fid+1:04d}.png"),
            os.path.join(seq_dir, f"{fid:05d}.png"),
            os.path.join(seq_dir, f"{fid:06d}.png"),
            os.path.join(seq_dir, f"{fid:05d}.jpg"),
            os.path.join(seq_dir, f"{fid:06d}.jpg"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        # Fallback: pick the fid-th sorted image in directory
        files = sorted([f for f in os.listdir(seq_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if len(files) == 0:
            raise FileNotFoundError(f"No images found in {seq_dir}")
        fid_clip = max(min(fid, len(files) - 1), 0)
        return os.path.join(seq_dir, files[fid_clip])

    def load_images(self, seq: str, ids: list):
        imgs = []
        for fid in ids:
            img_path = self._resolve_image_path(seq, fid)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(img)
        return imgs

    def load_cams(self, sequence, ids):
        # From visualize_lightspeed_pose.py: poses are stored in poses.pkl keyed by sequence
        # We must return (projs, poses) where poses are camera-to-world 4x4 reduced to 3x4 [R|t]
        seq_poses = self._poses_by_seq[sequence]
        projs = []
        poses = []

        # Intrinsics unknown; attempt to infer from first image size and assume pinhole with fx=fy, cx=w/2, cy=h/2
        # This is consistent with process_proj usage that rescales principal point with image resizing/cropping
        first_img = cv2.imread(self._resolve_image_path(sequence, ids[0]))
        h, w = first_img.shape[:2]
        fx = fy = max(h, w)  # simple heuristic; downstream often learns focal length candidates
        K = np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], dtype=np.float32)

        for fid in ids:
            # Stored poses are expected to be world-to-camera; convert to camera-to-world
            w2c = seq_poses[fid].astype(np.float32)
            if w2c.shape == (3, 4):
                w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
            c2w = np.linalg.inv(w2c).astype(np.float32)
            poses.append(c2w)
            projs.append(K.copy())

        return projs, poses

    def __getitem__(self, index):
        sequence, ids = self._index_to_seq_ids(index)

        imgs = self.load_images(sequence, ids)
        original_size = imgs[0].shape[:2]
        target_size, crop = get_target_size_and_crop(self.image_size, original_size)

        imgs = np.stack([process_img(img, target_size, crop) for img in imgs])

        projs, poses = self.load_cams(sequence, ids)
        projs = np.stack([process_proj(proj, original_size, target_size, crop) for proj in projs])
        poses = np.stack(poses)

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "ids": np.array(ids, dtype=np.int64),
            "data_id": index,
        }

        return data

    def get_img_paths(self, index):
        sequence, ids = self._index_to_seq_ids(index)
        img_paths = [self._resolve_image_path(sequence, fid) for fid in ids]
        return img_paths

    def get_sequence(self, index: int):
        sequence, _ = self._index_to_seq_ids(index)
        return sequence


