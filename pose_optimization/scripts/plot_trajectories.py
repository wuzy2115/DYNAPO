import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from evo.core import trajectory
from evo.core.trajectory import PosePath3D
from evo.core import metrics

HAS_EVO = True




def load_sintel_gt_poses(data_path, sequence, frame_ids):
    """
    Load ground-truth camera poses from Sintel dataset
    
    Args:
        data_path: Path to Sintel dataset
        sequence: Sequence name
        frame_ids: List of 0-based frame IDs to load poses for
        
    Returns:
        List of ground-truth camera poses as torch tensors
    """
    gt_poses = []
    TAG_FLOAT = 202021.25

    for id in frame_ids:
        cam_path = os.path.join(data_path, "camdata_left", sequence, f"frame_{id+1:04d}.cam")

        with open(cam_path, 'rb') as f:
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert check == TAG_FLOAT, f'Wrong tag in flow file (should be: {TAG_FLOAT}, is: {check})'
            _ = np.fromfile(f, dtype='float64', count=9)  # Skip intrinsics
            N = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))

            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = N
            pose = np.linalg.inv(pose).astype(np.float32)

            gt_poses.append(torch.tensor(pose))

    return gt_poses


def load_lightspeed_gt_poses(data_path, sequence, frame_ids):
    """
    Load ground-truth camera poses from a Lightspeed dataset poses.pkl
    (world-to-camera), convert to camera-to-world, and return as torch tensors.
    
    Args:
        data_path: Path to Lightspeed frames root or its parent
        sequence: Sequence name (key in poses.pkl)
        frame_ids: List of 0-based frame IDs to load poses for
    
    Returns:
        List of ground-truth camera poses (camera-to-world) as torch tensors
    """
    import pickle as pkl

    candidates = [
        os.path.join(data_path, "poses.pkl"),
        os.path.join(os.path.dirname(data_path), "poses.pkl"),
    ]

    poses_by_seq = None
    for cand in candidates:
        if os.path.exists(cand):
            with open(cand, "rb") as f:
                poses_by_seq = pkl.load(f)
            break
    if poses_by_seq is None:
        raise FileNotFoundError(f"poses.pkl not found next to lightspeed frames. Tried: {candidates}")

    if sequence not in poses_by_seq:
        raise KeyError(f"Sequence {sequence} not found in poses.pkl")

    seq_poses = poses_by_seq[sequence]

    gt_poses = []
    num_available = len(seq_poses)
    for idx in frame_ids:
        idx_clip = max(min(idx, num_available - 1), 0)
        w2c = seq_poses[idx_clip].astype(np.float32)
        if w2c.shape == (3, 4):
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        c2w = np.linalg.inv(w2c).astype(np.float32)
        gt_poses.append(torch.tensor(c2w))

    return gt_poses


def align_trajectories(pred_traj, gt_traj):
    """
    Align predicted trajectory to ground truth trajectory with scale correction
    
    Args:
        pred_traj: List or array of predicted camera poses as torch tensors or numpy arrays
        gt_traj: List or array of ground truth camera poses as torch tensors or numpy arrays
        
    Returns:
        List of aligned predicted camera poses as torch tensors
    """
    if not HAS_EVO:
        print("Warning: evo package not available for proper trajectory alignment.")
        return pred_traj

    # Convert to numpy arrays if tensors
    pred_numpy = np.stack([pose.cpu().numpy() if isinstance(pose, torch.Tensor) else pose for pose in pred_traj]).astype(np.float64)
    gt_numpy = np.stack([pose.cpu().numpy() if isinstance(pose, torch.Tensor) else pose for pose in gt_traj]).astype(np.float64)

    # Create PosePath3D objects for alignment
    pred_path = PosePath3D(poses_se3=pred_numpy)
    gt_path = PosePath3D(poses_se3=gt_numpy)

    # Align predicted trajectory to ground truth with scale correction
    pred_path.align(gt_path, correct_scale=True)
    print("Successfully aligned trajectories with scale correction")

    # Convert aligned trajectory back to torch tensors
    aligned_traj = [torch.tensor(pose, dtype=torch.float32) for pose in pred_path.poses_se3]

    return aligned_traj


def calculate_pose_metrics(pred_traj, gt_traj):
    """
    Calculate ATE, RTE, and RRE for a predicted trajectory against a ground truth trajectory.
    """
    if not HAS_EVO:
        print("Warning: evo package not available for metrics calculation.")
        return {"ate": -1, "rte": -1, "rre": -1}

    # Ensure trajectories are numpy arrays
    if isinstance(pred_traj, list) or isinstance(pred_traj, torch.Tensor):
        pred_traj = np.stack([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in pred_traj])
    if isinstance(gt_traj, list) or isinstance(gt_traj, torch.Tensor):
        gt_traj = np.stack([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in gt_traj])

    # Ensure same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]

    pred_traj_evo = PosePath3D(poses_se3=pred_traj.astype(np.float64))
    gt_traj_evo = PosePath3D(poses_se3=gt_traj.astype(np.float64))

    # Align trajectories
    pred_traj_evo.align(gt_traj_evo, correct_scale=True)

    # Calculate metrics
    ape = metrics.APE()
    rre = metrics.RPE(pose_relation=metrics.PoseRelation.rotation_angle_deg)
    rte = metrics.RPE(pose_relation=metrics.PoseRelation.translation_part)

    ape.process_data((pred_traj_evo, gt_traj_evo))
    rre.process_data((pred_traj_evo, gt_traj_evo))
    rte.process_data((pred_traj_evo, gt_traj_evo))

    return {
        "ate": ape.error.mean(),
        "rte": rte.error.mean(),
        "rre": rre.error.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize camera trajectories.")
    parser.add_argument("--seq_name", type=str, required=True, help="Sequence name to process.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["sintel", "lightspeed", "davis"],
                        help="Type of dataset.")
    parser.add_argument("--folders", type=str, nargs='+', required=True,
                        help="List of folder paths containing trajectory data.")
    parser.add_argument("--names", type=str, nargs='+', required=True,
                        help="List of legend names for each trajectory.")
    parser.add_argument("--gt_data_path", type=str, help="Path to ground truth data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")

    args = parser.parse_args()

    # Load predicted trajectories from poses.npy files
    pred_trajectories = [np.load(Path(folder) / args.dataset_type / args.seq_name / "poses.npy") for folder in args.folders]
    gt_trajectory = None

    if args.dataset_type != "davis":
        if not args.gt_data_path:
            raise ValueError("--gt_data_path is required for sintel and lightspeed datasets")

        num_poses = pred_trajectories[0].shape[0]
        
        # Frame IDs are 0-based
        frame_ids = list(range(num_poses))

        # Load ground truth trajectories (functions expect 0-based frame IDs)
        if args.dataset_type == "sintel":
            gt_trajectory_list = load_sintel_gt_poses(args.gt_data_path, args.seq_name, frame_ids)
        elif args.dataset_type == "lightspeed":
            gt_trajectory_list = load_lightspeed_gt_poses(args.gt_data_path, args.seq_name, frame_ids)
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")

        gt_trajectory = torch.stack(gt_trajectory_list).numpy()

        aligned_trajectories = []
        for pred_traj in pred_trajectories:
            aligned_traj = align_trajectories(pred_traj, gt_trajectory)
            aligned_trajectories.append(torch.stack(aligned_traj).numpy())
        pred_trajectories = aligned_trajectories

    num_plots = len(pred_trajectories)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), squeeze=False)
    axes = axes.flatten()

    for i, (pred_trajectory, name) in enumerate(zip(pred_trajectories, args.names)):
        ax = axes[i]
        
        if gt_trajectory is not None:
            ax.plot(gt_trajectory[:, 0, 3], gt_trajectory[:, 2, 3], label="Ground Truth", color='lightgreen', linewidth=6)
        
        ax.plot(pred_trajectory[:, 0, 3], pred_trajectory[:, 2, 3], label=name)
        
        if gt_trajectory is not None:
            # Calculate and print metrics
            pose_metrics = calculate_pose_metrics(pred_trajectory, gt_trajectory)
            ate = pose_metrics["ate"]
            rte = pose_metrics["rte"]
            rre = pose_metrics["rre"]
            
            print(f"\nMetrics for {name} on {args.seq_name}:")
            print(f"  ATE: {ate:.4f}")
            print(f"  RTE: {rte:.4f}")
            print(f"  RRE: {rre:.4f}")
            
            # Annotate ATE on the subplot at a consistent position across all subfigures
            ax.text(
                0.02, 0.98, f"ATE: {ate:.4f}",
                transform=ax.transAxes,
                ha='left', va='top'
            )
        
        ax.set_xlabel("Z (m)")
        if i == 0:
            ax.set_ylabel("X (m)")
        
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.axis('equal')
        ax.set_title(name)

    plt.tight_layout()

    method_name = Path(args.folders[0]).name
    output_path = os.path.join(args.output_dir, f"{args.dataset_type}_{method_name}_{args.seq_name}.png")
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
