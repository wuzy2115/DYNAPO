import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
import uuid
import cv2
from pathlib import Path
from omegaconf import OmegaConf
import logging
from minipytorch3d.rotation_conversions import quaternion_to_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from anycam.models.megasam_wrapper import MegaSamWrapper
from anycam.utils.bundle_adjustment import compute_depth_flow
from anycam.scripts.ba_refinement_opt_tracks_global import ba_refinement_opt_tracks_global
from anycam.utils.geometry import average_pose
from dotdict import dotdict
from anycam.loss import make_loss

import rerun as rr

logger = logging.getLogger(__name__)

def visualize_motion_prob_video(motion_prob_tensor, seq_imgs, save_path=None, fps=10, threshold=0.7):
    """
    Create a video visualization of motion probability masks over time.
    
    Args:
        motion_prob_tensor: Tensor of shape (n-1, 1, H, W) or (n-1, H, W)
        seq_imgs: Tensor of shape (n, 3, H, W) - original images
        save_path: Path to save the video (should end with .mp4)
        fps: Frames per second for the output video
    """
    # Ensure motion_prob_tensor has the right shape
    if motion_prob_tensor.dim() == 3:
        motion_prob_tensor = motion_prob_tensor.unsqueeze(1)  # Add channel dim
    
    n_motion_frames = motion_prob_tensor.shape[0]
    n_img_frames = seq_imgs.shape[0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Motion Probability Analysis Over Time', fontsize=16)
    
    # Initialize empty plots
    ax_img = axes[0, 0]
    ax_motion = axes[0, 1]
    ax_overlay = axes[1, 0]
    ax_hist = axes[1, 1]
    
    # Set up the plots
    ax_img.set_title('Original Image')
    ax_img.axis('off')
    
    ax_motion.set_title('Motion Probability\n(High=Static, Low=Dynamic)')
    ax_motion.axis('off')
    
    ax_overlay.set_title('Overlay (Red=Dynamic, Green=Static)')
    ax_overlay.axis('off')
    
    ax_hist.set_title('Motion Probability Distribution')
    ax_hist.set_xlabel('Motion Probability')
    ax_hist.set_ylabel('Pixel Count')
    ax_hist.set_xlim(0, 1)
    
    # Create custom colormap for motion probability
    colors = ['red', 'yellow', 'green']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('motion', colors, N=n_bins)
    
    # Initialize plot elements
    im_img = ax_img.imshow(np.zeros((100, 100, 3)))
    im_motion = ax_motion.imshow(np.zeros((100, 100)), cmap=cmap, vmin=0, vmax=1)
    im_overlay = ax_overlay.imshow(np.zeros((100, 100, 3)))
    
    # Add colorbar for motion probability
    cbar = plt.colorbar(im_motion, ax=ax_motion, shrink=0.8)
    cbar.set_label('Motion Probability')
    
    # Text elements for statistics
    stats_text = ax_motion.text(0.02, 0.98, '', transform=ax_motion.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        # Use the same frame for both image and motion (motion has n-1 frames)
        img_frame = min(frame, n_img_frames - 1)
        motion_frame = min(frame, n_motion_frames - 1)
        
        # Get image and motion data
        img = seq_imgs[img_frame].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        motion_mask = motion_prob_tensor[motion_frame, 0].cpu().numpy()
        
        # Update image
        im_img.set_array(img)
        
        # Update motion probability
        im_motion.set_array(motion_mask)
        
        # Create overlay
        overlay = img.copy()
        
        # Dynamic regions (red overlay)
        dynamic_mask = motion_mask > threshold
        overlay[dynamic_mask] = overlay[dynamic_mask] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
        
        # Static regions (green overlay)
        static_mask = motion_mask < threshold
        overlay[static_mask] = overlay[static_mask] * 0.7 + np.array([0.0, 1.0, 0.0]) * 0.3
        
        im_overlay.set_array(overlay)
        
        # Update histogram
        ax_hist.clear()
        ax_hist.hist(motion_mask.flatten(), bins=50, alpha=0.7, color='blue', density=True)
        ax_hist.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label='Dynamic threshold')
        ax_hist.axvline(x=threshold, color='green', linestyle='--', alpha=0.7, label='Static threshold')
        ax_hist.set_xlabel('Motion Probability')
        ax_hist.set_ylabel('Density')
        ax_hist.set_xlim(0, 1)
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Update statistics
        dynamic_ratio = (motion_mask > threshold).mean()
        static_ratio = (motion_mask < threshold).mean()
        mean_prob = motion_mask.mean()
        
        stats_str = f'Frame {frame+1}/{max(n_img_frames, n_motion_frames)}\n'
        stats_str += f'Mean: {mean_prob:.3f}\n'
        stats_str += f'Dynamic: {dynamic_ratio:.1%}\n'
        stats_str += f'Static: {static_ratio:.1%}'
        
        stats_text.set_text(stats_str)
        
        return [im_img, im_motion, im_overlay, stats_text]
    
    # Create animation
    n_frames = max(n_img_frames, n_motion_frames)
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False, repeat=True)
    
    # Save video if path provided
    if save_path:
        print(f"Saving motion probability video to: {save_path}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='AnyCam'), bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Video saved successfully!")
    
    # Show the animation
    plt.tight_layout()
    plt.show()
    
    return anim


def visualize_flow_occlusion_video(seq_flow_occs_fwd, seq_imgs, save_path=None, fps=10, flow_scale=1.0, arrow_subsample=8):
    """
    Create a comprehensive video visualization of optical flow and occlusion data.
    
    Args:
        seq_flow_occs_fwd: Tensor of shape (n-1, 3, H, W) where channels are [flow_x, flow_y, occlusion]
        seq_imgs: Tensor of shape (n, 3, H, W) - original images
        save_path: Path to save the video (should end with .mp4)
        fps: Frames per second for the output video
        flow_scale: Scale factor for flow visualization
        arrow_subsample: Subsample factor for flow arrows (every N pixels)
    """
    # Ensure inputs have the right shape
    if seq_flow_occs_fwd.dim() == 3:
        seq_flow_occs_fwd = seq_flow_occs_fwd.unsqueeze(0)  # Add batch dim if needed
    
    n_flow_frames = seq_flow_occs_fwd.shape[0]
    n_img_frames = seq_imgs.shape[0]
    h, w = seq_imgs.shape[2], seq_imgs.shape[3]
    
    print(f"Visualizing flow data:")
    print(f"  Flow frames: {n_flow_frames}, Image frames: {n_img_frames}")
    print(f"  Flow shape: {seq_flow_occs_fwd.shape}")
    print(f"  Image shape: {seq_imgs.shape}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Optical Flow and Occlusion Analysis', fontsize=16)
    
    # Set up the plots
    ax_img = axes[0, 0]
    ax_flow_color = axes[0, 1] 
    ax_flow_arrows = axes[0, 2]
    ax_flow_mag = axes[1, 0]
    ax_occlusion = axes[1, 1]
    ax_overlay = axes[1, 2]
    
    ax_img.set_title('Original Image')
    ax_img.axis('off')
    
    ax_flow_color.set_title('Flow Color Coding')
    ax_flow_color.axis('off')
    
    ax_flow_arrows.set_title('Sampled Grid')
    ax_flow_arrows.axis('off')
    
    ax_flow_mag.set_title('Flow Magnitude')
    ax_flow_mag.axis('off')
    
    ax_occlusion.set_title('Occlusion Mask')
    ax_occlusion.axis('off')
    
    ax_overlay.set_title('Flow + Occlusion Overlay')
    ax_overlay.axis('off')
    
    # Initialize plot elements
    im_img = ax_img.imshow(np.zeros((h, w, 3)))
    im_flow_color = ax_flow_color.imshow(np.zeros((h, w, 3)))
    im_flow_arrows = ax_flow_arrows.imshow(np.zeros((h, w, 3)))
    im_flow_mag = ax_flow_mag.imshow(np.zeros((h, w)), cmap='hot', vmin=0, vmax=1)
    im_occlusion = ax_occlusion.imshow(np.zeros((h, w)), cmap='RdYlGn', vmin=0, vmax=1)
    im_overlay = ax_overlay.imshow(np.zeros((h, w, 3)))
    
    # Add colorbars
    cbar_mag = plt.colorbar(im_flow_mag, ax=ax_flow_mag, shrink=0.8)
    cbar_mag.set_label('Flow Magnitude')
    
    cbar_occ = plt.colorbar(im_occlusion, ax=ax_occlusion, shrink=0.8)
    cbar_occ.set_label('Occlusion (0=occluded, 1=visible)')
    
    # Text elements for statistics
    stats_text = ax_flow_mag.text(0.02, 0.98, '', transform=ax_flow_mag.transAxes, 
                                 verticalalignment='top', 
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def flow_to_color(flow_x, flow_y):
        """Convert flow to HSV color representation"""
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)
        
        # Normalize angle to [0, 1] for hue
        hue = (angle + np.pi) / (2 * np.pi)
        
        # Use magnitude for saturation (normalized)
        max_mag = np.percentile(magnitude, 99) if magnitude.max() > 0 else 1.0
        saturation = np.clip(magnitude / max_mag, 0, 1)
        
        # Full value (brightness)
        value = np.ones_like(magnitude)
        
        # Create HSV image
        hsv = np.stack([hue, saturation, value], axis=-1)
        
        # Convert to RGB
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        
        return rgb
    
    def create_flow_arrows(img, flow_x, flow_y, subsample=8, scale=1.0):
        """Create colored grid visualization instead of arrows"""
        img_grid = img.copy()
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h:subsample, 0:w:subsample]
        
        # Sample flow at grid positions
        flow_x_sampled = flow_x[::subsample, ::subsample]
        flow_y_sampled = flow_y[::subsample, ::subsample]
        
        # Create unique colors for each grid point
        n_points_y, n_points_x = y_coords.shape
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_points_y * n_points_x)))
        
        # Draw colored circles at each grid point
        for i in range(n_points_y):
            for j in range(n_points_x):
                y, x = y_coords[i, j], x_coords[i, j]
                
                # Get unique color for this grid point
                color_idx = (i * n_points_x + j) % len(colors)
                color = (colors[color_idx][:3] * 255).astype(int)
                color_bgr = (int(color[2]), int(color[1]), int(color[0]))  # Convert RGB to BGR for OpenCV
                
                # Draw circle at grid point
                cv2.circle(img_grid, (int(x), int(y)), radius=4, color=color_bgr, thickness=-1)
                
                # Optional: draw a small border for better visibility
                cv2.circle(img_grid, (int(x), int(y)), radius=4, color=(0, 0, 0), thickness=1)
        
        return img_grid
    
    def animate(frame):
        # Use the same frame for both image and flow (flow has n-1 frames)
        img_frame = min(frame, n_img_frames - 1)
        flow_frame = min(frame, n_flow_frames - 1)
        
        # Get image and flow data
        img = seq_imgs[img_frame].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        # Get flow and occlusion data
        flow_data = seq_flow_occs_fwd[flow_frame].cpu().numpy()
        flow_x = flow_data[0]  # Normalized flow X
        flow_y = flow_data[1]  # Normalized flow Y
        occlusion = flow_data[2]  # Occlusion mask
        
        # Update original image
        im_img.set_array(img)
        
        # Create flow color visualization
        flow_color = flow_to_color(flow_x, flow_y)
        im_flow_color.set_array(flow_color)
        
        # Create flow arrows
        img_uint8 = (img * 255).astype(np.uint8)
        img_arrows = create_flow_arrows(img_uint8, flow_x, flow_y, 
                                       subsample=arrow_subsample, scale=flow_scale)
        im_flow_arrows.set_array(img_arrows)
        
        # Flow magnitude
        flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
        im_flow_mag.set_array(flow_magnitude)
        
        # Update colorbar range for flow magnitude
        vmax = np.percentile(flow_magnitude, 99) if flow_magnitude.max() > 0 else 1.0
        im_flow_mag.set_clim(vmin=0, vmax=vmax)
        
        # Occlusion mask
        im_occlusion.set_array(occlusion)
        
        # Create overlay: flow color with occlusion transparency
        overlay = flow_color.copy()
        # Make occluded regions more transparent/darker
        occlusion_alpha = np.stack([occlusion, occlusion, occlusion], axis=-1)
        overlay = overlay * occlusion_alpha + img * (1 - occlusion_alpha) * 0.3
        im_overlay.set_array(overlay)
        
        # Update statistics
        mean_flow_mag = flow_magnitude.mean()
        max_flow_mag = flow_magnitude.max()
        occlusion_ratio = (occlusion < 0.5).mean()  # Fraction of occluded pixels
        visible_ratio = (occlusion >= 0.5).mean()   # Fraction of visible pixels
        
        stats_str = f'Frame {frame+1}/{max(n_img_frames, n_flow_frames)}\n'
        stats_str += f'Mean Flow Mag: {mean_flow_mag:.3f}\n'
        stats_str += f'Max Flow Mag: {max_flow_mag:.3f}\n'
        stats_str += f'Occluded: {occlusion_ratio:.1%}\n'
        stats_str += f'Visible: {visible_ratio:.1%}'
        
        stats_text.set_text(stats_str)
        
        return [im_img, im_flow_color, im_flow_arrows, im_flow_mag, im_occlusion, im_overlay, stats_text]
    
    # Create animation
    n_frames = max(n_img_frames, n_flow_frames)
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False, repeat=True)
    
    # Save video if path provided
    if save_path:
        print(f"Saving flow visualization video to: {save_path}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='AnyCam'), bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Flow visualization video saved successfully!")
    
    # Show the animation
    plt.tight_layout()
    plt.show()
    
    return anim


def visualize_pixel_tracks(seq_flow_occs_fwd, seq_imgs, save_path=None, fps=10, flow_scale=1.0, arrow_subsample=8):
    """
    Create a video visualization of sampled pixel tracks from optical flow.
    
    Args:
        seq_flow_occs_fwd: Tensor of shape (n-1, 3, H, W) where channels are [flow_x, flow_y, occlusion]
        seq_imgs: Tensor of shape (n, 3, H, W) - original images
        save_path: Path to save the video (should end with .mp4)
        fps: Frames per second for the output video
        flow_scale: Scale factor for flow visualization
        arrow_subsample: Subsample factor for flow arrows (every N pixels)
    """
    # Ensure inputs have the right shape
    if seq_flow_occs_fwd.dim() == 3:
        seq_flow_occs_fwd = seq_flow_occs_fwd.unsqueeze(0)  # Add batch dim if needed
    
    n_flow_frames = seq_flow_occs_fwd.shape[0]
    n_img_frames = seq_imgs.shape[0]
    h, w = seq_imgs.shape[2], seq_imgs.shape[3]
    
    print(f"Visualizing pixel tracks:")
    print(f"  Flow frames: {n_flow_frames}, Image frames: {n_img_frames}")
    print(f"  Flow shape: {seq_flow_occs_fwd.shape}")
    print(f"  Image shape: {seq_imgs.shape}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sampled Pixel Tracks', fontsize=16)
    
    # Set up the plots
    ax_img = axes[0, 0]
    ax_flow_x = axes[0, 1]
    ax_flow_y = axes[0, 2]
    ax_flow_mag = axes[1, 0]
    ax_occlusion = axes[1, 1]
    ax_overlay = axes[1, 2]
    
    ax_img.set_title('Original Image')
    ax_img.axis('off')
    
    ax_flow_x.set_title('Sampled Flow X')
    ax_flow_x.axis('off')
    
    ax_flow_y.set_title('Sampled Flow Y')
    ax_flow_y.axis('off')
    
    ax_flow_mag.set_title('Sampled Flow Magnitude')
    ax_flow_mag.axis('off')
    
    ax_occlusion.set_title('Sampled Occlusion')
    ax_occlusion.axis('off')
    
    ax_overlay.set_title('Sampled Flow + Occlusion Overlay')
    ax_overlay.axis('off')
    
    # Initialize plot elements
    im_img = ax_img.imshow(np.zeros((h, w, 3)))
    im_flow_x = ax_flow_x.imshow(np.zeros((h, w)), cmap='RdYlGn', vmin=-1, vmax=1)
    im_flow_y = ax_flow_y.imshow(np.zeros((h, w)), cmap='RdYlGn', vmin=-1, vmax=1)
    im_flow_mag = ax_flow_mag.imshow(np.zeros((h, w)), cmap='hot', vmin=0, vmax=1)
    im_occlusion = ax_occlusion.imshow(np.zeros((h, w)), cmap='RdYlGn', vmin=0, vmax=1)
    im_overlay = ax_overlay.imshow(np.zeros((h, w, 3)))
    
    # Add colorbars
    cbar_mag = plt.colorbar(im_flow_mag, ax=ax_flow_mag, shrink=0.8)
    cbar_mag.set_label('Flow Magnitude')
    
    cbar_occ = plt.colorbar(im_occlusion, ax=ax_occlusion, shrink=0.8)
    cbar_occ.set_label('Occlusion (0=occluded, 1=visible)')
    
    # Text elements for statistics
    stats_text = ax_flow_mag.text(0.02, 0.98, '', transform=ax_flow_mag.transAxes, 
                                 verticalalignment='top', 
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        # Use the same frame for both image and flow (flow has n-1 frames)
        img_frame = min(frame, n_img_frames - 1)
        flow_frame = min(frame, n_flow_frames - 1)
        
        # Get image and flow data
        img = seq_imgs[img_frame].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        # Get flow and occlusion data
        flow_data = seq_flow_occs_fwd[flow_frame].cpu().numpy()
        flow_x = flow_data[0]  # Normalized flow X
        flow_y = flow_data[1]  # Normalized flow Y
        occlusion = flow_data[2]  # Occlusion mask
        
        # Update original image
        im_img.set_array(img)
        
        # Update sampled flow X
        im_flow_x.set_array(flow_x)
        
        # Update sampled flow Y
        im_flow_y.set_array(flow_y)
        
        # Flow magnitude
        flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
        im_flow_mag.set_array(flow_magnitude)
        
        # Update colorbar range for flow magnitude
        vmax = np.percentile(flow_magnitude, 99) if flow_magnitude.max() > 0 else 1.0
        im_flow_mag.set_clim(vmin=0, vmax=vmax)
        
        # Occlusion mask
        im_occlusion.set_array(occlusion)
        
        # Create overlay: flow color with occlusion transparency
        overlay = flow_to_color(flow_x, flow_y) # Use flow_to_color for consistency
        occlusion_alpha = np.stack([occlusion, occlusion, occlusion], axis=-1)
        overlay = overlay * occlusion_alpha + img * (1 - occlusion_alpha) * 0.3
        im_overlay.set_array(overlay)
        
        # Update statistics
        mean_flow_mag = flow_magnitude.mean()
        max_flow_mag = flow_magnitude.max()
        occlusion_ratio = (occlusion < 0.5).mean()  # Fraction of occluded pixels
        visible_ratio = (occlusion >= 0.5).mean()   # Fraction of visible pixels
        
        stats_str = f'Frame {frame+1}/{max(n_img_frames, n_flow_frames)}\n'
        stats_str += f'Mean Flow Mag: {mean_flow_mag:.3f}\n'
        stats_str += f'Max Flow Mag: {max_flow_mag:.3f}\n'
        stats_str += f'Occluded: {occlusion_ratio:.1%}\n'
        stats_str += f'Visible: {visible_ratio:.1%}'
        
        stats_text.set_text(stats_str)
        
        return [im_img, im_flow_x, im_flow_y, im_flow_mag, im_occlusion, im_overlay, stats_text]
    
    # Create animation
    n_frames = max(n_img_frames, n_flow_frames)
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False, repeat=True)
    
    # Save video if path provided
    if save_path:
        print(f"Saving pixel tracks video to: {save_path}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='AnyCam'), bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Pixel tracks video saved successfully!")
    
    # Show the animation
    plt.tight_layout()
    plt.show()
    
    return anim


def visualize_sampled_pixel_tracks(pixel_tracks, seq_imgs, indices, visibles, save_path=None, fps=10, track_subsample=4):
    """
    Create a video visualization of sampled pixel tracks from compute_pixel_tracks.
    
    Args:
        pixel_tracks: Tensor of shape (n, wc, gs, tl, 2) - sampled pixel tracks
        seq_imgs: Tensor of shape (n, 3, H, W) - original images
        indices: Tensor of shape (n, wc, gs, tl, 1) - frame indices for each track point
        visibles: Tensor of shape (n, wc, gs, tl, 1) - visibility mask for each track point
        save_path: Path to save the video (should end with .mp4)
        fps: Frames per second for the output video
        track_subsample: Subsample factor for tracks (show every N tracks)
    """
    n, wc, gs, tl, _ = pixel_tracks.shape
    seq_len, _, h, w = seq_imgs.shape
    
    print(f"Visualizing sampled pixel tracks:")
    print(f"  Pixel tracks shape: {pixel_tracks.shape}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of track groups: {wc}")
    print(f"  Grid size: {gs}")
    print(f"  Track length: {tl}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sampled Pixel Tracks from compute_pixel_tracks', fontsize=16)
    
    # Set up the plots
    ax_img = axes[0, 0]
    ax_tracks = axes[0, 1]
    ax_current_group = axes[1, 0]
    ax_stats = axes[1, 1]
    
    ax_img.set_title('Original Image')
    ax_img.axis('off')
    
    ax_tracks.set_title('All Active Tracks')
    ax_tracks.axis('off')
    
    ax_current_group.set_title('Current Track Group')
    ax_current_group.axis('off')
    
    ax_stats.set_title('Track Statistics')
    ax_stats.axis('off')
    
    # Initialize plot elements
    im_img = ax_img.imshow(np.zeros((h, w, 3)))
    im_tracks = ax_tracks.imshow(np.zeros((h, w, 3)))
    im_current_group = ax_current_group.imshow(np.zeros((h, w, 3)))
    
    # Convert tensors to numpy for easier processing
    pixel_tracks_np = pixel_tracks.cpu().numpy()
    indices_np = indices.cpu().numpy()
    visibles_np = visibles.cpu().numpy()
    seq_imgs_np = seq_imgs.cpu().numpy()
    
    # Create unique colors for track groups
    group_colors = plt.cm.tab20(np.linspace(0, 1, min(20, wc)))
    
    def denormalize_coords(coords, h, w):
        """Convert normalized coordinates [-1, 1] to pixel coordinates"""
        x = (coords[..., 0] + 1) * w / 2
        y = (coords[..., 1] + 1) * h / 2
        return np.stack([x, y], axis=-1)
    
    def animate(frame):
        # Get current image
        img = seq_imgs_np[frame].transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        # Update original image
        im_img.set_array(img)
        
        # Create tracks visualization
        img_tracks = (img * 255).astype(np.uint8).copy()
        img_current_group = (img * 255).astype(np.uint8).copy()
        
        active_tracks_count = 0
        visible_tracks_count = 0
        current_group_idx = None
        
        # Draw all active tracks
        for group_idx in range(wc):
            group_color = (group_colors[group_idx % len(group_colors)][:3] * 255).astype(int)
            group_color_bgr = (int(group_color[2]), int(group_color[1]), int(group_color[0]))
            
            # Check if this group is active at current frame
            group_active = False
            
            for track_idx in range(0, gs, track_subsample):
                for time_idx in range(tl):
                    if indices_np[0, group_idx, track_idx, time_idx, 0] == frame:
                        group_active = True
                        current_group_idx = group_idx
                        
                        # Get track coordinates
                        track_coords = pixel_tracks_np[0, group_idx, track_idx, time_idx, :]
                        visible = visibles_np[0, group_idx, track_idx, time_idx, 0] > 0.5
                        
                        if visible:
                            # Denormalize coordinates
                            pixel_coords = denormalize_coords(track_coords.reshape(1, 1, 2), h, w)[0, 0]
                            x, y = int(pixel_coords[0]), int(pixel_coords[1])
                            
                            # Check bounds
                            if 0 <= x < w and 0 <= y < h:
                                # Draw track point
                                cv2.circle(img_tracks, (x, y), radius=3, color=group_color_bgr, thickness=-1)
                                cv2.circle(img_tracks, (x, y), radius=3, color=(0, 0, 0), thickness=1)
                                
                                active_tracks_count += 1
                                visible_tracks_count += 1
                                
                                # Draw in current group visualization if this is the active group
                                if group_idx == current_group_idx:
                                    cv2.circle(img_current_group, (x, y), radius=4, color=group_color_bgr, thickness=-1)
                                    cv2.circle(img_current_group, (x, y), radius=4, color=(0, 0, 0), thickness=1)
                                    
                                    # Add track index as text
                                    cv2.putText(img_current_group, str(track_idx), (x+6, y+6), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Update track visualizations
        im_tracks.set_array(img_tracks)
        im_current_group.set_array(img_current_group)
        
        # Update statistics
        ax_stats.clear()
        ax_stats.set_title('Track Statistics')
        ax_stats.axis('off')
        
        stats_text = f'Frame: {frame+1}/{seq_len}\n'
        stats_text += f'Active track groups: {wc}\n'
        stats_text += f'Active tracks: {active_tracks_count}\n'
        stats_text += f'Visible tracks: {visible_tracks_count}\n'
        if current_group_idx is not None:
            stats_text += f'Current group: {current_group_idx}\n'
        stats_text += f'Grid size: {gs}\n'
        stats_text += f'Track length: {tl}\n'
        stats_text += f'Subsampling: every {track_subsample} tracks'
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add color legend for track groups
        legend_y = 0.4
        ax_stats.text(0.05, legend_y, 'Track Group Colors:', transform=ax_stats.transAxes, 
                     fontsize=10, weight='bold')
        
        for i in range(min(5, wc)):  # Show first 5 groups
            color = group_colors[i % len(group_colors)][:3]
            ax_stats.add_patch(plt.Rectangle((0.05, legend_y - 0.05 - i*0.04), 0.03, 0.03, 
                                           facecolor=color, transform=ax_stats.transAxes))
            ax_stats.text(0.1, legend_y - 0.04 - i*0.04, f'Group {i}', 
                         transform=ax_stats.transAxes, fontsize=9)
        
        return [im_img, im_tracks, im_current_group]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=seq_len, interval=1000//fps, blit=False, repeat=True)
    
    # Save video if path provided
    if save_path:
        print(f"Saving sampled pixel tracks video to: {save_path}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='AnyCam'), bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Sampled pixel tracks video saved successfully!")
    
    # Show the animation
    plt.tight_layout()
    plt.show()
    
    return anim


def quaternion_translation_to_matrix(quaternion, translation, use_conjugation=True, coord_transform="y_up_to_y_down"):
    """
    Convert quaternion and translation to 4x4 transformation matrix.
    Fixed to match MegaSam's original pose format and coordinate system.
    
    Args:
        quaternion: (4,) tensor in [qx, qy, qz, qw] format (as used by MegaSam)
        translation: (3,) tensor [tx, ty, tz] (camera position in world coordinates)
        use_conjugation: Whether to conjugate the quaternion (reverse rotation direction)
        coord_transform: Type of coordinate transformation to apply
                        Options: None, "y_up_to_y_down", "z_flip", "x_flip", "translation_flip_x", "translation_flip_y", "translation_flip_z", "translation_flip_yz"
    
    Returns:
        4x4 camera-to-world transformation matrix
    """
    # Convert from MegaSam's [qx, qy, qz, qw] format to minipytorch3d's [qw, qx, qy, qz] format
    if quaternion.shape[-1] == 4:
        # Assume input is [qx, qy, qz, qw] format (MegaSam standard)
        qx, qy, qz, qw = quaternion
        
        if use_conjugation:
            # Conjugate the quaternion (negate the imaginary parts) to reverse rotation direction
            quaternion_wxyz = torch.tensor([qw, -qx, -qy, -qz], dtype=quaternion.dtype, device=quaternion.device)
        else:
            quaternion_wxyz = torch.tensor([qw, qx, qy, qz], dtype=quaternion.dtype, device=quaternion.device)
    else:
        raise ValueError(f"Expected quaternion of length 4, got {quaternion.shape}")
    
    # Convert quaternion to rotation matrix (camera-to-world)
    rotation_matrix = quaternion_to_matrix(quaternion_wxyz.unsqueeze(0)).squeeze(0)
    
    # Create 4x4 camera-to-world transformation matrix
    # This matches the original MegaSam format where:
    # - rotation_matrix is camera-to-world rotation
    # - translation is camera position in world coordinates
    transform = torch.eye(4, device=quaternion.device, dtype=quaternion.dtype)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    
    # Apply coordinate system transformation
    if coord_transform == "y_up_to_y_down":
        # Y-up to Y-down conversion (flip Y and Z axes)
        coord_transform_matrix = torch.tensor([[1, 0, 0, 0],
                                              [0, -1, 0, 0],
                                              [0, 0, -1, 0],
                                              [0, 0, 0, 1]], dtype=transform.dtype, device=transform.device)
        transform = coord_transform_matrix @ transform
    elif coord_transform == "z_flip":
        # Left-handed to right-handed conversion (flip Z axis)
        coord_transform_matrix = torch.tensor([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, -1, 0],
                                              [0, 0, 0, 1]], dtype=transform.dtype, device=transform.device)
        transform = coord_transform_matrix @ transform
    elif coord_transform == "x_flip":
        # Flip X axis
        coord_transform_matrix = torch.tensor([[-1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], dtype=transform.dtype, device=transform.device)
        transform = coord_transform_matrix @ transform
    elif coord_transform == "translation_flip_x":
        # Flip only X translation (keep rotation unchanged) - fixes horizontal movement inversion
        transform[0, 3] = -transform[0, 3]
    elif coord_transform == "translation_flip_y":
        # Flip only Y translation (keep rotation unchanged)
        transform[1, 3] = -transform[1, 3]
    elif coord_transform == "translation_flip_z":
        # Flip only Z translation (keep rotation unchanged)
        transform[2, 3] = -transform[2, 3]
    elif coord_transform == "translation_flip_yz":
        # Flip both Y and Z translation (keep rotation unchanged)
        transform[1, 3] = -transform[1, 3]
        transform[2, 3] = -transform[2, 3]
    # If coord_transform is None, no transformation is applied
    
    return transform


def load_megasam(model_path, checkpoint=None, loaded_config=None):
    """
    Load MegaSamWrapper model from AnyCam checkpoint, filtering out pose_predictor weights.
    
    Args:
        model_path: Path to the model directory containing training_config.yaml and checkpoints
        checkpoint: Specific checkpoint name (optional, defaults to latest)
        
    Returns:
        model: MegaSamWrapper instance with loaded weights
        criterion: Loss criterion for compatibility
    """
    model_path = Path(model_path)
    config = OmegaConf.load(model_path / "training_config.yaml")

    prefix = "training_checkpoint_"
    ckpts = list(model_path.glob(f"{prefix}*.pt"))

    model_conf = config["model"]
    model_conf['use_provided_flow'] = loaded_config['prediction']['use_provided_flow'] if loaded_config is not None else False
    model_conf['use_provided_masks'] = loaded_config['prediction']['use_provided_masks'] if loaded_config is not None else False
    model_conf['use_provided_depth'] = loaded_config['prediction']['use_provided_depth'] if loaded_config is not None else False
    model_conf["train_directions"] = "forward"
    model_conf['depth_predictor']['type'] = loaded_config['prediction']['depth_predictor'] if loaded_config is not None else 'unidepth'
    model_conf['flow_model'] = loaded_config['prediction']['flow_model'] if loaded_config is not None else 'unimatch'
    model_conf['mask_path'] = loaded_config['prediction']['mask_path'] if loaded_config is not None else None
    
    # Create MegaSamWrapper instead of AnyCamWrapper
    model = MegaSamWrapper(model_conf)

    criterion = [make_loss(cfg) for cfg in config.get("loss", [])][0]

    training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]

    if training_steps:
        if checkpoint is None:
            ckpt_path = f"{prefix}{max(training_steps)}.pt"
        else:
            ckpt_path = checkpoint

        ckpt_path = model_path / ckpt_path

        print(f"Loading checkpoint: {ckpt_path}")

        cp = torch.load(ckpt_path, map_location="cpu")

        # Filter out pose_predictor related weights since MegaSamWrapper doesn't have pose_predictor
        filtered_state_dict = {}
        for key, value in cp["model"].items():
            if not key.startswith("pose_predictor"):
                filtered_state_dict[key] = value
            else:
                print(f"Skipping pose_predictor weight: {key}")

        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (expected for MegaSam): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print(f"Successfully loaded {len(filtered_state_dict)} parameters")

    return model, criterion


def load_images(input_path):
    """Load images from input path - same as original fit_video.py"""
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if input_path.is_file():
        # Single video file
        import cv2
        cap = cv2.VideoCapture(str(input_path))
        imgs = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame.transpose(2, 0, 1)).float() / 255.0
            imgs.append(frame)
        cap.release()
        return torch.stack(imgs)
    else:
        # Directory with images
        image_files = sorted(list(input_path.glob("*.jpg"))) + sorted(list(input_path.glob("*.png")))
        imgs = []
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img.transpose(2, 0, 1)).float() / 255.0
            imgs.append(img)
        return torch.stack(imgs)


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


def fit_video_megasam_wrapper(config, model, criterion, imgs, device, recon_data_path, seq_name, gt_proj=None):
    """Wrapper function to maintain compatibility with original interface"""
    return fit_video_megasam(config=config, model=model, criterion=criterion, imgs=imgs, device=device, recon_data_path=recon_data_path, seq_name=seq_name, gt_proj=gt_proj, return_extras=False)


@torch.autocast(device_type="cuda", enabled=True)
@torch.no_grad()
def fit_video_megasam(config, model, criterion, imgs, device="cuda", recon_data_path=None, seq_name=None, 
                      return_extras=False, gt_proj=None):
    """
    Main function for video fitting using MegaSam predictions instead of pose_predictor.
    
    Args:
        config: Configuration dict
        model: MegaSamWrapper instance 
        criterion: Loss criterion (not used but kept for compatibility)
        imgs: Input images
        device: Device to run on
        recon_data_path: Path to MegaSam prediction files
        seq_name: Sequence name for loading MegaSam data
        return_extras: Whether to return extra information
        gt_proj: Ground truth projection (optional)
    """
    print(seq_name)
    print(config)

    if recon_data_path is None:
        raise ValueError("recon_data_path must be provided")
    if seq_name is None:
        raise ValueError("seq_name must be provided")

    dataset_config = config.get("dataset", {})
    do_ba_refinement = config.get("do_ba_refinement", False)
    ba_refinement_level = config.get("ba_refinement_level", 0) + 1
    ba_refinement_config = config.get("ba_refinement", {})
    save_data_path = config['prediction'].get("save_data_path", None)
    if save_data_path is not None:
        save_data_path = Path(save_data_path) / config.get('dataset_type') / seq_name
        save_data_path.mkdir(parents=True, exist_ok=True)
    print(f"dataset_config: {dataset_config}")
    print(f"do_ba_refinement: {do_ba_refinement}")
    print(f"ba_refinement_level: {ba_refinement_level}")
    print(f"ba_refinement_config: {ba_refinement_config}")

    # Create dataset
    dataset = make_dataset(dataset_config, imgs, device="cpu")

    if config.get("with_rerun", False):
        rr.init("MegaSam Prediction", recording_id=uuid.uuid4())
        rr.connect()

        for i, img in enumerate(dataset.imgs):
            rr.set_time_sequence("timestep", i)
            rr.log(f"world/img", rr.Image((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).compress(jpeg_quality=95))

    logger.info("Loading MegaSam predictions")
    
    # Load MegaSam predictions
    megasam_data = model.load_megasam_predictions(recon_data_path, seq_name)
    logger.info(f"Loaded MegaSam data: poses {megasam_data['poses'].shape}, intrinsics {megasam_data['intrinsics'].shape}")

    # predict depths
    generate_depth = ba_refinement_config.get("generate_depth", False)
    if generate_depth:
        output = model.generate_depths(dataset.imgs, visualize=False, viz_output_path=f"depths_{seq_name}.mp4")
        megasam_data['disps'] = 1 / output['depth']
        megasam_data['mask'] = output['mask']
        megasam_data['intrinsics'] = output['intrinsics']
        megasam_data['disps'][~megasam_data['mask']] = 0 # set invalid disps to 0
    # Preprocess images for depth and flow
    logger.info("Preprocessing images")
    c, h, w = dataset.imgs.shape[1:]
    seq_imgs = dataset.imgs

    # Compute depth and flow using the model (same as original)
    seq_imgs, seq_depths, seq_flow_occs_fwd, seq_flow_occs_bwd = compute_depth_flow(
        model, seq_imgs, megasam_disps=megasam_data.get('disps')
    )
    
    # Visualize the flow and occlusion data with improved colored grid
    # print("Creating flow and occlusion visualization with colored grid...")
    # flow_video_path = f"flow_occlusion_video_{seq_name}.mp4" if seq_name else "flow_occlusion_video.mp4"
    # visualize_flow_occlusion_video(
    #     seq_flow_occs_fwd,
    #     seq_imgs,
    #     save_path=flow_video_path,
    #     fps=5,  # Slower for better viewing
    #     flow_scale=2.0,  # Scale up flow for better visibility
    #     arrow_subsample=12  # Subsample grid points for cleaner visualization
    # )
    
    print(f"After compute_depth_flow:")
    print(f"  seq_imgs shape: {seq_imgs.shape}")
    print(f"  seq_depths shape: {seq_depths.shape}")
    print(f"  seq_flow_occs_fwd shape: {seq_flow_occs_fwd.shape}")
    print(f"  seq_flow_occs_bwd shape: {seq_flow_occs_bwd.shape}")
    
    # Convert MegaSam poses from quaternion+translation to 4x4 matrices
    megasam_poses_raw = megasam_data['poses'].cpu().numpy()
    print(f"Raw MegaSam poses shape: {megasam_poses_raw.shape}")
    
    # CRITICAL FIX: MegaSam's internal poses are world-to-camera, but AnyCam expects camera-to-world
    # We need to invert them, just like MegaSam does in test_demo.py line 134:
    # cam_c2w = SE3(poses_th).inv().matrix().numpy()
    
    print("Converting MegaSam poses from world-to-camera to camera-to-world (following test_demo.py approach)")
    
    # Import SE3 from lietorch (same as MegaSam uses)
    from lietorch import SE3
    
    # Convert to SE3 format and invert to get camera-to-world poses
    poses_th = torch.as_tensor(megasam_poses_raw, device="cpu")
    cam_c2w_matrices = SE3(poses_th).inv().matrix().numpy()
    
    print(f"Converted {len(cam_c2w_matrices)} poses from world-to-camera to camera-to-world")
    
    # Use the inverted poses directly as 4x4 matrices
    best_trajectory = []
    for i in range(len(cam_c2w_matrices)):
        pose_matrix = torch.tensor(cam_c2w_matrices[i], dtype=torch.float32)
        best_trajectory.append(pose_matrix.numpy())
        
        # Debug: Print first few converted matrices
        if i < 3:
            print(f"Inverted pose {i} (camera-to-world):")
            print(pose_matrix.numpy())
            print()
    
    if len(best_trajectory) != len(dataset.imgs):
        best_trajectory = best_trajectory[:len(dataset.imgs)]
    
    print(f"Converted trajectory: {len(best_trajectory)} poses")
    
    # Extract projection matrix from MegaSam intrinsics
    megasam_intrinsics = megasam_data['intrinsics'].cpu().numpy()
    if megasam_intrinsics.ndim == 3:
        # Take first intrinsic matrix if multiple
        proj = megasam_intrinsics[0].copy()
    else:
        proj = megasam_intrinsics.copy()
    
    # Convert motion probabilities to uncertainties format expected by BA
    motion_prob = megasam_data['motion_prob'].cpu().numpy()

    if len(motion_prob) != len(dataset.imgs):
        motion_prob = motion_prob[:len(dataset.imgs)]
    
    print(f"Motion prob shape: {motion_prob.shape}")
    
    # Use motion_prob directly as uncertainties
    # High motion_prob (static) = low uncertainty = high confidence in BA
    # Low motion_prob (dynamic) = high uncertainty = low confidence in BA
    ba_uncertainties = motion_prob
    
    # Ensure proper shape for BA: should be (n-1, 1, H, W) for n images
    if ba_uncertainties.ndim == 3:
        ba_uncertainties = ba_uncertainties[:, None, :, :]  # Add channel dimension
    
    # # Take first n-1 frames for BA (since BA works on image pairs)
    # n_imgs = len(seq_imgs)
    # if ba_uncertainties.shape[0] >= n_imgs:
    #     ba_uncertainties = ba_uncertainties[:n_imgs-1]
    
    print(f"BA uncertainties shape after processing: {ba_uncertainties.shape}")
    
    # Convert to tensor and upsample to match image dimensions
    ba_uncertainties_tensor = torch.tensor(ba_uncertainties, dtype=torch.float32)
    
    # Get target dimensions from seq_imgs
    target_h, target_w = seq_imgs.shape[2], seq_imgs.shape[3]
    current_h, current_w = ba_uncertainties_tensor.shape[2], ba_uncertainties_tensor.shape[3]
    
    if current_h != target_h or current_w != target_w:
        print(f"Resizing uncertainties from {current_h}x{current_w} to {target_h}x{target_w}")
        ba_uncertainties_tensor = F.interpolate(
            ba_uncertainties_tensor, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
    
    print(f"Final ba_uncertainties_tensor shape: {ba_uncertainties_tensor.shape}")
    
    # Create motion probability video
    # print("Creating motion probability video...")
    # motion_video_path = f"motion_prob_video_{seq_name}.mp4" if seq_name else "motion_prob_video.mp4"
    # visualize_motion_prob_video(
    #     ba_uncertainties_tensor,
    #     seq_imgs,
    #     save_path=motion_video_path,
    #     fps=5,  # Slower for better viewing
    #     threshold=0.05
    # )
    
    # Validate shapes before BA
    print(f"Shape validation before BA:")
    print(f"  seq_imgs: {seq_imgs.shape}")
    print(f"  seq_depths: {seq_depths.shape}")
    print(f"  seq_flow_occs_fwd: {seq_flow_occs_fwd.shape}")
    print(f"  seq_flow_occs_bwd: {seq_flow_occs_bwd.shape}")
    print(f"  ba_uncertainties_tensor: {ba_uncertainties_tensor.shape}")
    print(f"  best_trajectory: {len(best_trajectory)} poses")

    logger.info(f"Loaded trajectory: {len(best_trajectory)} poses")
    logger.info(f"Loaded uncertainties: {ba_uncertainties_tensor.shape}")

    # Bundle Adjustment Refinement (same as original fit_video.py)
    if do_ba_refinement:
        logger.info("Starting BA refinement with MegaSam initial trajectory")
        
        if ba_refinement_level > 1:
            # Subsample images and recompute flows (matching original fit_video.py)
            # seq_imgs_ba = dataset.imgs[::ba_refinement_level]
            
            # Recompute depth and flow from subsampled images
            imgs0 = dataset.imgs[:-1:ba_refinement_level]
            imgs1 = dataset.imgs[1::ba_refinement_level]
            
            # Subsample MegaSam disparities to match BA refinement level
            megasam_disps_ba = None
            if megasam_data.get('disps') is not None:
                megasam_disps_ba = megasam_data['disps'][:-1:ba_refinement_level]
            
            seq_imgs_ba, seq_depths_ba, seq_flow_occs_fwd_ba, seq_flow_occs_bwd_ba = compute_depth_flow(
                model, imgs0=imgs0, imgs1=imgs1, megasam_disps=megasam_disps_ba
            )
            
            # Subsample trajectory and uncertainties to match the new sequence length
            trajectory_ba = best_trajectory[::ba_refinement_level][:len(seq_imgs_ba)]
            
            # For uncertainties, we need to subsample to match the flow length (n-1 for n images)
            ba_uncertainties_ba = ba_uncertainties_tensor[::ba_refinement_level][:len(seq_imgs_ba)]
        else:
            seq_imgs_ba = seq_imgs
            seq_depths_ba = seq_depths
            seq_flow_occs_fwd_ba = seq_flow_occs_fwd
            seq_flow_occs_bwd_ba = seq_flow_occs_bwd
            ba_uncertainties_ba = ba_uncertainties_tensor
            trajectory_ba = best_trajectory

        print(f"BA input shapes:")
        print(f"  seq_imgs_ba: {seq_imgs_ba.shape}")
        print(f"  seq_depths_ba: {seq_depths_ba.shape}")
        print(f"  seq_flow_occs_fwd_ba: {seq_flow_occs_fwd_ba.shape}")
        print(f"  seq_flow_occs_bwd_ba: {seq_flow_occs_bwd_ba.shape}")
        print(f"  ba_uncertainties_ba: {ba_uncertainties_ba.shape}")
        print(f"  trajectory_ba: {len(trajectory_ba)} poses")

        # Run BA refinement
        best_trajectory_ba, proj_ba, ba_extras = ba_refinement_opt_tracks_global(
            ba_refinement_config, 
            trajectory_ba, 
            proj, 
            ba_uncertainties_ba,
            seq_imgs_ba, 
            seq_depths_ba, 
            seq_flow_occs_fwd_ba, 
            seq_flow_occs_bwd_ba, 
            device=device,
            model=model
        )
        
        logger.info(f"BA refinement completed: {len(best_trajectory_ba)} refined poses")
        
        # Interpolate poses back to original frame rate (same as original fit_video.py)
        interpolated_poses = []
        l = len(best_trajectory_ba)
        keyframe_trajectory = best_trajectory_ba
        
        for i in range(len(dataset.imgs)):
            if i % ba_refinement_level == 0 and (i // ba_refinement_level) < l:
                interpolated_poses.append(torch.tensor(best_trajectory_ba[i // ba_refinement_level]))
            else:
                if i // ba_refinement_level + 1 < l:
                    prev_pose = best_trajectory_ba[i // ba_refinement_level]
                    next_pose = best_trajectory_ba[(i // ba_refinement_level) + 1]

                    t = (i % ba_refinement_level) / ba_refinement_level
                    interpolated_pose = average_pose(torch.stack([torch.tensor(prev_pose), torch.tensor(next_pose)]), weight=t)
                    interpolated_poses.append(interpolated_pose)
                else:
                    # Extrapolate using previous motion
                    last1 = interpolated_poses[-2]
                    last0 = interpolated_poses[-1]
                    rel = torch.inverse(last1.to(torch.float32)) @ last0
                    next = last0 @ rel
                    interpolated_poses.append(torch.tensor(next))

        interpolated_poses = torch.stack(interpolated_poses).cpu()
        best_trajectory = interpolated_poses
        proj = proj_ba

    else:
        ba_extras = None
        # Convert best_trajectory to tensor format
        best_trajectory = torch.stack([torch.tensor(pose) for pose in best_trajectory])

    # Rescale intrinsics to match the resolution used in compute_depth_flow.
    # MegaSam intrinsics correspond to the disparity resolution. Since disparities
    # were resized inside compute_depth_flow to match seq_imgs, we must scale K
    # from disparity resolution to seq_imgs resolution (per-axis).
    
    target_h, target_w = int(seq_imgs.shape[2]), int(seq_imgs.shape[3])
    if megasam_data.get('disps') is not None:
        disp0 = megasam_data['disps'][0]
        # Support shapes (H,W), (1,H,W) or (C,H,W)
        if hasattr(disp0, 'shape'):
            depth_h, depth_w = int(disp0.shape[-2]), int(disp0.shape[-1])
            if depth_h > 0 and depth_w > 0:
                scale_x = target_w / depth_w
                scale_y = target_h / depth_h
                # Scale fx, fy, cx, cy accordingly
                proj[0, 0] = proj[0, 0] * scale_x
                proj[1, 1] = proj[1, 1] * scale_y
                proj[0, 2] = proj[0, 2] * scale_x
                proj[1, 2] = proj[1, 2] * scale_y
                print(f"Rescaled intrinsics using disparity->image scales: sx={scale_x:.6f}, sy={scale_y:.6f}")

    if save_data_path is not None:
        # Save poses as poses.npy (N, 4, 4)
        if isinstance(best_trajectory, torch.Tensor):
            poses_np = best_trajectory.detach().cpu().numpy().astype(np.float32)
        else:
            poses_np = np.asarray(best_trajectory, dtype=np.float32)
        np.save(save_data_path / "poses.npy", poses_np)
        print(f"Saved poses to {save_data_path / 'poses.npy'} with shape {poses_np.shape}")

        # Save intrinsics as intrinsics.npy with a leading frame dimension to match loader's [0] access
        if isinstance(proj, torch.Tensor):
            intr_np = proj.detach().cpu().numpy().astype(np.float32)
        else:
            intr_np = np.asarray(proj, dtype=np.float32)
        intr_np_batched = intr_np[None, ...]  # (1, 3, 3)
        np.save(save_data_path / "intrinsics.npy", intr_np_batched)
        print(f"Saved intrinsics to {save_data_path / 'intrinsics.npy'} with shape {intr_np_batched.shape}")

        # Save disparities as disps.npy (N, H, W) - convert from depths
        if isinstance(seq_depths, torch.Tensor):
            disps_t = 1.0 / torch.clamp(seq_depths, min=1e-6)
            disps_np = disps_t.detach().cpu().numpy().astype(np.float32)
        else:
            # Assume numpy depths
            disps_np = (1.0 / np.clip(seq_depths, 1e-6, None)).astype(np.float32)
        np.save(save_data_path / "disps.npy", disps_np)
        print(f"Saved disparities to {save_data_path / 'disps.npy'} with shape {disps_np.shape}")

        # Save motion probabilities as motion_prob.npy (N, H, W)
        if isinstance(ba_uncertainties_tensor, torch.Tensor):
            motion_prob_np = ba_uncertainties_tensor.squeeze(1).detach().cpu().numpy().astype(np.float32)
        else:
            motion_prob_np = np.asarray(ba_uncertainties_tensor, dtype=np.float32)
            if motion_prob_np.ndim == 4 and motion_prob_np.shape[1] == 1:
                motion_prob_np = motion_prob_np[:, 0]
        np.save(save_data_path / "motion_prob.npy", motion_prob_np)
        print(f"Saved motion probabilities to {save_data_path / 'motion_prob.npy'} with shape {motion_prob_np.shape}")

        # Save images for convenience (not used by loader)
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
            "ba_uncertainties": ba_uncertainties_tensor if do_ba_refinement else None,
            "candidate_trajectories": [best_trajectory],  # Single trajectory from MegaSam
            "pred_labels": torch.tensor([1.0]),  # Single prediction
            "flow_labels": torch.tensor([1.0]),
            "images": seq_imgs,
            "seq_depths": seq_depths,
            "seq_flow_occs_fwd": seq_flow_occs_fwd,
            "seq_flow_occs_bwd": seq_flow_occs_bwd,
            "uncertainties": ba_uncertainties_tensor,
            "best_candidate": 0,  # Only one candidate from MegaSam
            "focal_length_candidates": torch.tensor([[proj[0, 0]]]),  # Single focal length
            "ba_optimized_windows_data": ba_extras.get("optimized_windows_data") if ba_extras is not None else None,
            "ba_param_sigma_depth": ba_extras.get("ba_param_sigma_depth") if ba_extras is not None else None,
            "ba_global_track_data": ba_extras.get("global_track_data") if ba_extras is not None else None,
            "keyframe_trajectory": keyframe_trajectory if do_ba_refinement else best_trajectory,
            "megasam_data": megasam_data,  # Include original MegaSam data
        }
        return best_trajectory, proj, extras_dict, ba_extras
