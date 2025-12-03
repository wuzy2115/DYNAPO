import argparse
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import cv2
import numpy as np
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('image_folder', help='Path to image file')
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-8B")
    parser.add_argument('--work-dir', default=None, help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please describe the video content.")
    parser.add_argument('--select', type=int, default=-1)
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second for output video')
    parser.add_argument('--save-frames', action='store_true', help='Also save individual frames')
    args = parser.parse_args()
    return args


def visualize_frame(pred_mask, image_path):
    """Visualize a single frame and return the result as numpy array"""
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()
    return visual_result


def save_individual_frame(visual_result, image_path, work_dir):
    """Save individual frame to disk"""
    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)


def create_video_from_frames(frames, output_path, fps=30.0):
    """Create video from list of frames"""
    if not frames:
        print("No frames to create video")
        return
    
    # Get dimensions from first frame
    height, width, channels = frames[0].shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in frames:
        out.write(frame)
    
    # Release everything
    out.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    image_files = []
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    for filename in sorted(list(os.listdir(cfg.image_folder))):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)
            image_paths.append(os.path.join(cfg.image_folder, filename))

    vid_frames = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        vid_frames.append(img)


    if cfg.select > 0:
        img_frame = vid_frames[cfg.select - 1]

        print(f"Selected frame {cfg.select}")
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            image=img_frame,
            text=cfg.text,
            tokenizer=tokenizer,
        )
    else:
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            video=vid_frames,
            text=cfg.text,
            tokenizer=tokenizer,
        )

    prediction = result['prediction']
    print(f"The output is:\n{prediction}")

    if '[SEG]' in prediction and Visualizer is not None:
        _seg_idx = 0
        pred_masks = result['prediction_masks'][_seg_idx]
        
        # Collect all visualized frames for video creation
        video_frames = []
        
        # Determine output directory
        if cfg.work_dir:
            output_dir = cfg.work_dir
        else:
            output_dir = './temp_visualize_results'
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx in range(len(vid_frames)):
            pred_mask = pred_masks[frame_idx]
            # Generate visualization for this frame
            visual_result = visualize_frame(pred_mask, image_paths[frame_idx])
            video_frames.append(visual_result)
            
            # Optionally save individual frames
            if cfg.save_frames:
                save_individual_frame(visual_result, image_paths[frame_idx], output_dir)
        
        # Create video from all frames
        video_output_path = os.path.join(output_dir, 'segmentation_video.mp4')
        create_video_from_frames(video_frames, video_output_path, cfg.fps)
        
    else:
        print("No segmentation found in prediction or Visualizer not available")
