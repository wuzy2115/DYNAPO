import argparse
import os
import sys
import importlib.util

from PIL import Image

import cv2
import numpy as np
from imageio import get_writer

try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

# Ensure workspace root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'projects', 'llava_sam2', 'hf', 'models')


def _import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Prefer package imports (works when PYTHONPATH includes workspace), fallback to file-path import
try:
    from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel  # type: ignore
    from projects.llava_sam2.hf.models.configuration_sa2va_chat import Sa2VAChatConfig  # type: ignore
    from projects.llava_sam2.hf.models.qwen2_5.modeling_sa2va_qwen import Sa2VAChatModelQwen  # type: ignore
    from projects.llava_sam2.hf.models.qwen2_5.configuration_sa2va_chat_qwen import Sa2VAChatConfigQwen  # type: ignore
    from projects.llava_sam2.hf.models.qwen3.modeling_sa2va_qwen3 import Sa2VAChatModelQwen as Sa2VAChatModelQwen3  # type: ignore
    from projects.llava_sam2.hf.models.qwen3.configuration_sa2va_chat_qwen3 import Sa2VAChatConfigQwen as Sa2VAChatConfigQwen3  # type: ignore
    from projects.llava_sam2.hf.models.tokenization_internlm2_fast import InternLM2TokenizerFast  # type: ignore
    from projects.llava_sam2.hf.models.tokenization_internlm2 import InternLM2Tokenizer  # type: ignore
except Exception:
    _modeling_path = os.path.join(MODELS_DIR, 'modeling_sa2va_chat.py')
    _configuration_path = os.path.join(MODELS_DIR, 'configuration_sa2va_chat.py')
    _modeling_qwen_path = os.path.join(MODELS_DIR, 'qwen2_5', 'modeling_sa2va_qwen.py')
    _configuration_qwen_path = os.path.join(MODELS_DIR, 'qwen2_5', 'configuration_sa2va_chat_qwen.py')
    _modeling_qwen3_path = os.path.join(MODELS_DIR, 'qwen3', 'modeling_sa2va_qwen3.py')
    _configuration_qwen3_path = os.path.join(MODELS_DIR, 'qwen3', 'configuration_sa2va_chat_qwen3.py')
    _tok_fast_path = os.path.join(MODELS_DIR, 'tokenization_internlm2_fast.py')
    _tok_slow_path = os.path.join(MODELS_DIR, 'tokenization_internlm2.py')
    modeling_mod = _import_module_from_path('modeling_sa2va_chat', _modeling_path)
    config_mod = _import_module_from_path('configuration_sa2va_chat', _configuration_path)
    modeling_qwen_mod = _import_module_from_path('modeling_sa2va_qwen', _modeling_qwen_path)
    config_qwen_mod = _import_module_from_path('configuration_sa2va_chat_qwen', _configuration_qwen_path)
    modeling_qwen3_mod = _import_module_from_path('modeling_sa2va_qwen3', _modeling_qwen3_path)
    config_qwen3_mod = _import_module_from_path('configuration_sa2va_chat_qwen3', _configuration_qwen3_path)
    tok_fast_mod = _import_module_from_path('tokenization_internlm2_fast', _tok_fast_path)
    tok_slow_mod = _import_module_from_path('tokenization_internlm2', _tok_slow_path)
    Sa2VAChatModel = modeling_mod.Sa2VAChatModel
    Sa2VAChatConfig = config_mod.Sa2VAChatConfig
    Sa2VAChatModelQwen = modeling_qwen_mod.Sa2VAChatModelQwen
    Sa2VAChatConfigQwen = config_qwen_mod.Sa2VAChatConfigQwen
    Sa2VAChatModelQwen3 = modeling_qwen3_mod.Sa2VAChatModelQwen
    Sa2VAChatConfigQwen3 = config_qwen3_mod.Sa2VAChatConfigQwen
    InternLM2TokenizerFast = tok_fast_mod.InternLM2TokenizerFast
    InternLM2Tokenizer = tok_slow_mod.InternLM2Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation (Local Model Loader)')
    parser.add_argument('input_path', help='Path to an image folder or a video file')
    parser.add_argument('--model_path', default='ByteDance/Sa2VA-8B', help='HF repo id or local checkpoint dir')
    parser.add_argument('--work-dir', default=None, help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please describe the video content.")
    parser.add_argument('--select', type=int, default=-1)
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second for output video')
    parser.add_argument('--save-frames', action='store_true', help='Also save individual frames')
    parser.add_argument('--image-mode', action='store_true', help='Process video as sequence of single images')
    parser.add_argument('--keyframes-json', type=str, default=None, help='Path to MLLM keyframes JSON (from thinkvideo_mllm_keyframe_selector).')
    parser.add_argument('--query-prompts', action='store_true', help='Enable periodic language prompt injections at query frames')
    parser.add_argument('--query-interval', type=int, default=16, help='Interval (in frames) between query prompt injections')
    args = parser.parse_args()
    return args


def visualize_frame(pred_mask, image_bgr):
    """Visualize a single frame and return the result as numpy array (BGR)."""
    visualizer = Visualizer()
    visualizer.set_image(image_bgr)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result_bgr = visualizer.get_image()  # BGR
    return visual_result_bgr


def save_individual_frame(visual_result, frame_basename, work_dir):
    """Save individual frame to disk"""
    output_path = os.path.join(work_dir, frame_basename)
    cv2.imwrite(output_path, visual_result)


def create_video_from_frames(frames, output_path, fps=30.0):
    """Create video from list of frames using imageio for better compatibility"""
    if not frames:
        print("No frames to create video")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with get_writer(output_path, format='FFMPEG', fps=fps) as writer:
        for frame in frames:
            arr = frame
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                max_val = float(arr.max()) if hasattr(arr, 'max') else 255.0
                if max_val <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            writer.append_data(arr)

    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path

    # Detect model type by trying to load config
    is_qwen2_5_model = False
    is_qwen3_model = False
    try:
        from transformers import AutoConfig
        auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Check if it's a Qwen model by examining architectures or nested configs
        architectures = getattr(auto_cfg, 'architectures', [])
        if architectures and 'Sa2VAChatModelQwen' in architectures:
            # Check text_config and vision_config to distinguish Qwen3 from Qwen2.5
            text_config = getattr(auto_cfg, 'text_config', None)
            vision_config = getattr(auto_cfg, 'vision_config', None)

            # Qwen3 has text_config.model_type == "qwen3_vl_text" and vision_config.model_type == "qwen3_vl"
            text_model_type = ''
            vision_model_type = ''

            if text_config:
                if isinstance(text_config, dict):
                    text_model_type = text_config.get('model_type', '')
                else:
                    text_model_type = getattr(text_config, 'model_type', '')

            if vision_config:
                if isinstance(vision_config, dict):
                    vision_model_type = vision_config.get('model_type', '')
                else:
                    vision_model_type = getattr(vision_config, 'model_type', '')

            # Check for Qwen3 indicators
            if 'qwen3_vl' in text_model_type.lower() or 'qwen3_vl' in vision_model_type.lower():
                is_qwen3_model = True
            # Check for Qwen2.5 indicators
            elif 'qwen2_5_vl' in text_model_type.lower() or 'qwen2_5_vl' in vision_model_type.lower():
                is_qwen2_5_model = True
            else:
                # Default to Qwen2.5 if it's Sa2VAChatModelQwen but can't determine version
                is_qwen2_5_model = True
    except Exception as e:
        print(f"Warning: Could not detect model type: {e}")
        # Try to infer from model_path name
        if 'qwen3' in model_path.lower() or 'qwen-3' in model_path.lower():
            is_qwen3_model = True
        elif 'qwen' in model_path.lower():
            is_qwen2_5_model = True

    # Load model based on detected type
    if is_qwen3_model:
        print(f"Loading Qwen3-VL model from {model_path}")
        try:
            config = Sa2VAChatConfigQwen3.from_pretrained(model_path)
        except Exception:
            from transformers import AutoConfig
            auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config = Sa2VAChatConfigQwen3(**auto_cfg.to_dict())

        import torch
        model = Sa2VAChatModelQwen3.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # For Qwen3 models, use AutoProcessor
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer = None  # Qwen3 uses processor, not tokenizer
    elif is_qwen2_5_model:
        print(f"Loading Qwen2.5-VL model from {model_path}")
        try:
            config = Sa2VAChatConfigQwen.from_pretrained(model_path)
        except Exception:
            from transformers import AutoConfig
            auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config = Sa2VAChatConfigQwen(**auto_cfg.to_dict())

        import torch
        model = Sa2VAChatModelQwen.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # For Qwen models, use processor instead of tokenizer
        from transformers import Qwen2_5_VLProcessor
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer = None  # Qwen uses processor, not tokenizer
    else:
        print(f"Loading InternVL model from {model_path}")
        # Load model code locally, weights from model_path (HF repo id or local dir)
        try:
            config = Sa2VAChatConfig.from_pretrained(model_path)
        except Exception:
            from transformers import AutoConfig
            auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config = Sa2VAChatConfig(**auto_cfg.to_dict())

        model = Sa2VAChatModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype="auto",
            device_map="cuda",
        )

        # Initialize tokenizer using local files when InternLM2 is used
        tokenizer = None
        # try:
        #     llm_arch = config.llm_config.architectures[0]
        # except Exception:
        #     llm_arch = None

        # if llm_arch == 'InternLM2ForCausalLM':
        #     # Resolve vocab file path
        #     vocab_file = None
        #     if os.path.isdir(model_path):
        #         candidate = os.path.join(model_path, 'tokenizer.model')
        #         if os.path.isfile(candidate):
        #             vocab_file = candidate
        #     if vocab_file is None:
        #         try:
        #             from huggingface_hub import hf_hub_download
        #             vocab_file = hf_hub_download(repo_id=model_path, filename='tokenizer.model')
        #         except Exception:
        #             vocab_file = None
        #     if vocab_file is not None:
        #         # try:
        #         #     tokenizer = InternLM2TokenizerFast(vocab_file=vocab_file)
        #         # except Exception:
        #         tokenizer = InternLM2Tokenizer(vocab_file=vocab_file)

        if tokenizer is None:
            # Fallback when non-InternLM2 arch or vocab resolution failed
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        processor = None  # InternVL uses tokenizer, not processor

    vid_frames = []
    raw_frames_bgr = []
    frame_basenames = []

    if os.path.isdir(cfg.input_path):
        print(f"Processing image folder: {cfg.input_path}")
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        image_files = sorted([f for f in os.listdir(cfg.input_path) if os.path.splitext(f)[1].lower() in image_extensions])
        for filename in image_files:
            image_path = os.path.join(cfg.input_path, filename)
            # For model processing
            img = Image.open(image_path).convert('RGB')
            vid_frames.append(img)
            # For visualization
            raw_frame = cv2.imread(image_path)
            raw_frames_bgr.append(raw_frame)
            # For saving frames
            frame_basenames.append(filename)
    elif os.path.isfile(cfg.input_path):
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        if os.path.splitext(cfg.input_path)[1].lower() in video_extensions:
            print(f"Processing video file: {cfg.input_path}")
            cap = cv2.VideoCapture(cfg.input_path)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # For visualization (frame is BGR from cv2)
                raw_frames_bgr.append(frame)
                
                # For model processing (convert to RGB PIL Image)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                vid_frames.append(pil_img)
                
                # For saving frames
                frame_basenames.append(f"frame_{frame_idx:06d}.png")
                frame_idx += 1
            cap.release()
        else:
            print(f"Error: Unsupported file type for {cfg.input_path}")
            sys.exit(1)
    else:
        print(f"Error: Input path not found: {cfg.input_path}")
        sys.exit(1)

    if cfg.select > 0:
        img_frame = vid_frames[cfg.select - 1]

        print(f"Selected frame {cfg.select}")
        print(f"The input is:\n{cfg.text}")
        if is_qwen3_model or is_qwen2_5_model:
            result = model.predict_forward(
                image=img_frame,
                text=cfg.text,
                processor=processor,
            )
        else:
            result = Sa2VAChatModel.predict_forward(
                model,
                image=img_frame,
                text=cfg.text,
                tokenizer=tokenizer,
            )
    else:
        print(f"The input is:\n{cfg.text}")
        if cfg.image_mode:
            per_frame_masks = []
            predictions = []
            any_seg = False
            for img_frame in vid_frames:
                if is_qwen3_model or is_qwen2_5_model:
                    res = model.predict_forward(
                        image=img_frame,
                        text=cfg.text,
                        processor=processor,
                    )
                else:
                    res = Sa2VAChatModel.predict_forward(
                        model,
                        image=img_frame,
                        text=cfg.text,
                        tokenizer=tokenizer,
                    )
                predictions.append(res.get('prediction', ''))
                if '[SEG]' in res.get('prediction', ''):
                    any_seg = True
                masks_list = res.get('prediction_masks', [])
                if len(masks_list) > 0 and masks_list[0] is not None:
                    mask_arr = masks_list[0]
                    if hasattr(mask_arr, 'ndim'):
                        if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
                            mask2d = mask_arr[0]
                        elif mask_arr.ndim == 2:
                            mask2d = mask_arr
                        elif mask_arr.ndim == 3:
                            mask2d = mask_arr[0]
                        else:
                            w, h = img_frame.size
                            mask2d = np.zeros((h, w), dtype=bool)
                    else:
                        w, h = img_frame.size
                        mask2d = np.zeros((h, w), dtype=bool)
                else:
                    w, h = img_frame.size
                    mask2d = np.zeros((h, w), dtype=bool)
                per_frame_masks.append(mask2d)

            stacked_masks = np.stack(per_frame_masks, axis=0) if len(per_frame_masks) > 0 else []
            final_pred = predictions[-1] if len(predictions) > 0 else ''
            if any_seg and '[SEG]' not in final_pred:
                final_pred = final_pred + ' [SEG]'
            result = {'prediction': final_pred, 'prediction_masks': [stacked_masks] if isinstance(stacked_masks, np.ndarray) else []}
        else:
            if cfg.keyframes_json is not None and os.path.isfile(cfg.keyframes_json):
                if is_qwen3_model:
                    result = Sa2VAChatModelQwen3.segment_with_keyframes(
                        model,
                        video=vid_frames,
                        processor=processor,
                        keyframes_json=cfg.keyframes_json,
                        enable_query_prompts=bool(cfg.query_prompts),
                        query_interval=int(cfg.query_interval),
                    )
                elif is_qwen2_5_model:
                    result = Sa2VAChatModelQwen.segment_with_keyframes(
                        model,
                        video=vid_frames,
                        processor=processor,
                        keyframes_json=cfg.keyframes_json,
                        enable_query_prompts=bool(cfg.query_prompts),
                        query_interval=int(cfg.query_interval),
                    )
                else:
                    result = Sa2VAChatModel.segment_with_keyframes(
                        model,
                        video=vid_frames,
                        tokenizer=tokenizer,
                        keyframes_json=cfg.keyframes_json,
                        enable_query_prompts=bool(cfg.query_prompts),
                        query_interval=int(cfg.query_interval),
                    )
            else:
                if is_qwen3_model or is_qwen2_5_model:
                    result = model.predict_forward(
                        video=vid_frames,
                        text=cfg.text,
                        processor=processor,
                    )
                else:
                    result = Sa2VAChatModel.predict_forward(
                        model,
                        video=vid_frames,
                        text=cfg.text,
                        tokenizer=tokenizer,
                    )

    prediction = result['prediction']
    print(f"The output is:\n{prediction}")

    if '[SEG]' in prediction and Visualizer is not None:
        _seg_idx = 0
        pred_masks_field = result['prediction_masks']

        # Check if prediction_masks is empty (no objects to segment)
        if not pred_masks_field or len(pred_masks_field) == 0:
            print("No objects detected for segmentation, skipping visualization.")
            pred_masks = None
        else:
            # Support multiple objects by unioning masks across objects when needed
            if isinstance(pred_masks_field, list) and len(pred_masks_field) > 0 and hasattr(pred_masks_field[0], 'ndim'):
                if isinstance(pred_masks_field[0], np.ndarray) and pred_masks_field[0].ndim == 3:
                    # List of [T, H, W] -> union across objects -> [T, H, W]
                    try:
                        pred_masks = np.any(np.stack(pred_masks_field, axis=0), axis=0)
                    except Exception:
                        pred_masks = pred_masks_field[_seg_idx]
                else:
                    pred_masks = pred_masks_field[_seg_idx]
            else:
                pred_masks = pred_masks_field[_seg_idx]

        video_frames = []
        
        if cfg.work_dir:
            output_dir = cfg.work_dir
        else:
            output_dir = './temp_visualize_results'
        os.makedirs(output_dir, exist_ok=True)

        # Skip visualization if no masks were generated
        if pred_masks is None:
            print("Skipping video generation as no segmentation masks were produced.")
        else:
            # NEW: store binary mask-only frames for evaluation video
            mask_frames = []

            for frame_idx in range(len(vid_frames)):
                pred_mask = pred_masks[frame_idx]
                visual_result_bgr = visualize_frame(pred_mask, raw_frames_bgr[frame_idx])
                # Convert to RGB for imageio writer
                visual_result_rgb = cv2.cvtColor(visual_result_bgr, cv2.COLOR_BGR2RGB)
                video_frames.append(visual_result_rgb)

                # Build 3-channel binary mask frame (0/255) for evaluation
                mask_uint8 = (pred_mask.astype(np.uint8) * 255) if hasattr(pred_mask, 'astype') else (np.array(pred_mask, dtype=np.uint8) * 255)
                mask_rgb = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)
                mask_frames.append(mask_rgb)

                if cfg.save_frames:
                    # Save BGR image with OpenCV
                    save_individual_frame(visual_result_bgr, frame_basenames[frame_idx], output_dir)

            video_output_path = os.path.join(output_dir, 'segmentation_video.mp4')
            create_video_from_frames(video_frames, video_output_path, cfg.fps)

            # NEW: write mask-only video for evaluation (J/F)
            mask_output_path = os.path.join(output_dir, 'segmentation_mask.mp4')
            create_video_from_frames(mask_frames, mask_output_path, cfg.fps)

    else:
        print("No segmentation found in prediction or Visualizer not available") 