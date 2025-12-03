import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from third_parts.video_io import VideoReader

if TYPE_CHECKING:
  from PIL.Image import Image as PILImageType


def _encode_pil_to_jpeg_base64(img: 'PILImageType') -> str:
  """Encode a PIL image to base64 JPEG string."""
  import io
  import base64
  buf = io.BytesIO()
  img.convert('RGB').save(buf, format='JPEG', quality=90)
  return base64.b64encode(buf.getvalue()).decode('utf-8')


def _build_internvl_transform(input_size: int = 448):
  """Build transform for InternVL3.5 image preprocessing."""
  import torchvision.transforms as T
  from torchvision.transforms.functional import InterpolationMode

  IMAGENET_MEAN = (0.485, 0.456, 0.406)
  IMAGENET_STD = (0.229, 0.224, 0.225)

  transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
  ])
  return transform


def _find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[Tuple[int, int]], width: int, height: int, image_size: int) -> Tuple[int, int]:
  """Find the closest aspect ratio for InternVL3.5 dynamic preprocessing."""
  best_ratio_diff = float('inf')
  best_ratio = (1, 1)
  area = width * height

  for ratio in target_ratios:
    target_aspect_ratio = ratio[0] / ratio[1]
    ratio_diff = abs(aspect_ratio - target_aspect_ratio)
    if ratio_diff < best_ratio_diff:
      best_ratio_diff = ratio_diff
      best_ratio = ratio
    elif ratio_diff == best_ratio_diff:
      if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
        best_ratio = ratio
  return best_ratio


def _dynamic_preprocess_internvl(image: 'PILImageType', min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False) -> List['PILImageType']:
  """Dynamic preprocessing for InternVL3.5."""
  orig_width, orig_height = image.size
  aspect_ratio = orig_width / orig_height

  # Calculate the existing image aspect ratio
  target_ratios = set(
    (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
    i * j <= max_num and i * j >= min_num)
  target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

  # Find the closest aspect ratio to the target
  target_aspect_ratio = _find_closest_aspect_ratio(
    aspect_ratio, target_ratios, orig_width, orig_height, image_size)

  # Calculate the target width and height
  target_width = image_size * target_aspect_ratio[0]
  target_height = image_size * target_aspect_ratio[1]
  blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

  # Resize the image
  resized_img = image.resize((target_width, target_height))
  processed_images = []

  for i in range(blocks):
    box = (
      (i % (target_width // image_size)) * image_size,
      (i // (target_width // image_size)) * image_size,
      ((i % (target_width // image_size)) + 1) * image_size,
      ((i // (target_width // image_size)) + 1) * image_size
    )
    # Split the image
    split_img = resized_img.crop(box)
    processed_images.append(split_img)

  assert len(processed_images) == blocks
  if use_thumbnail and len(processed_images) != 1:
    thumbnail_img = image.resize((image_size, image_size))
    processed_images.append(thumbnail_img)

  return processed_images


def _load_image_internvl(image: 'PILImageType', input_size: int = 448, max_num: int = 12) -> torch.Tensor:
  """Load and preprocess image for InternVL3.5."""
  transform = _build_internvl_transform(input_size=input_size)
  images = _dynamic_preprocess_internvl(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
  pixel_values = [transform(img) for img in images]
  pixel_values = torch.stack(pixel_values)
  return pixel_values


def _ensure_dir(path: Path) -> None:
  path.mkdir(parents=True, exist_ok=True)


def _uniform_sample_indices(total: int, num_samples: int) -> List[int]:
  if num_samples >= total:
    return list(range(total))
  # Evenly spaced indices over [0, total-1]
  positions = np.linspace(0, total - 1, num=num_samples)
  indices = sorted({int(round(p)) for p in positions})
  # Ensure cardinality by adding neighbors if deduped too much
  i = 0
  while len(indices) < num_samples and i < total:
    if i not in indices:
      indices.append(i)
    i += max(1, total // max(2, num_samples))
  return sorted(indices[:num_samples])


# New: sample indices by a fixed stride N (every N frames)
def _stride_sample_indices(total: int, stride: int) -> List[int]:
  """Return indices sampled every `stride` frames starting from 0.

  The sampling is 0-based: [0, stride, 2*stride, ...] and stops before `total`.
  The number of sampled frames is floor((total - 1) / stride) + 1 when total > 0.
  """
  if total <= 0:
    return []
  s = max(1, int(stride))
  return list(range(0, total, s))


def _load_images_sorted(images_dir: Path) -> List[Path]:
  if images_dir.is_dir():
    frame_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    ])
    assert len(frame_paths) > 0, f'No images found under {images_dir}'
    return frame_paths

  # If not a directory, allow a single video file path
  assert images_dir.is_file(), f'Path not found: {images_dir}'
  video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpg', '.mpeg', '.m4v'}
  is_video = images_dir.suffix.lower() in video_exts
  assert is_video, f'Unsupported path (expect images dir or video): {images_dir}'

  # Extract frames next to the video to a deterministic folder
  frames_dir = images_dir.parent / f"{images_dir.stem}_frames"
  _ensure_dir(frames_dir)

  # If frames already exist, reuse them; otherwise extract
  existing = sorted([
      p for p in frames_dir.iterdir()
      if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
  ])
  if len(existing) == 0:
    vr = VideoReader(str(images_dir))
    assert vr.opened and vr.frame_cnt > 0, f'Failed to open video or empty: {images_dir}'
    # Save as sequential JPEG files
    vr.cvt2frames(str(frames_dir), file_start=0, filename_tmpl='{:06d}.jpg', start=0, max_num=0, show_progress=False)
    existing = sorted([
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    ])
  assert len(existing) > 0, f'No frames extracted from video: {images_dir}'
  return existing


def _build_sequence_prompt(num_frames: int, user_query: str) -> str:
  instructions = (
      "You will act as a keyframe selection agent for a video reasoning task. During each inference, you will "
      "be given multiple keyframes sampled from a long video. The keyframes are "
      "presented as separate images in strict temporal order (Frame 1, Frame 2, ...). "
      "You need to find all moving objects in the given video. "
      "You need to think in chain of thoughts to analyze each keyframe and find the best keyframes for each "
      "target object, where a segmentation model can find the target object in that frame with less effort. "
      "Your chain of thoughts should begin with what can be seen in each keyframe, how many objects in "
      "total. etc. Some of the objects may be seriously obscured "
      "or blocked by other objects. Some of the objects may be camouflaged in their surroundings. Analyze "
      "each frame separately to get all the visible objects. This chain of thoughts should follow the output "
      "format:\n"
      '"Chain of Thoughts:\n'
      '- Frame 1: <analysis of frame 1>;\n'
      '- Frame 2: <analysis of frame 2>;\n'
      '...\". For the analysis of each frame, you also have to follow the chain-of-thought format:\n'
      '"- *<question 1>* <answer 1>;\n'
      '- *<question 2>* <answer 2>;\n'
      '...\', where you have to ask questions to yourself and answer them. Your answer should be as detailed as '
      'possible. You should start with broader questions, like "what can be seen in the frame?" to some '
      'detailed questions like "how many and which '
      'objects are moving?", "What is the relationship between these objects?", etc. There will be many questions and answers in the analysis of each frame, '
      "helping you to fully understand the frame. The actual questions vary by cases. Generate the questions and answers based on the analysis of each frame. "
      "Additionally, determine whether any target object becomes fully occluded or goes out of frame and later reappears. "
      "Count the number of continuous visibility periods for each object (from first appearance to disappearance/end, then from reappearance to next disappearance/end). "
      "Your thinking process aims to find the keyframe for each target object of interest "
      "(find all the target objects, each of which may correspond to one keyframe per continuous visibility period) related to the user query. Lastly, you have to output a list of dictionaries with the format:\n"
      '"Output list: [{object_index: 1, keyframe: k_1, object_description: <description of the object 1 in '
      'keyframe k_1>}, {object_index: 2, keyframe: k_2, object_description: <description of the object 2 in '
      'keyframe k_2>}, ...]"\n'
      "where each element in the list is a dictionary with three items: object index, keyframe index, "
      "object description. k is the k-th keyframe in the sequence (1-based index), counted by temporal order from Frame 1 to Frame N. "
      "object_index is a numbering integer starting with 1. object_description implies the description for that object in a particular frame, "
      'helping the model to find the object in that particular frame. For example, a valid element in an '
      'output list can be like \'Output list: [{object_index: 1, keyframe: 4, object_description: "the man at '
      'the top left corner of the image"}]\'. In case there are multiple objects in the same keyframe, you need to discriniate and describe them separetly. DO NOT describe multiple objects in the same keyframe as a single object. '
      'Here are some badcases: \'Output list: [{object_index: 1, keyframe: 1, object_description: "a group of penguins on ice, period 1 (first visible)"}\''
      '\'Output list: [{object_index: 1, keyframe: 1, object_description: "the person on the bicycle, clearly visible"}]\', a better one should be \'Output list: [{object_index: 1, keyframe: 1, object_description: "the person on the bicycle"}, {object_index: 2, keyframe: 1, object_description: "the bicycle"}]\''
      '\'Output list: [{object_index: 1, keyframe: 1, object_description: "the blue cart with two people approaching the ramp"}]\', a better one should be \'Output list: [{object_index: 1, keyframe: 1, object_description: "the blue cart"}, {object_index: 2, keyframe: 1, object_description: "the people "}]\''
      "Include the objects even if they are only partially visible. If an object disappears (fully occluded or out of frame) and later reappears, "
      "treat each continuous visibility span as a separate period and select one best keyframe per period. Reuse the same object_index for all periods of the same object and output multiple entries—one per period. "
      "Consider the period in your question when analyzing the frame. While choosing the keyframe for any object, you should prioritize those frames where objects are not overlapped. This will help the model to "
      "better recognize the object. Keep the output list in text format. Don't use JSON formatting. The output "
      'list begins with the prefix "Output list: ", followed by a square bracket with multiple curly brackets. '
      'The square bracket should be in the same line, following the format "Output list: [...]". Don\'t start '
      "with a new line.\n"
      + f"Here is a sequence with {num_frames} keyframes presented as separate images in temporal order. "
      + "Follow the instruction and output the index of the best keyframe."
  )
  return instructions


def _resize_image_keep_rgb(path: Path, resize_size: int) -> 'PILImageType':
  img = PILImage.open(path).convert('RGB')
  s = int(resize_size)
  w, h = img.size
  if w <= 0 or h <= 0 or s <= 0:
    return img
  scale = min(s / float(w), s / float(h))
  new_w = max(1, int(round(w * scale)))
  new_h = max(1, int(round(h * scale)))
  resized = img.resize((new_w, new_h), PILImage.BILINEAR)
  canvas = PILImage.new('RGB', (s, s), (0, 0, 0))
  paste_x = (s - new_w) // 2
  paste_y = (s - new_h) // 2
  canvas.paste(resized, (paste_x, paste_y))
  return canvas


def _prepare_message_content(
  images: List['PILImageType'],
  user_query: str,
  use_base64_sequence: bool = False,
) -> Tuple[List[Dict], str]:
  """Return (content, prompt_text) for the chat message."""
  prompt = _build_sequence_prompt(num_frames=len(images), user_query=user_query)
  # Optionally send as pre-encoded base64 image_url entries (GPT-4o typical)
  if use_base64_sequence:
    content = [{'type': 'text', 'text': prompt}]
    for img in images:
      b64 = _encode_pil_to_jpeg_base64(img)
      content.append({
        'type': 'image_url',
        'image_url': {
          'url': f"data:image/jpeg;base64,{b64}",
          'detail': 'high',
        }
      })
    return content, prompt
  else:
    content: List[Dict] = []
    for img in images:
      content.append({'type': 'image', 'image': img})
    content.append({'type': 'text', 'text': prompt})
    return content, prompt


def _run_mllm(
  messages: List[Dict],
  model_path: str,
  max_new_tokens: int,
  vlm_type: str,
  openai_api_key: str = None,
  openai_model: str = 'gpt-4o',
  google_api_key: str = None,
  google_model: str = 'gemini-2.5-pro',
  max_retries: int = 5,
  retry_initial_delay: float = 1.0,
  retry_backoff_base: float = 2.0,
  openai_image_format: str = 'png',
) -> Tuple[str, Dict]:
  vlm = (vlm_type or 'qwen').lower()

  if vlm in ('gpt4o', 'openai', 'gpt-4o'):
    # OpenAI GPT-4o path (image + text via Chat Completions API)
    import io
    import base64

    # Extract first conversation, text prompt, and images (potentially multiple)
    conversation = messages[0] if isinstance(messages, list) and len(messages) > 0 else messages
    if isinstance(conversation, list):
      conv_msgs = conversation
    elif isinstance(conversation, dict):
      conv_msgs = [conversation]
    else:
      conv_msgs = []

    user_contents = []
    for m in conv_msgs:
      if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), list):
        user_contents.extend(m['content'])

    prompt_texts: List[str] = []
    pil_images: List['PILImageType'] = []
    passthrough_image_urls: List[Dict] = []
    for c in user_contents:
      if not isinstance(c, dict):
        continue
      t = c.get('type')
      if t == 'text' and isinstance(c.get('text'), str):
        prompt_texts.append(c['text'])
      elif t in ('image', 'input_image'):
        img_obj = c.get('image') or c.get('image_url')
        if isinstance(img_obj, PILImage.Image):
          pil_images.append(img_obj)
        elif isinstance(img_obj, np.ndarray):
          pil_images.append(PILImage.fromarray(img_obj))
      elif t == 'image_url' and isinstance(c.get('image_url'), (dict, str)):
        # Pass through already-encoded image_url items as-is
        image_url_obj = c.get('image_url')
        if isinstance(image_url_obj, str):
          passthrough_image_urls.append({
            'type': 'image_url',
            'image_url': {
              'url': image_url_obj,
              'detail': 'high',
            }
          })
        else:
          # assume dict with url + optional detail
          url = image_url_obj.get('url')
          detail = image_url_obj.get('detail', 'high')
          if isinstance(url, str) and url:
            passthrough_image_urls.append({
              'type': 'image_url',
              'image_url': {
                'url': url,
                'detail': detail,
              }
            })
    prompt_combined = "\n\n".join(prompt_texts).strip()
    # Compute text token count deterministically when possible (prefer o200k_base for GPT-4o)
    text_tokens_est = None
    try:
      import tiktoken
      try:
        enc = tiktoken.get_encoding('o200k_base')
      except Exception:
        enc = tiktoken.get_encoding('cl100k_base')
      text_tokens_est = len(enc.encode(prompt_combined))
    except Exception:
      text_tokens_est = None

    # Encode images as data URLs for Chat Completions
    image_data_urls: List[str] = []
    for pil_image in pil_images:
      buf = io.BytesIO()
      fmt = (openai_image_format or 'png').upper()
      if fmt not in ('PNG', 'JPEG', 'JPG'):
        fmt = 'PNG'
      pil_image.convert('RGB').save(buf, format=('JPEG' if fmt in ('JPEG', 'JPG') else 'PNG'))
      b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
      mime = 'image/jpeg' if fmt in ('JPEG', 'JPG') else 'image/png'
      image_data_urls.append(f"data:{mime};base64,{b64}")
    # Count how many visual inputs are included
    num_input_images = len(image_data_urls) + len(passthrough_image_urls)

    import time
    import random
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    # Build messages for Chat Completions API
    content_items = []
    if prompt_combined:
      content_items.append({"type": "text", "text": prompt_combined})
    # Include any pre-encoded image_url items (e.g., base64 JPEGs)
    if passthrough_image_urls:
      content_items.extend(passthrough_image_urls)
    for url in image_data_urls:
      content_items.append({
        "type": "image_url",
        "image_url": {"url": url, "detail": "high"}
      })

    chat_messages = [
      {"role": "user", "content": content_items}
    ]

    def _is_rate_limit_error(err: Exception) -> bool:
      msg = str(err).lower()
      return ('rate limit' in msg) or ('429' in msg)

    last_err: Exception = None
    response = None
    delay = max(0.0, float(retry_initial_delay))
    for attempt in range(max(1, int(max_retries))):
      try:
        response = client.chat.completions.create(
          model=openai_model or 'gpt-4o',
          messages=chat_messages,
          max_tokens=max_new_tokens,
          temperature=0.1,
        )
        break
      except Exception as e:
        last_err = e
        if _is_rate_limit_error(e) and attempt < max(int(max_retries) - 1, 0):
          jitter = random.uniform(0.0, 0.25 * delay)
          time.sleep(delay + jitter)
          delay = delay * (retry_backoff_base if retry_backoff_base and retry_backoff_base > 1.0 else 2.0)
          continue
        else:
          return f"[GPT-4o API error] {e}", {
            'estimation_method': 'api_error',
            'note': 'Request failed before usage could be recorded',
          }

    if not response or not getattr(response, 'choices', None):
      return "", {
        'estimation_method': 'empty_response',
        'note': 'Empty response from API',
      }

    output_text = response.choices[0].message.content or ""

    # Try to read token usage from API
    usage_block = getattr(response, 'usage', None)
    usage_dict: Dict = {}
    if usage_block is not None:
      # OpenAI Chat Completions return prompt_tokens/completion_tokens/total_tokens
      input_tokens = getattr(usage_block, 'prompt_tokens', None)
      output_tokens = getattr(usage_block, 'completion_tokens', None)
      total_tokens = getattr(usage_block, 'total_tokens', None)
      if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
      # Try to extract any available per-modality token details if provided by the API
      input_text_tokens = None
      input_visual_tokens = None
      input_visual_tokens_method = None
      try:
        details = getattr(usage_block, 'prompt_tokens_details', None)
        if details is None and hasattr(usage_block, 'model_dump'):
          dumped = usage_block.model_dump()
          details = dumped.get('prompt_tokens_details') if isinstance(dumped, dict) else None
        if details is not None:
          # details may be an object or dict; try common fields
          if isinstance(details, dict):
            input_text_tokens = details.get('text_tokens') or details.get('text')
            input_visual_tokens = details.get('image_tokens') or details.get('vision_tokens') or details.get('image')
          else:
            input_text_tokens = getattr(details, 'text_tokens', None)
            if input_text_tokens is None:
              input_text_tokens = getattr(details, 'text', None)
            input_visual_tokens = getattr(details, 'image_tokens', None)
            if input_visual_tokens is None:
              input_visual_tokens = getattr(details, 'vision_tokens', None)
          if input_visual_tokens is not None:
            input_visual_tokens_method = 'api'
      except Exception:
        input_text_tokens = None
        input_visual_tokens = None
      # If API didn't provide text/visual breakdown, fall back to local text count
      # If API didn't provide text/visual breakdown, fall back to estimates
      if input_text_tokens is None:
        input_text_tokens = text_tokens_est
      # Visual tokens are generally not provided by Chat Completions; if missing, infer as prompt_tokens - text_tokens
      if input_visual_tokens is None and (input_tokens is not None) and (input_text_tokens is not None):
        try:
          inferred = int(input_tokens) - int(input_text_tokens)
          input_visual_tokens = inferred if inferred > 0 else 0
          input_visual_tokens_method = 'inferred'
        except Exception:
          input_visual_tokens = None
      usage_dict = {
        'estimation_method': 'api_usage',
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'input_text_tokens': input_text_tokens,
        'input_visual_tokens': input_visual_tokens,
        'input_visual_tokens_method': input_visual_tokens_method,
        'input_images': num_input_images,
        'note': getattr(usage_block, 'note', None) or 'Per-modality counts may be missing; text tokens computed with o200k_base when available. Visual tokens often not reported.',
      }
    else:
      # Fallback: estimate tokens from text using cl100k_base; images excluded
      try:
        import tiktoken
        try:
          enc = tiktoken.get_encoding('o200k_base')
        except Exception:
          enc = tiktoken.get_encoding('cl100k_base')
        approx_in = len(enc.encode(prompt_combined))
        approx_out = len(enc.encode(output_text))
      except Exception:
        approx_in = len(prompt_combined.split())
        approx_out = len(output_text.split())
      usage_dict = {
        'estimation_method': 'text_only_estimate',
        'note': 'Images not included in token estimate; visual tokens unavailable.',
        'input_tokens': approx_in,
        'output_tokens': approx_out,
        'total_tokens': approx_in + approx_out,
        'input_text_tokens': approx_in,
        'input_visual_tokens': None,
        'input_images': num_input_images,
      }

    return output_text, usage_dict

  if vlm in ('gemini', 'google', 'gemini-2.5-pro'):
    # Google Gemini path (multimodal via google-generativeai)
    import io
    import base64
    try:
      import google.generativeai as genai
    except Exception as e:
      return f"[Gemini import error] {e}", {
        'estimation_method': 'api_error',
        'note': 'google-generativeai not installed',
      }

    # Extract prompt text and images from our message format
    conversation = messages[0] if isinstance(messages, list) and len(messages) > 0 else messages
    if isinstance(conversation, list):
      conv_msgs = conversation
    elif isinstance(conversation, dict):
      conv_msgs = [conversation]
    else:
      conv_msgs = []

    user_contents = []
    for m in conv_msgs:
      if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), list):
        user_contents.extend(m['content'])

    prompt_texts: List[str] = []
    pil_images: List['PILImageType'] = []
    for c in user_contents:
      if not isinstance(c, dict):
        continue
      t = c.get('type')
      if t == 'text' and isinstance(c.get('text'), str):
        prompt_texts.append(c['text'])
      elif t in ('image', 'input_image'):
        img_obj = c.get('image') or c.get('image_url')
        if isinstance(img_obj, PILImage.Image):
          pil_images.append(img_obj)
        elif isinstance(img_obj, np.ndarray):
          pil_images.append(PILImage.fromarray(img_obj))
      elif t == 'image_url':
        # Not supported directly in Gemini without fetching; ignore here
        pass

    prompt_combined = "\n\n".join(prompt_texts).strip()

    # Configure client
    try:
      genai.configure(api_key=google_api_key)
      model_name = google_model or 'gemini-2.5-pro'
      model = genai.GenerativeModel(model_name)
    except Exception as e:
      return f"[Gemini client error] {e}", {
        'estimation_method': 'api_error',
        'note': 'Failed to configure Gemini client',
      }

    # Build parts: text + inline images
    parts: List = []
    if prompt_combined:
      parts.append(prompt_combined)

    img_fmt = (openai_image_format or 'png').upper()
    if img_fmt not in ('PNG', 'JPEG', 'JPG'):
      img_fmt = 'PNG'
    mime = 'image/jpeg' if img_fmt in ('JPEG', 'JPG') else 'image/png'

    for pil_image in pil_images:
      buf = io.BytesIO()
      pil_image.convert('RGB').save(buf, format=('JPEG' if img_fmt in ('JPEG', 'JPG') else 'PNG'))
      parts.append({
        'mime_type': mime,
        'data': buf.getvalue(),
      })

    try:
      response = model.generate_content(
        parts,
        generation_config={
          'max_output_tokens': max_new_tokens,
          'temperature': 0.1,
        },
      )
    except Exception as e:
      return f"[Gemini API error] {e}", {
        'estimation_method': 'api_error',
        'note': 'Request failed before usage could be recorded',
      }

    output_text = getattr(response, 'text', None) or ""

    # Token usage
    um = getattr(response, 'usage_metadata', None)
    input_tokens = None
    output_tokens = None
    total_tokens = None
    if um is not None:
      # Different SDK versions expose different field names
      input_tokens = getattr(um, 'prompt_token_count', None)
      if input_tokens is None:
        input_tokens = getattr(um, 'input_token_count', None)
      output_tokens = getattr(um, 'candidates_token_count', None)
      if output_tokens is None:
        output_tokens = getattr(um, 'output_token_count', None)
      total_tokens = getattr(um, 'total_token_count', None)
      try:
        # Sometimes usage is a dict
        if input_tokens is None:
          input_tokens = um.get('prompt_token_count') if isinstance(um, dict) else None
        if input_tokens is None and isinstance(um, dict):
          input_tokens = um.get('input_token_count')
        if output_tokens is None and isinstance(um, dict):
          output_tokens = um.get('candidates_token_count') or um.get('output_token_count')
        if total_tokens is None and isinstance(um, dict):
          total_tokens = um.get('total_token_count')
      except Exception:
        pass

    usage_dict = {
      'estimation_method': 'api_usage',
      'note': 'Gemini usage from usage_metadata when available.',
      'input_tokens': input_tokens,
      'output_tokens': output_tokens,
      'total_tokens': total_tokens,
      'input_text_tokens': None,
      'input_visual_tokens': None,
      'input_images': len(pil_images),
    }

    return output_text, usage_dict

  if vlm == 'internvl':
    # InternVL3.5 path
    from transformers import AutoModel, AutoTokenizer

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # Extract first conversation, text prompt, and images
    conversation = messages[0] if isinstance(messages, list) and len(messages) > 0 else messages
    if isinstance(conversation, list):
      conv_msgs = conversation
    elif isinstance(conversation, dict):
      conv_msgs = [conversation]
    else:
      conv_msgs = []

    user_contents = []
    for m in conv_msgs:
      if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), list):
        user_contents.extend(m['content'])

    prompt_texts: List[str] = []
    pil_images: List['PILImageType'] = []
    for c in user_contents:
      if not isinstance(c, dict):
        continue
      t = c.get('type')
      if t == 'text' and isinstance(c.get('text'), str):
        prompt_texts.append(c['text'])
      elif t in ('image', 'input_image'):
        img_obj = c.get('image') or c.get('image_url')
        if isinstance(img_obj, PILImage.Image):
          pil_images.append(img_obj)
        elif isinstance(img_obj, np.ndarray):
          pil_images.append(PILImage.fromarray(img_obj))

    prompt_combined = "\n\n".join(prompt_texts).strip()

    # Process images for InternVL3.5
    if pil_images:
      # Concatenate all processed images
      pixel_values_list = []
      for pil_image in pil_images:
        pixel_values = _load_image_internvl(pil_image, input_size=448, max_num=12)
        pixel_values_list.append(pixel_values)

      # Concatenate all pixel values
      pixel_values = torch.cat(pixel_values_list, dim=0)
      pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

      # Add image tokens to prompt
      num_images = len(pil_images)
      if num_images == 1:
        question = f"<image>\n{prompt_combined}"
      else:
        # Multiple images
        image_tokens = "".join([f"<image>" for _ in range(num_images)])
        question = f"{image_tokens}\n{prompt_combined}"
    else:
      pixel_values = None
      question = prompt_combined

    # Generate response
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    try:
      response = model.chat(tokenizer, pixel_values, question, generation_config)
      output_text = response if isinstance(response, str) else str(response)
    except Exception as e:
      return f"[InternVL3.5 error] {e}", {
        'estimation_method': 'api_error',
        'note': 'InternVL3.5 generation failed',
      }

    return output_text, {
      'estimation_method': 'not_applicable',
      'note': 'Local model; token usage not available',
    }

  if vlm == 'qwen3moe':
    # Qwen3-VL-235B MoE path
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

    reasoning_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    # messages is a batch-of-conversations; take the first conversation
    conversation = messages[0] if isinstance(messages, list) and len(messages) > 0 else messages

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt'
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else reasoning_model.device)

    with torch.no_grad():
      generated_ids = reasoning_model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=False,
      )
    generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (output_texts[0] if output_texts else ""), {
      'estimation_method': 'not_applicable',
      'note': 'Local model; token usage not available',
    }

  if vlm == 'qwen3':
    # Qwen3-VL path (non-MoE)
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    reasoning_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()
    processor = AutoProcessor.from_pretrained(model_path)

    # messages is a batch-of-conversations; take the first conversation
    conversation = messages[0] if isinstance(messages, list) and len(messages) > 0 else messages

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt'
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else reasoning_model.device)

    with torch.no_grad():
      generated_ids = reasoning_model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=False,
      )
    generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (output_texts[0] if output_texts else ""), {
      'estimation_method': 'not_applicable',
      'note': 'Local model; token usage not available',
    }

  if vlm == 'gemma':
    # Gemma 3 VLM path
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor

    reasoning_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map='auto',
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)

    # messages is a batch-of-conversations; take the first conversation
    conversation = messages[0] if isinstance(messages, list) and len(messages) > 0 else messages

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt'
    ).to("cuda" if torch.cuda.is_available() else reasoning_model.device)

    input_len = inputs["input_ids"].shape[-1]

    # Ensure the model can generate all requested new tokens by explicitly
    # setting max_length to current input length + max_new_tokens.
    eos_id = getattr(processor.tokenizer, 'eos_token_id', None)
    pad_id = getattr(processor.tokenizer, 'pad_token_id', None)
    
    with torch.no_grad():
      generation = reasoning_model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          max_length=input_len + max_new_tokens,
          eos_token_id=eos_id,
          pad_token_id=pad_id if pad_id is not None else eos_id,
          do_sample=False,
      )
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded, {
      'estimation_method': 'not_applicable',
      'note': 'Local model; token usage not available',
    }

  if vlm == 'qwen':
    # Qwen2.5-VL path
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    text_inputs = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text_inputs,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else reasoning_model.device)

    with torch.no_grad():
      generated_ids = reasoning_model.generate(
          **inputs,
          use_cache=True,
          max_new_tokens=max_new_tokens,
          do_sample=False,
      )
    generated_ids_trimmed = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (output_texts[0] if output_texts else ""), {
      'estimation_method': 'not_applicable',
      'note': 'Local model; token usage not available',
    }

  # Fallback for unknown vlm types
  return f"[Error] Unknown VLM type: {vlm}", {
    'estimation_method': 'error',
    'note': f'Unsupported VLM type: {vlm}',
  }


def _parse_output(text: str) -> Tuple[Dict, str]:
  """Return (output_json, think_text).

  This prioritizes parsing a free-form "Output list: [...]" section as instructed
  by the prompt, converting it into {"instances": [...]}.
  Falls back to attempting JSON extraction if present.
  """
  import re
  think_text = ""
  output_obj: Dict = {}

  # Extract <think>...</think>
  m_think = re.search(r"<think>([\s\S]*?)</think>", text)
  if m_think:
    think_text = m_think.group(1).strip()

  instances: List[Dict] = []

  # First, try to parse: Output list: [{object_index: 1, keyframe: 1, object_description: "..."}, ...]
  m_output_list = re.search(r"Output\s*list\s*:\s*\[(.*?)\]", text, flags=re.IGNORECASE | re.DOTALL)
  if m_output_list:
    body = m_output_list.group(1)
    for m_item in re.finditer(r"\{(.*?)\}", body, flags=re.DOTALL):
      item = m_item.group(1)
      m_idx = re.search(r"object_index\s*:\s*([0-9]+)", item, flags=re.IGNORECASE)
      m_kf = re.search(r"keyframe\s*:\s*([0-9]+)", item, flags=re.IGNORECASE)
      m_desc = re.search(
        r"object_description\s*:\s*(\"([^\"]*)\"|'([^']*)'|([^,\}]+))",
        item,
        flags=re.IGNORECASE | re.DOTALL,
      )
      if m_idx and m_kf and m_desc:
        try:
          obj_idx = int(m_idx.group(1))
          keyframe_idx = int(m_kf.group(1))
        except Exception:
          continue
        desc = m_desc.group(2) or m_desc.group(3) or m_desc.group(4) or ""
        desc = desc.strip()
        instances.append({
          'object_index': obj_idx,
          'keyframe': keyframe_idx,
          'object_description': desc,
        })

  if instances:
    return {'instances': instances}, think_text

  # If no "Output list" parsed, fall back to JSON inside <output>...</output>
  m_out = re.search(r"<output>([\s\S]*?)</output>", text)
  json_str = None
  if m_out:
    candidate = m_out.group(1).strip()
    brace_start = candidate.find('{')
    brace_end = candidate.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
      json_str = candidate[brace_start:brace_end + 1]

  # Fallback: try to find first JSON object anywhere
  if json_str is None:
    m_any = re.search(r"\{[\s\S]*\}", text)
    if m_any:
      json_str = m_any.group(0)

  if json_str is not None:
    try:
      output_obj = json.loads(json_str)
    except Exception:
      output_obj = {}

  return output_obj, think_text


def _map_instances_to_global(instances: List[Dict], sample_indices: List[int]) -> List[Dict]:
  mapped = []
  for inst in instances:
    try:
      kf = int(inst.get('keyframe', 0))
      obj_idx = int(inst.get('object_index', 0))
      desc = str(inst.get('object_description', '')).strip()
    except Exception:
      continue
    if 1 <= kf <= len(sample_indices):
      global_idx = sample_indices[kf - 1]
      mapped.append({
        'object_index': obj_idx,
        'sample_keyframe': kf,
        'global_frame_index': global_idx,
        'object_description': desc,
      })
  return mapped


def main():
  parser = argparse.ArgumentParser(description='Dynamic Scene Reasoning')
  parser.add_argument('--scene_name', type=str, required=True, help='Scene name used under outputs directory.')
  parser.add_argument('--images_dir', type=str, required=True, help='Directory with input frames (sorted by name).')
  parser.add_argument('--query', type=str, help='User query for reasoning selection.')
  parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='HF path to the VLM model.')
  parser.add_argument('--vlm', type=str, default='qwen', choices=['qwen', 'qwen3', 'qwen3moe', 'gemma', 'gpt4o', 'gemini', 'internvl'], help='Which VLM backend to use.')
  parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API key (required when --vlm gpt4o).')
  parser.add_argument('--openai_model', type=str, default='gpt-4o', help='OpenAI model to use (e.g., gpt-4o).')
  parser.add_argument('--google_api_key', type=str, default=None, help='Google API key for Gemini (required when --vlm gemini).')
  parser.add_argument('--google_model', type=str, default='gemini-2.5-pro', help='Google Gemini model name (e.g., gemini-2.5-pro).')
  parser.add_argument('--num_samples', type=int, default=8, help='Number of frames to sample uniformly from the sequence.')
  parser.add_argument('--resize_size', type=int, default=840, help='Resize images to (S,S) for the MLLM input.')
  parser.add_argument('--max_new_tokens', type=int, default=768, help='Max new tokens for generation.')

  # Retry and backoff controls (for OpenAI path)
  parser.add_argument('--max_retries', type=int, default=5, help='Max retries on rate limit or transient errors.')
  parser.add_argument('--retry_initial_delay', type=float, default=1.0, help='Initial delay (seconds) before first retry.')
  parser.add_argument('--retry_backoff_base', type=float, default=2.0, help='Exponential backoff base for retries.')
  # Pricing controls (USD per 1K tokens)
  parser.add_argument('--pricing_mode', type=str, default='auto', choices=['auto', 'manual'], help='Use model-based pricing (auto) or manual per-1K values.')
  parser.add_argument('--price_input_per_1k', type=float, default=0.005, help='USD cost per 1K input tokens (manual mode).')
  parser.add_argument('--price_output_per_1k', type=float, default=0.015, help='USD cost per 1K output tokens (manual mode).')
  parser.add_argument('--output_dir', type=str, default='keyframe_selector/outputs', help='Directory to save outputs.')
  args = parser.parse_args()

  images_dir = Path(args.images_dir)
  frame_paths = _load_images_sorted(images_dir)
  total = len(frame_paths)

  sampling_strategy = 'stride'
  sampling_param = args.num_samples
  sample_indices = _stride_sample_indices(total, args.num_samples)
  sampled_paths = [frame_paths[i] for i in sample_indices]

  mode_to_use = 'sequence'
  use_base64_seq = bool(args.vlm in ('gpt4o', 'openai', 'gpt-4o'))

  # Prepare images
  cell_images = [_resize_image_keep_rgb(p, args.resize_size) for p in sampled_paths]

  # Build message content
  content, prompt = _prepare_message_content(
    images=cell_images,
    user_query=args.query,
    use_base64_sequence=use_base64_seq,
  )

  message = [{
    'role': 'user',
    'content': content,
  }]
  messages = [message]

  # Run MLLM
  output_text, usage = _run_mllm(
    messages=messages,
    model_path=args.model_path,
    max_new_tokens=args.max_new_tokens,
    vlm_type=args.vlm,
    openai_api_key=args.openai_api_key,
    openai_model=args.openai_model,
    google_api_key=args.google_api_key,
    google_model=args.google_model,
    max_retries=args.max_retries,
    retry_initial_delay=args.retry_initial_delay,
    retry_backoff_base=args.retry_backoff_base,
  )

  # Save raw MLLM output and metadata for downstream parsing
  out_root = Path(args.output_dir) / args.scene_name
  _ensure_dir(out_root)
  raw_json_path = out_root / 'mllm_raw.json'
  # Compute rough cost if usage available (images not included in tokenization estimate)
  input_tokens = None
  output_tokens = None
  total_tokens = None
  estimation_method = None
  usage_note = None
  input_text_tokens = None
  input_visual_tokens = None
  input_images = None
  if isinstance(usage, dict):
    input_tokens = usage.get('input_tokens')
    output_tokens = usage.get('output_tokens')
    total_tokens = usage.get('total_tokens')
    estimation_method = usage.get('estimation_method')
    usage_note = usage.get('note')
    input_text_tokens = usage.get('input_text_tokens')
    input_visual_tokens = usage.get('input_visual_tokens')
    input_images = usage.get('input_images')
  def _safe_num(v):
    try:
      return int(v) if v is not None else None
    except Exception:
      return None
  input_tokens = _safe_num(input_tokens)
  output_tokens = _safe_num(output_tokens)
  total_tokens = _safe_num(total_tokens)
  input_text_tokens = _safe_num(input_text_tokens)
  # input_visual_tokens may legitimately be None when unavailable; keep as-is if not numeric
  # Determine pricing per model when auto mode is enabled
  pricing_source = None
  price_in_per_1k = float(args.price_input_per_1k)
  price_out_per_1k = float(args.price_output_per_1k)
  if (args.vlm in ('gpt4o', 'openai', 'gpt-4o')) and (args.pricing_mode == 'auto'):
    model_l = (args.openai_model or '').strip().lower()
    # Pricing map (USD per 1K tokens) based on OpenAI pricing page
    PRICING_MAP = {
      'gpt-4o': (0.005, 0.015),
      'gpt-4o-2024-05-13': (0.005, 0.015),
      'gpt-4o-mini': (0.00015, 0.0006),
      'gpt-4o-mini-2024-07-18': (0.00015, 0.0006),
    }
    # Normalize possible aliases
    canonical = None
    if model_l in PRICING_MAP:
      canonical = model_l
    elif 'gpt-4o-mini' in model_l:
      canonical = 'gpt-4o-mini'
    elif 'gpt-4o' in model_l:
      canonical = 'gpt-4o'
    if canonical is not None:
      price_in_per_1k, price_out_per_1k = PRICING_MAP[canonical]
      pricing_source = f'auto:{canonical}'
    else:
      pricing_source = 'manual:fallback'
  else:
    pricing_source = 'manual:flags'

  estimated_cost_usd = None
  if (input_tokens is not None) or (output_tokens is not None):
    in_tokens = float(input_tokens or 0)
    out_tokens = float(output_tokens or 0)
    estimated_cost_usd = (in_tokens / 1000.0) * price_in_per_1k + (out_tokens / 1000.0) * price_out_per_1k
    estimated_cost_usd = round(estimated_cost_usd, 6)
  with open(raw_json_path, 'w') as f:
    json.dump({
      'scene_name': args.scene_name,
      'query': args.query,
      'prompt': prompt,
      'vlm': args.vlm,
      'model_path': args.model_path,
      'input_mode': mode_to_use,
      'sequence_encoding': ('base64_image_url' if use_base64_seq and mode_to_use == 'sequence' else 'tensor_image'),
      'resize_size': args.resize_size,
      'sampling': {
        'strategy': sampling_strategy,
        'param': sampling_param,
        'selected_json_path': None,
        'sample_indices': sample_indices,
        'sampled': [
          {
            'sample_index': i + 1,
            'global_frame_index': sample_indices[i],
            'frame_path': str(sampled_paths[i]),
          } for i in range(len(sampled_paths))
        ],
      },
      'max_new_tokens': args.max_new_tokens,
      'usage': {
        'estimation_method': estimation_method,
        'note': usage_note,
        'input_tokens': input_tokens,
        'input_text_tokens': input_text_tokens,
        'input_visual_tokens': input_visual_tokens,
        'input_images': input_images,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
      },
      'pricing': {
        'mode': args.pricing_mode,
        'model': (args.openai_model if args.vlm in ('gpt4o', 'openai', 'gpt-4o') else args.google_model if args.vlm in ('gemini', 'google', 'gemini-2.5-pro') else args.model_path),
        'source': pricing_source,
        'price_input_per_1k_usd': price_in_per_1k,
        'price_output_per_1k_usd': price_out_per_1k,
        'note': 'Image inputs may incur additional cost not reflected when usage excludes image tokens.',
        'estimated_cost_usd': estimated_cost_usd,
      },
      'raw_output_text': output_text,
    }, f, indent=2, ensure_ascii=False)

  print(f"[MLLM Keyframe Selector] scene={args.scene_name} frames={total} sampled={len(sample_indices)} mode={mode_to_use}")
  # Brief token and cost summary if available
  if input_tokens is not None or output_tokens is not None:
    in_tok = input_tokens if input_tokens is not None else 0
    out_tok = output_tokens if output_tokens is not None else 0
    cost_str = f" est_cost=${estimated_cost_usd}" if estimated_cost_usd is not None else ""
    model_str = f" model={args.openai_model} pricing={price_in_per_1k}/{price_out_per_1k} per1k ({args.pricing_mode})"
    print(f"[Usage] in_tokens={in_tok} out_tokens={out_tok} total={total_tokens if total_tokens is not None else in_tok + out_tok} method={estimation_method}{cost_str}{model_str}")
  print(f"[MLLM Keyframe Selector] saved raw output -> {raw_json_path}")


if __name__ == '__main__':
  main()
