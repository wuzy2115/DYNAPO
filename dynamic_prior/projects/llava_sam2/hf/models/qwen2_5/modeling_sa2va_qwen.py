import torch
from torch import nn
from transformers import (AutoModel, GenerationConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen2ForCausalLM)
from transformers.modeling_utils import PreTrainedModel

import numpy as np
from torchvision.transforms.functional import to_pil_image

import torch.nn.functional as F

from qwen_vl_utils import process_vision_info

import os
import sys

# Ensure project root is importable when this file is loaded standalone (no package)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

# Prefer relative imports; fall back to absolute project imports when executed standalone
try:
    from .configuration_sa2va_chat_qwen import Sa2VAChatConfigQwen
except Exception:
    from projects.llava_sam2.hf.models.configuration_sa2va_chat_qwen import Sa2VAChatConfigQwen

try:
    from .sam2 import SAM2
except Exception:
    from projects.llava_sam2.hf.models.sam2 import SAM2



class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModelQwen(PreTrainedModel):
    config_class = Sa2VAChatConfigQwen
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['Qwen2_5_VisionTransformerPretrainedModel', 'Qwen2_5_VLDecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True



    def __init__(self, config: Sa2VAChatConfigQwen, model=None, use_flash_attn=True):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1024, )

        self.min_pixels = 512 * 28 * 28
        self.max_pixels = 2048 * 28 * 28

        self.torch_dtype = torch.bfloat16

        if model is not None:
            self.model=model
        else:
            self.model = Qwen2_5_VLForConditionalGeneration(config)
        
        self.model._tied_weights_keys = None

        llm_hidden_size = config.text_config.hidden_size

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
    ):
        assert processor is not None
        self.processor = processor
        
        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids('[SEG]')

        text = text.replace('<image>', "")

        if image is None and video is None and '<image>' not in past_text:
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": past_text + text},
                    ],
                }
            ]

            # Preparation for inference
            processsed_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            mm_inputs = self.processor(
                text=[processsed_text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            mm_inputs = mm_inputs.to(self.device)

            ret_masks = []
        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                extra_pixel_values = []
                images = []
                content = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    # assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if frame_idx < 5:
                        content.append({"type": "image", "image": frame_image},)


                content.append({"type": "text", "text": text})
                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]

                # Preparation for inference
                processsed_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)
                mm_inputs = self.processor(
                    text=[processsed_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                mm_inputs = mm_inputs.to(self.device)

                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                num_frames = min(5, len(video))

            else:
                ori_image_size = image.size
                
                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                extra_pixel_values = [g_pixel_values]
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ]

                # Preparation for inference
                processsed_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)
                mm_inputs = self.processor(
                    text=[processsed_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                mm_inputs = mm_inputs.to(self.device)

                num_frames = 1
            
            input_dict['g_pixel_values'] = g_pixel_values
            ret_masks = []

        generate_output = self.model.generate(
            **mm_inputs,
            max_new_tokens=2048,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        generate_output_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
        ]

        predict = self.processor.batch_decode(generate_output_trimmed, skip_special_tokens=False)[0].strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks,}

    def segment_with_keyframes(self, video, processor, keyframes_json: str, enable_query_prompts: bool = False, query_interval: int = 16):
        """
        Segment a video using MLLM-selected keyframes and per-object descriptions (Qwen version).
        - Parse keyframes JSON (supports either pre-parsed 'instances' or parses 'raw_output_text').
        - Map keyframe indices to global frame indices via 'sampling.sample_indices'.
        - Compute a language embedding per object description using the LLM (mean pooled last hidden states),
          project via self.text_hidden_fcs to SAM2 hidden_dim.
        - Inject each object's embedding at its keyframe via SAM2 add_language_embd, then propagate across the video.
        Returns a dict: { 'prediction': '[SEG]', 'prediction_masks': [np.ndarray(T,H,W) per object] }.
        """
        import json as _json
        import re as _re
        from typing import List as _List, Dict as _Dict

        assert processor is not None
        self.processor = processor
        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids('[SEG]')

        # 1) Load keyframe JSON
        with open(keyframes_json, 'r') as f:
            data = _json.load(f)

        # 2) Build mapping: sample keyframe (1-based) -> global frame index (0-based)
        sample_indices = None
        if isinstance(data, dict):
            sampling = data.get('sampling', {}) if isinstance(data.get('sampling'), dict) else {}
            if isinstance(sampling.get('sample_indices'), list) and len(sampling['sample_indices']) > 0:
                sample_indices = sampling['sample_indices']
        if sample_indices is None:
            # Try to reconstruct from 'sampled'
            sampled = data.get('sampling', {}).get('sampled', []) if isinstance(data.get('sampling'), dict) else []
            # sampled entries have 'sample_index' (1-based) and 'global_frame_index'
            mapping = {}
            for item in sampled:
                try:
                    mapping[int(item['sample_index'])] = int(item['global_frame_index'])
                except Exception:
                    continue
            # normalize to list in order 1..K
            if len(mapping) > 0:
                max_k = max(mapping.keys())
                sample_indices = [mapping.get(i, 0) for i in range(1, max_k + 1)]
        assert sample_indices is not None and len(sample_indices) > 0, "Invalid or missing sample_indices in keyframes JSON"

        # 3) Extract instances: [{object_index:int, keyframe:int, object_description:str}, ...]
        instances = []
        if isinstance(data.get('instances'), list):
            for it in data['instances']:
                try:
                    instances.append({
                        'object_index': int(it.get('object_index')),
                        'keyframe': int(it.get('keyframe')),
                        'object_description': str(it.get('object_description', '')).strip(),
                    })
                except Exception:
                    continue
        if not instances:
            raw_text = str(data.get('raw_output_text', '') or '')
            if raw_text:
                # Support multiple formats: standard format, markdown JSON format, backtick format
                # First try standard bracket format
                m_output_list = _re.search(
                                r"(?:\*\*Output\s*list\s*:\*\*|Output\s*list\s*:)(?:.*?```json\s*)?\s*\[.*?\]",
                                raw_text,
                                flags=_re.IGNORECASE | _re.DOTALL
                            )
                body = ""
                found_output_list = False
                if m_output_list:
                    found_output_list = True
                    # Extract the content between [ and ]
                    full_match = m_output_list.group(0)
                    bracket_start = full_match.find('[')
                    bracket_end = full_match.rfind(']')
                    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
                        body = full_match[bracket_start + 1:bracket_end].strip()
                else:
                    # If bracket format not found, try to find backtick format objects
                    output_section = _re.search(
                        r"(?:\*\*Output\s*list\s*:\*\*|Output\s*list\s*:)(.*?)(?:\n\n|\Z)",
                        raw_text,
                        flags=_re.IGNORECASE | _re.DOTALL
                    )
                    if output_section:
                        found_output_list = True
                        section_text = output_section.group(1)
                        # Find all backtick-enclosed objects
                        backtick_objects = _re.findall(r'`\s*\{[^}]*\}\s*`', section_text)
                        if backtick_objects:
                            # Convert backtick objects to standard format
                            cleaned_objects = []
                            for obj in backtick_objects:
                                # Remove backticks
                                cleaned = obj.strip('`').strip()
                                cleaned_objects.append(cleaned)
                            body = ', '.join(cleaned_objects)

                # If Output list is found but body is empty (empty list case), this is normal
                if found_output_list and not body:
                    # Empty list case, instances remain as empty list, this is normal
                    pass
                elif found_output_list and body:
                    # Robustly split multiple { ... } items by tracking brace depth
                    items_blocks = []
                    depth = 0
                    start_idx = None
                    for idx, ch in enumerate(body):
                        if ch == '{':
                            if depth == 0:
                                start_idx = idx
                            depth += 1
                        elif ch == '}':
                            if depth > 0:
                                depth -= 1
                                if depth == 0 and start_idx is not None:
                                    items_blocks.append(body[start_idx:idx + 1])
                                    start_idx = None

                    for item in items_blocks:
                        m_idx = _re.search(r"object_index\s*:\s*([0-9]+)", item, flags=_re.IGNORECASE)
                        m_kf = _re.search(r"keyframe\s*:\s*([0-9]+)", item, flags=_re.IGNORECASE)
                        m_desc = _re.search(r"object_description\s*:\s*(\"([^\"]*)\"|'([^']*)'|([^,\}]+))", item, flags=_re.IGNORECASE | _re.DOTALL)
                        if m_idx and m_kf and m_desc:
                            try:
                                obj_idx = int(m_idx.group(1))
                                keyf = int(m_kf.group(1))
                            except Exception:
                                continue
                            desc = m_desc.group(2) or m_desc.group(3) or m_desc.group(4) or ""
                            desc = desc.strip()
                            instances.append({'object_index': obj_idx, 'keyframe': keyf, 'object_description': desc})
        # 空列表是合法的情况，表示没有需要分割的对象
        if len(instances) == 0:
            # 返回空的预测结果
            return {
                'prediction': '[SEG] no objects to segment',
                'prediction_masks': [],
            }

        # 4) Map sample keyframe -> real frame index in the video
        #    keyframe indices in instances are 1-based positions in sampled frames
        def map_keyframe_to_frame_idx(kf_1_based: int) -> int:
            assert 1 <= kf_1_based <= len(sample_indices), f"keyframe {kf_1_based} out of range"
            return int(sample_indices[kf_1_based - 1])

        # 5) Precompute grounding images once
        extra_pixel_values = []
        for frame_image in video:
            g_image = np.array(frame_image)
            g_image = self.extra_image_processor.apply_image(g_image)
            g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            extra_pixel_values.append(g_image)
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
        ]).to(self.torch_dtype)

        # 6) Compute language embeddings per unique object and keep for later per-object propagation
        device = self.device
        # Group multiple descriptions/keyframes per object
        obj_seeds_by_id = {}
        for it in instances:
            try:
                oid = int(it['object_index'])
                kf_global = map_keyframe_to_frame_idx(int(it['keyframe']))
                desc = str(it.get('object_description', ''))
            except Exception:
                continue
            obj_seeds_by_id.setdefault(oid, []).append((kf_global, desc))
        obj_ids_sorted = sorted(obj_seeds_by_id.keys())

        T = len(video)
        H = video[0].size[1]
        W = video[0].size[0]

        # Helper merge utilities (adapted from ref_SegAnyMo/sam2/run_sam2.py)
        def _compute_iou(m1: np.ndarray, m2: np.ndarray) -> float:
            inter = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            return float(inter) / float(union) if union > 0 else 0.0

        def _is_subset(m1: np.ndarray, m2: np.ndarray, coverage_threshold: float = 0.9) -> bool:
            a1 = (m1 > 0).sum()
            inter = np.logical_and(m1 > 0, m2 > 0).sum()
            if a1 == 0:
                return False
            return (inter / a1) >= coverage_threshold

        def _analyze_frame_merges(video_segments: dict, iou_threshold: float = 0.9):
            potential_merges = {}
            frame_count = len(video_segments)
            for per_obj_output_mask in video_segments.values():
                visited = set()
                for obj_id1, mask1 in per_obj_output_mask.items():
                    for obj_id2, mask2 in per_obj_output_mask.items():
                        if obj_id1 == obj_id2 or (obj_id1, obj_id2) in visited or (obj_id2, obj_id1) in visited:
                            continue
                        iou = _compute_iou(mask1, mask2)
                        if iou > iou_threshold or _is_subset(mask1, mask2) or _is_subset(mask2, mask1):
                            potential_merges.setdefault(obj_id1, {}).setdefault(obj_id2, 0)
                            potential_merges[obj_id1][obj_id2] += 1
                        visited.add((obj_id1, obj_id2))
            from collections import defaultdict as _dd
            final_merges = _dd(list)
            for obj_id1, merge_counts in potential_merges.items():
                for obj_id2, count in merge_counts.items():
                    if count / frame_count >= 0.3:
                        if obj_id2 not in final_merges[obj_id1]:
                            final_merges[obj_id1].append(obj_id2)
            groups = []
            visited_ids = set()
            for obj_id1, merged_ids in final_merges.items():
                if obj_id1 in visited_ids:
                    continue
                current_group = set([obj_id1] + merged_ids)
                for obj_id2 in merged_ids:
                    current_group.update(final_merges.get(obj_id2, []))
                groups.append(sorted(current_group))
                visited_ids.update(current_group)
            all_obj_ids = set()
            for f in video_segments.values():
                all_obj_ids.update(f.keys())
            unmerged = all_obj_ids - set([x for g in groups for x in g])
            for oid in unmerged:
                groups.append([oid])
            return {i + 1: group for i, group in enumerate(groups)}

        def _merge_masks(video_segments: dict, merge_groups: dict):
            merged_video_segments = {}
            for out_frame_idx, per_obj_output_mask in video_segments.items():
                merged = {}
                for new_obj_id, obj_ids in merge_groups.items():
                    combined = None
                    for obj_id in obj_ids:
                        m = per_obj_output_mask.get(obj_id)
                        if m is None:
                            continue
                        combined = (m.copy() if combined is None else np.logical_or(combined, m))
                    if combined is not None:
                        merged[new_obj_id] = combined
                merged_video_segments[out_frame_idx] = merged
            return merged_video_segments

        # 7) Per-object propagation (forward + reverse) starting at each object's keyframe
        per_obj_full_masks: _Dict[int, np.ndarray] = {}

        for obj_id in obj_ids_sorted:
            seeds = obj_seeds_by_id[obj_id]
            # Preserve original seeds for logic like reverse propagation decision
            original_seeds = list(seeds)
            # Augment seeds with periodic query frames so we regenerate lang_embd_3d per query
            if enable_query_prompts and len(original_seeds) > 0:
                seed_frames_set = {int(kf) for kf, _ in original_seeds}
                # Use the last provided description for this object for query frames
                desc_for_queries = str(original_seeds[-1][1])
                step = max(1, int(query_interval))
                for qf in range(0, T, step):
                    if qf in seed_frames_set:
                        continue
                    seeds.append((int(qf), desc_for_queries))
                # Sort seeds to keep chronological order (optional but tidy)
                seeds.sort(key=lambda x: int(x[0]))

            # Fresh SAM2 inference state per object (equivalent to predictor.reset_state)
            sam_state_obj = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            add_obj_id = int(obj_id) + 100
            added_any_seed = False

            # Add all language seeds for this object at their respective keyframes
            for keyframe_idx, desc in seeds:
                # Build a small window to obtain text-driven SEG embedding for this seed
                window_size = 5
                if keyframe_idx + window_size > T:
                    start = max(0, T - window_size)
                    end = T
                else:
                    start = keyframe_idx
                    end = min(T, keyframe_idx + window_size)
                window_frames = video[start:end]

                # Prepare messages in Qwen format
                content = []
                for frame_image in window_frames[:5]:  # Qwen uses max 5 frames
                    content.append({"type": "image", "image": frame_image})

                text = f"Please segment {desc}".strip()
                content.append({"type": "text", "text": text})

                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]

                # Preparation for inference
                processsed_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                mm_inputs = self.processor(
                    text=[processsed_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                mm_inputs = mm_inputs.to(self.device)

                # Call MLLM to obtain <SEG> token hidden states
                generate_output = self.model.generate(
                    **mm_inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                hidden_states = generate_output.hidden_states
                last_hidden_states = [item[-1][0] for item in hidden_states]
                last_hidden_states = torch.cat(last_hidden_states, dim=0)
                seg_hidden_states = get_seg_hidden_states(
                    last_hidden_states,
                    generate_output.sequences[0][:-1],
                    seg_id=self.seg_token_idx,
                )
                if seg_hidden_states.shape[0] == 0:
                    # Skip this seed if no [SEG] token is produced
                    continue
                all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)
                # Use the last SEG occurrence
                seg_embd = all_seg_hidden_states[-1].unsqueeze(0)  # [1, C]
                lang_embd_3d = seg_embd.unsqueeze(0)  # [1, 1, C]

                self.grounding_encoder.sam2_model.add_language_embd(
                    sam_state_obj,
                    frame_idx=keyframe_idx,
                    obj_id=add_obj_id,
                    language_embd=lang_embd_3d,
                    inference=True,
                )
                added_any_seed = True

            # Collect masks per frame for this object
            obj_masks = [None] * T
            if added_any_seed:
                # Only reverse propagate if earliest seed is not at frame 0
                earliest_seed_frame = min(k for k, _ in original_seeds) if len(original_seeds) > 0 else 0
                require_reverse = (earliest_seed_frame != 0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Forward direction from earliest seed to end
                    for out_frame_idx, out_obj_ids, video_res_masks in self.grounding_encoder.sam2_model.propagate_in_video(sam_state_obj):
                        if isinstance(video_res_masks, torch.Tensor):
                            resized = F.interpolate(video_res_masks, size=(H, W), mode='bilinear', align_corners=False)
                            bin_masks = (resized[:, 0].sigmoid() > 0.5).detach().cpu().numpy()
                        else:
                            if hasattr(video_res_masks, 'ndim') and video_res_masks.ndim == 4:
                                t_m = torch.from_numpy(video_res_masks).to(dtype=torch.float32)
                            elif hasattr(video_res_masks, 'ndim') and video_res_masks.ndim == 3:
                                t_m = torch.from_numpy(video_res_masks)[:, None, ...].to(dtype=torch.float32)
                            else:
                                t_m = torch.as_tensor(video_res_masks, dtype=torch.float32)
                                if t_m.ndim == 3:
                                    t_m = t_m[:, None, ...]
                            resized = F.interpolate(t_m, size=(H, W), mode='bilinear', align_corners=False)
                            bin_masks = (resized[:, 0] > 0.5).detach().cpu().numpy()
                        for i, oid in enumerate(out_obj_ids):
                            if oid is None:
                                continue
                            true_oid = int(oid) - 100 if int(oid) >= 100 else int(oid)
                            if true_oid == obj_id:
                                obj_masks[out_frame_idx] = bin_masks[i]

                    # Reverse direction back to 0 if required
                    if require_reverse:
                        for out_frame_idx, out_obj_ids, video_res_masks in self.grounding_encoder.sam2_model.propagate_in_video(sam_state_obj, reverse=True):
                            if isinstance(video_res_masks, torch.Tensor):
                                resized = F.interpolate(video_res_masks, size=(H, W), mode='bilinear', align_corners=False)
                                bin_masks = (resized[:, 0].sigmoid() > 0.5).detach().cpu().numpy()
                            else:
                                if hasattr(video_res_masks, 'ndim') and video_res_masks.ndim == 4:
                                    t_m = torch.from_numpy(video_res_masks).to(dtype=torch.float32)
                                elif hasattr(video_res_masks, 'ndim') and video_res_masks.ndim == 3:
                                    t_m = torch.from_numpy(video_res_masks)[:, None, ...].to(dtype=torch.float32)
                                else:
                                    t_m = torch.as_tensor(video_res_masks, dtype=torch.float32)
                                    if t_m.ndim == 3:
                                        t_m = t_m[:, None, ...]
                                resized = F.interpolate(t_m, size=(H, W), mode='bilinear', align_corners=False)
                                bin_masks = (resized[:, 0] > 0.5).detach().cpu().numpy()
                            for i, oid in enumerate(out_obj_ids):
                                if oid is None:
                                    continue
                                true_oid = int(oid) - 100 if int(oid) >= 100 else int(oid)
                                if true_oid == obj_id:
                                    obj_masks[out_frame_idx] = bin_masks[i]

            # Fill missing frames with zeros
            for t_idx in range(T):
                if obj_masks[t_idx] is None:
                    obj_masks[t_idx] = np.zeros((H, W), dtype=bool)
            per_obj_full_masks[obj_id] = np.stack(obj_masks, axis=0)

        # 8) Build frame-wise segments to run merging like run_sam2.py
        video_segments = {}
        for t_idx in range(T):
            per_frame = {}
            for oid, masks in per_obj_full_masks.items():
                per_frame[oid] = masks[t_idx]
            video_segments[t_idx] = per_frame

        merge_groups = _analyze_frame_merges(video_segments, iou_threshold=0.9)
        merged_video_segments = _merge_masks(video_segments, merge_groups)

        # 9) Convert merged segments back to per-object arrays ordered by new ids
        new_ids_sorted = sorted(merge_groups.keys())
        prediction_masks = []
        for new_oid in new_ids_sorted:
            frames = []
            for t_idx in range(T):
                mask = merged_video_segments.get(t_idx, {}).get(new_oid, None)
                if mask is None:
                    mask = np.zeros((H, W), dtype=bool)
                frames.append(mask)
            prediction_masks.append(np.stack(frames, axis=0))

        return {
            'prediction': '[SEG] keyframe-driven',
            'prediction_masks': prediction_masks,
        }

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]