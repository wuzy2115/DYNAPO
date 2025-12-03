# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import Any, List, Optional, Tuple, Union

import os
import sys

# Ensure project root is importable when this file is loaded standalone (no package)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

# Prefer relative imports; fall back to absolute project imports when executed standalone
try:
    from .modeling_internlm2 import InternLM2ForCausalLM
except Exception:
    from projects.llava_sam2.hf.models.modeling_internlm2 import InternLM2ForCausalLM
try:
    from .modeling_phi3 import Phi3ForCausalLM
except Exception:
    from projects.llava_sam2.hf.models.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import StoppingCriteriaList, StoppingCriteria

try:
    from .configuration_sa2va_chat import Sa2VAChatConfig
except Exception:
    from projects.llava_sam2.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
try:
    from .modeling_intern_vit import InternVisionModel, has_flash_attn
except Exception:
    from projects.llava_sam2.hf.models.modeling_intern_vit import InternVisionModel, has_flash_attn

try:
    from .sam2 import SAM2
except Exception:
    from projects.llava_sam2.hf.models.sam2 import SAM2
try:
    from .templates import PROMPT_TEMPLATE
except Exception:
    from projects.llava_sam2.hf.models.templates import PROMPT_TEMPLATE

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from types import MethodType
import torch.nn.functional as F

try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except Exception:
    try:
        from projects.llava_sam2.hf.models.flash_attention import FlashAttention
        has_flash_attn = True
    except Exception:
        print('FlashAttention is not installed.')
        has_flash_attn = False

logger = logging.get_logger(__name__)

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModel(PreTrainedModel):
    config_class = Sa2VAChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: Sa2VAChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.template = self.template.replace('-', '_')
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = PROMPT_TEMPLATE[self.template]
        self.template = self.conv_template
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.init_prediction_config = False

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        if 'vp_overall_mask' not in data.keys():
            vp_overall_mask = None
        else:
            vp_overall_mask = data['vp_overall_mask']

        if 'prompt_masks' in data.keys():
            prompt_masks = data['prompt_masks']
        else:
            prompt_masks = None

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
            vp_overall_mask=vp_overall_mask,
            prompt_masks=prompt_masks,
        )

        return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            vp_overall_mask=None,
            prompt_masks=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
        fast_vit_embeds = None

        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        if vp_overall_mask is not None and prompt_masks is not None:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1
            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        if vp_embeds is None:
            try:
                input_embeds[selected] = vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vit_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vit_embeds) + 1
                    vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vit_embeds[:n_token]
        else:
            try:
                input_embeds[selected] = vp_embeds.reshape(-1, C)
            except Exception as e:
                vp_embeds = vp_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape='
                      f'{input_embeds[selected].shape}, '
                      f'vp_embeds.shape={vp_embeds.shape}')
                n_token = selected.sum()
                if n_token > len(vp_embeds):
                    print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
                    expand_ratio = n_token // len(vp_embeds) + 1
                    vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

                input_embeds[selected] = vp_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prompt_masks=None,
            vp_overall_mask=None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.device
        assert self.img_context_token_id is not None

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)

                vit_embeds = self.extract_feature(pixel_values.to(device))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            if vp_overall_mask is not None and prompt_masks is not None:
                vp_embeds = []
                vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
                prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

                vp_overall_mask = vp_overall_mask[image_flags == 1]
                overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

                i_vp_img = 0
                for i_img in range(len(vit_embeds)):
                    vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                    if vp_overall_mask[i_img]:
                        tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                        objects_prompt_masks = prompt_masks[i_vp_img]
                        n_obj = len(objects_prompt_masks)
                        tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                        objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                        vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                        i_vp_img += 1

                vp_embeds = torch.cat(vp_embeds, dim=0)
            else:
                vp_embeds = None

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            if vp_embeds is None:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            else:
                if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                    print("Shape mismatch, selected is {}, vp embeds is {} !!!" \
                          .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                    min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                    input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
                else:
                    input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        # set stop criteria and generation configs for model
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        self.bot_name = 'BOT'
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.to(torch_dtype)
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'

        # change phi3 prepare for generation fuction
        if self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            self.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.language_model)

        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.img_context_token_id = img_context_token_id
        self.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
        return

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
    ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        if image is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': None,
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': None,
            }
            ret_masks = []
        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                extra_pixel_values = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if frame_idx < 5:
                        img = self.transformer(frame_image)
                        pixel_values.append(img)

                pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)
                num_image_tokens = self.patch_token
                num_frames = len(pixel_values)

                input_dict['vp_overall_mask'] = None
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

                images = dynamic_preprocess(image, self.min_dynamic_patch,
                                            self.max_dynamic_patch,
                                            self.image_size, self.use_thumbnail)

                if mask_prompts is not None:
                    vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
                    input_dict['vp_overall_mask'] = vp_overall_mask
                else:
                    input_dict['vp_overall_mask'] = None

                pixel_values = [self.transformer(image) for image in images]
                pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
                num_image_tokens = pixel_values.shape[0] * self.patch_token
                num_frames = 1
            input_dict['g_pixel_values'] = g_pixel_values
            input_dict['pixel_values'] = pixel_values

            if mask_prompts is not None:
                # reshape mask prompts to feature size
                mask_prompts = [torch.Tensor(item).to(pixel_values.device) for item in mask_prompts]
                mask_prompts = [F.interpolate(
                    item.unsqueeze(0),
                    size=(int(self.image_size // self.patch_size * self.downsample_ratio),
                          int(self.image_size // self.patch_size * self.downsample_ratio)),
                    mode='nearest').squeeze(0) for item in mask_prompts]
                region_pixels = []
                for mask_prompt in mask_prompts[0]:
                    region_pixels.append(mask_prompt.bool().to(torch.int64).sum())

                vp_token_str = '\nThere are {} part regions in the picture: '.format(len(mask_prompts[0]))
                for i in range(len(mask_prompts[0])):
                    vp_token_str = vp_token_str + \
                                   f"region{i + 1}" + self.VP_START_TOKEN + \
                                   self.IMG_CONTEXT_TOKEN * region_pixels[i] + \
                                   self.VP_END_TOKEN
                    if i == len(mask_prompts[0]) - 1:
                        vp_token_str = vp_token_str + '.\n'
                    else:
                        vp_token_str = vp_token_str + ', '
            else:
                vp_token_str = ''

            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            image_token_str = image_token_str + '\n'
            image_token_str = image_token_str * num_frames
            image_token_str = image_token_str.strip()

            ret_masks = []

            if '<image>' in text or mask_prompts is not None:
                assert past_text is None or len(past_text) == 0
            text = text.replace('<image>', image_token_str + vp_token_str)
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': input_dict['pixel_values'],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': mask_prompts,
                'vp_overall_mask': input_dict['vp_overall_mask'],
            }

        generate_output = self.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()

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

    def segment_with_keyframes(self, video, tokenizer, keyframes_json: str, enable_query_prompts: bool = False, query_interval: int = 16):
        """
        Segment a video using MLLM-selected keyframes and per-object descriptions.
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

        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

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
                # 支持多种格式：标准格式、markdown JSON格式、反引号格式
                # 首先尝试标准的方括号格式
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
                    # 如果没有找到方括号格式，尝试查找反引号格式的对象
                    output_section = _re.search(
                        r"(?:\*\*Output\s*list\s*:\*\*|Output\s*list\s*:)(.*?)(?:\n\n|\Z)",
                        raw_text,
                        flags=_re.IGNORECASE | _re.DOTALL
                    )
                    if output_section:
                        found_output_list = True
                        section_text = output_section.group(1)
                        # 查找所有反引号包围的对象
                        backtick_objects = _re.findall(r'`\s*\{[^}]*\}\s*`', section_text)
                        if backtick_objects:
                            # 将反引号对象转换为标准格式
                            cleaned_objects = []
                            for obj in backtick_objects:
                                # 移除反引号
                                cleaned = obj.strip('`').strip()
                                cleaned_objects.append(cleaned)
                            body = ', '.join(cleaned_objects)

                # 如果找到了Output list但body为空（空列表的情况），这是正常的
                if found_output_list and not body:
                    # 空列表情况，instances保持为空列表，这是正常的
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
        # Empty list is a valid case, indicating no objects to segment
        if len(instances) == 0:
            # Return empty prediction result
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

                # Prepare pixel_values as in predict_forward (video path)
                pixel_values = []
                for frame_image in window_frames:
                    img = self.transformer(frame_image)
                    pixel_values.append(img)
                pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
                num_frames = pixel_values.shape[0]

                # Compose image token string as in predict_forward
                num_image_tokens = self.patch_token
                image_token_str = f"{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * num_image_tokens}{self.IMG_END_TOKEN}"
                image_token_str = (image_token_str + '\n') * num_frames
                image_token_str = image_token_str.strip()

                # Text prompt: include <image> placeholder, then replace
                text = f"<image> Please segment {desc}".strip()
                text = text.replace('<image>', image_token_str)
                input_text = ''
                input_text += self.template['INSTRUCTION'].format(
                    input=text, round=1, bot_name=self.bot_name)
                ids = self.tokenizer.encode(input_text)
                ids = torch.tensor(ids).cuda().unsqueeze(0)
                attention_mask = torch.ones_like(ids, dtype=torch.bool)

                mm_inputs = {
                    'pixel_values': pixel_values,
                    'input_ids': ids,
                    'attention_mask': attention_mask,
                    'position_ids': None,
                    'past_key_values': None,
                    'labels': None,
                    'prompt_masks': None,
                    'vp_overall_mask': None,
                }

                # Call MLLM to obtain <SEG> token hidden states
                generate_output = self.generate(
                    **mm_inputs,
                    generation_config=self.gen_config,
                    streamer=None,
                    bos_token_id=self.tokenizer.bos_token_id,
                    stopping_criteria=self.stop_criteria,
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

            # (No extra injection block needed: query frames were appended to seeds and handled above)

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
    # Ensure seg_mask is on the same device as hidden_states
    seg_mask = seg_mask.to(hidden_states.device)
    return hidden_states[-n_out:][seg_mask]

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
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

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation_phi3(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs

