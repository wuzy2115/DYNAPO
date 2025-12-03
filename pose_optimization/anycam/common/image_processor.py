from collections import namedtuple
import logging
import math
import os

import requests
import torch
from torch import nn
# import lpips
import torch.nn.functional as F

from torchvision import transforms as tfs

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from torch.amp import autocast

from anycam.common.cameras.pinhole import (
    center_crop,
    no_crop,
    normalize_camera_intrinsics,
    random_crop,
    resize_to_canonical_frame,
)
from anycam.common.geometry import compute_occlusions

logger = logging.getLogger("image_processor")


def make_image_processor(config, **kwargs):
    type = config.get("type", "RGB").lower()
    if type == "rgb":
        ip = RGBProcessor()
    elif type == "perceptual":
        ip = PerceptualProcessor(config.get("layers", 1))
    elif type == "patch":
        ip = PatchProcessor(config.get("patch_size", 3))
    elif type == "raft":
        ip = RaftExtractor()
    elif type == "flow":
        ip = FlowProcessor()
    elif type == "flow_occlusion":
        ip = FlowOcclusionProcessor(
            n_pairs=config.get("n_pairs", 2), 
            flow_model=kwargs.get("flow_model", config.get("flow_model", "unimatch")), 
            use_existing_flow=kwargs.get("use_provided_flow", config.get("use_existing_flow", False)),
            pair_mode=kwargs.get("pair_mode", config.get("pair_mode", "one-to-many")),
        )
    else:
        raise NotImplementedError(f"Unsupported image processor type: {type}")
    return ip


class RGBProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 3

    def forward(self, images):
        images = images * 0.5 + 0.5
        return images


class PerceptualProcessor(nn.Module):
    def __init__(self, layers=1) -> None:
        super().__init__()
        self.lpips_module = lpips.LPIPS(net="vgg")
        self._layers = layers
        self.channels = sum(self.lpips_module.chns[: self._layers])

    def forward(self, images):
        n, v, c, h, w = images.shape
        images = images.view(n * v, c, h, w)

        in_input = self.lpips_module.scaling_layer(images)

        x = self.lpips_module.net.slice1(in_input)
        h_relu1_2 = x
        x = self.lpips_module.net.slice2(x)
        h_relu2_2 = x
        x = self.lpips_module.net.slice3(x)
        h_relu3_3 = x

        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3"])
        outs = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)

        feats = []

        for kk in range(self._layers):
            f = lpips.normalize_tensor(outs[kk])
            f = F.interpolate(f, (h, w))
            feats.append(f)

        feats = torch.cat(feats, dim=1)

        feats = feats.view(n, v, self.channels, h, w)

        return feats


class PatchProcessor(nn.Module):
    def __init__(self, patch_size) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.channels = 3 * (patch_size**2)

        self._hps = self.patch_size // 2

    def forward(self, images):
        n, v, c, h, w = images.shape
        images = images.view(n * v, c, h, w) * 0.5 + 0.5

        images = F.pad(images, pad=(self.patch_size // 2,) * 4, mode="replicate")
        h_, w_ = images.shape[-2:]

        parts = []

        for y in range(0, self.patch_size):
            for x in range(0, self.patch_size):
                parts.append(
                    images[
                        :,
                        :,
                        y : h_ - (self.patch_size - y - 1),
                        x : w_ - (self.patch_size - x - 1),
                    ]
                )

        patch_images = torch.cat(parts, dim=1)
        patch_images = patch_images.view(n, v, self.channels, h, w)

        return patch_images


class DinoExtractor(nn.Module):
    def __init__(self, variant):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/dino:main", variant)
        self.model.eval()

    def load_checkpoint(self, ckpt_file, checkpoint_key="model"):
        state_dict = torch.load(ckpt_file, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded with msg: {}".format(msg))

    def forward(self, img: torch.Tensor, transform=True, upsample=True):
        n, c, h_in, w_in = img.shape
        if transform:
            img = self.transform(img, 256)  # Nx3xHxW
        with torch.no_grad():
            out = self.model.get_intermediate_layers(img.to(self.device), n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token
            h, w = int(img.shape[2] / self.model.patch_embed.patch_size), int(
                img.shape[3] / self.model.patch_embed.patch_size
            )
            dim = out.shape[-1]
            out = out.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
            if upsample:
                out = torch.nn.functional.interpolate(
                    out, (h_in, w_in), mode="bilinear"
                )
        return out

    @staticmethod
    def transform(img, image_size):

        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        transforms = tfs.Compose([tfs.Resize(image_size), tfs.Normalize(MEAN, STD)])
        img = transforms(img)
        return img

    @property
    def device(self):
        return next(self.parameters()).device


class RaftExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        raft_weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = raft_weights.transforms()
        self.raft = raft_large(raft_weights)
        self.raft.eval()
        for param in self.raft.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, img: torch.Tensor, upsample=True):
        n, v, c, h_in, w_in = img.shape
        img = img.reshape(n * v, c, h_in, w_in)
        img, _ = self.raft_transforms(img * 0.5 + 0.5, img * 0.5 + 0.5)
        feats = self.raft.feature_encoder(img)
        if upsample:
            feats = F.interpolate(feats, (h_in, w_in), mode="bilinear")
            feats = feats.view(n, v, -1, h_in, w_in)
        else:
            feats = feats.view(n, v, -1, feats.shape[-2], feats.shape[-1])
        return feats

    @property
    def device(self):
        return next(self.parameters()).device


class FlowProcessor(nn.Module):
    def __init__(self):
        super().__init__()

        raft_weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = raft_weights.transforms()
        self.raft = raft_large(raft_weights)
        self.raft.eval()
        for param in self.raft.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, img: torch.Tensor, upsample=True):
        n, v, c, h, w = img.shape
        img = img.reshape(n * v // 2, 2, c, h, w)
        img0 = img[:, 0]
        img1 = img[:, 1]
        img0, img1 = self.raft_transforms(img0 * 0.5 + 0.5, img1 * 0.5 + 0.5)
        flow_fwd = self.raft(img0, img1)[-1]
        flow_bwd = self.raft(img1, img0)[-1]
        flow0_r = torch.cat(
            (flow_fwd[:, 0:1, :, :] * 2 / w, flow_fwd[:, 1:2, :, :] * 2 / h), dim=1
        )
        flow1_r = torch.cat(
            (flow_bwd[:, 0:1, :, :] * 2 / w, flow_bwd[:, 1:2, :, :] * 2 / h), dim=1
        )
        flow = torch.stack((flow0_r, flow1_r), dim=1)

        img = torch.cat((img, flow), dim=2)

        img = img.reshape(n, v, -1, h, w)

        return img

    @property
    def device(self):
        return next(self.parameters()).device


class FlowOcclusionProcessor(nn.Module):
    def __init__(self, n_pairs=2, flow_model="unimatch", use_existing_flow=False, pair_mode="one-to-many", **kwargs):
        super().__init__()

        self.n_pairs = n_pairs

        self.flow_model = flow_model

        self.use_existing_flow = use_existing_flow

        self.pair_mode = pair_mode

        assert self.pair_mode in ("one-to-many", "sequential"), f"Unknown pair mode: {self.pair_mode}"

        # TODO: Temporary solution
        logger.info(f"Loading flow model: {self.flow_model}")
        if True or not self.use_existing_flow:
            if self.flow_model == "raft":
                raft_weights = Raft_Large_Weights.DEFAULT
                self.raft_transforms = raft_weights.transforms()
                self.raft = raft_large(raft_weights)
                self.raft.eval()

                for param in self.raft.parameters():
                    param.requires_grad = False

            elif self.flow_model == "unimatch":
                from unimatch.unimatch import UniMatch

                ckpt_path = os.path.join(os.environ["HOME"], ".cache", "torch", "checkpoints", "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth")
                logger.info(f"Loading pretrained model from {ckpt_path}")
                if not os.path.exists(ckpt_path):
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    logger.info(f"Downloading from https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth")
                    r = requests.get('https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
                    with open(ckpt_path , 'wb') as f:
                        f.write(r.content)
                self.unimatch = UniMatch(
                    feature_channels=128,
                    num_scales=2,
                    upsample_factor=4,
                    ffn_dim_expansion=4,
                    num_transformer_layers=6,
                    reg_refine=True,
                    task="flow")
                self.unimatch.load_state_dict(torch.load(ckpt_path)['model'], strict=True)
                self.unimatch.eval()
                for param in self.unimatch.parameters():
                    param.requires_grad = False

                # self.unimatch = torch.compile(self.unimatch)
                
            elif self.flow_model == "alltracker":
                # import sys
                # sys.path.append(os.path.join(os.environ["PYTHONPATH"], "alltracker"))
                from alltracker.nets.alltracker import Net as AllTrackerNet
                
                # Initialize AllTracker model
                self.alltracker = AllTrackerNet(
                    seqlen=16,  # Default sequence length
                    use_attn=True,
                    use_mixer=False,
                    use_conv=False,
                    use_convb=False,
                    use_basicencoder=False,
                    use_sinmotion=False,
                    use_relmotion=False,
                    use_sinrelmotion=False,
                    use_feats8=False,
                    no_time=False,
                    no_space=False,
                    no_split=False,
                    no_ctx=False,
                    full_split=False,
                    corr_levels=5,
                    corr_radius=4,
                    num_blocks=3,
                    dim=128,
                    hdim=128,
                    init_weights=True,
                )
                
                # Load pretrained weights
                ckpt_path = os.path.join(os.environ["HOME"], ".cache", "torch", "checkpoints", "alltracker.pth")
                logger.info(f"Loading AllTracker model from {ckpt_path}")
                if not os.path.exists(ckpt_path):
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    logger.info(f"Downloading AllTracker weights from https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth")
                    r = requests.get('https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth')
                    with open(ckpt_path, 'wb') as f:
                        f.write(r.content)
                
                # Load state dict
                state_dict = torch.load(ckpt_path, map_location='cpu')
                self.alltracker.load_state_dict(state_dict['model'], strict=True)
                self.alltracker.eval()
                
                # Freeze parameters
                for param in self.alltracker.parameters():
                    param.requires_grad = False
                
                logger.info("AllTracker model loaded successfully")
                
                # Initialize cache for sequence processing
                self.alltracker_cache = {}
                    
            else:
                raise ValueError(f"Unknown flow model: {self.flow_model}")

    def flow_raft(self, img0, img1):
        n, c, h, w = img0.shape

        target_h = h
        target_w = w

        target_h = math.ceil(target_h / 32) * 32
        target_w = math.ceil(target_w / 32) * 32

        if target_h != h or target_w != w:
            img0 = F.interpolate(img0, (target_h, target_w), mode='bilinear', align_corners=True)
            img1 = F.interpolate(img1, (target_h, target_w), mode='bilinear', align_corners=True)

        img0, img1 = self.raft_transforms(img0 * 0.5 + 0.5, img1 * 0.5 + 0.5)
        flow_fwd = self.raft(img0, img1)[-1]
        flow_bwd = self.raft(img1, img0)[-1]

        if target_h != h or target_w != w:
            flow_fwd = F.interpolate(flow_fwd, (h, w), mode='bilinear', align_corners=True)
            flow_bwd = F.interpolate(flow_bwd, (h, w), mode='bilinear', align_corners=True)

        return flow_fwd, flow_bwd
    
    @autocast(enabled=False, device_type="cuda")
    def flow_unimatch(self, img0, img1):
        n, c, h, w = img0.shape

        max_size = 320
        smaller = min(h, w)

        if smaller > max_size:
            scale_factor = max_size / smaller
            target_h = h * scale_factor
            target_w = w * scale_factor
        else:
            target_h = h
            target_w = w
        
        target_h = math.ceil(target_h / 32) * 32
        target_w = math.ceil(target_w / 32) * 32

        if target_h != h or target_w != w:
            img0 = F.interpolate(img0, (target_h, target_w), mode='bilinear', align_corners=True)
            img1 = F.interpolate(img1, (target_h, target_w), mode='bilinear', align_corners=True)

        img0 = (img0 * 0.5 + 0.5) * 255
        img1 = (img1 * 0.5 + 0.5) * 255

        attn_type = 'swin'
        attn_splits_list = [2, 8]
        corr_radius_list = [-1, 4]
        prop_radius_list = [-1, 1]
        num_reg_refine = 6

        if target_h > target_w:
            img0 = img0.permute(0, 1, 3, 2)
            img1 = img1.permute(0, 1, 3, 2)

        results_dict = self.unimatch(img0, img1,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task="flow",
                            pred_bidir_flow=True,
                            )
        
        flows = results_dict['flow_preds'][-1]

        if target_h > target_w:
            flows = flows.permute(0, 1, 3, 2)
            flows = flows[:, [1, 0], :, :]

        if target_h != h or target_w != w:
            flows = F.interpolate(flows, (h, w), mode='bilinear', align_corners=True)

        flow_fwd = flows[:n]
        flow_bwd = flows[n:]
        return flow_fwd, flow_bwd

    @autocast(enabled=False, device_type="cuda")
    def flow_alltracker(self, img0, img1):
        """
        AllTracker processes entire sequences, not pairs.
        For pairwise processing, we need to handle this differently.
        """
        n, c, h, w = img0.shape
        
        # Stack images to create a mini-sequence
        img_sequence = torch.stack([img0, img1], dim=1)  # (n, 2, c, h, w)
        
        # Convert from [-1, 1] to [0, 255] (AllTracker expects [0, 255])
        img_sequence = (img_sequence * 0.5 + 0.5) * 255.0
        
        # AllTracker expects input in format (B, T, C, H, W)
        # img_sequence is already in this format
        
        with torch.no_grad():
            # Process with AllTracker
            flows, visconfs, _, _ = self.alltracker(
                img_sequence, 
                iters=4, 
                sw=None, 
                is_training=False
            )
            
            # flows shape: (B, T, 2, H, W) - flows relative to first frame
            # visconfs shape: (B, T, 2, H, W) - visibility confidence
            
            # For our case with 2 frames, we get:
            # flows[:, 0] = flow from frame 0 to frame 0 (should be zero)
            # flows[:, 1] = flow from frame 0 to frame 1 (forward flow)
            
            # Extract forward flow (from frame 0 to frame 1)
            flow_fwd = flows  # (n, 2, h, w)
            
            # For backward flow, we need to run AllTracker in reverse
            # Create reverse sequence
            img_sequence_rev = torch.stack([img1, img0], dim=1)  # (n, 2, c, h, w)
            
            flows_rev, _, _, _ = self.alltracker(
                img_sequence_rev, 
                iters=4, 
                sw=None, 
                is_training=False
            )
            
            # Extract backward flow (from frame 1 to frame 0)
            flow_bwd = flows_rev  # (n, 2, h, w)
            
        return flow_fwd, flow_bwd

    def process_sequence_with_alltracker(self, img_sequence, data=None):
        """
        Process entire sequence with AllTracker to get flows between all consecutive frames.
        This is used when we have access to the full sequence.
        """
        n, v, c, h, w = img_sequence.shape
        
        # Convert from [-1, 1] to [0, 255] (AllTracker expects [0, 255])
        img_sequence = (img_sequence * 0.5 + 0.5) * 255.0
        
        with torch.no_grad():
            # Process with AllTracker to get tracks
            flows, visconfs, _, _ = self.alltracker(
                img_sequence, 
                iters=4, 
                sw=None, 
                is_training=False
            )
            
            # flows shape: (B, T, 2, H, W) - flows relative to first frame
            # Convert to consecutive frame flows
            
            flow_fwd_list = []
            flow_bwd_list = []
            
            # flows[:, t] represents flow from frame 0 to frame t
            # We need flows from frame t to frame t+1
            
            for t in range(v - 1):
                if t == 0:
                    # For t=0 to t=1, use flows[:, 1] directly
                    flow_fwd = flows[:, t+1]  # Flow from frame 0 to frame 1
                else:
                    # For t to t+1, compute the difference
                    # Flow from frame t to frame t+1 = flow(0->t+1) - flow(0->t)
                    flow_curr = flows[:, t]     # Flow from frame 0 to frame t  
                    flow_next = flows[:, t+1]   # Flow from frame 0 to frame t+1
                    
                    # The flow from t to t+1 is approximately the difference
                    flow_fwd = flow_next - flow_curr
                
                flow_fwd_list.append(flow_fwd)
                
                # For backward flow, we need to reverse the direction
                # For now, approximate it as negative of forward flow
                flow_bwd = -flow_fwd
                flow_bwd_list.append(flow_bwd)
            
            # Stack all flows
            flows_fwd = torch.stack(flow_fwd_list, dim=1)  # (n, v-1, 2, h, w)
            flows_bwd = torch.stack(flow_bwd_list, dim=1)  # (n, v-1, 2, h, w)
            
        return flows_fwd, flows_bwd

    def extract_alltracker_tracks(self, img_sequence, grid_size=16):
        """
        Extract tracks directly from AllTracker without converting to flows.
        This is a more direct approach that preserves AllTracker's tracking quality.
        """
        n, v, c, h, w = img_sequence.shape
        
        # Convert from [-1, 1] to [0, 255] (AllTracker expects [0, 255])
        img_sequence = (img_sequence * 0.5 + 0.5) * 255.0
        
        # Create sampling grid 
        device = img_sequence.device
        x = torch.linspace(0, w-1, grid_size, device=device)
        y = torch.linspace(0, h-1, grid_size, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        sample_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (grid_size^2, 2)
        
        with torch.no_grad():
            # Run AllTracker to get trajectory predictions
            flows, visconfs, _, _ = self.alltracker(
                img_sequence, 
                iters=4, 
                sw=None, 
                is_training=False
            )
            
            # flows shape: (B, T, 2, H, W) - flows relative to first frame
            # visconfs shape: (B, T, 2, H, W) - visibility confidence
            # Extract tracks by sampling at grid points
            
            tracks_list = []
            visibles_list = []
            
            for t in range(v):
                if t == 0:
                    # Initial positions 
                    current_positions = sample_points.clone()  # (grid_size^2, 2)
                    # Convert to normalized coordinates [-1, 1]
                    current_positions_norm = torch.zeros_like(current_positions)
                    current_positions_norm[:, 0] = (current_positions[:, 0] / (w-1)) * 2 - 1
                    current_positions_norm[:, 1] = (current_positions[:, 1] / (h-1)) * 2 - 1
                    
                    tracks_list.append(current_positions_norm.unsqueeze(0))  # (1, grid_size^2, 2)
                    # For the first frame, all points should be visible by definition
                    visibles_list.append(torch.ones(1, grid_size**2, 1, device=device))
                else:
                    # Get flow at current positions
                    flow_t = flows[:, t]  # (1, 2, h, w) - flow from frame 0 to frame t
                    
                    # Sample flow at initial grid positions (in pixel coordinates)
                    grid_norm = sample_points.clone()
                    grid_norm[:, 0] = (grid_norm[:, 0] / (w-1)) * 2 - 1  # Normalize x to [-1, 1]
                    grid_norm[:, 1] = (grid_norm[:, 1] / (h-1)) * 2 - 1  # Normalize y to [-1, 1]
                    grid_norm = grid_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size^2, 2)
                    
                    # Sample flow
                    sampled_flow = F.grid_sample(
                        flow_t, grid_norm, 
                        mode='bilinear', align_corners=False
                    ).squeeze(2)  # (1, 2, grid_size^2)
                    sampled_flow = sampled_flow.permute(0, 2, 1)  # (1, grid_size^2, 2)
                    
                    # Convert flow to position (flow is in pixels)
                    new_positions = sample_points.unsqueeze(0) + sampled_flow  # (1, grid_size^2, 2)
                    
                    # Convert to normalized coordinates [-1, 1]
                    new_positions_norm = torch.zeros_like(new_positions)
                    new_positions_norm[:, :, 0] = (new_positions[:, :, 0] / (w-1)) * 2 - 1
                    new_positions_norm[:, :, 1] = (new_positions[:, :, 1] / (h-1)) * 2 - 1
                    
                    # Check bounds
                    bounds_valid = (new_positions_norm.abs() < 0.99).all(dim=-1, keepdim=True)  # (1, grid_size^2, 1)
                    
                    # Sample visibility confidence at new positions
                    if visconfs is not None:
                        new_positions_norm_vis = new_positions_norm.unsqueeze(1)  # (1, 1, grid_size^2, 2)
                        
                        # Use visconfs[:, :, 0, :, :] as suggested by user
                        vis_conf_t = visconfs[:, t, 0:1]  # (1, 1, h, w) - first channel of visibility
                        sampled_vis = F.grid_sample(
                            vis_conf_t, new_positions_norm_vis,
                            mode='bilinear', align_corners=False
                        ).squeeze(2)  # (1, 1, grid_size^2)
                        sampled_vis = sampled_vis.permute(0, 2, 1)  # (1, grid_size^2, 1)
                        
                        # Convert visibility confidence to binary visibility (threshold at 0.5)
                        vis_valid = (sampled_vis > 0.5).float()
                    else:
                        vis_valid = torch.ones_like(bounds_valid)
                    
                    # Combine bounds and visibility validity
                    valid = bounds_valid.float() * vis_valid
                    
                    tracks_list.append(new_positions_norm)
                    visibles_list.append(valid)
            
            # Stack results
            tracks = torch.stack(tracks_list, dim=2)  # (1, grid_size^2, v, 2)
            visibles = torch.stack(visibles_list, dim=2)  # (1, grid_size^2, v, 1)
            
        return tracks, visibles

    @torch.no_grad()
    def forward(self, img: torch.Tensor, data=None):
        n, v, c, h, w = img.shape

        if self.pair_mode == "one-to-many":
            img = img.reshape((n * v) // self.n_pairs, self.n_pairs, c, h, w)
            img0 = img[:, :1].expand(-1, self.n_pairs-1, -1, -1, -1).reshape(-1, c, h, w)
            img1 = img[:, 1:].reshape(-1, c, h, w)
        elif self.pair_mode == "sequential":
            img = img.reshape(n, v, c, h, w)
            img0 = img[:, :-1].reshape(-1, c, h, w)
            img1 = img[:, 1:].reshape(-1, c, h, w)

        if not self.use_existing_flow or (not "flows_fwd" in data) or len(data["flows_fwd"]) == 0:
            if self.flow_model == "raft":
                flow_fwd, flow_bwd = self.flow_raft(img0, img1)
            elif self.flow_model == "unimatch":
                flow_fwd, flow_bwd = self.flow_unimatch(img0, img1)
            elif self.flow_model == "alltracker":
                if self.pair_mode == "sequential" and v > 2:
                    # For sequential mode with multiple frames, use the sequence processing
                    flows_fwd, flows_bwd = self.process_sequence_with_alltracker(img, data)
                    flow_fwd = flows_fwd.reshape(-1, 2, h, w)
                    flow_bwd = flows_bwd.reshape(-1, 2, h, w)
                else:
                    # For pairwise processing or short sequences
                    flow_fwd, flow_bwd = self.flow_alltracker(img0, img1)
        else:
            # assert v == 2 or v == 3
            flow_fwd = data["flows_fwd"].reshape(-1, 2, h, w)
            flow_bwd = data["flows_bwd"].reshape(-1, 2, h, w)

        occ0, occ1 = compute_occlusions(flow_fwd, flow_bwd)
        flow0_r = torch.cat(
            (flow_fwd[:, 0:1, :, :] * 2 / w, flow_fwd[:, 1:2, :, :] * 2 / h), dim=1
        )
        flow1_r = torch.cat(
            (flow_bwd[:, 0:1, :, :] * 2 / w, flow_bwd[:, 1:2, :, :] * 2 / h), dim=1
        )

        if self.pair_mode == "one-to-many":
            flow = torch.stack((flow0_r, flow1_r), dim=1)
            occ = torch.stack((occ0, occ1), dim=1)

            img = torch.stack((img0, img1), dim=1)

            img_ip = torch.cat((img * .5 + .5, flow, occ), dim=2)

            new_v = v // self.n_pairs * (self.n_pairs - 1) * 2

            img_ip = img_ip.reshape(n, new_v, -1, h, w)

            return img_ip
        
        elif self.pair_mode == "sequential":
            flow0_r = flow0_r.reshape(n, v-1, 2, h, w)
            flow1_r = flow1_r.reshape(n, v-1, 2, h, w)
            occ0 = occ0.reshape(n, v-1, 1, h, w)
            occ1 = occ1.reshape(n, v-1, 1, h, w)

            flow0_r = torch.cat((flow0_r, torch.zeros_like(flow0_r[:, :1])), dim=1)
            flow1_r = torch.cat((torch.zeros_like(flow1_r[:, :1]), flow1_r), dim=1)
            occ0 = torch.cat((occ0, torch.ones_like(occ0[:, :1])), dim=1)
            occ1 = torch.cat((torch.ones_like(occ1[:, :1]), occ1), dim=1)

            img_ip_fwd = torch.cat((img * .5 + .5, flow0_r, occ0), dim=2)
            img_ip_bwd = torch.cat((torch.flip(img, dims=(1,)) * .5 + .5, flow1_r, occ1), dim=2)

            return img_ip_fwd, img_ip_bwd

    @property
    def device(self):
        return next(self.parameters()).device


class AutoMaskingWrapper(nn.Module):

    # Adds the corresponding color from the input frame for reference
    def __init__(self, image_processor):
        super().__init__()
        self.image_processor = image_processor

        self.channels = self.image_processor.channels + 1

    def forward(self, images, threshold):
        n, v, c, h, w = images.shape
        processed_images = self.image_processor(images)
        thresholds = threshold.view(n, 1, 1, h, w).expand(n, v, 1, h, w)
        processed_images = torch.stack((processed_images, thresholds), dim=2)
        return processed_images


CANONICAL_PROJ = torch.tensor(
    [
        [276.2771, 0.0000, 341.0247],
        [0.0000, 276.2771, 119.3848],
        [0.0000, 0.0000, 0.5000],
    ]
)


class CanonicalProcessor(nn.Module):
    def __init__(self, canonical_proj: torch.Tensor = CANONICAL_PROJ):
        super().__init__()
        self.canonical_proj = canonical_proj

    def forward(
        self,
        img_data: torch.Tensor,
        Ks: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        canonical_K_d = self.canonical_proj.to(img_data.device)

        flatten = Ks.ndim == 4
        img_shape = img_data.shape
        Ks_shape = Ks.shape
        if flatten:
            assert img_data.ndim == 5
            img_data = img_data.flatten(0, 1)
            Ks = Ks.flatten(0, 1)
        else:
            assert img_data.ndim == 4

        img_data, Ks = resize_to_canonical_frame(img_data, Ks, canonical_K_d)

        if flatten:
            img_data = img_data.view(*img_shape[:2], *img_data.shape[1:])
            Ks = Ks.view(*Ks_shape[:2], *Ks.shape[1:])

        return img_data, Ks


class CroppingProcessor(nn.Module):
    # Crops the input image to the specified size, adapts the intrisics accordingly
    def __init__(self, crop_type: str, img_size: tuple[int, int]):
        super().__init__()
        self.crop_type = crop_type
        self.img_size = img_size

    def forward(
        self,
        img_data: torch.Tensor,
        Ks: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flatten = Ks.ndim == 4
        img_shape = img_data.shape
        Ks_shape = Ks.shape
        if flatten:
            assert img_data.ndim == 5
            img_data = img_data.flatten(0, 1)
            Ks = Ks.flatten(0, 1)
        else:
            assert img_data.ndim == 4

        match self.crop_type:
            case "random":
                crop_fn = random_crop
            case "center":
                crop_fn = center_crop
            case "no_crop":
                crop_fn = no_crop
            case _:
                raise ValueError(
                    f"Unknown crop type: {self.crop_type}. Supported types are: random, center, no_crop"
                )
        img_data, Ks = crop_fn(img_data, Ks, self.img_size)

        if flatten:
            img_data = img_data.view(*img_shape[:2], *img_data.shape[1:])
            Ks = Ks.view(*Ks_shape[:2], *Ks.shape[1:])

        return img_data, Ks


class NormalizationProcessor(nn.Module):
    # Normalizes the input image to the specified range
    def __init__(self, normalization_config):
        super().__init__()
        self.normalization_config = normalization_config

    def forward(
        self,
        img_data: torch.Tensor,
        Ks: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flatten = Ks.ndim == 4
        img_shape = img_data.shape
        Ks_shape = Ks.shape
        if flatten:
            assert img_data.ndim == 5
            img_data = img_data.flatten(0, 1)
            Ks = Ks.flatten(0, 1)
        else:
            assert img_data.ndim == 4

        Ks = normalize_camera_intrinsics(
            Ks,
            torch.tensor(img_data.shape[-2:], device=Ks.device)
            .unsqueeze(0)
            .expand(Ks.shape[0], -1),
        )

        if flatten:
            img_data = img_data.view(*img_shape[:2], *img_data.shape[1:])
            Ks = Ks.view(*Ks_shape[:2], *Ks.shape[1:])

        return img_data, Ks
