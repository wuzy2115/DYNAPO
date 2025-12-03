# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Consistent video depth initialization saver."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from lietorch import SE3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_grad", type=float, default=2.0, help="w_grad")
    parser.add_argument("--w_normal", type=float, default=6.0, help="w_normal")
    parser.add_argument(
        "--output_dir", type=str, default="outputs_wo_cvd", help="outputs direcotry"
    )
    parser.add_argument("--scene_name", default='alley_1', type=str, help="scene name")
    args = parser.parse_args()

    # 保持原始数据加载流程
    rootdir = os.getcwd() + "/reconstructions_tumrgbd"
    scene_name = args.scene_name
    
    img_data = np.load(os.path.join(rootdir, scene_name, "images.npy"))[:, ::-1, ...]
    disp_data = (
        np.load(
            os.path.join(rootdir, scene_name.replace("_opt", ""), "disps.npy")
        )
        + 1e-6
    )
    intrinsics = np.load(os.path.join(rootdir, scene_name, "intrinsics.npy"))
    poses = np.load(os.path.join(rootdir, scene_name, "poses.npy"))

    # 保持原始Tensor转换
    img_data_pt = (
        torch.from_numpy(np.ascontiguousarray(img_data)).float().cuda() / 255.0
    )
    init_disp = torch.from_numpy(disp_data).float().cuda()
    poses_th = torch.as_tensor(poses, device="cpu").float().cuda()

    intrinsics = intrinsics[0]
    # 保持原始相机参数计算
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    K = torch.from_numpy(K).float().cuda()
    K_o = K.clone()

    # 保持原始预处理流程
    RESIZE_FACTOR = 0.5
    init_disp = torch.nn.functional.interpolate(
        init_disp.unsqueeze(1),
        scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
        mode="bilinear",
    ).squeeze(1)

    # 保持原始上采样方式
    init_disp_hr = torch.nn.functional.interpolate(
        init_disp.unsqueeze(1), scale_factor=(2, 2), mode="bilinear"
    ).squeeze(1)

    # 严格保持原始保存格式
    cam_c2w = SE3(poses_th).inv().matrix()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    np.savez(
        "%s/%s_sgd_cvd_hr.npz" % (args.output_dir, scene_name),
        images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0),
        depths=np.clip(np.float16(1.0 / init_disp_hr.cpu().numpy()), 1e-3, 1e2),
        intrinsic=K_o.detach().cpu().numpy(),
        cam_c2w=cam_c2w.detach().cpu().numpy(),
    )
