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

"""Evaluate Sintel dataset."""

# pylint: disable=invalid-name

import os
from evaluate_rpe import evaluate_trajectory
from lietorch import SE3  # pylint: disable=g-importing-member
import numpy as np
import torch


def rotmat2qvec(R):
  """Rotation matrix to quaternion."""
  Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
  K = (
      np.array([
          [Rxx - Ryy - Rzz, 0, 0, 0],
          [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
          [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
          [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
      ])
      / 3.0
  )
  eigvals, eigvecs = np.linalg.eigh(K)
  qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
  if qvec[0] < 0:
    qvec *= -1
  return qvec


def align_trajectories(model, data):
  """Align two trajectories using the method of Horn (closed-form).

  Args:
    model: first trajectory (3xn)
    data: second trajectory (3xn)

  Returns:
    rot: rotation matrix (3x3)
    trans: translation vector (3x1)
    trans_error: translational error per point (1xn)
  """
  np.set_printoptions(precision=3, suppress=True)
  model_mean = [[model.mean(1)[0]], [model.mean(1)[1]], [model.mean(1)[2]]]
  data_mean = [[data.mean(1)[0]], [data.mean(1)[1]], [data.mean(1)[2]]]
  model_zerocentered = model - model_mean
  data_zerocentered = data - data_mean

  W = np.zeros((3, 3))
  for column in range(model.shape[1]):
    W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
  U, _, Vh = np.linalg.linalg.svd(W.transpose())
  S = np.matrix(np.identity(3))
  if np.linalg.det(U) * np.linalg.det(Vh) < 0:
    S[2, 2] = -1
  rot = U * S * Vh  # pylint: disable=redefined-outer-name

  rotmodel = rot * model_zerocentered
  dots = 0.0
  norms = 0.0

  for column in range(data_zerocentered.shape[1]):
    dots += np.dot(
        data_zerocentered[:, column].transpose(), rotmodel[:, column]
    )
    normi = np.linalg.norm(model_zerocentered[:, column])
    norms += normi * normi

  s = float(dots / norms)

  # print ("scale: %f " % s)
  trans = data_mean - s * rot * model_mean  # pylint: disable=redefined-outer-name

  model_aligned = s * rot * model + trans
  alignment_error = model_aligned - data

  trans_error = np.sqrt(  # pylint: disable=redefined-outer-name
      np.sum(np.multiply(alignment_error, alignment_error), 0)
  ).A[0]

  return rot, trans, trans_error, s, model_aligned


if __name__ == "__main__":
  scene_names = []
  scene_names += ["alley_1", "alley_2", "temple_2", "temple_3", "market_5"]
  scene_names += [
      "mountain_1",
      "bamboo_2",
      "bamboo_1",
  ]
  scene_names += ["ambush_4", "ambush_5", "ambush_6"]
  scene_names += ["market_2", "market_6", "cave_4"]
  scene_names += ["cave_2", "shaman_3", "sleeping_1", "sleeping_2"]

  ate = []
  rte = []
  rre = []

  gt_root_dir = "/home/zhengqili/filestore/droid_slam/data/Sintel"
  rootdir = "%s/reconstructions" % os.getcwd()

  for scene_name in scene_names:
    gt_path = os.path.join(gt_root_dir, scene_name, "extrinsics.npy")
    gt_cam2w = np.load(gt_path)
    poses = np.load(os.path.join(rootdir, scene_name, "poses.npy"))
    cam_c2w = SE3(
        torch.as_tensor(poses, device="cpu")
    ).inv()  # .matrix().numpy()
    est_cam2w = cam_c2w.matrix().numpy()
    num_cams = gt_cam2w.shape[0]

    tstamps = [float(i / 30.0) for i in range(num_cams)]
    assert gt_cam2w.shape[0] == est_cam2w.shape[0]

    full_t = np.dot(np.linalg.inv(gt_cam2w[-1]), gt_cam2w[0])
    normalize_scale = np.linalg.norm(full_t[:3, 3]) + 1e-8
    gt_cam2w[:, :3, 3] /= normalize_scale
    print(normalize_scale)

    rot, trans, trans_error, scale, align_tj = align_trajectories(
        est_cam2w[:, :3, 3].transpose(1, 0), gt_cam2w[:, :3, 3].transpose(1, 0)
    )

    est_cam2w[:, :3, 3] = (
        scale * rot * est_cam2w[:, :3, 3].transpose(1, 0) + trans
    ).transpose(1, 0)

    for k in range(num_cams):
      est_cam2w[k, :3, :3] = rot @ est_cam2w[k, :3, :3]

    traj_est_dict = [est_cam2w[i, ...] for i in range(est_cam2w.shape[0])]
    traj_gt_dict = [gt_cam2w[i, ...] for i in range(gt_cam2w.shape[0])]
    rpe_result = evaluate_trajectory(
        traj_gt_dict, traj_est_dict, param_fixed_delta=True, param_delta=1
    )

    rte_error = np.array(rpe_result)[:, 2]
    rre_error = np.array(rpe_result)[:, 3]

    trans_error_mean = np.sqrt(np.mean(rte_error**2))
    rot_error_mean = np.sqrt(np.mean(rre_error**2))

    print(scene_name)
    print(
        "absolute_translational_error.rmse %f m"
        % np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
    )
    print("relative translational_error %f m" % trans_error_mean)
    print("relative rotational_error %f deg" % np.rad2deg(rot_error_mean))

    ate.append(np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)))
    rte.append(trans_error_mean)
    rre.append(np.rad2deg(rot_error_mean))

  print("Average ATE: ", np.mean(ate))
  print("Average RTE: ", np.mean(rte))
  print("Average RRE: ", np.mean(rre))
