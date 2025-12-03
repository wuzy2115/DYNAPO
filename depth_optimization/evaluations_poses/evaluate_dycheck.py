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

"""Evaluate dycheck poses."""

# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
# pylint: disable=invalid-name
# pylint: disable=g-explicit-length-test

import os
import sys
from evaluate_rpe import evaluate_trajectory
from lietorch import SE3  # pylint: disable=g-importing-member
import numpy as np
import torch

sys.path.append(os.path.realpath("."))
import camera_tracking_scripts.colmap_read_model as read_model


def load_colmap_data(realdir):
  """Load colmap data."""
  camerasfile = os.path.join(realdir, "sparse/cameras.bin")
  camdata = read_model.read_cameras_binary(camerasfile)

  list_of_keys = list(camdata.keys())
  cam = camdata[list_of_keys[0]]
  print("Cameras", len(cam))

  imagesfile = os.path.join(realdir, "sparse/images.bin")
  imdata = read_model.read_images_binary(imagesfile)

  w2c_mats = []
  bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])

  names = [imdata[k].name for k in imdata]
  img_keys = [k for k in imdata]

  print("Images #", len(names))
  perm = np.argsort(names)

  points3dfile = os.path.join(realdir, "sparse/points3D.bin")
  pts3d = read_model.read_points3d_binary(points3dfile)

  # extract point 3D xyz
  point_cloud = []
  for key in pts3d:
    point_cloud.append(pts3d[key].xyz)

  upper_bound = 100000

  if upper_bound < len(img_keys):
    print("Only keeping " + str(upper_bound) + " images!")

  for i in perm[0 : min(upper_bound, len(img_keys))]:
    im = imdata[img_keys[i]]
    if "2_" in im.name:
      continue

    if "1_" in im.name:
      continue

    # print(im.name)
    R = im.qvec2rotmat()
    t = im.tvec.reshape([3, 1])
    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    w2c_mats.append(m)

  w2c_mats = np.stack(w2c_mats, 0)
  # bounds_mats = np.stack(bounds_mats, 0)
  c2w_mats = np.linalg.inv(w2c_mats)

  return c2w_mats


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

  trans = data_mean - s * rot * model_mean  # pylint: disable=redefined-outer-name

  model_aligned = s * rot * model + trans
  alignment_error = model_aligned - data

  trans_error = np.sqrt(  # pylint: disable=redefined-outer-name
      np.sum(np.multiply(alignment_error, alignment_error), 0)
  ).A[0]

  return rot, trans, trans_error, s, model_aligned


if __name__ == "__main__":
  scene_names = []
  scene_names += ["apple", "backpack", "block", "creeper"]
  scene_names += ["handwavy", "haru-sit", "mochi-high-five", "pillow"]
  scene_names += ["spin", "sriracha-tree", "teddy", "paper-windmill"]

  datapath = "/home/zhengqili/dycheck"
  rootdir = "%s/reconstructions" % os.getcwd()
  ate = []
  rte = []
  rre = []

  for scene_name in scene_names:
    gt_cam2w = load_colmap_data("%s/%s/dense" % (datapath, scene_name))

    poses = np.load(os.path.join(rootdir, scene_name, "poses.npy"))
    cam_c2w = SE3(
        torch.as_tensor(poses, device="cpu")
    ).inv()  # .matrix().numpy()

    est_cam2w = cam_c2w.matrix().numpy()
    num_cams = gt_cam2w.shape[0]

    assert gt_cam2w.shape[0] == est_cam2w.shape[0]

    full_t = np.dot(np.linalg.inv(gt_cam2w[-1]), gt_cam2w[0])
    normalize_scale = np.linalg.norm(full_t[:3, 3]) + 1e-8
    gt_cam2w[:, :3, 3] /= normalize_scale

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

    # breakpoint()
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

  # print("exp_name ", exp_name)
  print("Average ATE: ", np.mean(ate))
  print("Average RTE: ", np.mean(rte))
  print("Average RRE: ", np.mean(rre))
