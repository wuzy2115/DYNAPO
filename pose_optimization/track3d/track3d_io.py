import argparse
import os.path as osp
import pickle
import numpy as np
import track3d.utils as utils
from scipy.spatial.transform import Rotation


def load_rgbd_cam_megasam(vid: str, root_dir: str):
    """Load RGB, depth, and camera data from the second script's format."""
    input_dict = {'left': {'camera': [], 'depth': [], 'video': []}}
    
    # Load images (video frames)
    img_path = osp.join(root_dir, vid, "images.npy")
    img_data = np.load(img_path)[:, ::-1, ...]  # Assuming vertical flip as in the second script
    n_fr = img_data.shape[0]
    input_dict['nfr'] = n_fr
    H, W = img_data.shape[1], img_data.shape[2]
    
    # Load camera poses (extrinsics)
    poses_path = osp.join(root_dir, vid, "poses.npy")
    poses = np.load(poses_path)  # Shape (n_fr, 7)
    
    # Load camera intrinsics
    intrinsics_path = osp.join(root_dir, vid, "intrinsics.npy")
    intrinsics = np.load(intrinsics_path)  # [fx, fy, cx, cy]
    
    # Process each frame to create CameraAZ objects
    for fid in range(n_fr):
        # Compute extrinsics (world to camera matrix)
        pose = poses[fid]  # 4x4 camera-to-world matrix
        C, q = pose[:3], pose[3:]        
        rotation = Rotation.from_quat([q[0], q[1], q[2], q[3]])
        R_c2w = rotation.as_matrix()  # Camera-to-world rotation
        R_w2c = R_c2w.T  # World-to-camera rotation
        t_w2c = -R_w2c @ C  # Translation in camera coordinates
        extr = np.hstack([R_w2c, t_w2c.reshape(-1, 1)])
        
        # Normalize intrinsics based on image dimensions
        fx_normalized = intrinsics[0][0] / W
        fy_normalized = intrinsics[0][1] / H
        cx_normalized = intrinsics[0][2] / W
        cy_normalized = intrinsics[0][3] / H
        
        intr_normalized = {
            'fx': fx_normalized,
            'fy': fy_normalized,
            'cx': cx_normalized,
            'cy': cy_normalized,
            'k1': 0,
            'k2': 0,
        }
        input_dict['left']['camera'].append(
            utils.CameraAZ(
                from_json={
                    'extr': extr,
                    'intr_normalized': intr_normalized,
                }
            )
        )
    
    # Convert image data to list of uint8 frames
    rgbs = [img_data[fid].astype(np.uint8) for fid in range(n_fr)]
    input_dict['left']['video'] = np.transpose(np.array(rgbs), (0, 2, 3, 1))
    
    # Load disparity and compute depth
    disp_path = osp.join(root_dir, vid.replace("_opt", ""), "disps.npy")
    disp_data = np.load(disp_path) + 1e-6  # Avoid division by zero
    depths = 1.0 / disp_data
    
    # Apply depth thresholding and clipping
    depths[depths > 20] = 0
    depths[depths < 0] = 0
    input_dict['left']['depth'] = depths
    
    return input_dict

def load_tracks(scene_name: str, save_root: str, recon_results: str=''):
  with open(
      osp.join(save_root, scene_name, scene_name + '-optimized_tracks.pkl'),
      'rb',
  ) as f:
    opt_track3d = pickle.load(f)
    track3d = utils.Track3d(load_from_json=opt_track3d)
  
  motion_mag = utils.get_scene_motion_2d_displacement(track3d)
  track3d_dynamic = track3d.get_new_track((motion_mag > 16).any(axis=1))
  
  
  return track3d_dynamic

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--scene_name', help='video id, in the format of <raw-video-id>_<timestamp>', type=str, default='alley_1')
  parser.add_argument('--output_folder', help='output folder', type=str, default='/home/zhuoyuanwu/stereo4d-code/sintel_processed_grid_size_16')
  parser.add_argument('--recon_results', type=str, default='/home/zhuoyuanwu/mega-sam/reconstructions_orig')

  args = parser.parse_args()

  track3d = load_tracks(args.scene_name, args.output_folder, args.recon_results)

if __name__ == '__main__':
  main()