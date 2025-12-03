import copy
import numpy as np
import math
import numpy as np
import einops
import tqdm
from typing import List, Optional
import cv2
from matplotlib.collections import LineCollection
import matplotlib

import matplotlib.pyplot as plt

class CameraAZ:
  def __init__(
      self,
      from_json=None,
      from_jaxcam=None,
  ):
    """
    Initialize the object with either JSON data or JAX camera data.

    Parameters:
    from_json (dict, optional): A dictionary containing 'extr' and 'intr_normalized' keys.
    from_jaxcam (jaxcam, optional): Initialize from JAX camera.
    """
    if from_json is not None:
      self.extr = from_json['extr']
      self.intr_normalized = from_json['intr_normalized']
    elif from_jaxcam is not None:
      self._init_from_jaxcam(from_jaxcam)
    else:
      raise NotImplementedError()
    
  def __str__(self):
    return f"extr: \n{self.extr}\n intr_normalized: \n{self.intr_normalized}"

  def _init_from_jaxcam(self, jax_camera):
    self.extr = np.asarray(jax_camera.world_to_camera_matrix[:3])
    self.intr_normalized = {
        'fx': (
            jax_camera.intrinsic_matrix[0][0] / jax_camera.image_size_x
        ).item(),
        'fy': (
            jax_camera.intrinsic_matrix[1][1] / jax_camera.image_size_y
        ).item(),
        'cx': (
            jax_camera.intrinsic_matrix[0][2] / jax_camera.image_size_x
        ).item(),
        'cy': (
            jax_camera.intrinsic_matrix[1][2] / jax_camera.image_size_y
        ).item(),
        'k1': 0,
        'k2': 0,
    }

  def to_json_format(self):
    return {
        'extr': self.extr,
        'intr_normalized': self.intr_normalized,
    }

  def get_c2w(self):
    """
    Get the camera-to-world transformation matrix.
    Returns:
      numpy.ndarray: A 4x4 camera-to-world transformation matrix.
    """
    w2c = np.concatenate((self.extr, np.array([[0, 0, 0, 1]])), axis=0)
    c2w = np.linalg.inv(w2c)
    return c2w

  def get_hfov_deg(self):
    """
    Get the horizontal field of view (HFOV) in degrees.
    """
    return math.degrees(2 * np.arctan(0.5 / self.intr_normalized['fx']))


  def get_intri_matrix(self, imh: int, imw: int):
    """
    Get the intrinsic matrix

    Parameters:
    imh (int): The height of the image.
    imw (int): The width of the image.

    Returns:
    numpy.ndarray: A 3x3 intrinsic matrix.
    """
    return np.array([
        [self.intr_normalized['fx'] * imw, 0, self.intr_normalized['cx'] * imw],
        [0, self.intr_normalized['fy'] * imh, self.intr_normalized['cy'] * imh],
        [0, 0, 1],
    ])
  
  def pix_2_world_np(
      self,
      xy: np.ndarray,
      depth: np.ndarray,
      valid_depth_min: float,
      valid_depth_max: float,
  ):
    """unproject points from ndc from to world frame.

    depth: h x w xy definition:

        xy.shape [:, 2]
        left to right: [0, w]
        top to bottom: [0, h]
    """

    _, dim = xy.shape
    assert dim == 2
    imh, imw = depth.shape

    valid_mask = (
        (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < imw) & (xy[:, 1] < imh)
    )

    x_cam = (
        xy[..., 0] / imw - self.intr_normalized['cx']
    ) / self.intr_normalized['fx']
    y_cam = (
        xy[..., 1] / imh - self.intr_normalized['cy']
    ) / self.intr_normalized['fy']
    z_cam = np.ones_like(xy[..., 0])
    xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    x_query = np.clip(np.round(xy[:, 0]).astype(int), 0, imw - 1)
    y_query = np.clip(np.round(xy[:, 1]).astype(int), 0, imh - 1)
    depth_values = depth[y_query, x_query]

    valid_mask = (
        valid_mask
        & (depth_values > valid_depth_min)
        & (depth_values < valid_depth_max)
    )

    xyz_cam = depth_values[:, None] * xyz_cam

    xyz_world = (self.extr[:3, :3].T @ (xyz_cam - self.extr[:3, 3]).T).T
    return xyz_world, valid_mask

  

  def world_2_pix_np(
      self, xyz_world: np.ndarray, imh: int, imw: int, min_depth: float = 0.01
  ):
    """project points from world frame to screen space.

    xyz_world: [:, 3] array of points in world frame.
    """
    xyz_world_hom = np.concatenate(
        (xyz_world, np.ones_like(xyz_world[:, :1])), axis=-1
    )
    xyz_homo = (
        self.get_intri_matrix(imh, imw) @ self.extr @ xyz_world_hom.T
    ).T  # npt, 3
    depth = xyz_homo[:, 2]
    xy = xyz_homo[:, :2] / xyz_homo[:, 2:]
    valid_mask = (
        (xy[:, 0] >= 0.5)
        & (xy[:, 1] >= 0.5)
        & (xy[:, 0] < imw - 0.5)
        & (xy[:, 1] < imh - 0.5)
        & (depth > min_depth)
    )
    return xy, valid_mask, depth


class Track3d:

  def __init__(
      self,
      tracks: Optional[np.ndarray] = None,
      visibles: Optional[np.ndarray] = None,
      depths: Optional[np.ndarray] = None,
      cameras: Optional[List[CameraAZ]] = None,
      video: Optional[np.ndarray] = None,
      query_points: Optional[np.ndarray] = None,
      valid_depth_min=0,
      valid_depth_max=20,
      load_from_json=None,
      track3d: Optional[np.ndarray] = None,
      visible_list: Optional[np.ndarray] = None,
      color_values: Optional[np.ndarray] = None,
  ):
    """tracks: npt x nframe x d2

    visibles: npt x nframe
    depths: nframe x imh x imw
    cameras: nframe x CameraAZ
    video: nframe x imh x imw
    query_points: npt x 3(t, h, w)
    valid_depth_min: float, min value for valid depth
    valid_depth_max: float, max value for valid depth
    """
    if load_from_json is not None:
      self._load_from_json(load_from_json)
    else:
      # sanity check
      if tracks is not None:
        npt, nframe, d2 = tracks.shape
        assert d2 == 2, 'tracks dimension should be 2'
        assert (npt, nframe) == visibles.shape, f'Wrong shape visibles.shape {visibles.shape}, expected {npt, nframe}' # pytype: disable=attribute-error
        assert depths.shape[0] == nframe, ( # pytype: disable=attribute-error
            f'Wrong shape depths.shape[0] {depths.shape[0]}, expected nframe' # pytype: disable=attribute-error
            f' {nframe}'
        )
        _, imh, imw = depths.shape # pytype: disable=attribute-error
        assert len(cameras) == nframe, f'Wrong shape cameras.shape {len(cameras)}, expected nframe {nframe}'
        assert (nframe, imh, imw, 3) == video.shape, f'Wrong shape video.shape {video.shape}, expected (nframe, imh, imw, 3) {nframe, imh, imw, 3}' # pytype: disable=attribute-error
        if query_points is not None:
          assert query_points.shape == (npt, 3), f'query_points should be npt x 3, got {query_points.shape}'

        # Filter out tracks that would result in NaN values
        # valid_track_mask = self._verify_tracks_for_nan(tracks, visibles, depths, valid_depth_min, valid_depth_max)
        # print(f"Filtering tracks: {np.sum(~valid_track_mask)} out of {npt} tracks removed due to potential NaN values")
        
        # # Apply filtering
        # tracks = tracks[valid_track_mask]
        # visibles = visibles[valid_track_mask]
        # if query_points is not None:
        #   query_points = query_points[valid_track_mask]
        
        # Update npt after filtering
        npt = tracks.shape[0]

        # unproject track
        track3d = []
        visible_list = []
        for t in range(len(video)):
          xyz_world, valid_mask = cameras[t].pix_2_world_np(
              tracks[:, t],
              depths[t],
              valid_depth_min,
              valid_depth_max,
          )
          visible_list.append(visibles[:, t] & valid_mask)
          track3d.append(xyz_world)
        track3d = einops.rearrange(
            np.stack(track3d, axis=0), 't npt d3->npt t d3'
        )
        visible_list = einops.rearrange(
            np.stack(visible_list, axis=0), 't npt->npt t'
        )
        
        # # Final verification: remove any tracks that still contain NaN values
        # nan_mask = np.any(np.isnan(track3d), axis=(1, 2))
        # if np.any(nan_mask):
        #   print(f"Final NaN check: Removing {np.sum(nan_mask)} additional tracks with NaN values")
        #   valid_final_mask = ~nan_mask
        #   track3d = track3d[valid_final_mask]
        #   visible_list = visible_list[valid_final_mask]
        #   if query_points is not None:
        #     query_points = query_points[valid_final_mask]
      elif track3d is not None:
        npt, nframe = track3d.shape[:2]
        assert track3d.shape[2] == 3, f'track3d should be npt x nframe x 3, got {track3d.shape}'
        assert (npt, nframe) == visible_list.shape, f'Wrong shape visible_list.shape {visible_list.shape}, expected {npt, nframe}' # pytype: disable=attribute-error
        assert len(cameras) == nframe, f'Wrong shape cameras.shape {len(cameras)}, expected nframe {nframe}'
        assert nframe == video.shape[0], f'Wrong shape video.shape[0] {video.shape[0]}, expected nframe {nframe}' # pytype: disable=attribute-error
        _, imh, imw, _ = video.shape # pytype: disable=attribute-error
      else:
        raise NotImplementedError
      if color_values is not None:
        pass
      elif query_points is not None:
        # get point color
        color_values = video[
            query_points[:, 0],
            query_points[:, 1].astype(int),
            query_points[:, 2].astype(int),
        ]  # npt, 3
      else:
        color_values = None

      self.cameras = cameras
      self.track3d = track3d
      self.imh = imh
      self.imw = imw
      self.visible_list = visible_list
      self.color_values = color_values
      self.video = video
      self.query_points = query_points

  def _verify_tracks_for_nan(
      self, 
      tracks: np.ndarray, 
      visibles: np.ndarray, 
      depths: np.ndarray, 
      valid_depth_min: float, 
      valid_depth_max: float
  ) -> np.ndarray:
    """
    Verify tracks to filter out those that would result in NaN values in 3D coordinates.
    
    Args:
      tracks: 2D tracks array of shape (npt, nframe, 2)
      visibles: visibility mask of shape (npt, nframe)
      depths: depth maps of shape (nframe, imh, imw)
      valid_depth_min: minimum valid depth value
      valid_depth_max: maximum valid depth value
    
    Returns:
      np.ndarray: Boolean mask of shape (npt,) indicating which tracks are valid
    """
    npt, nframe, _ = tracks.shape
    _, imh, imw = depths.shape
    valid_track_mask = np.ones(npt, dtype=bool)
    
    for track_idx in range(npt):
      track_valid = True
      
      for frame_idx in range(nframe):
        # Only check frames where the track is supposed to be visible
        if not visibles[track_idx, frame_idx]:
          continue
          
        # Get the 2D coordinates for this track at this frame
        xy = tracks[track_idx, frame_idx]  # shape (2,)
        
        # Check if coordinates are within image bounds
        if xy[0] < 0 or xy[1] < 0 or xy[0] >= imw or xy[1] >= imh:
          continue
          
        # Get the depth value at this location
        x_query = int(np.clip(np.round(xy[0]), 0, imw - 1))
        y_query = int(np.clip(np.round(xy[1]), 0, imh - 1))
        depth_value = depths[frame_idx, y_query, x_query]
        
        # Check if depth value would cause NaN in 3D coordinates
        if (np.isnan(depth_value) or 
            np.isinf(depth_value) or 
            depth_value <= 0 or
            depth_value < valid_depth_min or 
            depth_value > valid_depth_max):
          track_valid = False
          break
      
      valid_track_mask[track_idx] = track_valid
    
    return valid_track_mask

  def _load_from_json(self, load_from_json):
    if load_from_json['cameras'] is not None:
      self.cameras = [
          CameraAZ(from_json=camera)
          for camera in load_from_json['cameras']  # FIXME
          # for camera in load_from_json['camera']
      ]
    self.track3d = np.stack(load_from_json['track3d'], axis=0)
    self.imh = load_from_json['imh']
    self.imw = load_from_json['imw']
    self.visible_list = load_from_json['visible_list']
    self.color_values = load_from_json['color_values']
    self.video = load_from_json['video']
    self.query_points = load_from_json['query_points']

  def to_json_format(self, save_video=False, save_camera=True):
    return {
        'cameras': [camera.to_json_format() for camera in self.cameras] if save_camera else None,
        'track3d': self.track3d,
        'imh': self.imh,
        'imw': self.imw,
        'visible_list': self.visible_list,
        'color_values': self.color_values,
        'video': self.video if save_video else None,
        'query_points': getattr(self, 'query_points', None)
    }

  def get_new_track(self, track_mask=None, percentage=None | float):
    """
    Generate a new track by applying a mask or a random selection based on a given percentage.

    Parameters:
    track_mask (numpy.ndarray, optional): A boolean mask array to select specific tracks. If None, a random mask will be generated.
    percentage (float, optional): The percentage of tracks to randomly select if track_mask is None. Should be a value between 0 and 1.

    Returns:
    new_track (object): A new instance of the track object with the selected tracks.
    """
    if track_mask is None:
      track_mask = np.random.uniform(size=self.track3d.shape[0]) < percentage
    new_track = copy.deepcopy(self)
    new_track.track3d = new_track.track3d[track_mask]
    new_track.visible_list = new_track.visible_list[track_mask]
    if new_track.color_values is not None:
      new_track.color_values = new_track.color_values[track_mask]
    if new_track.query_points is not None:
      new_track.query_points = new_track.query_points[track_mask]
    return new_track

  @staticmethod
  def combine_tracks(track_list: List['Track3d']) -> 'Track3d':
    """
    Combine multiple Track3d objects into a single Track3d object.
    
    Args:
      track_list: List of Track3d objects to combine
      
    Returns:
      Track3d: A new Track3d object with combined tracks
    """
    if not track_list:
      raise ValueError("track_list cannot be empty")
    
    if len(track_list) == 1:
      return copy.deepcopy(track_list[0])
    
    # Use the first track as reference for camera, video, and image dimensions
    reference_track = track_list[0]
    
    # Combine track3d arrays
    combined_track3d = np.concatenate([track.track3d for track in track_list], axis=0)
    
    # Combine visible_list arrays
    combined_visible_list = np.concatenate([track.visible_list for track in track_list], axis=0)
    
    # Combine color_values if they exist
    combined_color_values = None
    if all(track.color_values is not None for track in track_list):
      combined_color_values = np.concatenate([track.color_values for track in track_list], axis=0)
    
    # Combine query_points if they exist
    combined_query_points = None
    if all(track.query_points is not None for track in track_list):
      combined_query_points = np.concatenate([track.query_points for track in track_list], axis=0)
    
    # Create new Track3d object
    combined_track = Track3d(
      track3d=combined_track3d,
      visible_list=combined_visible_list,
      cameras=reference_track.cameras,
      video=reference_track.video,
      color_values=combined_color_values,
      query_points=combined_query_points
    )
    
    return combined_track
  


def get_scene_motion_2d_displacement(
    track3d: Track3d,
    tracks_leave_trace=16,
):
  """Get 2D point trajectories of scene motion.

  Returns max 2D displacement of 3D points over tracks_leave_trace frames as if
  the camera is static. This measurement decouples camera motion.

  Args:
      track3d: An instance of Track3d containing 3D tracks and visibility info.
      tracks_leave_trace: Number of frames over which to compute displacement.

  Returns:
      displacement: A (npt, nframe) array of max 2D displacements over
      tracks_leave_trace frames.
  """
  all_points = track3d.track3d  # Shape: (npt, nframe, 3)
  npt, nframe, _ = all_points.shape
  displacement = np.zeros_like(track3d.visible_list, dtype=np.float32)
  for t in tqdm.tqdm(range(nframe), desc='Computing 2D displacement'):
    s_start = max(0, t - tracks_leave_trace)
    s_end = t + 1  # Include current frame
    s_list = np.arange(s_start, s_end)  # Shape: (L,)
    L = len(s_list)
    if L < 2:
      # Not enough frames to compute displacement
      continue
    # Extract positions and visibilities for relevant frames
    positions = all_points[:, s_list, :]  # Shape: (npt, L, 3)
    visibilities = track3d.visible_list[:, s_list]  # Shape: (npt, L)
    # Flatten positions for projection
    positions_flat = positions.reshape(-1, 3)
    # Project all positions using the camera at frame t
    points_2d_flat, valid_mask_flat, _ = track3d.cameras[t].world_2_pix_np(
        positions_flat,
        track3d.imh,
        track3d.imw,
    )
    # Reshape back to (npt, L, 2)
    points_2d = points_2d_flat.reshape(npt, L, -1)
    valid_mask = valid_mask_flat.reshape(npt, L)

    # Extract positions and masks at time t
    points_2d_t = points_2d[:, -1, :]  # Shape: (npt, 2)
    valid_mask_t = valid_mask[:, -1]
    visibilities_t = visibilities[:, -1]

    # Compute displacements to previous frames
    deltas = (
        points_2d[:, :-1, :] - points_2d_t[:, None, :]
    )  # Shape: (npt, L-1, 2)
    distances = np.linalg.norm(deltas, axis=2)  # Shape: (npt, L-1)
    # print("distances: ", distances.shape)
    # Validity mask
    valid = (
        valid_mask[:, :-1]
        & valid_mask_t[:, None]
        & visibilities[:, :-1]
        & visibilities_t[:, None]
    )
    # Apply validity mask
    distances[~valid] = 0
    # Compute maximum displacement
    max_displacement = np.max(distances, axis=1)  # Shape: (npt,)
    displacement[:, t] = max_displacement
  return displacement



def flow_to_depth(flow: np.ndarray, hfov_deg, baseline) -> np.ndarray:
  """Calculates depth map from the flow field and camera metadata.

  assumes cx2 - cx1 = 0, valid disparity should be positive

  Args:
      flow: The optical flow field (numpy array).
      hfov_deg: Horizontal field of view (degree).
      baseline: The baseline value in meters (float).

  Returns:
      The calculated depth map (numpy array).
  """
  # disp = np.abs(flow[..., 0])
  disp = np.clip(flow[..., 0], 0, None)
  imh, imw = disp.shape
  fx = imw / np.tan(np.radians(hfov_deg / 2)) / 2
  depth = (fx * baseline) / disp
  return depth



def inverse_warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
  """Warps an image based on the provided flow field.

  Args:
      img: The image to warp (numpy array).
      flow: The optical flow field (numpy array).

  Returns:
      The warped image (numpy array).
  """
  h, w = flow.shape[:2]
  flow_new = flow.copy()
  flow_new[:, :, 0] += np.arange(w)
  flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

  res = cv2.remap(
      img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
  )
  return res


def plot_3d_tracks_plt(
    video: np.ndarray,
    track3d: Track3d,
    tracks_leave_trace=16,
    point_size: int = 10,
):
  """Visualize 2D point trajectories.

  The trail shows where the previous 3D point in the current camera frame. This
  visualization decouples camera motion
  """
  num_points, num_frames = track3d.track3d.shape[:2]
  figure_dpi = 64

  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

  point_colors = np.zeros((num_points, 3))
  for i in range(num_points):
    point_colors[i] = np.array(color_map(cmap_norm(i)))[:3]

  disp = []
  for t in range(num_frames):
    frame = video[t].copy()

    # Draw tracks on the frame
    all_points = track3d.track3d  # npt, nframe, xyz
    npt, nframe, _ = all_points.shape
    all_points = einops.rearrange(
        all_points, 'npt nframe xyz->(npt nframe) xyz'
    )

    points_at_frame, valid_mask, _ = track3d.cameras[t].world_2_pix_np(
        all_points,
        track3d.imh,
        track3d.imw,
    )
    points_at_frame = einops.rearrange(
        points_at_frame,
        '(npt nframe) xyz->nframe npt xyz',
        npt=npt,
        nframe=nframe,
    )
    valid_mask = einops.rearrange(
        valid_mask, '(npt nframe)-> npt nframe', npt=npt, nframe=nframe
    )
    valid_mask = valid_mask & track3d.visible_list
    valid_mask = valid_mask.transpose(1, 0)
    line_tracks = points_at_frame[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = valid_mask[max(0, t - tracks_leave_trace) : t + 1]
    fig = plt.figure(
        figsize=(frame.shape[1] / figure_dpi, frame.shape[0] / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w',
    )
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(frame / 255.0)

    for s in range(line_tracks.shape[0] - 1):
      # Collect lines and colors for the track
      visible_line_mask = line_visibles[s] & line_visibles[s + 1] & line_visibles[-1]
      pt1 = line_tracks[s, visible_line_mask]
      pt2 = line_tracks[s + 1, visible_line_mask]
      lines = np.concatenate([pt1, pt2], axis=1)
      lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in lines]
      c = point_colors[visible_line_mask]
      alpha = (s + 1) / (line_tracks.shape[0] - 1)
      c = np.concatenate([c, np.ones_like(c[..., :1]) * alpha], axis=1)
      lc = LineCollection(lines, colors=c, linewidths=1)
      ax.add_collection(lc)
    visibles_mask = valid_mask[t].astype(bool)
    colalpha = point_colors[visibles_mask]
    plt.scatter(
        points_at_frame[t, visibles_mask, 0],
        points_at_frame[t, visibles_mask, 1],
        s=point_size,
        c=colalpha,
    )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]  # pytype: disable=attribute-error
    disp.append(np.copy(img))
    plt.close(fig)
    del fig, ax

  disp = np.stack(disp, axis=0)
  return disp


def load_dataset_npz(path):
  """
  Load released npz format
  """
  with open(path, 'rb') as f:
    data_zip = np.load(f)
    data = {}
    for k in data_zip.keys():
      data[k] = data_zip[k]
  # --------------
  # Camera intrinsics
  # --------------
  data['meta_fov'] = {
    'start_yaw_in_degrees': data['fov_bounds'][0],
    'end_yaw_in_degrees': data['fov_bounds'][1],
    'start_tilt_in_degrees': data['fov_bounds'][2],
    'end_tilt_in_degrees': data['fov_bounds'][3],
  }
  data.pop('fov_bounds')
  # --------------
  # Camera poses
  # --------------
  c2w = data['camera2world']  # (T, 3, 4)
  R = c2w[:, :, :3]
  t = c2w[:, :, 3:]

  # Compute inverse: R^T and new translation
  R_inv = np.transpose(R, (0, 2, 1))  # Transpose R
  t_inv = -np.matmul(R_inv, t)
  data['extrs_rectified'] = np.concatenate([R_inv, t_inv], axis=-1)
  data.pop('camera2world')
  # --------------
  # 3D tracks
  # --------------
  lengths = data['track_lengths']
  shape = (len(lengths), len(data['timestamps']), 3)
  tracks = np.full(shape, np.nan)
  tracks[
    np.repeat(np.arange(lengths.shape[0]), lengths),
    data['track_indices'], :
  ] = data['track_coordinates']
  data['track3d'] = tracks
  data.pop('track_lengths')
  data.pop('track_indices')
  data.pop('track_coordinates')
  return data


def compute_per_track_motion_scores(
    track3d: Track3d,
    tracks_leave_trace: int = 16,
) -> np.ndarray:
  """Compute per-track per-frame motion scores.

  Uses 2D displacement in the current camera frame between the last
  tracks_leave_trace frames (max over window) to decouple camera motion.

  Returns:
    motion_scores: array of shape (npt, nframe)
  """
  motion_scores = get_scene_motion_2d_displacement(
      track3d, tracks_leave_trace=tracks_leave_trace
  )
  return motion_scores


def plot_3d_tracks_plt_with_motion(
    video: np.ndarray,
    track3d: Track3d,
    motion_scores: Optional[np.ndarray] = None,
    tracks_leave_trace: int = 16,
    point_size: int = 10,
    text_fontsize: int = 8,
):
  """Visualize 2D point trajectories colored by motion score with colorbar.

  If motion_scores is not provided, it is computed using
  compute_per_track_motion_scores(track3d, tracks_leave_trace).

  Args:
    video: Frames as (nframe, H, W, 3) uint8
    track3d: Track3d instance (subset already applied if needed)
    motion_scores: Optional (npt, nframe) per-frame scores
    tracks_leave_trace: Tail length for trails and motion computation
    point_size: Scatter size for points
    text_fontsize: Unused (kept for API compatibility)

  Returns:
    disp: (nframe, H, W, 3) uint8
  """
  if motion_scores is None:
    motion_scores = compute_per_track_motion_scores(
        track3d, tracks_leave_trace=tracks_leave_trace
    )

  num_points, num_frames = track3d.track3d.shape[:2]
  figure_dpi = 64

  # Normalize scores globally for consistent coloring across frames
  scores_flat = motion_scores.reshape(-1)
  finite_mask = np.isfinite(scores_flat)
  if np.any(finite_mask):
    vmin = np.percentile(scores_flat[finite_mask], 5)
    vmax = np.percentile(scores_flat[finite_mask], 95)
  else:
    vmin, vmax = 0.0, 1.0
  if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    vmin, vmax = 0.0, max(1.0, float(vmax))

  cmap = matplotlib.colormaps.get_cmap('turbo')
  norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

  # Reserve a narrow strip for colorbar on the right
  colorbar_px = 48

  disp = []
  for t in range(num_frames):
    frame = video[t].copy()

    all_points = track3d.track3d
    npt, nframe, _ = all_points.shape
    all_points = einops.rearrange(
        all_points, 'npt nframe xyz->(npt nframe) xyz'
    )

    points_at_frame, valid_mask, _ = track3d.cameras[t].world_2_pix_np(
        all_points,
        track3d.imh,
        track3d.imw,
    )
    points_at_frame = einops.rearrange(
        points_at_frame,
        '(npt nframe) xyz->nframe npt xyz',
        npt=npt,
        nframe=nframe,
    )
    valid_mask = einops.rearrange(
        valid_mask, '(npt nframe)-> npt nframe', npt=npt, nframe=nframe
    )
    valid_mask = valid_mask & track3d.visible_list
    valid_mask = valid_mask.transpose(1, 0)
    line_tracks = points_at_frame[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = valid_mask[max(0, t - tracks_leave_trace) : t + 1]

    fig = plt.figure(
        figsize=((frame.shape[1] + colorbar_px) / figure_dpi, frame.shape[0] / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w',
    )
    # Two-column layout: image+tracks | colorbar
    gs = fig.add_gridspec(1, 2, width_ratios=[frame.shape[1], colorbar_px], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    # Don't turn off cax axis - we need it for colorbar ticks
    ax.imshow(frame / 255.0)

    # Draw trails in light grayscale with fading alpha
    for s in range(line_tracks.shape[0] - 1):
      visible_line_mask = line_visibles[s] & line_visibles[s + 1] & line_visibles[-1]
      if not np.any(visible_line_mask):
        continue
      pt1 = line_tracks[s, visible_line_mask]
      pt2 = line_tracks[s + 1, visible_line_mask]
      lines = np.concatenate([pt1, pt2], axis=1)
      lines = [[(x1, y1), (x2, y2)] for x1, y1, x2, y2 in lines]
      alpha = (s + 1) / max(1, (line_tracks.shape[0] - 1))
      rgb = np.ones((len(lines), 3)) * 0.9
      rgba = np.concatenate([rgb, np.ones((len(lines), 1)) * (alpha * 0.6)], axis=1)
      lc = LineCollection(lines, colors=rgba, linewidths=1)
      ax.add_collection(lc)

    # Scatter current points colored by current motion score (unnormalized values shown via colorbar)
    visibles_mask = valid_mask[t].astype(bool)
    xs = points_at_frame[t, visibles_mask, 0]
    ys = points_at_frame[t, visibles_mask, 1]
    scores_t = motion_scores[visibles_mask, t]
    colors = cmap(norm(scores_t)) if xs.size > 0 else None
    if xs.size > 0:
      ax.scatter(xs, ys, s=point_size, c=colors)

    # Colorbar reflecting unnormalized motion scores
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Motion score', rotation=270, labelpad=15, fontsize=10)
    # Label vmin, vmax and annotate 20
    ticks = [vmin, vmax]
    if vmin < 20 < vmax:
      ticks = [vmin, 20, vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks], fontsize=9)
    # Move ticks to the right side of the colorbar (away from the image)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    # Hide colorbar axis spines but keep ticks visible
    for spine in cb.ax.spines.values():
      spine.set_visible(False)
    if not (vmin < 20 < vmax):
      y = 0.0 if 20 < vmin else 1.0
      va = 'bottom' if y == 0.0 else 'top'
      cb.ax.text(0.5, y, '20', ha='center', va=va, transform=cb.ax.transAxes, fontsize=15)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]  # pytype: disable=attribute-error
    disp.append(np.copy(img))
    plt.close(fig)
    del fig, ax, cax

  disp = np.stack(disp, axis=0)
  return disp


def create_motion_colorbar_image(
    motion_scores: np.ndarray,
    output_path: str,
    width_px: int = 100,
    height_px: int = 400,
    dpi: int = 100,
):
  """
  Create a standalone colorbar image for motion scores.
  
  Args:
    motion_scores: Array of motion scores (npt, nframe)
    output_path: Path to save the colorbar image
    width_px: Width of the colorbar image in pixels
    height_px: Height of the colorbar image in pixels
    dpi: DPI for the output image
  """
  # Compute vmin and vmax from motion scores
  scores_flat = motion_scores.reshape(-1)
  finite_mask = np.isfinite(scores_flat)
  if np.any(finite_mask):
    vmin = np.percentile(scores_flat[finite_mask], 5)
    vmax = np.percentile(scores_flat[finite_mask], 95)
  else:
    vmin, vmax = 0.0, 1.0
  if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    vmin, vmax = 0.0, max(1.0, float(vmax))
  
  cmap = matplotlib.colormaps.get_cmap('turbo')
  norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  
  # Create figure
  fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
  ax = fig.add_axes([0.2, 0.05, 0.3, 0.9])  # [left, bottom, width, height]
  
  # Create colorbar
  sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  cb = plt.colorbar(sm, cax=ax)
  cb.set_label('Motion Score', rotation=90, labelpad=15, fontsize=14)
  
  # Set ticks
  ticks = [vmin, vmax]
  if vmin < 20 < vmax:
    ticks = [vmin, 20, vmax]
  cb.set_ticks(ticks)
  cb.set_ticklabels([f"{t:.1f}" for t in ticks], fontsize=12)
  
  # Add annotation for 20 if it's not in ticks
  if not (vmin < 20 < vmax):
    y = 0.0 if 20 < vmin else 1.0
    va = 'bottom' if y == 0.0 else 'top'
    cb.ax.text(1.5, y, '20', ha='left', va=va, transform=cb.ax.transAxes, fontsize=12)
  
  plt.savefig(output_path, bbox_inches='tight', dpi=dpi, facecolor='white')
  plt.close(fig)
  print(f"Colorbar saved to {output_path}")


