# mask-leval evaluation
import sys
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import math
import numpy as np
import cv2
from glob import glob
from PIL import Image
import os
from scipy.optimize import linear_sum_assignment
import pandas
import re
from imageio import get_writer

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j

def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res

def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F

def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def evaluate_unsupervised(all_gt_masks, all_res_masks, metric, all_void_masks=None):
    j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
    f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
    for ii in range(all_gt_masks.shape[0]):
        for jj in range(all_res_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
    if 'J' in metric and 'F' in metric:
        all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
    else:
        all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
    row_ind, col_ind = linear_sum_assignment(-all_metrics)
    return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]

def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def load_mask(path):
    image = cv2.imread(path)

    contains_black = np.any(np.all(image == [0, 0, 0], axis=2))
    
    if contains_black:
        binary_mask = np.any(image > 0, axis=2)
    else:
        white_threshold = 250
        mask = np.all(image > white_threshold, axis=2)
        binary_mask = ~mask

    binary_mask = binary_mask.astype(np.uint8)
    
    return binary_mask

def read_masks_fbms(mask_dir, indices=None):
    # Get all potential mask file paths in the directory
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png"))) + \
                 sorted(glob(os.path.join(mask_dir, "*.jpg"))) + \
                 sorted(glob(os.path.join(mask_dir, "*.jpeg"))) + \
                 sorted(glob(os.path.join(mask_dir, "*.bmp")))

    # Convert to zero-based indexing for direct access
    if indices is not None:
        selected_paths = [mask_paths[i] for i in indices if 0 <= i < len(mask_paths)]
    else:
        selected_paths = mask_paths

    mask_list = []
    for path in selected_paths:
        mask_img = load_mask(path)
        mask_img = (mask_img > 0).astype(np.uint8)
        mask_list.append(mask_img)

    # Stack mask list along a new axis if there are masks, otherwise create an empty array
    dynamic_mask = np.stack(mask_list, axis=0) if mask_list else None

    return dynamic_mask

def read_masks(mask_dir, exp_masks=None):
    # Check if there are subfolders containing per-object annotations (e.g., SegTrackv2)
    subdirs = [os.path.join(mask_dir, d) for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))]

    def _collect_image_paths(root):
        return (
            sorted(glob(os.path.join(root, "*.png")))
            + sorted(glob(os.path.join(root, "*.jpg")))
            + sorted(glob(os.path.join(root, "*.jpeg")))
            + sorted(glob(os.path.join(root, "*.bmp")))
        )

    candidate_subdirs = []
    for sd in subdirs:
        sd_paths = _collect_image_paths(sd)
        if len(sd_paths) > 0:
            candidate_subdirs.append((sd, sd_paths))

    if len(candidate_subdirs) > 0:
        # Merge per-object masks by frame name (union across objects)
        frame_to_masks = {}
        for _, sd_paths in candidate_subdirs:
            for path in sd_paths:
                fname_no_ext = os.path.splitext(os.path.basename(path))[0]
                mask_img, _ = load_ann_png(path)
                if mask_img.ndim == 3:
                    mask_img = mask_img[..., 0]
                mask_bin = (mask_img > 0).astype(np.uint8)
                if fname_no_ext not in frame_to_masks:
                    frame_to_masks[fname_no_ext] = []
                frame_to_masks[fname_no_ext].append(mask_bin)

        def _frame_sort_key(name):
            # name is basename without extension; try to extract trailing/frame index
            m = re.search(r'_(\d+)', name)
            if m:
                return int(m.group(1))
            m2 = re.search(r'(\d+)', name)
            return int(m2.group(1)) if m2 else name

        sorted_names = sorted(frame_to_masks.keys(), key=_frame_sort_key)
        merged_list = []
        for name in sorted_names:
            masks = frame_to_masks[name]
            base_h, base_w = masks[0].shape[:2]
            aligned = []
            for m in masks:
                if m.shape[0] != base_h or m.shape[1] != base_w:
                    m_resized = cv2.resize(m.astype(np.uint8), (base_w, base_h), interpolation=cv2.INTER_NEAREST)
                else:
                    m_resized = m
                aligned.append(m_resized > 0)
            union_mask = np.any(np.stack(aligned, axis=0), axis=0).astype(np.uint8)
            merged_list.append(union_mask)

        if len(merged_list) == 0:
            return np.zeros_like(exp_masks, dtype=np.uint8)
        return np.stack(merged_list, axis=0)

    # Fallback: read masks directly from the root directory (single-object or already merged)
    mask_paths = _collect_image_paths(mask_dir)
    mask_list = []
    for path in mask_paths:
        mask_img, _ = load_ann_png(path)
        mask_img = (mask_img > 0).astype(np.uint8)
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        mask_list.append(mask_img)
    if not mask_list:
        dynamic_mask = np.zeros_like(exp_masks, dtype=np.uint8)
    else:
        dynamic_mask = np.stack(mask_list, axis=0)
    return dynamic_mask

def _find_video_file(mask_dir):
    preferred_basenames = ["segmentation_mask", "mask", "masks"]
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"]

    # Prefer known mask filenames first
    for base in preferred_basenames:
        for ext in video_exts:
            cand = os.path.join(mask_dir, f"{base}{ext}")
            # import sys;sys.stdin = open("/dev/tty");import pdb;pdb.set_trace()
            if os.path.isfile(cand):
                return cand

    # Fallback: any video file in the directory
    for ext in video_exts:
        cands = sorted(glob(os.path.join(mask_dir, f"*{ext}")))
        if len(cands) > 0:
            # import sys;sys.stdin = open("/dev/tty");import pdb;pdb.set_trace()
            return cands[0]
    return None

def _read_masks_from_video(video_path, indices=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(total_frames)) if indices is None else [i for i in indices if 0 <= i < total_frames]

    mask_list = []
    current_index = 0
    target_ptr = 0
    next_target = frame_indices[target_ptr] if len(frame_indices) > 0 else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if next_target is None:
            break
        if current_index < next_target:
            current_index += 1
            continue

        # Convert frame to binary mask: any non-zero pixel is foreground
        # Expectation: mask video uses 0 (bg) and 255 (fg)
        binary_mask = (np.any(frame > 127, axis=2)).astype(np.uint8)
        mask_list.append(binary_mask)

        target_ptr += 1
        next_target = frame_indices[target_ptr] if target_ptr < len(frame_indices) else None
        current_index += 1

    cap.release()

    if len(mask_list) == 0:
        return None
    return np.stack(mask_list, axis=0)

def read_masks_any(mask_dir, exp_masks=None, indices=None):
    # Try images first
    img_paths = (
        sorted(glob(os.path.join(mask_dir, "*.png")))
        + sorted(glob(os.path.join(mask_dir, "*.jpg")))
        + sorted(glob(os.path.join(mask_dir, "*.jpeg")))
        + sorted(glob(os.path.join(mask_dir, "*.bmp")))
    )
    if len(img_paths) > 0:
        if indices is not None:
            img_paths = [img_paths[i] for i in indices if 0 <= i < len(img_paths)]
        mask_list = []
        for path in img_paths:
            mask_img, _ = load_ann_png(path)
            mask_img = (mask_img > 0).astype(np.uint8)
            if mask_img.ndim == 3:
                mask_img = mask_img[..., 0]
            mask_list.append(mask_img)
        dynamic_mask = np.stack(mask_list, axis=0) if len(mask_list) > 0 else None
        if dynamic_mask is None and exp_masks is not None:
            dynamic_mask = np.zeros_like(exp_masks, dtype=np.uint8)
        return dynamic_mask

    # Try video file
    #import sys;sys.stdin = open("/dev/tty");import pdb;pdb.set_trace()
    video_path = _find_video_file(mask_dir)
    if video_path is not None:
        masks = _read_masks_from_video(video_path, indices=indices)
        if masks is None and exp_masks is not None:
            return np.zeros_like(exp_masks, dtype=np.uint8)
        return masks

    # Fallback to zeros if expected shape given
    if exp_masks is not None:
        return np.zeros_like(exp_masks, dtype=np.uint8)
    return None

def extract_frame_number(filename):
    # Try to match the number after an underscore, if present
    match = re.search(r'_(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # If no underscore number, match the first number in the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def get_matching_pred_indices(pred_dir, gt_dir):
    # Get and sort all pred and gt mask paths
    pred_paths = sorted(glob(os.path.join(pred_dir, "*.png"))) + sorted(glob(os.path.join(pred_dir, "*.jpg")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))

    # Extract frame numbers from pred mask filenames
    pred_indices = [extract_frame_number(os.path.basename(path)) for path in pred_paths]

    # Extract frame numbers from gt mask filenames and find matching indices in pred masks
    matching_pred_indices = []
    for gt_path in gt_paths:
        gt_index = extract_frame_number(os.path.basename(gt_path))
        if gt_index in pred_indices:
            matching_pred_indices.append(pred_indices.index(gt_index))  # Get index in pred list

    return matching_pred_indices

def get_fbms_labeled_indices(seq_dir):
    """Return labeled frame indices from FBMS *Def.dat under seq_dir/GroundTruth."""
    gt_root = os.path.join(seq_dir, 'GroundTruth')
    dat_candidates = sorted(glob(os.path.join(gt_root, '*Def.dat')))
    if len(dat_candidates) == 0:
        return []
    _, _, labeled_frames, _, _ = parse_fbms_definition_file(dat_candidates[0])
    return labeled_frames

def _colorize_masks_overlay(background_shape, gt_mask, pred_mask):
    """Create a color overlay image showing GT and Pred masks.
    - GT: green
    - Pred: red
    - Intersection: yellow
    Ensures both masks are visible via alpha blending over a black background.
    """
    h, w = background_shape[:2]
    gt_u8 = (gt_mask > 0).astype(np.uint8)
    pred_u8 = (pred_mask > 0).astype(np.uint8)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Colors: pred=red, gt=green, intersection=yellow
    # Start with red for pred
    overlay[pred_u8 > 0] = (0, 0, 255)
    # Add green for gt; intersection becomes yellow (0,255,255) after blending logic below
    gt_indices = gt_u8 > 0
    overlay[gt_indices & (pred_u8 == 0)] = (0, 255, 0)
    overlay[gt_indices & (pred_u8 > 0)] = (0, 255, 255)

    # Alpha-blend over black just to ensure consistent appearance
    alpha = 0.8
    result = (overlay.astype(np.float32) * alpha).astype(np.uint8)
    return result

def _put_text_lines(img, lines, origin=(10, 25), line_height=22, color=(255, 255, 255)):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += line_height

def _collect_image_paths_for_rgb(root):
    return (
        sorted(glob(os.path.join(root, "*.png")))
        + sorted(glob(os.path.join(root, "*.jpg")))
        + sorted(glob(os.path.join(root, "*.jpeg")))
        + sorted(glob(os.path.join(root, "*.bmp")))
    )

def _overlay_mask_on_rgb(rgb, mask, color_bgr=(0, 255, 0), alpha=0.5):
    """Overlay a single binary mask on an RGB image with a given color and alpha."""
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    h, w = rgb.shape[:2]
    mask_u8 = (mask > 0).astype(np.uint8)
    result = rgb.astype(np.float32).copy()
    color_arr = np.zeros_like(result)
    color_arr[:, :] = np.array(color_bgr, dtype=np.float32)
    # Blend only at mask locations
    mask_3 = np.repeat(mask_u8[:, :, None], 3, axis=2).astype(bool)
    result[mask_3] = (1.0 - alpha) * result[mask_3] + alpha * color_arr[mask_3]
    return np.clip(result, 0, 255).astype(np.uint8)

def write_overlay_video(seq_name, out_dir, gt_masks, pred_masks, j_per_frame, f_per_frame, fps=10, rgb_dir=None, labeled_indices=None):
    """Write an overlay video for a sequence.
    Inputs:
      - gt_masks, pred_masks: uint8 arrays of shape [T, H, W] with values in {0,1}
      - j_per_frame, f_per_frame: arrays of shape [T]
      - out_dir: directory to save the video file
      - fps: frames per second
      - rgb_dir: directory containing original RGB frames (optional)
      - labeled_indices: optional list[int] of labeled frame indices (0-based) to match RGB filenames
    Output video filename: {seq_name}_overlay.mp4
    """
    os.makedirs(out_dir, exist_ok=True)

    num_frames = min(gt_masks.shape[0], pred_masks.shape[0])
    h, w = gt_masks.shape[1], gt_masks.shape[2]

    # Prepare RGB frames if provided
    img_paths = []
    if rgb_dir is not None and os.path.isdir(rgb_dir):
        img_paths = _collect_image_paths_for_rgb(rgb_dir)
        if len(img_paths) > 0:
            # If labeled indices are given (FBMS), select matching RGB files by frame number
            if labeled_indices is not None and len(labeled_indices) > 0:
                # Build number->path mapping from filenames
                num_to_path = {}
                for p in img_paths:
                    num = extract_frame_number(os.path.basename(p))
                    if num is not None:
                        num_to_path[num] = p
                # Determine offset between labeled indices (0-based) and filename numbering (often 1-based)
                offset = 0
                first_idx = labeled_indices[0]
                if first_idx not in num_to_path and (first_idx + 1) in num_to_path:
                    offset = 1
                # Select paths per labeled index
                selected_paths = []
                for li in labeled_indices[:num_frames]:
                    sel = num_to_path.get(li + offset, None)
                    selected_paths.append(sel)
                img_paths = selected_paths
            # Limit by available frames if not using labeled selection
            num_frames = min(num_frames, len(img_paths)) if (labeled_indices is None) else num_frames

    out_path = os.path.join(out_dir, f"{seq_name}_overlay.mp4")
    writer = get_writer(out_path, format='FFMPEG', fps=fps)

    for t in range(num_frames):
        # Build three panels
        # Read and resize RGB if available; otherwise use black background
        if len(img_paths) > 0:
            rgb_path = img_paths[t] if t < len(img_paths) else None
            rgb = cv2.imread(rgb_path) if (rgb_path is not None) else None
            if rgb is None:
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                if rgb.shape[0] != h or rgb.shape[1] != w:
                    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

        gt_on_rgb = _overlay_mask_on_rgb(rgb, gt_masks[t], color_bgr=(0, 255, 0), alpha=0.5)
        pred_on_rgb = _overlay_mask_on_rgb(rgb, pred_masks[t], color_bgr=(0, 0, 255), alpha=0.5)
        overlay = _colorize_masks_overlay((h, w, 3), gt_masks[t], pred_masks[t])

        j_val = float(j_per_frame[t]) if j_per_frame is not None else None
        f_val = float(f_per_frame[t]) if f_per_frame is not None else None
        text_lines = [f"Seq: {seq_name}", f"Frame: {t+1}/{num_frames}"]
        if j_val is not None:
            text_lines.append(f"J: {j_val:.3f}")
        if f_val is not None:
            text_lines.append(f"F: {f_val:.3f}")
        # Compose 3-panel and write text on right panel area
        composite = np.concatenate([gt_on_rgb, pred_on_rgb, overlay], axis=1)
        _put_text_lines(composite, text_lines, origin=(2 * w + 10, 25))

        if composite.dtype != np.uint8:
            max_val = float(composite.max()) if hasattr(composite, 'max') else 255.0
            if max_val <= 1.0:
                composite = (composite * 255).astype(np.uint8)
            else:
                composite = composite.astype(np.uint8)
        writer.append_data(composite)

    writer.close()
    return out_path

def parse_fbms_definition_file(def_path):
    """Parse FBMS GroundTruth definition file (*Def.dat).
    Returns:
      - num_regions (int)
      - region_color_values (list[int]): color scale values per region index (0..num_regions-1)
      - labeled_frames (list[int]): zero-based frame indices that are labeled
      - gt_filenames (list[str]): corresponding GT PPM filenames (as listed in the .dat)
      - input_filenames (list[str]): original input filenames (not used here)
    """
    num_regions = None
    region_color_values = []
    labeled_frames = []
    gt_filenames = []
    input_filenames = []

    with open(def_path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines()]

    i = 0
    # Find number of regions
    while i < len(lines):
        if lines[i].lower().startswith('total number of regions'):
            i += 1
            num_regions = int(lines[i])
            i += 1
            break
        i += 1

    # Read region scales (color values)
    for ridx in range(num_regions if num_regions is not None else 0):
        # Expect lines like: "Scale of region k:" then the value line
        while i < len(lines) and not lines[i].lower().startswith('scale of region'):
            i += 1
        if i < len(lines) and lines[i].lower().startswith('scale of region'):
            i += 1
            region_color_values.append(int(lines[i]))
            i += 1

    # Skip to frame info section
    # Find "Total number of labeled frames for this shot:" then entries of Frame number/File name/Input file name
    while i < len(lines) and not lines[i].lower().startswith('total number of labeled frames'):
        i += 1
    if i < len(lines) and lines[i].lower().startswith('total number of labeled frames'):
        i += 1
        # labeled_count = int(lines[i]); not strictly needed
        i += 1

    # Parse labeled frames blocks
    while i < len(lines):
        if lines[i].lower().startswith('frame number'):
            i += 1
            frame_idx = int(lines[i]); i += 1
            # File name
            if i < len(lines) and lines[i].lower().startswith('file name'):
                i += 1
                gt_name = lines[i]; i += 1
            else:
                gt_name = None
            # Input file name
            if i < len(lines) and lines[i].lower().startswith('input file name'):
                i += 1
                in_name = lines[i]; i += 1
            else:
                in_name = None
            if gt_name is not None:
                labeled_frames.append(frame_idx)
                gt_filenames.append(gt_name)
                input_filenames.append(in_name)
        else:
            i += 1

    return num_regions, region_color_values, labeled_frames, gt_filenames, input_filenames


def _ppm_to_foreground_mask(ppm_path, fg_region_color_values, bg_region_color_value=None):
    """Load a PPM FBMS GT file and return a binary foreground mask.
    Arguments:
      - ppm_path: path to *_gt.ppm
      - fg_region_color_values: list of 24-bit color integers for foreground regions
      - bg_region_color_value: optional 24-bit color int for background (unused)
    Returns np.uint8 mask with values {0,1}.
    """
    img = Image.open(ppm_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    r = arr[..., 0].astype(np.uint32)
    g = arr[..., 1].astype(np.uint32)
    b = arr[..., 2].astype(np.uint32)
    color_int = r * 65536 + g * 256 + b

    fg_mask = np.zeros(color_int.shape, dtype=bool)
    for cval in fg_region_color_values:
        fg_mask |= (color_int == cval)
    return fg_mask.astype(np.uint8)

def _pgm_to_foreground_mask(pgm_path, fg_region_gray_values, bg_region_gray_value=None):
    """Load a PGM FBMS GT file and return a binary foreground mask.
    Arguments:
      - pgm_path: path to *_gt.pgm
      - fg_region_gray_values: list of grayscale integers (could be up to 65535) for foreground regions
      - bg_region_gray_value: optional grayscale value for background (unused)
    Returns np.uint8 mask with values {0,1}.
    """
    arr = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        # Fallback to PIL if OpenCV failed
        arr = np.array(Image.open(pgm_path))
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr_u32 = arr.astype(np.uint32)
    fg_mask = np.zeros(arr_u32.shape, dtype=bool)
    for gval in fg_region_gray_values:
        fg_mask |= (arr_u32 == gval)
    return fg_mask.astype(np.uint8)


def read_fbms_gt(seq_dir):
    """Read FBMS ground-truth for one sequence directory containing a 'GroundTruth' subdir.
    Returns:
      - gt_masks: np.uint8 array [T_labeled, H, W] with values {0,1}
      - labeled_frames: list[int] zero-based frame indices
    """
    gt_root = os.path.join(seq_dir, 'GroundTruth')
    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"GroundTruth directory not found under: {seq_dir}")
    dat_candidates = sorted(glob(os.path.join(gt_root, '*Def.dat')))
    if len(dat_candidates) == 0:
        raise FileNotFoundError(f"No *Def.dat found in {gt_root}")
    def_path = dat_candidates[0]

    num_regions, region_color_values, labeled_frames, gt_filenames, _ = parse_fbms_definition_file(def_path)
    if num_regions is None or len(region_color_values) == 0:
        raise RuntimeError(f"Failed to parse regions from {def_path}")

    # Determine background regions, with special-cases for FBMS-59 sequences
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    if seq_name == 'marple2':
        bg_indices = {0, 2, 4}
    elif seq_name in ('marple7', 'marple10'):
        bg_indices = {0, 2}
    elif seq_name in {'marple13'}:
        bg_indices = {0, 2}
    else:
        bg_indices = {0}

    # Guard against malformed definitions with fewer regions than expected
    bg_indices = {idx for idx in bg_indices if idx < len(region_color_values)}

    # Representative background color (not used in conversion, but kept for completeness)
    bg_color = region_color_values[0]
    # Foreground colors are all region colors not marked as background
    fg_colors = [c for idx, c in enumerate(region_color_values) if idx not in bg_indices]

    masks = []
    for ppm_name in gt_filenames:
        ppm_path = os.path.join(gt_root, ppm_name)
        if not os.path.isfile(ppm_path):
            raise FileNotFoundError(f"Missing GT file: {ppm_path}")
        ext = os.path.splitext(ppm_name)[1].lower()
        if ext == '.ppm':
            mask = _ppm_to_foreground_mask(ppm_path, fg_colors, bg_color)
        elif ext == '.pgm':
            mask = _pgm_to_foreground_mask(ppm_path, fg_colors, bg_color)
        else:
            raise ValueError(f"Unsupported GT extension '{ext}' in {ppm_name}; expected .ppm or .pgm")
        masks.append(mask)

    if len(masks) == 0:
        return None, labeled_frames
    gt_masks = np.stack(masks, axis=0).astype(np.uint8)
    return gt_masks, labeled_frames


def convert_fbms_gt_to_png(seq_dir, out_dir):
    """Utility to dump FBMS GT as binary PNG masks for convenience.
    Writes one PNG per labeled frame index: {frame_idx:06d}.png.
    """
    masks, labeled_frames = read_fbms_gt(seq_dir)
    if masks is None:
        return []
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for idx, frame_idx in enumerate(labeled_frames):
        mask = (masks[idx] > 0).astype(np.uint8) * 255
        out_path = os.path.join(out_dir, f"{frame_idx:06d}.png")
        Image.fromarray(mask).save(out_path)
        written.append(out_path)
    return written

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train trajectory-based motion segmentation network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--res_dir', type=str,default="current_work_dir/exp_res/sam_res/ablation/no_tracks/initial_preds")
    parser.add_argument('--eval_seq_list', type=str, default=None)
    parser.add_argument('--eval_dir', type=str,default="current_work_dir/baseline/DAVIS/Annotations_unsupervised/480p")
    parser.add_argument('--img_dir', type=str,default="current-data-dir/baseline/davis/Testset")
    parser.add_argument('--visualize', action='store_true', help='Enable video visualization of GT vs Pred masks')
    parser.add_argument('--vis_out_dir', type=str, default="visualizations", help='Directory to save visualization videos')
    parser.add_argument('--vis_fps', type=int, default=10, help='FPS for visualization videos')

    args = parser.parse_args()
    
    # eval_seq_path = "baseline/DAVIS/ImageSets/2016/moving_val.txt"
    eval_seq_path = args.eval_seq_list
    eval_dir = args.eval_dir
    # eval_dir = "baseline/DAVIS/Annotations_unsupervised/480p"
    if args.eval_seq_list is None:
        eval_seq_name = [name for name in os.listdir(args.res_dir) if os.path.isdir(os.path.join(args.res_dir, name))]
    else:
        eval_seq_path = args.eval_seq_list
        with open(eval_seq_path, 'r') as file:
            eval_seq_name = [line.strip() for line in file]
    
    metric=('J', 'F')

    # Containers
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    # eval_seq_name = ['marple2', 'marple7', 'marple10'] # HACK
    for seq in tqdm(eval_seq_name):
        # seq = eval_seq_name[11]
        gt_dir = os.path.join(eval_dir, seq)
        res_dir = os.path.join(args.res_dir, seq)
        
        use_fbms = os.path.isdir(os.path.join(gt_dir, 'GroundTruth')) and ("FBMS" in args.eval_dir)
        if use_fbms:
            # Load FBMS GT from GroundTruth/*.ppm using *Def.dat
            gt_masks, labeled_indices = read_fbms_gt(gt_dir)

            # Prepare prediction indices from labeled frames
            pred_indices = labeled_indices

            # Prefer reading predictions from video if present; fallback to images
            pred_masks = read_masks_any(res_dir, indices=pred_indices)
            if pred_masks is None:
                pred_masks = read_masks_fbms(res_dir, indices=pred_indices)
        else:
            gt_masks = read_masks(gt_dir)
            # Prefer reading predictions from video if present; fallback to images
            pred_masks = read_masks_any(res_dir, exp_masks=gt_masks)
            if pred_masks is None:
                pred_masks = read_masks(res_dir, gt_masks)

        # Skip evaluation if no prediction masks were generated (e.g., empty keyframes JSON)
        if pred_masks is None:
            print(f"Warning: No prediction masks found for sequence '{seq}', skipping evaluation.")
            continue

        # if gt_masks.shape[0] != pred_masks.shape[0]:
        #     gt_masks = gt_masks[:-1]
        min_shape = min(gt_masks.shape[0], pred_masks.shape[0])
        gt_masks = gt_masks[:min_shape]
        pred_masks = pred_masks[:min_shape]
        
        # Ensure spatial alignment (H, W) between prediction and ground truth
        if gt_masks.shape[1] != pred_masks.shape[1] or gt_masks.shape[2] != pred_masks.shape[2]:
            gt_h, gt_w = gt_masks.shape[1], gt_masks.shape[2]
            resized_pred_list = []
            for t in range(pred_masks.shape[0]):
                frame_u8 = pred_masks[t].astype(np.uint8)
                resized = cv2.resize(frame_u8, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
                resized_pred_list.append((resized > 0).astype(np.uint8))
            pred_masks = np.stack(resized_pred_list, axis=0)
        
        gt_masks = np.expand_dims(gt_masks, axis=0)
        pred_masks = np.expand_dims(pred_masks, axis=0)
        
        j_metrics_res, f_metrics_res = evaluate_unsupervised(gt_masks, pred_masks, metric=metric)
        
        for ii in range(gt_masks.shape[0]):
            seq_name = f'{seq}_{ii+1}'
            if 'J' in metric:
                [JM, JR, JD] = db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
                metrics_res['J']["M_per_object"][seq_name] = JM
            if 'F' in metric:
                [FM, FR, FD] = db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)
                metrics_res['F']["M_per_object"][seq_name] = FM

        # Optional visualization: use per-frame J and F for the single object case
        if args.visualize:
            # Collapse batch dimension for video writing
            gt_seq = gt_masks[0]
            pred_seq = pred_masks[0]
            # If both J and F available, take them; otherwise pass None where missing
            j_seq = j_metrics_res[0] if 'J' in metric else None
            f_seq = f_metrics_res[0] if 'F' in metric else None

            out_dir = os.path.join(args.vis_out_dir, seq)
            rgb_dir = os.path.join(args.img_dir, seq)
            # Pass labeled indices for FBMS so RGB aligns with annotated frames
            vis_labeled = labeled_indices if ('labeled_indices' in locals()) else None
            write_overlay_video(seq, out_dir, gt_seq.astype(np.uint8), pred_seq.astype(np.uint8), j_seq, f_seq, fps=args.vis_fps, rgb_dir=rgb_dir, labeled_indices=vis_labeled)
                
    J, F = metrics_res['J'], metrics_res['F']

    seq_names = list(J['M_per_object'].keys())
    sys.stdout.write("----------------Global results in CSV---------------\n")
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    table_g = pandas.DataFrame(data=np.reshape(g_res, [1, len(g_res)]), columns=g_measures)
    table_g.to_csv(sys.stdout, index=False, float_format="%0.5f")
    save_path_g = os.path.join(args.res_dir, "table_g.csv")
    table_g.to_csv(save_path_g, index=False, float_format="%0.5f")

    sys.stdout.write("\n\n------------Per sequence results in CSV-------------\n")
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pandas.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    table_seq.to_csv(sys.stdout, index=False, float_format="%0.5f")
    save_path_s = os.path.join(args.res_dir, "table_seq.csv")
    table_seq.to_csv(save_path_s, index=False, float_format="%0.5f")
