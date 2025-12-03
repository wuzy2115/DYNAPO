### **Prompt for Code Agent: Implementing a Hybrid 4D Reconstruction Pipeline with Scale Alignment**

**Project Goal:**
The overall goal is to build a robust 4D reconstruction pipeline for dynamic scenes by combining the strengths of a COLMAP-style Bundle Adjustment (BA) for static scene elements and the Stereo4D pipeline for dynamic object tracking. A critical step will be to ensure metric scale consistency between these two components.

I will provide the `Stereo4D` optimization code scripts later for modifications. Your task in this first step is to implement the *framework* for the two phases and the scale alignment logic.

---

**1. Overview of the Hybrid Pipeline**

Our pipeline will operate in two distinct phases:

*   **Phase 1: Static Scene & Camera Pose Reconstruction (COLMAP-style BA)**
    *   **Input:** Video frames, 2D feature tracks corresponding to *static* scene points, initial (rough) camera intrinsics, and possibly initial camera poses (e.g., from a feed-forward network like AnyCam).
    *   **Output:** Highly accurate, globally consistent **metric-scaled** camera poses for all frames (`T_cw_final_scaled_i`) and a dense 3D point cloud of the **static background** in metric scale (`P_static_final_scaled`).
    *   **Key Concept:** This phase will *not* use the Stereo4D loss components. It will use a traditional Bundle Adjustment formulation (which you will provide/implement separately) that inherently recovers structure up to scale. The metric scale will be enforced *after* the BA.

*   **Phase 2: Dynamic Object Trajectory Optimization (Stereo4D)**
    *   **Input:** The **fixed, scaled camera poses** from Phase 1 (`T_cw_final_scaled_i`), raw stereo images/depth maps, 2D tracks corresponding to *dynamic* objects, and initial depths for these dynamic objects from a monocular/stereo depth estimation model.
    *   **Output:** Smooth, accurate 4D trajectories for each dynamic object.
    *   **Key Concept:** This phase will utilize your existing `Stereo4D` loss components (primarily `L_dynamic`). The camera poses are *fixed inputs*, not variables.

---

**2. Your Task: Implement the Framework for Phase 1 & Scale Alignment**

You need to implement the following Python functions/classes. Assume that the internal workings of the "COLMAP-style BA" are handled by an external (or placeholder) function that returns unscaled results.

#### **A. `run_colmap_style_ba(frames, static_2d_tracks, initial_intrinsics, initial_poses) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]`**

*   **Description:** This function simulates or wraps your COLMAP-style Bundle Adjustment for static features.
*   **Input:**
    *   `frames`: A list of image frames (or paths to them).
    *   `static_2d_tracks`: A dictionary mapping unique static track IDs to lists of `(frame_idx, pixel_u, pixel_v)` tuples.
    *   `initial_intrinsics`: Camera intrinsic parameters (e.g., focal length `f` and principal point `(cx, cy)`).
    *   `initial_poses`: A dictionary mapping frame IDs to initial `4x4` camera-to-world pose matrices (unscaled).
*   **Output:**
    *   `unscaled_camera_poses`: A dictionary mapping frame IDs to optimized `4x4` camera-to-world pose matrices (rotation and translation) from the BA. These poses are **geometrically consistent but in arbitrary scale**.
    *   `unscaled_static_3d_points`: A dictionary mapping unique static track IDs to optimized `(x, y, z)` 3D coordinates in the world frame. These points are also **geometrically consistent but in arbitrary scale**.
*   **Implementation:** For now, you can implement this as a placeholder that returns dummy, unscaled but consistent poses and points (e.g., by scaling a set of known metric poses/points by a random factor). *I will provide the actual BA implementation later if needed.*

#### **B. `align_scale_to_metric(unscaled_camera_poses, unscaled_static_3d_points, metric_depth_model, camera_intrinsics) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], float]`**

*   **Description:** This is the core function for scale alignment. It takes the unscaled BA output and aligns it to a metric scale using a reference from your depth estimation model.
*   **Input:**
    *   `unscaled_camera_poses`: Output from `run_colmap_style_ba`.
    *   `unscaled_static_3d_points`: Output from `run_colmap_style_ba`.
    *   `metric_depth_model`: A callable object/function that, given an `(image, camera_intrinsics)`, returns a metric depth map for that image.
    *   `camera_intrinsics`: The fixed camera intrinsics (focal length `f` and principal point `(cx, cy)`) used during the BA.
*   **Output:**
    *   `scaled_camera_poses`: The camera poses, now in metric scale.
    *   `scaled_static_3d_points`: The static 3D points, now in metric scale.
    *   `scale_factor`: The scalar `s` used for the alignment.
*   **Implementation Steps (as discussed):**
    1.  **Choose a Reference Frame:** Select one representative frame (e.g., `frame_idx = 0`).
    2.  **Estimate Metric Depths from Model:** For a subset of the static 2D tracks visible in this reference frame:
        *   Use `metric_depth_model` to predict a depth map `D_metric` for the reference frame.
        *   For each static 2D point `(u, v)` in the reference frame, extract its depth `d_metric = D_metric[v, u]`.
        *   Unproject this `(u, v, d_metric)` to obtain a 3D point `P_j_metric` in the *camera frame* of the reference image.
        *   Transform `P_j_metric` to the *world frame* using the *unscaled* camera pose `unscaled_camera_poses[frame_idx]` to get a set of `P_metric_world_j` points.
    3.  **Retrieve Corresponding BA Points:** For the same static features, retrieve their unscaled 3D coordinates `P_BA_world_j` from `unscaled_static_3d_points`.
    4.  **Compute Scale Factor `s`:** Calculate the optimal scale factor `s` using the closed-form solution:
        $$
        s = \frac{\sum_j (P_{BA\_world\_j} \cdot P_{metric\_world\_j})}{\sum_j ||P_{BA\_world\_j}||^2}
        $$
        (Ensure `P_BA_world_j` and `P_metric_world_j` are treated as vectors for the dot product and norm calculation).
    5.  **Apply Scale Factor:**
        *   Create `scaled_static_3d_points` by multiplying all points in `unscaled_static_3d_points` by `s`.
        *   Create `scaled_camera_poses` by multiplying the *translation component* of each `4x4` pose matrix in `unscaled_camera_poses` by `s`. The rotation part remains untouched.

#### **C. `reproject_static_point(point_3d_world, camera_pose_4x4, intrinsics) -> Tuple[float, float]`**

*   **Description:** A helper function to reproject a 3D world point into a 2D pixel coordinate in a given camera frame.
*   **Input:**
    *   `point_3d_world`: `(x, y, z)` numpy array.
    *   `camera_pose_4x4`: `4x4` camera-to-world pose matrix.
    *   `intrinsics`: Camera intrinsic parameters (focal length `f` and principal point `(cx, cy)`).
*   **Output:** `(u, v)` pixel coordinates.
*   **Implementation:** Standard camera projection formula.

---

**3. Clarification and Guidance for Coding:**

*   **Numpy/Tensor Operations:** Encourage the use of NumPy for efficient vector/matrix operations. If this is part of a PyTorch/TensorFlow pipeline, equivalent tensor operations should be used.
*   **Coordinate Systems:** Maintain strict adherence to coordinate system definitions:
    *   `camera_pose_4x4` is always Camera-to-World (C-W). To project a world point into a camera, you need to use the World-to-Camera (W-C) inverse.
    *   `unscaled_static_3d_points` are in the World Frame.
*   **Function Signatures:** Adhere to the provided function signatures.
*   **Placeholders:** For `run_colmap_style_ba` and `metric_depth_model`, simple dummy implementations are acceptable initially, as the focus is on the framework and `align_scale_to_metric`.