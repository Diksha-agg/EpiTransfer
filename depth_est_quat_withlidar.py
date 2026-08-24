# # COLOR CAMERA

# Camera Matrix K:
# [[641.44458008   0.         654.18164062]
#  [  0.         640.63641357 362.02645874]
#  [  0.           0.           1.        ]]

# Distortion Coefficients:
# [-5.77447191e-02  7.11912960e-02 -5.22341143e-05  4.32054076e-04
#  -2.29621753e-02]
# Extrinsics from depth camera to color camera
# Rotation:
# [0.9999973177909851, 0.000735334528144449, -0.0022035259753465652, -0.0007433785940520465, 0.9999930262565613, -0.0036519591230899096, 0.002200825372710824, 0.003653587307780981, 0.9999908804893494]

# Translation (meters):
# [-0.05914340913295746, 6.826478056609631e-06, 0.0004445803933776915]



# DEPTH CAMERA

# Camera Matrix K:
# [[388.66162109   0.         320.1713562 ]
#  [  0.         388.66162109 240.05702209]
#  [  0.           0.           1.        ]]

# Distortion Coefficients:
# [0. 0. 0. 0. 0.]

# camera z axis = -x axis of mocap
# camera x axis = -z axis of mocap
# camera y axis = -y axis of mocap

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import math
import yaml
import argparse

import sys
sys.path.append('/home/diksha/Documents/USL/RAFT/RAFT/core')

sys.path.append('/home/diksha/Documents/USL/RAFT/RAFT')
from raft import RAFT
import torch
from lidar_utils import (
    compare_depth_at_lidar_points,
    load_pcd_file,
    load_transformation_matrix,
    plot_lidar_depth_percent_error,
    project_points_to_image,
    transform_points_lidar_to_camera,
    filter_lidar_points
)

def load_raft_model(model_path, device='cuda'):
    """Load RAFT with map_location so CUDA checkpoints also work on CPU."""
    args = argparse.Namespace(
        model=model_path,
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )
    model = torch.nn.DataParallel(RAFT(args))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.module
    model.to(device)
    model.eval()
    print(f"✅ RAFT model loaded from {model_path}")
    return model

def RotationMatrix(phi, theta, psi):
  # Rotation matrix around the z-axis
  Rz = np.array([[math.cos(psi), -math.sin(psi), 0],
                  [math.sin(psi),  math.cos(psi), 0],
                  [0,              0,             1]])

  # Rotation matrix around the y-axis
  Ry = np.array([[math.cos(theta),  0, math.sin(theta)],
                  [0,                1, 0],
                  [-math.sin(theta), 0, math.cos(theta)]])

  # Rotation matrix around the x-axis
  Rx = np.array([[1, 0,                0],
                  [0, math.cos(phi), -math.sin(phi)],
                  [0, math.sin(phi),  math.cos(phi)]])

  return Rx, Ry, Rz

def calculate_mocap_averages(csv_file_path):
    """
    Calculate average rotation (quaternion) and position from mocap CSV file.
    """
    df = pd.read_csv(csv_file_path, skiprows=6)
    
    # Extract quaternion components (columns 2-5: qx, qy, qz, qw)
    quat_x_avg = df.iloc[:, 2].mean()
    quat_y_avg = df.iloc[:, 3].mean()
    quat_z_avg = df.iloc[:, 4].mean()
    quat_w_avg = df.iloc[:, 5].mean()
    
    # Position averages (columns 6-8)
    pos_x_avg = df.iloc[:, 6].mean()
    pos_y_avg = df.iloc[:, 7].mean()
    pos_z_avg = df.iloc[:, 8].mean()
    
    # Normalize quaternion
    quat = np.array([quat_x_avg, quat_y_avg, quat_z_avg, quat_w_avg])
    quat = quat / np.linalg.norm(quat)
    
    return {
        'quaternion': quat,  # (x, y, z, w)
        'position_avg': np.array([pos_x_avg, pos_y_avg, pos_z_avg])
    }
    
def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    """
    x, y, z, w = quat
    
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),       2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R

def findFundamental(t, R, K):
    """Compute fundamental matrix from translation and rotation"""
    T = np.array([[0, -t[2], t[1]],
                  [t[2], 0, -t[0]],
                  [-t[1], t[0], 0]])
    F = np.linalg.inv(K).T @ T @ R @ np.linalg.inv(K)
    return F

def findLines(Transfer_vec_curCam, Rot_curr, ptsCurr, Transfer_vec_prevCam, Rot_prev, ptsPrev, K):
    """Compute epilines for stereo matching"""
    R_curr = np.eye(3)
    print("Transfer vector current camera:", Transfer_vec_curCam)
    FCurr = findFundamental(Transfer_vec_curCam, R_curr, K)
    linesCurr = cv2.computeCorrespondEpilines(ptsCurr, 1, FCurr)
    linesCurr = linesCurr.reshape(-1, 3)
    print("Transfer vector previous camera:", Transfer_vec_prevCam)
    R_prev = Rot_curr @ Rot_prev.T
    FPrev = findFundamental(Transfer_vec_prevCam, R_prev, K)
    linesPrev = cv2.computeCorrespondEpilines(ptsPrev, 1, FPrev)
    linesPrev = linesPrev.reshape(-1, 3)
    
    return linesCurr, linesPrev

# def findDepth(pixLeft, pixRight, baseline, fx, fy):
#     """Calculate depth from disparity"""
#     if abs(pixLeft[0] - pixRight[0]) > abs(pixLeft[1] - pixRight[1]):
#         f = fx
#         disparity = abs(pixLeft[0] - pixRight[0])
#     else:
#         f = fy
#         disparity = abs(pixLeft[1] - pixRight[1])
    
#     if disparity < 1e-6:
#         return 0
#     depth = baseline * f / disparity
#     return depth

def findDepth(pixLeft, pixRight, baseline, fx, fy):
    pixLeft = np.asarray(pixLeft, dtype=np.float32).reshape(-1)
    pixRight = np.asarray(pixRight, dtype=np.float32).reshape(-1)

    if abs(pixLeft[0] - pixRight[0]) > abs(pixLeft[1] - pixRight[1]):
        f = fx
        disparity = abs(pixLeft[0] - pixRight[0])
    else:
        f = fy
        disparity = abs(pixLeft[1] - pixRight[1])

    if disparity < 1e-6:
        return 0
    return baseline * f / disparity

def _to_raft_tensor(img, device):
    """Convert a grayscale/BGR image to the RGB tensor layout RAFT expects."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

def get_raft_matches_dense(
    imgPrev,
    imgCurr,
    model,
    device,
    grid_step=1,
    fb_threshold=5.0,
    min_flow=0.05,
    max_flow=None,
    iters=32,
):
    """
    Get dense RAFT correspondences with a tunable forward-backward check.

    The RAFT helper in demo.py uses a hard-coded 2 px FB threshold. That is
    often too strict for real scenes with occlusion and larger viewpoint change.
    """
    h, w = imgPrev.shape[:2]
    img1_tensor = _to_raft_tensor(imgPrev, device)
    img2_tensor = _to_raft_tensor(imgCurr, device)

    with torch.no_grad():
        _, flow_fwd_t = model(img1_tensor, img2_tensor, iters=iters, test_mode=True)
        _, flow_bwd_t = model(img2_tensor, img1_tensor, iters=iters, test_mode=True)

    flow_fwd = flow_fwd_t[0].detach().cpu().numpy().transpose(1, 2, 0)
    flow_bwd = flow_bwd_t[0].detach().cpu().numpy().transpose(1, 2, 0)

    y_coords, x_coords = np.mgrid[0:h:grid_step, 0:w:grid_step]
    pts_prev = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).astype(np.float32)
    pts_prev_int = pts_prev.astype(np.int32)

    fwd = flow_fwd[pts_prev_int[:, 1], pts_prev_int[:, 0]]
    pts_curr = pts_prev + fwd
    pts_curr_int = np.round(pts_curr).astype(np.int32)

    in_bounds = (
        (pts_curr_int[:, 0] >= 0)
        & (pts_curr_int[:, 0] < w)
        & (pts_curr_int[:, 1] >= 0)
        & (pts_curr_int[:, 1] < h)
    )

    fb_error = np.full(len(pts_prev), np.inf, dtype=np.float32)
    bwd_sampled = flow_bwd[pts_curr_int[in_bounds, 1], pts_curr_int[in_bounds, 0]]
    fb_error[in_bounds] = np.linalg.norm(fwd[in_bounds] + bwd_sampled, axis=1)

    flow_magnitude = np.linalg.norm(fwd, axis=1)
    flow_valid = flow_magnitude > min_flow
    if max_flow is not None:
        flow_valid &= flow_magnitude < max_flow

    valid = in_bounds & (fb_error < fb_threshold) & flow_valid
    print(
        "RAFT candidate stats: "
        f"grid={len(pts_prev)}, in_bounds={np.sum(in_bounds)}, "
        f"fb<{fb_threshold}px={np.sum(fb_error < fb_threshold)}, "
        f"flow_valid={np.sum(flow_valid)}"
    )
    print(
        "Flow magnitude stats: "
        f"min={flow_magnitude.min():.3f}, "
        f"mean={flow_magnitude.mean():.3f}, "
        f"median={np.median(flow_magnitude):.3f}, "
        f"max={flow_magnitude.max():.3f}"
    )
    print(f"RAFT matches kept before RANSAC: {np.sum(valid)}")

    return pts_prev[valid], pts_curr[valid], fb_error[valid]

def filter_matches_fundamental_ransac(
    pts_prev,
    pts_curr,
    reproj_threshold=3.0,
    confidence=0.999,
    max_iters=10000,
):
    """Filter RAFT correspondences with OpenCV fundamental-matrix RANSAC."""
    if len(pts_prev) < 8:
        print(f"Skipping F RANSAC: only {len(pts_prev)} matches")
        return pts_prev, pts_curr, None, None

    F, inliers = cv2.findFundamentalMat(
        pts_prev.astype(np.float32),
        pts_curr.astype(np.float32),
        cv2.FM_RANSAC,
        ransacReprojThreshold=reproj_threshold,
        confidence=confidence,
        maxIters=max_iters,
    )

    if F is None or inliers is None:
        print("F RANSAC failed; keeping unfiltered RAFT matches")
        return pts_prev, pts_curr, F, None

    inlier_mask = inliers.ravel().astype(bool)
    print(
        f"F RANSAC kept {np.sum(inlier_mask)}/{len(pts_prev)} matches "
        f"(threshold={reproj_threshold}px)"
    )
    return pts_prev[inlier_mask], pts_curr[inlier_mask], F, inlier_mask

def save_match_visualization(
    img_prev,
    img_curr,
    pts_prev,
    pts_curr,
    output_path,
    max_matches=2000,
):
    """Save a side-by-side sampled match plot: previous image left, current right."""
    if len(pts_prev) == 0:
        print("No matches available for visualization")
        return

    rng = np.random.default_rng(0)
    sample_count = min(max_matches, len(pts_prev))
    sample_idx = rng.choice(len(pts_prev), size=sample_count, replace=False)

    prev_bgr = cv2.cvtColor(img_prev, cv2.COLOR_GRAY2BGR)
    curr_bgr = cv2.cvtColor(img_curr, cv2.COLOR_GRAY2BGR)
    h = max(prev_bgr.shape[0], curr_bgr.shape[0])
    w_prev = prev_bgr.shape[1]
    canvas = np.zeros((h, w_prev + curr_bgr.shape[1], 3), dtype=np.uint8)
    canvas[: prev_bgr.shape[0], :w_prev] = prev_bgr
    canvas[: curr_bgr.shape[0], w_prev:] = curr_bgr

    for idx in sample_idx:
        p0 = tuple(np.round(pts_prev[idx]).astype(int))
        p1 = tuple(np.round(pts_curr[idx] + np.array([w_prev, 0])).astype(int))
        color = tuple(int(c) for c in rng.integers(40, 255, size=3))
        cv2.circle(canvas, p0, 1, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 1, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p0, p1, color, 1, lineType=cv2.LINE_AA)

    cv2.imwrite(output_path, canvas)
    print(f"Saved sampled RAFT/F-RANSAC match visualization to {output_path}")

def save_match_coverage_visualization(
    img_prev,
    img_curr,
    pts_prev,
    pts_curr,
    output_path,
    max_points=30000,
):
    """Save matched-point coverage without lines so dense overlap is readable."""
    if len(pts_prev) == 0:
        print("No matches available for coverage visualization")
        return

    rng = np.random.default_rng(1)
    sample_count = min(max_points, len(pts_prev))
    sample_idx = rng.choice(len(pts_prev), size=sample_count, replace=False)

    prev_bgr = cv2.cvtColor(img_prev, cv2.COLOR_GRAY2BGR)
    curr_bgr = cv2.cvtColor(img_curr, cv2.COLOR_GRAY2BGR)

    for p in np.round(pts_prev[sample_idx]).astype(int):
        cv2.circle(prev_bgr, tuple(p), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    for p in np.round(pts_curr[sample_idx]).astype(int):
        cv2.circle(curr_bgr, tuple(p), 1, (255, 255, 0), -1, lineType=cv2.LINE_AA)

    h = max(prev_bgr.shape[0], curr_bgr.shape[0])
    canvas = np.zeros((h, prev_bgr.shape[1] + curr_bgr.shape[1], 3), dtype=np.uint8)
    canvas[: prev_bgr.shape[0], : prev_bgr.shape[1]] = prev_bgr
    canvas[: curr_bgr.shape[0], prev_bgr.shape[1]:] = curr_bgr

    cv2.imwrite(output_path, canvas)
    print(f"Saved matched-point coverage visualization to {output_path}")




def draw_epilines_and_intersection(img, lines1, lines2, pts):
    print(img.shape)
    r, c = img.shape
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    intersections = []
    intersection_indices = []

    for i, (r1, r2, pt) in enumerate(zip(lines1, lines2, pts)):
        # Draw lines for visual confirmation
        color1 = (0, 0, 255)  # Red for t1
        color2 = (0, 255, 0)  # Green for t2
        # print(r1, r2)

        if np.abs(r1[1])<=1e-6:
            x0_1, y0_1 = map(int, [-r1[2] / r1[0], 0])
            x1_1, y1_1 = map(int, [-(r1[2] + r1[1] * c) / r1[0], c]) 
        else:
            x0_1, y0_1 = map(int, [0, -r1[2] / r1[1]])
            x1_1, y1_1 = map(int, [c, -(r1[2] + r1[0] * c) / r1[1]])

        if np.abs(r2[1])<=1e-6:
            x0_2, y0_2 = map(int, [-r2[2] / r2[0], 0])
            x1_2, y1_2 = map(int, [-(r2[2] + r2[1] * c) / r2[0], c])
        else:
            x0_2, y0_2 = map(int, [0, -r2[2] / r2[1]])
            x1_2, y1_2 = map(int, [c, -(r2[2] + r2[0] * c) / r2[1]])

        img_color = cv2.line(img_color, (x0_1, y0_1), (x1_1, y1_1), color1, 3)
        img_color = cv2.line(img_color, (x0_2, y0_2), (x1_2, y1_2), color2, 3)

        # Find intersection of the lines
        a1, b1, c1 = r1
        a2, b2, c2 = r2

        # Solve the linear equations
        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([-c1, -c2])

        try:
            x, y = np.linalg.solve(A, B)
            intersections.append((x, y))
            intersection_indices.append(i)
            img_color = cv2.circle(img_color, (int(x), int(y)), 15, (255, 0, 0), -1)  # Blue circle for intersection
            # plt.figure(figsize=(10, 10))
            # plt.imshow(img_color)
            # plt.show()
            
        except np.linalg.LinAlgError:
            # Lines are parallel and do not intersect
            pass

    return img_color, np.array(intersections), intersection_indices


# def draw_epilines_and_intersection(img, linesCurr, linesPrev, ptsCurr):
#     """Draw epilines and find intersections"""
#     h, w = img.shape[:2]
#     img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     intersections = []
#     intersection_indices = []
    
#     for i, (line_curr, line_prev, pt_curr) in enumerate(zip(linesCurr, linesPrev, ptsCurr)):
#         # Find intersection
#         A = np.array([[line_curr[0], line_curr[1]],
#                       [line_prev[0], line_prev[1]]])
#         b = np.array([-line_curr[2], -line_prev[2]])
        
#         try:
#             intersection = np.linalg.solve(A, b)
            
#             # Check if intersection is within image bounds
#             # if 0 <= intersection[0] < w and 0 <= intersection[1] < h:
#             intersections.append(intersection)
#             intersection_indices.append(i)
                
#             # Draw point
#             cv2.circle(img_color, tuple(intersection.astype(int)), 2, (0, 255, 0), -1)
#         except np.linalg.LinAlgError:
#             continue
    
#     return img_color, np.array(intersections), intersection_indices

def remove_padded_region_points(mkpts_prev, mkpts_curr, 
                                 h_crop, w_crop):
    """Remove matches where either point falls in the padded border"""
    valid = (
        (mkpts_prev[:, 0] < w_crop) & (mkpts_prev[:, 1] < h_crop) &
        (mkpts_curr[:, 0] < w_crop) & (mkpts_curr[:, 1] < h_crop)
    )
    return mkpts_prev[valid], mkpts_curr[valid]

def quaternion_to_euler(quat):
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians.
    Uses ZYX convention (yaw-pitch-roll).
    """
    x, y, z, w = quat
    
    # Roll (rotation around x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (rotation around y-axis)
    sinp = 2 * (w * y - z * x)
    # Clamp to avoid numerical errors with asin
    sinp = np.clip(sinp, -1, 1)
    pitch = math.asin(sinp)
    
    # Yaw (rotation around z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def main():

    # Load RAFT model
    raft_model_path = "/home/diksha/Documents/USL/RAFT/RAFT/models/raft-things.pth" # changed to small for faster inference

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raft_model = load_raft_model(raft_model_path, device=device)
    print("✅ RAFT model loaded")

    start_time = time.time()

    K = np.array([[641.44458008, 0., 654.18164062],
                [0., 640.63641357, 362.02645874],
                [0., 0., 1.]])    
    D = np.array([-5.77447191e-02, 7.11912960e-02, -5.22341143e-05, 4.32054076e-04, -2.29621753e-02])

    ## load data
    img_dir = "/home/diksha/Documents/USL/real-sense/captured_images/images/"
    base_dir = "/home/diksha/Documents/USL/real-sense"
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    current_dir = os.path.join(img_dir,"take13")
    current_color_dir = os.path.join(current_dir, "color")
    # current_depth_dir = os.path.join(current_dir, "depth")
    current_csv_file = os.path.join(current_dir,"take13_q.csv")


    prev_dir = os.path.join(img_dir,"take12")
    prev_color_dir = os.path.join(prev_dir, "color")
    # prev_depth_dir = os.path.join(prev_dir, "depth")
    prev_csv_file = os.path.join(prev_dir,"take12_q.csv")

    results_curr = calculate_mocap_averages(current_csv_file)
    quat_curr = results_curr['quaternion']  # (x, y, z, w)
    print("Current Average Quaternion (x, y, z, w):", quat_curr)
    pos_x_curr,pos_y_curr,pos_z_curr = results_curr['position_avg']

    results_prev = calculate_mocap_averages(prev_csv_file)
    quat_prev = results_prev['quaternion']  # (x, y, z, w)
    print("Previous Average Quaternion (x, y, z, w):", quat_prev)
    pos_x_prev,pos_y_prev,pos_z_prev = results_prev['position_avg']
    from scipy.spatial.transform import Rotation as R
    # Convert quaternions to rotation matrices
    # rotation from object to world frame
    Rot_curr_obj2world = R.from_quat(quat_curr).as_matrix()  
    Rot_prev_obj2world = R.from_quat(quat_prev).as_matrix()
    # print("Current Rotation Matrix:\n", Rot_curr_obj2world)
    
    Rot_cam2object = np.array([[0, 0, -1],
                              [0, -1, 0],
                              [-1, 0, 0]])
    # rotation from camera to world frame
    Rot_curr = Rot_curr_obj2world @ Rot_cam2object 
    Rot_prev = Rot_prev_obj2world @ Rot_cam2object 

    print("Current Rotation Matrix:\n", Rot_curr)
    print("Previous Rotation Matrix:\n", Rot_prev)
    # Convert positions to camera coordinate system
    t_curr_w = np.array([pos_x_curr,pos_y_curr,pos_z_curr])  # Convert mm to meters
    t_prev_w = np.array([pos_x_prev,pos_y_prev,pos_z_prev])  # Convert mm to meters


    t_curr_prev_w = t_curr_w - t_prev_w
    print("Translation from previous to current camera (world frame):", t_curr_prev_w)
    # Virtual stereo baseline (in meters)
    baseline = 2

    # Virtual stereo camera position in world coordinates (shifted along the camera's y-axis)
    t_virtual_stereo_world = baseline*(Rot_curr[:,1])
    print("Virtual stereo camera translation (world frame):", t_virtual_stereo_world)
    # Virtual stereo camera position in current camera's coordinate system
    t_virtual_stereo_cam = Rot_curr.T @ t_virtual_stereo_world
    print("Virtual stereo camera translation (current camera frame):", t_virtual_stereo_cam)
    print("Translation from previous to current camera (current camera frame):", Rot_curr.T @ t_curr_prev_w)
    final_tran = t_virtual_stereo_world + t_curr_prev_w
    print("Final translation vector (world frame):",final_tran)
    Transfer_vec_prev_virtual_stereo = Rot_curr.T @ final_tran
    print("Final translation vector (current camera frame):",Transfer_vec_prev_virtual_stereo)
    
    time_load = time.time()
    print(f"⏱️ Time to load images: {time_load - start_time:.2f} seconds")
    start_time_load = time.time()
    # Load images
    selected_image_idx = 0
    prev_images = sorted(f for f in os.listdir(prev_color_dir) if f.endswith('.png'))
    curr_images = sorted(f for f in os.listdir(current_color_dir) if f.endswith('.png'))

    if not prev_images or not curr_images:
        raise ValueError("No PNG images found")

    if selected_image_idx >= len(prev_images) or selected_image_idx >= len(curr_images):
        raise ValueError(
            f"selected_image_idx={selected_image_idx} is out of range: "
            f"{len(prev_images)} previous images, {len(curr_images)} current images"
        )

    prev_image = prev_images[selected_image_idx]
    curr_image = curr_images[selected_image_idx]
    print(f"Using previous image: {prev_image}")
    print(f"Using current image: {curr_image}")
    
    img_path_prev = os.path.join(prev_color_dir, prev_image)
    img_path_curr = os.path.join(current_color_dir, curr_image)

    imgPrev = cv2.imread(img_path_prev, cv2.IMREAD_GRAYSCALE)
    imgCurr = cv2.imread(img_path_curr, cv2.IMREAD_GRAYSCALE)

    if imgPrev is None or imgCurr is None:
        raise ValueError("Could not load images")

    # Undistort images 
    h, w = imgCurr.shape
    raw_image_shape = imgCurr.shape
    K_newn, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    x, y, w_roi, h_roi = roi
    print("new camera matrix K_new:", K_newn)
    # undistort images and crop to valid region
    imgPrev = cv2.undistort(imgPrev, K, D, None, K_newn)
    imgCurr = cv2.undistort(imgCurr, K, D, None, K_newn)
    imgPrev = imgPrev[y:y+h_roi, x:x+w_roi]
    imgCurr = imgCurr[y:y+h_roi, x:x+w_roi]
    print(f"Undistorted images to size: {imgCurr.shape},{imgPrev.shape}")
    K_newn[0, 2] -= x
    K_newn[1, 2] -= y
    print("Adjusted new camera matrix K_new after cropping:", K_newn)
    
    # Resize images for faster RAFT computation
    scale = 0.5  # 50% of original size (adjust 0.25-0.75 based on speed/accuracy tradeoff)
    h_small = int(imgCurr.shape[0] * scale)
    w_small = int(imgCurr.shape[1] * scale)

    imgPrev = cv2.resize(imgPrev, (w_small, h_small))
    imgCurr = cv2.resize(imgCurr, (w_small, h_small))

    print(f"Resized images to {imgCurr.shape}")

    # Adjust camera matrix for resized images
    K_new = K_newn.copy()
    K_new[0, 0] *= scale  # fx
    K_new[1, 1] *= scale  # fy
    K_new[0, 2] *= scale  # cx
    K_new[1, 2] *= scale  # cy

    print("Adjusted camera matrix K_new for resized images:", K_new)
    
    epipole = K_new @ Transfer_vec_prev_virtual_stereo
    print("Epipole in pixel coordinates in virtual cam:", epipole[0]/epipole[2], epipole[1]/epipole[2])
    epipole_prev = K_new @ Rot_prev.T @ final_tran
    print("Epipole in pixel coordinates in prev cam:", epipole_prev[0]/epipole_prev[2], epipole_prev[1]/epipole_prev[2])
    epipole_curr = K_new @ Rot_curr.T @ t_curr_prev_w
    print("Epipole in pixel coordinates in current cam:", epipole_curr[0]/epipole_curr[2], epipole_curr[1]/epipole_curr[2])
    epipole_curr_prev = K_new @ Rot_prev.T @ t_curr_prev_w
    print("Epipole in pixel coordinates of current cam in prev cam:", epipole_curr_prev[0]/epipole_curr_prev[2], epipole_curr_prev[1]/epipole_curr_prev[2])
    
    # Pad images to make dimensions divisible by 8 (RAFT requirement)
    h_crop, w_crop = imgCurr.shape
    pad_h = (8 - h_crop % 8) % 8
    pad_w = (8 - w_crop % 8) % 8

    if pad_h > 0 or pad_w > 0:
        imgCurr = cv2.copyMakeBorder(imgCurr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        imgPrev = cv2.copyMakeBorder(imgPrev, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        print(f"Padded images to size: {imgCurr.shape} (divisible by 8)")
    
    end_time_load = time.time()
    time_load = end_time_load - start_time
    print(f"⏱️ Total loading and preprocessing time: {time_load:.2f} seconds")

    final_image_shape = imgCurr.shape
    # RAFT match tuning. If RANSAC is dropping too many points, first relax
    # ransac_reproj_threshold_px, then fb_threshold_px.
    raft_grid_step_px = 1
    raft_fb_threshold_px = 5.0
    raft_min_flow_px = 0.05
    raft_max_flow_px = None
    use_f_ransac = True
    ransac_reproj_threshold_px = 3.0

    start_time_raft = time.time()
    mkpts_prev, mkpts_curr, fb_error = get_raft_matches_dense(
        imgPrev,
        imgCurr,
        raft_model,
        device=device,
        grid_step=raft_grid_step_px,
        fb_threshold=raft_fb_threshold_px,
        min_flow=raft_min_flow_px,
        max_flow=raft_max_flow_px,
        iters=32,
    )
    if use_f_ransac:
        mkpts_prev, mkpts_curr, F_ransac, ransac_mask = filter_matches_fundamental_ransac(
            mkpts_prev,
            mkpts_curr,
            reproj_threshold=ransac_reproj_threshold_px,
        )

    mkpts_prev, mkpts_curr = remove_padded_region_points(
        mkpts_prev, mkpts_curr, h_crop, w_crop
    )
    end_time_raft = time.time()
    time_raft = end_time_raft - start_time_raft
    print(f"⏱️ RAFT matching time: {time_raft:.2f} seconds")
    print("Final match array shape:", mkpts_prev.shape)  # (N, 2)
    match_plot_path = os.path.join(
        output_dir,
        f"raft_matches_{os.path.basename(current_dir)}_{os.path.basename(prev_dir)}.png",
    )
    save_match_visualization(
        imgPrev,
        imgCurr,
        mkpts_prev,
        mkpts_curr,
        match_plot_path,
    )
    coverage_plot_path = os.path.join(
        output_dir,
        f"raft_match_coverage_{os.path.basename(current_dir)}_{os.path.basename(prev_dir)}.png",
    )
    save_match_coverage_visualization(
        imgPrev,
        imgCurr,
        mkpts_prev,
        mkpts_curr,
        coverage_plot_path,
    )
    # mkpts_prev = np.array([[329.8,181.8],[493.1,55.2]]).reshape(-1, 1, 2)
    # mkpts_curr = np.array([[311.1,213.2],[498.9,50]]).reshape(-1, 1, 2)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(imgPrev, cmap='gray')
    ax[0].set_title('Previous Image')
    ax[1].imshow(imgCurr, cmap='gray')
    ax[1].set_title('Current Image')
    # plt.show()
    start_time_lines = time.time()
    linesCurr, linesPrev = findLines(
        t_virtual_stereo_cam, Rot_curr.T, mkpts_curr,
        Transfer_vec_prev_virtual_stereo, Rot_prev.T, mkpts_prev, K_new
    )
    
    new_image = np.zeros_like(imgCurr)
    img_color, intersections, intersection_indices = draw_epilines_and_intersection(
        new_image, linesCurr, linesPrev, mkpts_curr
    )
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_color)
    # plt.title('Image with Epilines')

    # plt.show()
    depth_map = np.zeros_like(imgCurr, dtype=np.float32)
    fx, fy = K_new[0, 0], K_new[1, 1]
    # print(intersection_indices)
    for idx, i in enumerate(intersection_indices):
        # print(i)
        pixLeft = np.asarray(mkpts_curr[i], dtype=np.float32).reshape(-1)
        pixRight = np.asarray(intersections[idx], dtype=np.float32).reshape(-1)
        # print(f"Left pixel: {pixLeft}, Right pixel: {pixRight}")
        depth = findDepth(pixLeft, pixRight, baseline, fx, fy)
        depth_map[int(pixLeft[1]), int(pixLeft[0])] = depth
        # print(f"Depth at point {i}: {depth}")
        # pixLeft = mkpts_curr[i]#.squeeze()
        # pixRight = intersections[idx]
        # depth = findDepth(pixLeft, pixRight, baseline, fx, fy)
        # # print(depth)
        # depth_map[int(pixLeft[1]), int(pixLeft[0])] = depth
    positive_depth_count = np.count_nonzero(depth_map > 0)
    # display_depth_count = np.count_nonzero((depth_map > 0) & (depth_map < 10))
    # print(
    #     f"Depth samples written: {positive_depth_count}; "
    #     f"displayed with 0<depth<10 m: {display_depth_count}"
    # )

    # Lidar data
    lidar_base_dir = "/home/diksha/Documents/USL/real-sense/captured_images/lidar"
    lidar_take_dir = os.path.join(lidar_base_dir, os.path.basename(current_dir))
    lidar_pcd_path = os.path.join(lidar_take_dir, "cloud.pcd")
    lidar_calib_path = "/home/diksha/Documents/USL/real-sense/lidar_cam_calib_result.json"
    if os.path.exists(lidar_pcd_path) and os.path.exists(lidar_calib_path):
        R_cam_lidar, t_cam_lidar = load_transformation_matrix(lidar_calib_path)
        lidar_points, _ = load_pcd_file(lidar_pcd_path)
        # R_extra = cv2.Rodrigues(np.array([-0.1, 0, 0]))[0]
        # R_cam_lidar_corrected = R_cam_lidar @ R_extra
        lidar_cam_points = transform_points_lidar_to_camera(
            lidar_points,
            R_cam_lidar,
            t_cam_lidar,
        )
        
        # Project into the undistorted full image, then apply the same ROI crop
        # and resize used by the estimated depth map.
        lidar_img_points_full, lidar_depths_full, _ = project_points_to_image(
            lidar_cam_points,
            K_new,
            final_image_shape,
            min_depth=0.1,
            max_depth=100.0,
            D=None,
        )

        lidar_img_points_crop = lidar_img_points_full #- np.array([x, y], dtype=np.float32)
        in_crop = (
            (lidar_img_points_crop[:, 0] >= 0)
            & (lidar_img_points_crop[:, 0] < w_roi)
            & (lidar_img_points_crop[:, 1] >= 0)
            & (lidar_img_points_crop[:, 1] < h_roi)
        )
        lidar_img_points_est = lidar_img_points_crop[in_crop]# * scale
        lidar_depths_est = lidar_depths_full[in_crop]

        # Plot projected LIDAR points on the current image
        plt.figure(figsize=(10, 8))
        plt.imshow(imgCurr, cmap="gray")
        scatter = plt.scatter(
            lidar_img_points_est[:, 0],
            lidar_img_points_est[:, 1],
            c=lidar_depths_est,
            cmap="jet",
            s=12,
            alpha=0.75,
            edgecolors="none",
        )
        plt.colorbar(scatter, label="Lidar depth (m)")
        plt.title("Projected LIDAR points on current image")
        plt.tight_layout()
        # plt.show()
        
        valid = (
            (lidar_img_points_est[:, 0] >= 0)
            & (lidar_img_points_est[:, 0] < imgCurr.shape[1])
            & (lidar_img_points_est[:, 1] >= 0)
            & (lidar_img_points_est[:, 1] < imgCurr.shape[0])
            & np.isfinite(lidar_depths_est)
        )

        lidar_img_points_est = lidar_img_points_est[valid]
        lidar_depths_est = lidar_depths_est[valid]

        

        lidar_img_points_est, lidar_depths_est = filter_lidar_points(
            imgCurr,
            depth_map,
            lidar_img_points_est,
            lidar_depths_est,
            edge_dilate=10,
            var_threshold=0.5,
            window=5,
        )
        # Plot projected LIDAR points on the current image
        plt.figure(figsize=(10, 8))
        plt.imshow(imgCurr, cmap="gray")
        scatter = plt.scatter(
            lidar_img_points_est[:, 0],
            lidar_img_points_est[:, 1],
            c=lidar_depths_est,
            cmap="jet",
            s=12,
            alpha=0.75,
            edgecolors="none",
        )
        plt.colorbar(scatter, label="Lidar depth (m)")
        plt.title("Projected LIDAR points on current image")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"lidar_points_{os.path.basename(current_dir)}_{os.path.basename(prev_dir)}.png",
            ),
            dpi=200,
            bbox_inches="tight",
        )
        # plt.show()

        comparison = compare_depth_at_lidar_points(
            depth_map,
            lidar_img_points_est,
            lidar_depths_est,
            sample_radius_px=5,
            min_lidar_depth=0.1,
            max_lidar_depth=100.0,
        )
        sampled_pixels = comparison["sampled_pixels"]
        sampled_est_depths = comparison["estimated_depths"]

        plt.figure(figsize=(10, 8))
        plt.imshow(imgCurr, cmap="gray")
        scatter = plt.scatter(
            sampled_pixels[:, 0],
            sampled_pixels[:, 1],
            c=sampled_est_depths,
            cmap="jet",
            s=12,
            alpha=0.75,
            edgecolors="none",
        )
        plt.colorbar(scatter, label="Estimated depth (m)")
        plt.title("Projected estimated points on current image")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"estimated_points_{os.path.basename(current_dir)}_{os.path.basename(prev_dir)}.png",
            ),
            dpi=200,
            bbox_inches="tight",
        )


        percent_error = comparison["percent_error"]
        comparison_plot_path = os.path.join(output_dir, f"lidar_depth_percent_error_{os.path.basename(current_dir)}_{os.path.basename(prev_dir)}.png")
        
        plot_lidar_depth_percent_error(
            imgCurr,
            comparison,
            output_path=comparison_plot_path,
            show=False,
        )

        comparison_csv_path = os.path.join(output_dir, f"lidar_depth_percent_error_{os.path.basename(current_dir)}_{os.path.basename(prev_dir)}.csv")
        pd.DataFrame(
            {
                "lidar_x": comparison["lidar_pixels"][:, 0],
                "lidar_y": comparison["lidar_pixels"][:, 1],
                "sampled_est_x": comparison["sampled_pixels"][:, 0],
                "sampled_est_y": comparison["sampled_pixels"][:, 1],
                "lidar_depth_m": comparison["lidar_depths"],
                "estimated_depth_m": comparison["estimated_depths"],
                "abs_error_m": comparison["abs_error"],
                "percent_error": comparison["percent_error"],
            }
        ).to_csv(comparison_csv_path, index=False)

        print(
            f"Compared {len(percent_error)} estimated depth samples "
            f"against {len(lidar_img_points_est)} projected LIDAR points"
        )
        print(f"Saved LIDAR percent-error plot to {comparison_plot_path}")
        print(f"Saved LIDAR percent-error samples to {comparison_csv_path}")
        if len(percent_error) > 0:
            print(
                "Percent error summary: "
                f"mean={np.mean(percent_error):.2f}%, "
                f"median={np.median(percent_error):.2f}%, "
                f"max={np.max(percent_error):.2f}%"
            )
    else:
        print(
            "Skipping LIDAR comparison because required files were not found: "
            f"{lidar_pcd_path} or {lidar_calib_path}"
        )

    end_time_lines = time.time()
    time_lines = end_time_lines - start_time_lines
    print(f"⏱️ Line finding time: {time_lines:.2f} seconds")

    nonzero_coords = np.argwhere((depth_map > 0) & (depth_map < 100))
    x_vals = nonzero_coords[:, 1]
    y_vals = nonzero_coords[:, 0]
    depth_vals = depth_map[y_vals, x_vals]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n⏱️ Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imgCurr, cmap='gray')
    scatter = plt.scatter(x_vals, y_vals, c=depth_vals, cmap='jet', alpha=0.7, s=5)
    plt.colorbar(scatter, label='Depth (m)')
    plt.title('Depth Map')
    plt.subplot(1, 2, 2)
    plt.imshow(imgPrev, cmap='gray')
    plt.title('Previous Image')
    plt.savefig(os.path.join(output_dir, f"depth_map_{os.path.basename(current_dir)}_{os.path.basename(prev_dir)}.png"), dpi=100, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
