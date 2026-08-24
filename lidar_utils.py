# lidar_utils.py
"""
LIDAR utility functions for depth map generation from point cloud data.
Handles PCD file loading, coordinate transformation, and image projection.
"""

import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import warnings
import cv2 
import cv2
import numpy as np

def filter_lidar_points(image, depth_map, img_points, depths,
                        edge_dilate=5, var_threshold=0.5, window=5):
    """
    Keep only projected LIDAR points that land on valid estimated depth regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * edge_dilate + 1, 2 * edge_dilate + 1))
    edge_mask = cv2.dilate(edges, kernel)

    h, w = depth_map.shape
    half = window // 2
    keep = []

    for i, (u, v) in enumerate(img_points.astype(np.int32)):
        if u < 0 or v < 0 or u >= w or v >= h:
            continue

        # 1) Reject points on image edges or strong contours
        if edge_mask[v, u] > 0:
            continue

        # 2) Look at local estimated-depth neighborhood
        y0 = max(0, v - half)
        y1 = min(h, v + half + 1)
        x0 = max(0, u - half)
        x1 = min(w, u + half + 1)

        patch = depth_map[y0:y1, x0:x1]
        valid_patch = patch[patch > 0]

        # 3) Reject unstable depth neighborhoods
        if len(valid_patch) >= 2 and np.var(valid_patch) > var_threshold:
            continue

        # 4) Reject if there is no valid estimated depth at all
        if len(valid_patch) == 0:
            continue

        keep.append(i)

    return img_points[keep], depths[keep]

def sample_depth_map_at_points(depth_map, img_points, radius_px=3):
    """
    Sample a sparse estimated depth map at projected LIDAR pixel locations.

    If the exact pixel is empty, the nearest non-zero depth within radius_px is
    used. This is useful when the estimated map is sparse and does not land on
    exactly the same pixels as the projected LIDAR points.
    """
    depth_map = np.asarray(depth_map, dtype=np.float32)
    img_points = np.asarray(img_points, dtype=np.float32)

    h, w = depth_map.shape[:2]
    estimated_depths = np.full(len(img_points), np.nan, dtype=np.float32)
    sampled_pixels = np.full((len(img_points), 2), np.nan, dtype=np.float32)
    radius_px = int(max(radius_px, 0))

    for idx, (x_float, y_float) in enumerate(img_points):
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        if depth_map[y, x] > 0:
            estimated_depths[idx] = depth_map[y, x]
            sampled_pixels[idx] = (x, y)
            continue

        if radius_px == 0:
            continue

        x0 = max(0, x - radius_px)
        x1 = min(w, x + radius_px + 1)
        y0 = max(0, y - radius_px)
        y1 = min(h, y + radius_px + 1)
        window = depth_map[y0:y1, x0:x1]
        valid_y, valid_x = np.nonzero(window > 0)

        if len(valid_x) == 0:
            continue

        valid_x_img = valid_x + x0
        valid_y_img = valid_y + y0
        distances_sq = (valid_x_img - x) ** 2 + (valid_y_img - y) ** 2
        nearest = int(np.argmin(distances_sq))
        estimated_depths[idx] = depth_map[valid_y_img[nearest], valid_x_img[nearest]]
        sampled_pixels[idx] = (valid_x_img[nearest], valid_y_img[nearest])

    return estimated_depths, sampled_pixels


def compare_depth_at_lidar_points(
    estimated_depth_map,
    lidar_img_points,
    lidar_depths,
    sample_radius_px=3,
    min_lidar_depth=0.1,
    max_lidar_depth=100.0,
):
    """
    Compare estimated depth against LIDAR depth at projected LIDAR locations.

    Returns a dictionary of matched pixels, LIDAR depths, estimated depths,
    absolute errors, and percent errors.
    """
    lidar_img_points = np.asarray(lidar_img_points, dtype=np.float32)
    lidar_depths = np.asarray(lidar_depths, dtype=np.float32)

    estimated_depths, sampled_pixels = sample_depth_map_at_points(
        estimated_depth_map,
        lidar_img_points,
        radius_px=sample_radius_px,
    )

    valid = (
        np.isfinite(estimated_depths)
        & (estimated_depths > 0)
        & np.isfinite(lidar_depths)
        & (lidar_depths >= min_lidar_depth)
        & (lidar_depths <= max_lidar_depth)
    )

    lidar_valid = lidar_depths[valid]
    estimated_valid = estimated_depths[valid]
    abs_error = np.abs(estimated_valid - lidar_valid)
    squared_error = (estimated_valid - lidar_valid) ** 2
    percent_error = (abs_error / lidar_valid) * 100.0
    mae = float(np.mean(abs_error))

    # RMSE — root mean squared error, penalizes large errors more
    rmse = float(np.sqrt(np.mean(squared_error)))
    log_diff = np.log(estimated_valid) - np.log(lidar_valid)
    rmse_log = float(np.sqrt(np.mean(log_diff ** 2)))
    abs_rel = float(np.mean(abs_error / lidar_valid))

    # SqRel — squared relative error
    sq_rel = float(np.mean(squared_error / lidar_valid))
    ratio = np.maximum(estimated_valid / lidar_valid, lidar_valid / estimated_valid)
    delta1 = float(np.mean(ratio < 1.25))         # δ1
    delta2 = float(np.mean(ratio < 1.25 ** 2))    # δ2
    delta3 = float(np.mean(ratio < 1.25 ** 3))    # δ3


    return {
        "lidar_pixels": lidar_img_points[valid],
        "sampled_pixels": sampled_pixels[valid],
        "lidar_depths": lidar_valid,
        "estimated_depths": estimated_valid,
        "abs_error": abs_error,
        "percent_error": percent_error,

        # Scalar summary metrics (for Excel summary row)
        "mae":               mae,
        "rmse":              rmse,
        "rmse_log":          rmse_log,
        "abs_rel":           abs_rel,
        "sq_rel":            sq_rel,
        "delta1":            delta1,   # higher is better
        "delta2":            delta2,
        "delta3":            delta3,
        "n_points":          int(np.sum(valid)),
    }

def plot_lidar_depth_percent_error(
    image,
    comparison,
    output_path="lidar_depth_percent_error.png",
    show=False,
):
    """Plot percent error at projected LIDAR measurement locations."""
    import matplotlib.pyplot as plt
    import numpy as np  # Ensure numpy is imported

    percent_error = np.asarray(comparison["percent_error"], dtype=np.float32)
    lidar_pixels = np.asarray(comparison["lidar_pixels"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    ax.set_title("Depth percent error at LIDAR points")

    # 1. First, check if we actually have points to look at
    if len(percent_error) > 0:
        # 2. Create a mask to filter for points with error < 20%
        valid_mask = percent_error < 100
        
        filtered_pixels = lidar_pixels[valid_mask]
        filtered_errors = percent_error[valid_mask]

        # 3. Check if any points survived our filtering
        if len(filtered_errors) > 0:
            scatter = ax.scatter(
                filtered_pixels[:, 0],
                filtered_pixels[:, 1],
                c=filtered_errors,
                cmap="magma",
                s=12,
                alpha=0.85,
            )
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Percent error (%)")
        else:
            # All points had an error >= 20%
            ax.text(
                0.5,
                0.5,
                "All matching depth samples have >= 20% error",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 8},
            )
    else:
        # Zero points were provided in the dictionary
        ax.text(
            0.5,
            0.5,
            "No matching estimated depth samples near LIDAR points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 8},
        )

    # 4. Handle saving and showing the plot cleanly
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


        

def load_transformation_matrix(json_path):
    """
    Load transformation matrix from JSON file.
    
    Expected JSON format:
    {
        "rotation": [[...], [...], [...]],  # 3x3 rotation matrix
        "translation": [x, y, z],            # 3x1 translation
        # OR
        "T_cam_lidar": [[...], [...], [...], [0, 0, 0, 1]]  # 4x4 homogeneous transform
    }
    
    Args:
        json_path: Path to JSON file containing transformation matrix
        
    Returns:
        R (3x3): Rotation matrix
        t (3x1): Translation vector
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Try different possible keys in the JSON
    if "T_cam_lidar" in data:
        T = np.array(data["T_cam_lidar"])
        R = T[:3, :3]
        t = T[:3, 3]
    elif "rotation" in data and "translation" in data:
        R = np.array(data["rotation"])
        t = np.array(data["translation"]).reshape(3, 1)
    else:
        raise ValueError("JSON must contain either 'T_cam_lidar' (4x4) or 'rotation'+'translation'")
    
    return R, t

def load_camera_intrinsics(npz_path):
    """
    Load camera intrinsics from NPZ file.
    
    Expected NPZ keys:
    - 'K': Camera matrix (3x3)
    - 'D': Distortion coefficients (optional)
    
    Args:
        npz_path: Path to NPZ file with camera intrinsics
        
    Returns:
        K (3x3): Camera matrix
        D: Distortion coefficients (or None)
    """
    data = np.load(npz_path)
    
    if 'K' in data:
        K = data['K']
    elif 'camera_matrix' in data:
        K = data['camera_matrix']
    else:
        raise ValueError("NPZ file must contain 'K' or 'camera_matrix' (camera matrix)")
    
    D = data.get('D', None)
    if D is None and 'dist_coeff' in data:
        D = data['dist_coeff']
    return K, D

def load_pcd_file(pcd_path):
    """
    Load point cloud from PCD file.
    
    Args:
        pcd_path: Path to PCD file
        
    Returns:
        points (Nx3): XYZ coordinates of points
        colors (Nx3, optional): RGB colors if available
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.uint8)
    else:
        colors = None
    
    return points, colors

def transform_points_lidar_to_camera(lidar_points, R, t):
    """
    Transform LIDAR points from LIDAR frame to camera frame.
    
    p_camera = R @ p_lidar + t
    
    Args:
        lidar_points (Nx3): Points in LIDAR frame
        R (3x3): Rotation matrix (lidar_to_camera)
        t (3x1): Translation vector (lidar_to_camera)
        
    Returns:
        cam_points (Nx3): Points in camera frame
    """
    # Ensure correct shapes
    lidar_points = np.asarray(lidar_points, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32).reshape(3, 1)
    
    # Transform: p_cam = R @ p_lidar + t
    cam_points = (R @ lidar_points.T).T + t.T
    
    return cam_points

def project_points_to_image(cam_points, K, image_shape, min_depth=0.1, max_depth=100, D=None):
    """
    Project 3D camera points to 2D image plane.
    
    Args:
        cam_points (Nx3): Points in camera frame
        K (3x3): Camera intrinsic matrix
        image_shape (H, W): Image dimensions
        min_depth: Minimum depth threshold (points closer are filtered)
        max_depth: Maximum depth threshold (points farther are filtered)
        D: Distortion coefficients for the target image, or None if image is already undistorted
        
    Returns:
        img_points (Mx2): 2D pixel coordinates (valid points only)
        depths (M,): Depth values for valid points
        valid_indices (M,): Indices of valid points in original array
    """
    h, w = image_shape[:2]
    
    # Filter points in front of camera (z > 0)
    valid_z = cam_points[:, 2] > 0
    cam_points_front = cam_points[valid_z]
    indices_front = np.where(valid_z)[0]
    
    if len(cam_points_front) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=int),
        )
    
    if D is None:
        # Project to image plane: p_2d = K @ p_3d
        proj = K @ cam_points_front.T
        proj_2d = (proj[:2, :] / proj[2, :]).T
    else:
        import cv2

        proj_2d, _ = cv2.projectPoints(
            cam_points_front.astype(np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            K,
            np.asarray(D, dtype=np.float32).reshape(-1),
        )
        proj_2d = proj_2d.reshape(-1, 2)
    
    # Filter by image bounds
    in_bounds = (proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < w) & \
                (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < h)
    
    # Filter by depth range
    depths = cam_points_front[:, 2]
    depth_valid = (depths >= min_depth) & (depths <= max_depth)
    
    valid = in_bounds & depth_valid
    valid_indices = indices_front[valid]
    
    img_points = proj_2d[valid].astype(np.float32)
    valid_depths = depths[valid].astype(np.float32)
    
    return img_points, valid_depths, valid_indices

def create_sparse_depth_map(image_shape, img_points, depths, fill_gaps=False):
    """
    Create sparse depth map from projected points.
    
    Args:
        image_shape (H, W): Image dimensions
        img_points (Mx2): Pixel coordinates
        depths (M,): Depth values
        fill_gaps: If True, apply simple hole-filling (optional)
        
    Returns:
        depth_map (H, W): Sparse depth map (0 where no point projects)
    """
    h, w = image_shape[:2]
    depth_map = np.zeros((h, w), dtype=np.float32)
    
    # Round to nearest pixel
    x = np.round(img_points[:, 0]).astype(int)
    y = np.round(img_points[:, 1]).astype(int)

    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = x[in_bounds]
    y = y[in_bounds]
    depths = depths[in_bounds]
    
    # Handle multiple points projecting to same pixel (keep closest)
    unique_pixels = {}
    for xi, yi, d in zip(x, y, depths):
        key = (xi, yi)
        if key not in unique_pixels or d < unique_pixels[key]:
            unique_pixels[key] = d
    
    # Fill depth map
    for (xi, yi), d in unique_pixels.items():
        depth_map[yi, xi] = d
    
    # Optional: Fill small gaps using morphological operations
    if fill_gaps:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        depth_map = cv2.dilate(depth_map, kernel, iterations=1)
    
    return depth_map

def generate_lidar_depth_overlay(
    lidar_data_dir, 
    image_shape,
    K,
    R_cam_lidar,
    t_cam_lidar,
    D=None,
    pcd_filename=None,
    min_depth=0.1,
    max_depth=100,
    fill_gaps=False
):
    """
    Main function: Load LIDAR data and generate depth map overlay.
    
    Args:
        lidar_data_dir: Directory containing PCD files
        image_shape: (H, W) of the image
        K (3x3): Camera intrinsic matrix
        R_cam_lidar (3x3): Rotation matrix from LIDAR to camera
        t_cam_lidar (3x1): Translation vector from LIDAR to camera
        D: Distortion coefficients for the target image, or None if image is already undistorted
        pcd_filename: Specific PCD file to load (None = first file in directory)
        min_depth: Minimum depth threshold
        max_depth: Maximum depth threshold
        fill_gaps: Whether to fill holes in sparse depth map
        
    Returns:
        depth_map (H, W): Sparse depth map
        cam_points (Nx3): Transformed camera points
        img_points (Mx2): Projected image points
        depths (M,): Depth values
    """
    # Find PCD file
    if pcd_filename is None:
        pcd_files = sorted(Path(lidar_data_dir).glob("*.pcd"))
        if not pcd_files:
            raise FileNotFoundError(f"No PCD files found in {lidar_data_dir}")
        pcd_path = pcd_files[0]
        print(f"Using first PCD file: {pcd_path.name}")
    else:
        pcd_path = os.path.join(lidar_data_dir, pcd_filename)
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"PCD file not found: {pcd_path}")
    
    # Load point cloud
    print(f"Loading point cloud from {pcd_path}")
    lidar_points, colors = load_pcd_file(pcd_path)
    print(f"Loaded {len(lidar_points)} points from LIDAR")
    
    # Transform to camera frame
    cam_points = transform_points_lidar_to_camera(lidar_points, R_cam_lidar, t_cam_lidar)
    print(f"Transformed points to camera frame")
    
    # Project to image plane
    img_points, depths, valid_indices = project_points_to_image(
        cam_points, K, image_shape, min_depth, max_depth, D=D
    )
    print(f"Projected {len(img_points)} points to image plane")
    
    # Create sparse depth map
    depth_map = create_sparse_depth_map(image_shape, img_points, depths, fill_gaps)
    
    return depth_map, cam_points, img_points, depths

def get_lidar_data_paths(base_dir, take_number):
    """
    Helper function to get standard LIDAR data paths.
    
    Expected structure:
    base_dir/
        take{N}/
            *.pcd files
            transformation_matrices/ (optional)
            camera_intrinsics/ (optional)
    
    Args:
        base_dir: Base directory containing take folders
        take_number: Take number (1, 2, etc.)
        
    Returns:
        dict with paths to pcd_dir, transformation_file, intrinsics_file
    """
    take_dir = os.path.join(base_dir, f"take{take_number}")
    
    return {
        'pcd_dir': take_dir,
        'transformation_file': os.path.join(take_dir, 'T_cam_lidar.json'),
        'intrinsics_file': os.path.join(take_dir, 'camera_intrinsics.npz'),
    }
