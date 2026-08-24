# EpiTransfer

Monocular depth estimation from a moving RealSense camera using epipolar geometry, validated against LIDAR ground truth.

Given two RGB frames from known camera poses (position + orientation from motion-capture), EpiTransfer computes dense correspondences with [RAFT](https://github.com/princeton-vl/RAFT) optical flow, filters them with fundamental-matrix RANSAC, derives the epipolar geometry from the relative camera pose, and triangulates per-pixel depth. Estimated depth is then compared against LIDAR points projected into the same frame.

## Pipeline

1. **Undistort** raw RealSense color frames using the calibrated intrinsics (`undiistorted_images.py`).
2. **Estimate depth** between a previous and current frame (`depth_est_quat_withlidar.py`):
   - Load camera poses (quaternion + position) from mocap CSVs and average them per frame.
   - Compute dense RAFT correspondences between the frame pair, with a tunable forward-backward consistency check.
   - Filter matches using fundamental-matrix RANSAC.
   - Build the fundamental matrix from the relative pose and camera intrinsics, compute epilines, and triangulate depth from disparity.
   - Project LIDAR points into the same image and compare against estimated depth (percent error, per-point plots).
3. **Visualize** raw depth maps captured by the sensor for inspection (`visualize_depth.py`).

## Scripts

| Script | Purpose |
|---|---|
| `undiistorted_images.py` | Undistorts and resizes captured color images using the RealSense color camera intrinsics/distortion coefficients. |
| `depth_est_quat_withlidar.py` | Full depth estimation pipeline: RAFT correspondences → RANSAC filtering → epipolar-geometry depth → LIDAR comparison. |
| `visualize_depth.py` | Loads a raw depth/color image pair and renders a colorized (Turbo colormap) depth map. |

## Requirements

- Python 3
- `numpy`, `pandas`, `opencv-python`, `matplotlib`, `pyyaml`, `torch`
- [RAFT](https://github.com/princeton-vl/RAFT) (checked out locally; the path is currently added via `sys.path.append` in `depth_est_quat_withlidar.py` — update this to point at your own RAFT checkout)
- A pretrained RAFT model checkpoint
- Lidar utilities module (`lidar_utils.py`, providing `compare_depth_at_lidar_points`, `load_pcd_file`, `load_transformation_matrix`, `plot_lidar_depth_percent_error`, `project_points_to_image`, `transform_points_lidar_to_camera`, `filter_lidar_points`)

## Usage

Undistort captured frames:

```bash
python undiistorted_images.py
```

Run depth estimation with LIDAR comparison:

```bash
python depth_est_quat_withlidar.py
```

Visualize a raw depth/color pair:

```bash
python visualize_depth.py
```

> Note: file paths (image directories, model checkpoints, RAFT location) are currently hardcoded at the top of each script — update them for your own data layout before running.

## Example output

Depth map with LIDAR overlay:

![Depth map](docs/images/depth_map.png)

Projected LIDAR points on the current frame:

![LIDAR projection](docs/images/lidar_points.png)

RAFT correspondence matches between frame pairs:

![RAFT matches](docs/images/raft_matches.png)

*(Drop your own output images into `docs/images/` using these filenames, or update the paths above.)*
