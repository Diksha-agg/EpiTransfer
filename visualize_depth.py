import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Path to captured images
base_path = "/home/diksha/Documents/USL/real-sense/captured_images/images/take2"
depth_dir = os.path.join(base_path, "depth/dark")
color_dir = os.path.join(base_path, "color/dark")
MAX_DEPTH_MM = 10000

def normalize_depth(depth_image, max_depth_mm=7000):
    """
    Normalize depth image for visualization.
    Keeps only values <= 7000 mm and converts them to meters.
    """
    depth_copy = depth_image.astype(float)

    # Remove invalid and out-of-range values
    depth_copy[depth_copy <= 0] = np.nan
    depth_copy[depth_copy > max_depth_mm] = np.nan

    depth_m = depth_copy / 1000.0
    valid_depths = depth_m[~np.isnan(depth_m)]

    if len(valid_depths) == 0:
        return np.zeros_like(depth_image, dtype=np.uint8)

    min_depth = np.nanmin(depth_m)
    max_depth = np.nanmax(depth_m)

    normalized = np.zeros_like(depth_image, dtype=np.uint8)
    if max_depth > min_depth:
        normalized = ((depth_m - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    normalized[np.isnan(depth_copy)] = 0
    return normalized

def visualize_depth_map(image_num):
    """
    Visualize a single depth-color pair with various depth visualization methods.
    """
    # Load images
    depth_file = f"depth_{image_num:02d}_20260528_223210_756.png"
    color_file = f"color_{image_num:02d}_20260528_223210_756.png"
    
    # Try to find exact files
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.startswith(f"depth_{image_num:02d}")])
    color_files = sorted([f for f in os.listdir(color_dir) if f.startswith(f"color_{image_num:02d}")])
    
    if not depth_files or not color_files:
        print(f"Image pair {image_num:02d} not found")
        return
    
    depth_path = os.path.join(depth_dir, depth_files[0])
    color_path = os.path.join(color_dir, color_files[0])
    
    # Read images
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # Read as 16-bit
    color = cv2.imread(color_path)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    print(f"Image {image_num:02d}:")
    print(f"  Depth shape: {depth.shape}, dtype: {depth.dtype}, range: [{depth.min()}, {depth.max()}]")
    print(f"  Color shape: {color.shape}, dtype: {color.dtype}")
    
    depth_filtered = depth.astype(float)
    depth_filtered[depth_filtered <= 0] = np.nan
    depth_filtered[depth_filtered > MAX_DEPTH_MM] = np.nan
    depth_m = depth_filtered / 1000.0

    # Normalize depth
    depth_normalized = normalize_depth(depth, MAX_DEPTH_MM)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    # Create visualization
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # axes[0].imshow(color)
    # axes[0].set_title("Color Image")
    # axes[0].axis('off')

    # axes[1].imshow(depth_m, cmap='jet')
    # axes[1].set_title(f"Depth Map (<= {MAX_DEPTH_MM/1000:.1f} m)")
    # axes[1].axis('off')
    # cbar = plt.colorbar(axes[1].images[0], ax=axes[1])
    # cbar.set_label("Distance (m)")

    # axes[2].imshow(depth_colored)
    # axes[2].set_title("Depth Map (Turbo Colormap)")
    # axes[2].axis('off')
    # plt.tight_layout()
    # plt.savefig(f"/home/diksha/Documents/USL/real-sense/depth_viz_{image_num:02d}.png", dpi=100, bbox_inches='tight')
    # plt.show()

    # Create visualization
    # Create a single figure with only the Turbo depth map and colorbar
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.sca(ax)

    valid_depths = depth_m[~np.isnan(depth_m)]
    if len(valid_depths) > 0:
        vmin = float(np.nanmin(depth_m))
        vmax = float(np.nanmax(depth_m))
    else:
        vmin = 0.0
        vmax = 1.0

    im = ax.imshow(depth_m, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title("Depth Map (Turbo Colormap)")
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Distance (m)")

    plt.tight_layout()
    plt.savefig(f"/home/diksha/Documents/USL/real-sense/depth_viz_{image_num:02d}.png", dpi=100, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    print("Depth Map Visualization Script")
    print("=" * 50)
    
    # Visualize first image in detail
    print("\nVisualizing image 01 in detail...")
    visualize_depth_map(1)
    
    
    print("\nVisualization complete!")
    print("Saved images:")
    print("  - depth_viz_01.png (detailed view of first image)")
    print("  - depth_viz_all.png (grid of all depth maps)")
