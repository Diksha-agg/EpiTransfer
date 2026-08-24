import cv2
import numpy as np
from pathlib import Path




K = np.array([[641.44458008, 0., 654.18164062],
                    [0., 640.63641357, 362.02645874],
                    [0., 0., 1.]])    
D = np.array([-5.77447191e-02, 7.11912960e-02, -5.22341143e-05, 4.32054076e-04, -2.29621753e-02])




for i in range(1,18):
    take = f"take{i}"
    img_dir = f"/home/diksha/Documents/USL/real-sense/captured_images/images/{take}/color"
    png_files = sorted(Path(img_dir).glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No .png images found in {img_dir}")
    output_dir = Path(f"/home/diksha/Documents/USL/real-sense/captured_images/images/{take}/color_undistorted")
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [png_files[0]]
    img = cv2.imread(str(png_files[0]), cv2.IMREAD_GRAYSCALE)

    h, w = img.shape
    raw_image_shape = img.shape
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    x, y, w_roi, h_roi = roi
    print("new camera matrix K_new:", K_new)
    # undistort images and crop to valid region
    img_undistorted = cv2.undistort(img, K, D, None, K_new)
    img_undistorted = img_undistorted[y:y+h_roi, x:x+w_roi]
    scale = 0.5  # 50% of original size (adjust 0.25-0.75 based on speed/accuracy tradeoff)
    h_small = int(img_undistorted.shape[0] * scale)
    w_small = int(img_undistorted.shape[1] * scale)
    
    img_undistorted_resized = cv2.resize(img_undistorted, (w_small, h_small))

    # save the undistorted image
    output_path = output_dir / png_files[0].name
    cv2.imwrite(str(output_path), img_undistorted_resized)

K_new[0, 2] -= x
K_new[1, 2] -= y
K_new[0, 0] *= scale  # fx
K_new[1, 1] *= scale  # fy
K_new[0, 2] *= scale  # cx
K_new[1, 2] *= scale  # cy

print("Adjusted new camera matrix K_new after cropping and resizing:", K_new)