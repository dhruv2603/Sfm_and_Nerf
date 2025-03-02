import cv2
import numpy as np

# Replace these values with your actual intrinsic parameters:
# fx, fy: focal lengths; cx, cy: optical centers.
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# Replace with your distortion coefficients.
# They usually come as: [k1, k2, p1, p2, k3]
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Load your image.
img = cv2.imread("image.jpg")

# Optionally, compute a new optimal camera matrix to minimize unwanted pixels.
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

# Undistort the image using the original or the new camera matrix.
undistorted_img = cv2.undistort(
    img, camera_matrix, dist_coeffs, None, new_camera_matrix
)

# (Optional) Crop the image based on the valid pixels ROI.
x, y, w, h = roi
undistorted_img = undistorted_img[y : y + h, x : x + w]

# Save or display the calibrated image.
cv2.imwrite("undistorted_image.jpg", undistorted_img)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
