import cv2
import numpy as np

# Define the size of the calibration pattern (number of internal corners)
pattern_size = (8, 6)  # Change this to match your calibration pattern

# Create arrays to store object points and image points from all calibration images
object_points = []  # 3D points in real-world space
image_points = []   # 2D points in image plane

# Prepare object points: 3D coordinates of the calibration pattern
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Load calibration images (you should capture multiple images of the calibration pattern)
calibration_images = ['MS5.jpeg', 'MS6.jpeg', 'MS8.jpeg']

for image_path in calibration_images:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in the image
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # If corners are found, add object points and image points
        object_points.append(objp)
        image_points.append(corners)

# Perform camera calibration
image_size = gray.shape[::-1]
camera_matrix = np.zeros((3, 3), dtype=np.float64)
dist_coeffs = np.zeros((5, 1), dtype=np.float64)
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(object_points))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(object_points))]

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs)

# Save the obtained calibration parameters to files
np.save("camera_matrix.npy", camera_matrix)
np.save("distortion_coefficients.npy", dist_coeffs)

# Print the results
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
