from ultralytics import YOLO
import cv2 as cv
import numpy as np
import glob
import pickle

# Load a model
model = YOLO("yolov8n-pose.pt")  # Load your YOLO model

# Load camera calibration parameters
cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
dist = pickle.load(open("dist.pkl", "rb"))


def calculate_3d_coordinates(keypoints_2d):
    chessboardSize = (9, 6)
    frameSize = (2560, 1440)
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(
        -1, 2
    )

    size_of_chessboard_squares_mm = 20
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob("*.jpg")
    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        # if corners == True:
        #     print("corner is True")
        # else:
        #     print("corners is False")
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(1000)

    cv.destroyAllWindows()

    ############## CALIBRATION #######################################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, frameSize, None, None
    )

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"))
    pickle.dump(cameraMatrix, open("cameraMatrix.pkl", "wb"))
    pickle.dump(dist, open("dist.pkl", "wb"))

    print("Camera_Matrix", cameraMatrix)
    print("Distortion", dist)
    print("Rotational", rvecs[0])
    print("Translational", tvecs[0])

    points_2d = np.array([keypoints_2d], dtype=np.float32)
    points_3d = cv.projectPoints(
        points_2d, rvecs[0], tvecs[0], cameraMatrix, dist, None, None, 0
    )
    x_3d, y_3d, z_3d = points_3d[0][0]
    print("points_3d", x_3d, y_3d, z_3d)
    return x_3d, y_3d, z_3d


# Load an image or use YOLO to detect keypoints
source = "1.jpg"  # Load your image

# Use YOLO to detect keypoints
results = model(source, save=True, imgsz=640, conf=0.2)

# Assuming you have detected keypoints in 'results'
# keypoints should be a list of 2D coordinates, e.g., [(x1, y1), (x2, y2), ...]
keypoints = []

for r in results:
    for keypoint in r.keypoints.xy:
        keypoints.append(keypoint)

# Loop through keypoints and calculate 3D coordinates
for i, keypoint in enumerate(keypoints):
    keypoint_cpu = keypoint.cpu()  # Move the tensor from GPU to CPU
    keypoint_np = keypoint_cpu.numpy()  # Convert to NumPy array
    print(f"Keypoint {i}: {keypoint_np}")
    x_3d, y_3d, z_3d = calculate_3d_coordinates(keypoint_np)
    print(f"Keypoint {i}: 3D Coordinates: ({x_3d}, {y_3d}, {z_3d})")
