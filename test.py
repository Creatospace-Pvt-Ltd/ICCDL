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


import cv2
import numpy as np
import pickle

def calculate_3d_coordinates(points_2d):
    chessboardSize = (6, 6)
    frameSize = (500, 300)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    size_of_chessboard_squares_mm = 20
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane.

    images = glob.glob("*.jpg")
    for image in images:
        img = cv2.imread(image)
        img = cv2.resize(img, frameSize)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('chl ja', gray)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            print("Detecting?")
            # Draw and display the corners
            img_with_corners = cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv2.imshow("img", img_with_corners)
            cv2.waitKey(1000)

    cv2.destroyAllWindows()
    objectpoint_n = np.array(objpoints)
    print(objectpoint_n.shape[0])
    imgpoints_n = np.array(imgpoints)
    print(imgpoints_n.shape[0])

    ############## CALIBRATION #######################################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frameSize, None, None
    )

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    with open("calibration.pkl", "wb") as f:
        pickle.dump((cameraMatrix, dist), f)

    with open("cameraMatrix.pkl", "wb") as f:
        pickle.dump(cameraMatrix, f)

    with open("dist.pkl", "wb") as f:
        pickle.dump(dist, f)

    print("Camera_Matrix", cameraMatrix)
    print("Distortion", dist)
    print("Rotational", rvecs[0])
    print("Translational", tvecs[0])

    
    

    

    CameraMatrix = np.array(cameraMatrix, dtype=np.float32)
    rvec = np.array(rvecs[0], dtype=np.float32)
    tvec = np.array(tvecs[0], dtype=np.float32)
    dist_n = np.array(dist, dtype=np.float64)


# ... Previous code ...
    

# Double-check the format and correctness of input data
    # assert isinstance(objectpoint_n, np.ndarray), "objectPoints_n should be a NumPy array."
    # assert objectpoint_n.shape[1] == 3, "objectPoints_n should have 3 columns (3D points)."
    # assert isinstance(imgpoints_n, np.ndarray), "points_2d should be a NumPy array."
    
    # assert imgpoints_n[1] == 2, "points_2d should have 2 columns (2D points)."
    
    # # the number of rows should match!
    # assert imgpoints_n[0] == objectpoint_n.shape[0], "The number of 3D and 2D points should match."

    # assert isinstance(CameraMatrix, np.ndarray), "CameraMatrix should be a NumPy array."
    # assert CameraMatrix.shape == (3, 3), "CameraMatrix should be a 3x3 matrix."

    # assert isinstance(dist_n, np.ndarray), "dist_n should be a NumPy array."
    # assert dist_n.shape[1] in (4, 5), "dist_n should be a 1x4 or 1x5 array."



    def check_null_argument(args):
        for arg in args:
            if arg is None:
                return "dikkat hai"
        return "sb theek h"

    args = [points_2d, rvec, tvec, CameraMatrix, dist_n]
    print(check_null_argument(args))

    
    npoints = len(objectpoint_n)

    

    retval, rvec_n, tvec_n = cv2.solvePnP(objectpoint_n[0],imgpoints_n[0] , CameraMatrix, dist_n, flags=cv2.SOLVEPNP_ITERATIVE)

    return retval, rvec_n, tvec_n




# execution starts
# Load an image or use YOLO to detect keypoints
source = glob.glob("*.jpg")
#source = "2.jpg"  # Load your image

# Use YOLO to detect keypoints
for img in source:
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
    calculate_3d_coordinates(keypoint_np)
    retval, rev, tra = calculate_3d_coordinates(keypoint_np)
    print(f"values: ({retval}, {rev}, {tra})")
