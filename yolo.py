from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

print("Hello World")
import cv2 as cv
import numpy as np
import glob
import pickle


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################


# changing the chessboardSize (9,6)
chessboardSize = (9, 6)
frameSize = (640, 480)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)

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

print(cameraMatrix)
print(dist)
print(rvecs)
print(tvecs)


# Load camera calibration parameters
camera_matrix = np.load("camera_matrix.npy")  # Intrinsic parameters
dist_coeffs = np.load("distortion_coefficients.npy")
R = np.load("rotation_matrix.npy")  # Extrinsic parameters
T = np.load("translation_vector.npy")

# Load an image and depth map (or estimate depth)
image = cv.imread("1.jpg")
depth_map = estimate_depth(image)

# Project 2D image points to 3D space
points_2d = np.array([[x, y]], dtype=np.float32)
points_3d = cv.projectPoints(points_2d, R, T, camera_matrix, dist_coeffs)

x_3d, y_3d, z_3d = points_3d[0][0]

print(f"3D Coordinates: ({x_3d}, {y_3d}, {z_3d})")

# cap = cv2.VideoCapture("MatchSequence.mp4")
# #cap = cv2.imread('frame_1.jpg')
# while cap.isOpened():


#      success,frame = cap.read()
#      if success:

#         results = model(frame,save=True)
#         keypoints = results

#         # Print pose keypoints to the terminal
#         for i, keypoint in enumerate(keypoints):
#             x, y, conf = keypoint[0], keypoint[1], keypoint[2]
#             print(f"Keypoint {i}: x={x}, y={y}, confidence={conf}")

#         annotated_frame = results[0].plot()
#         cv2.imshow("frame",annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#      else:
#          break

# cap.release()


source = "anim_pose.png"
results = model(source, save=True, imgsz=640, conf=0.2)
for r in results:
    for keypoint_idx, keypoints in enumerate(r.keypoints.xy):
        print(keypoints)
        # cv.putText(source,str(keypoint_idx),(int(keypoints[0]),int(keypoints[1])),cv.FONT_HERSHEY_SIMPLEX)

# model.predict('frame_1.jpg', save=True, imgsz=320, conf=0.5,save_txt = True)
