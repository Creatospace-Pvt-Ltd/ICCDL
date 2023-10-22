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


def get_xyz(camera1_coords, camera1_M, camera1_R, camera1_T, camera2_coords, camera2_M, camera2_R, camera2_T):
    # Get the two key equations from camera1
    
    camera1_u, camera1_v = camera1_coords
    # Put the rotation and translation side by side and then multiply with camera matrix
    camera1_P = camera1_M.dot(np.column_stack((camera1_R,camera1_T)))
    # Get the two linearly independent equation referenced in the notes
    camera1_vect1 = camera1_v*camera1_P[2,:]-camera1_P[1,:]
    camera1_vect2 = camera1_P[0,:] - camera1_u*camera1_P[2,:]
    
    # Get the two key equations from camera2
    camera2_u, camera2_v = camera2_coords
    # Put the rotation and translation side by side and then multiply with camera matrix
    camera2_P = camera2_M.dot(np.column_stack((camera2_R,camera2_T)))
    # Get the two linearly independent equation referenced in the notes
    camera2_vect1 = camera2_v*camera2_P[2,:]-camera2_P[1,:]
    camera2_vect2 = camera2_P[0,:] - camera2_u*camera2_P[2,:]
    
    # Stack the 4 rows to create one 4x3 matrix
    full_matrix = np.row_stack((camera1_vect1, camera1_vect2, camera2_vect1, camera2_vect2))
    # The first three columns make up A and the last column is b
    A = full_matrix[:, :3]
    b = full_matrix[:, 3].reshape((4, 1))
    # Solve overdetermined system. Note b in the wikipedia article is -b here.
    # https://en.wikipedia.org/wiki/Overdetermined_system
    soln = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(-b)
    return soln


def calculate_3d_coordinates(keypoint_mat):
    chessboardSize = (6, 6)
    frameSize = (2560, 1440)
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

    images = glob.glob("*.jpeg")
    print(images)
    for image in images:
        img = cv2.imread(image)
        cv.imshow('nbueffiuejferfefefsa', img)
        cv.waitKey(0)
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

    # printing the matrix
    # print("Camera_Matrix", cameraMatrix)
    # print("Distortion", dist)
    # print("Rotational", rvecs[0])
    # print("Translational", tvecs[0])

    
    

    

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


    
    npoints = len(objectpoint_n)
    
    R_value = []
    T_value = []
    x =[]
    y =[]
    z =[]
    results=[]
    cord = []
    xp1 =[]
    yp1 =[]
    xp2 =[]
    yp2 =[]
    for i in range(0,2):
        retval, rvec_n, tvec_n = cv2.solvePnP(objectpoint_n[i],imgpoints_n[i] , CameraMatrix, dist_n, flags=cv2.SOLVEPNP_ITERATIVE)
        print("npn Fucntion results")
        print(retval,rvec_n,tvec_n)
        T_value.append(tvec_n)

        print("Calculating R matrix")
        R, jac = cv2.Rodrigues(rvec_n)
        print(R)
        R_value.append(R)
    print("Which Keypoint?",keypoint_mat[0][1])
    for i in range(0,17):
        print(get_xyz(keypoint_mat[0][i],CameraMatrix,R_value[0],T_value[0],keypoint_mat[1][i],CameraMatrix,R_value[1],T_value[1]))
        # x.append(get_xyz(keypoint_mat[0][i],CameraMatrix,R_value[0],T_value[0],keypoint_mat[1][i],CameraMatrix,R_value[1],T_value[1])[0][0])
        # y.append(get_xyz(keypoint_mat[0][i],CameraMatrix,R_value[0],T_value[0],keypoint_mat[1][i],CameraMatrix,R_value[1],T_value[1])[1][0])
        # z.append(get_xyz(keypoint_mat[0][i],CameraMatrix,R_value[0],T_value[0],keypoint_mat[1][i],CameraMatrix,R_value[1],T_value[1])[2][0])
        print(keypoint_mat[0][i])
        xp1.append(keypoint_mat[0][i][0])
        xp2.append(keypoint_mat[0][i][1])
        yp1.append(keypoint_mat[1][i][0])
        yp2.append(keypoint_mat[1][i][1])
        cord.append([get_xyz(keypoint_mat[0][i],CameraMatrix,R_value[0],T_value[0],keypoint_mat[1][i],CameraMatrix,R_value[1],T_value[1])[0][0],get_xyz(keypoint_mat[0][i],CameraMatrix,R_value[0],T_value[0],keypoint_mat[1][i],CameraMatrix,R_value[1],T_value[1])[1][0],
                     get_xyz(keypoint_mat[0][i],CameraMatrix,R_value[0],T_value[0],keypoint_mat[1][i],CameraMatrix,R_value[1],T_value[1])[2][0]])
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from mpl_toolkits.mplot3d import Axes3D

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create example data points (replace this with your data)
    cord = np.array(cord)
    # Extract x, y, and z coordinates
    x = cord[:, 0]
    y = cord[:, 1]
    z = cord[:, 2]
    print("Printing")
    print(x)
    print("y",y)
    print("z",z)
    # Plot the points
    ax.scatter(x, y, z, c='b', marker='o')

    
    # Set labels for the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Show the plot
    plt.show()
    
    # return retval, rvec_n, tvec_n

    xp1=np.array(xp1)
    yp1=np.array(yp1)
    xp2=np.array(xp2)
    yp2=np.array(yp2)

    print(xp1, xp2)
    # Define marker size (adjust as needed)
    marker_size = 50  # Set the size you prefer
    # Create a 2D scatter plot
    plt.figure()  # Create a new figure for the 2D plot
    # Create a scatter plot with marker size
    plt.scatter(xp1, yp1, label='Scatter Plot', color='b', marker='o', s=marker_size)

   
   

    # Add labels and a title for the 2D plot
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Scatter Plot 1')
    # plt.scatter(xp2, yp2, label='Scatter Plot', color='b', marker='o')
    plt.xlim(0, 2560)  # Set x-axis limits
    plt.ylim(0, 1440)  # Set y-axis limits
    # # Add labels and a title
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Scatter Plot 2')
    # # Add a legend
    # plt.legend()

    # Show the plot
    plt.show()

    plt.figure()
    plt.scatter(xp2, yp2, label='2D Scatter Plot', color='r', marker='o', s=marker_size)
    # Add labels and a title for the 2D plot
    plt.xlim(0, 2560)  # Set x-axis limits
    plt.ylim(0, 1440)  # Set y-axis limits
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Scatter Plot 2')
    plt.show()

# execution starts
# Load an image or use YOLO to detect keypoints
source = glob.glob("*.jpeg")
#source = "2.jpg"  # Load your image

# Use YOLO to detect keypoints
for img in source:
    results = model(source, save=True, imgsz=2560, conf=0.2)

# Assuming you have detected keypoints in 'results'
# keypoints should be a list of 2D coordinates, e.g., [(x1, y1), (x2, y2), ...]
keypoints = []

for r in results:
    for keypoint in r.keypoints.xy:
        keypoints.append(keypoint)

keypoint_mat = []

for i, keypoint in enumerate(keypoints):
    keypoint_cpu = keypoint.cpu()  # Move the tensor from GPU to CPU
    keypoint_np = keypoint_cpu.numpy()  # Convert to NumPy array
    print(keypoint_np)
    keypoint_mat.append(keypoint_np) 

calculate_3d_coordinates(keypoint_mat)
# print("Rotational_matrix (from calibration)")
# print(rvec,tra)
# cv2.solvePnP will give you a vector for the rotation matrix,
# so you need to convert this to a 3x3 matrix using the following line:


# Loop through keypoints and calculate 3D coordinates

#     print(f"Keypoint {i}: {keypoint_np}")
    

