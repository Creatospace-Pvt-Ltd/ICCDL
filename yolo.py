from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

print("Hello World")
import cv2

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


source = 'anim_pose.png'
results = model(source,save=True,imgsz=640,conf=0.2)
for r in results:
    for keypoint_idx,keypoints in enumerate(r.keypoints.xy):
        print(keypoints)
        cv2.putText(source,str(keypoint_idx),(int(keypoints[0]),int(keypoints[1])),cv2.FONT_HERSHEY_SIMPLEX)  

#model.predict('frame_1.jpg', save=True, imgsz=320, conf=0.5,save_txt = True)

