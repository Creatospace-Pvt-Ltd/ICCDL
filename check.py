from ultralytics import YOLO
import glob
model = YOLO("yolov8n-pose.pt") 

source = glob.glob("*.jpg")
#source = "2.jpg"  # Load your image

# Use YOLO to detect keypoints
for img in source:
    results = model(source, save=True, imgsz=640, conf=0.2)

print(results)