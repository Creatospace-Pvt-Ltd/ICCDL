from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg') 

# Extract keypoint
result_keypoint = results.keypoints.xyn.cpu().numpy()[0]