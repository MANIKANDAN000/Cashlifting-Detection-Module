Cash-Lifting Detection Module
![python-shield](https://img.shields.io/badge/Python-3.8+-blue.svg)

![pytorch-shield](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

![yolo-shield](https://img.shields.io/badge/YOLO-v8-00FFFF.svg)

![license-shield](https://img.shields.io/badge/License-MIT-green.svg)
An advanced surveillance system designed to automatically detect suspicious cash-handling behavior at a point-of-sale. This module uses state-of-the-art object detection and tracking to identify when a cashier pockets cash instead of placing it in the register, triggering real-time alerts.
![alt text](https://i.imgur.com/example.gif)

Replace with a GIF of your actual project in action.
🚀 Core Features
Real-time Detection: Utilizes the high-speed YOLOv8n model for efficient frame-by-frame analysis, suitable for live camera feeds.
Robust Tracking: Integrates the DeepSORT algorithm to maintain a unique ID for each hand, even through occlusions and rapid movements.
Contextual Rule Engine: Goes beyond simple detection by analyzing the sequence of events—tracking the path of a hand after it interacts with cash.
Customizable Zones: Easily define Regions of Interest (ROIs) for key areas like the cash register and employee pockets to adapt to any camera angle.
Automated Alerting: Automatically logs suspicious events and saves video clips of the incidents for review, creating an audit trail.
⚙️ How It Works: The Detection Pipeline
The system processes video streams through a multi-stage pipeline to identify and flag suspicious activities.
Frame Ingestion: The system captures frames from a video source (e.g., a pre-recorded file or a live RTSP stream) using OpenCV.
Object Detection (YOLOv8n): Each frame is passed to a custom-trained YOLOv8n model. The model identifies and returns bounding boxes for the key classes: hand and cash.
Object Tracking (DeepSORT): The detected bounding boxes are fed into the DeepSORT tracker. DeepSORT assigns a unique and persistent tracking ID to each hand, allowing the system to follow its movement across multiple frames.
State Management & Event Triggering: This is the core logic of the system. For each tracked hand, the system maintains a state:
Step 1: Interaction Detected: The system detects a significant overlap between a tracked hand and a cash bounding box. A holding_cash flag is set to True for that specific hand's ID.
Step 2: Path Monitoring: The system continuously tracks the centroid (center point) of the hand's bounding box.
Step 3: Rule Evaluation:
Suspicious Event: If the hand's center point enters the predefined Pocket Zone while its holding_cash flag is True, an alert is triggered.
Legitimate Event: If the hand's center point enters the Register Zone, the holding_cash flag is reset to False, and the transaction is considered legitimate.
Alerting & Logging: When the rule engine triggers an alert:
Logging: The event is logged to events.log with a timestamp, event type ("Suspicious Pocketing"), and the hand's tracking ID.
Video Clip Generation: A 10-second video clip of the incident (5 seconds before and 5 seconds after the event) is saved for evidentiary review.
🛠️ Technology Stack
Object Detection: YOLOv8n (pre-trained on COCO)
Object Tracking: DeepSORT
Deep Learning Framework: PyTorch
CV & Video Processing: OpenCV
File & System Operations: os, glob, PyYAML
📦 Setup and Installation
Clone the repository:
code
Bash
git clone https://github.com/your-username/cash-lifting-detection.git
cd cash-lifting-detection
Create a virtual environment (recommended):
code
Bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:
code
Bash
pip install -r requirements.txt
(Note: Ensure you have compatible versions of PyTorch, CUDA, and torchvision installed for GPU support.)
Download Model Weights:
Place your trained model weights (best.pt) into the weights/ directory. If the directory doesn't exist, create it.
🎬 Usage
The detection script can be run on a video file or a live RTSP stream. The critical Regions of Interest (ROIs) must be configured in a config.yaml file.
1. Configure your ROIs:
Create a config.yaml file and define the bounding box coordinates for your specific camera setup.
code
Yaml
# config.yaml
# Coordinates are in [x1, y1, x2, y2] format
rois:
  register_zone: [850, 400, 1100, 650]
  pocket_zone: [300, 500, 550, 750]
2. Run the detection script:
On a local video file:
code
Bash
python detect.py --source /path/to/your/video.mp4 --config config.yaml
On a live RTSP stream:
code
Bash
python detect.py --source "rtsp://user:pass@ip_address:port/stream" --config config.yaml
To save the output video:
code
Bash
python detect.py --source /path/to/your/video.mp4 --config config.yaml --save-video
🧠 Training a Custom Model
The included best.pt model was trained on a custom dataset. To train your own model for different environments or lighting conditions, follow these steps:
Data Collection: Gather images or extract frames from videos that represent your target environment.
Annotation: Annotate the images with bounding boxes for the hand and cash classes. Tools like Roboflow are highly recommended for this process, as they can export labels directly in YOLO format.
Data Configuration: Create a data.yaml file that points to your training and validation image sets.
code
Yaml
train: ../data/train/images
val: ../data/val/images

nc: 2
names: ['hand', 'cash']
Start Training: Use the YOLOv8 CLI to start the training process.
code
Bash
yolo train model=yolov8n.pt data=path/to/your/data.yaml epochs=50 imgsz=640 batch=16
The best performing model will be saved as runs/train/exp/weights/best.pt.
Public Dataset Resources
To augment your dataset or pre-train for better hand detection, consider these public datasets:
Roboflow Universe - Cash Handling: Directly relevant, already annotated datasets in YOLO format.
EgoHands Dataset: Excellent for robust hand detection from a first-person perspective.
100 Days of Hands: A large and diverse dataset to help your hand detector generalize well.
📈 Future Improvements
Web UI Dashboard: Develop a Flask/FastAPI-based web interface for live monitoring and reviewing alerts.
Performance Optimization: Convert the model to TensorRT for a significant inference speed-up on NVIDIA GPUs.
Advanced Rule Logic: Implement more complex rules, such as detecting hand-to-hand cash transfers away from the register.
VMS Integration: Integrate the alerting system with existing Video Management Systems (VMS) via API calls.
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
