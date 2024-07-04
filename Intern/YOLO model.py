# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install necessary packages and YOLOv5
!pip install -U -q PyYAML>=5.3.1
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
%pip install -qr requirements.txt

# Import necessary libraries
import torch
from IPython.display import Image, display  # to display images

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or 'yolov5m', 'yolov5l', 'yolov5x', 'custom'

# Custom data configuration (assumes dataset is already uploaded to Google Drive)
custom_data = """
train: /content/drive/MyDrive/your_dataset/images/train
val: /content/drive/MyDrive/your_dataset/images/val

nc: 1  # number of classes (only cars in this case)
names: ['car']  # class names
"""

with open("custom_data.yaml", "w") as f:
    f.write(custom_data)

# Train the YOLO model
!python train.py --img 640 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt

# Perform inference on a test image
test_image_path = '/content/drive/MyDrive/your_dataset/test.jpg'
results = model(test_image_path)

# Results
results.print()  
results.show()  

# Save results
results.save()  

# Display the result image with bounding boxes
display(Image(filename='runs/detect/exp/test.jpg'))
