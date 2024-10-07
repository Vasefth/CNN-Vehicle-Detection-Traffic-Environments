# YOLOv8 Car Object Detection

This project implements car object detection using the YOLOv8 model, leveraging the `ultralytics` library for training and evaluation. The model is trained on a custom dataset of car images and annotated bounding boxes, and the dataset is split into training and validation sets. YOLOv8 is used for its high-speed detection capabilities, making it ideal for real-time applications.

## Features

- **YOLOv8 Model**: Pre-trained YOLOv8 model is fine-tuned for car detection.
- **Dataset Handling**: The dataset consists of images and bounding boxes normalized for YOLOv8 format.
- **Training & Evaluation**: Model is trained for 100 epochs, with evaluation metrics such as precision, recall, and confusion matrices automatically generated.
- **Visualization**: Results of model performance are visualized, including loss curves, precision-recall plots, and more.

## Requirements

- Python 3.x
- Ultralytics YOLO (`pip install ultralytics`)
- NumPy, Pandas, Matplotlib, PyTorch

## Setup

1. **Install Dependencies**:
   ```bash
   pip install ultralytics
   ```

2. **Dataset Preparation**:
   - Place your training images in the specified directory.
   - Ensure your bounding box annotations are in the CSV format and normalized for YOLOv8.
   - The code splits the dataset into training and validation sets automatically.

3. **Run the Code**:
   - Modify the dataset paths in the script to match your file structure.
   - Run the training using the provided YOLOv8 model:
     ```bash
     model.train(data='path_to_your_data.yaml', epochs=100, imgsz=640)
     ```

## Kaggle Reference

This code is adapted and enhanced from a [YOLOv8 car detection notebook on Kaggle](https://www.kaggle.com/code/aruaru0/yolov8-car-object-detection), which provided the initial structure for training and testing the YOLOv8 model. Adjustments were made for dataset handling, model parameters, and additional evaluation metrics.

## Results

- After training, metrics such as precision, recall, and F1-score are displayed and visualized.
- Evaluation plots (precision-recall curves, F1-score, confusion matrix) are automatically saved for review.

## Future Improvements

- Implement real-time object detection on video streams.
- Experiment with other YOLOv8 models (e.g., YOLOv8m or YOLOv8l) for better accuracy.
