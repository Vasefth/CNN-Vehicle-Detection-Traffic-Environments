# Car Object Detection using YOLOv5

This project demonstrates car object detection using the YOLOv5 architecture. The model is trained on a custom dataset and utilizes Google Drive for data storage and retrieval. The implementation allows for the easy adaptation of the YOLOv5 model for detecting cars in images by splitting the dataset, normalizing bounding boxes, and training with customized settings.

## Features

- **YOLOv5 Model**: Utilizes YOLOv5n (nano version) for training and car detection.
- **Dataset Handling**: The dataset is loaded from Google Drive, split into training and validation sets, and bounding boxes are normalized for YOLOv5 format.
- **Training & Evaluation**: The model is trained for 100 epochs with visual results, including bounding boxes and metrics, automatically saved.
- **Result Visualization**: After training, the model is used to predict on test images, and the results are displayed, showing the detected cars with bounding boxes.

## Requirements

- Python 3.x
- Ultralytics YOLO
- Google Colab (optional, but required for mounting Google Drive)
- NumPy, Pandas, Matplotlib, OpenCV

## Setup

1. **Install Dependencies**:
   ```bash
   pip install ultralytics
   ```

2. **Mount Google Drive**:
   The code mounts your Google Drive to load the training and testing data:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Dataset Preparation**:
   - Images are stored in the `/content/drive/My Drive/working/datasets/cars/images/` directory.
   - Labels are stored in the `/content/drive/My Drive/working/datasets/cars/labels/` directory.
   - The dataset is automatically split into training and validation sets using `train_test_split()`.

4. **Training**:
   The model is trained using YOLOv5n:
   ```python
   model = YOLO('yolov5n.pt')
   model.train(data="dataset.yaml", epochs=100, batch=8)
   ```

5. **Inference**:
   After training, the model is used to make predictions on test images:
   ```python
   model = YOLO('./runs/detect/train/weights/last.pt')
   ret = model("/content/drive/My Drive/input/car-object-detection/data/testing_images", save=True, conf=0.2, iou=0.5)
   ```

## Result Visualization

Predicted images with bounding boxes are saved in the `./runs/detect/predict/` directory and displayed using the following:
```python
files = glob.glob("./runs/detect/predict/*")
for i in range(0, 30, 1):
    img = Image(files[i])
    display(img)
```

## Acknowledgements

This implementation was inspired by various open-source projects, including:

- [YOLOv8 Car Object Detection](https://www.kaggle.com/code/aruaru0/yolov8-car-object-detection#setting-parameters)
- [YOLOv9 Car Object Detection](https://www.kaggle.com/code/aruaru0/yolov9-car-object-detection#setup-yaml-file)
- [YOLOv5 Car Object Detection by Balraj98](https://www.kaggle.com/code/balraj98/yolo-v5-car-object-detection)
- [YOLOv5 Car Detection by Vexxingbanana](https://www.kaggle.com/code/vexxingbanana/yolov5-car-object-detection#Visualize-Predictions)

While these sources provided inspiration, this version of the code has been customized to utilize Google Drive for dataset loading and storage, along with adjustments to the training process. The code is based on open-source implementations, and further modifications have been made to fit the specific project requirements.

## Future Work

- Extend the model to detect multiple objects beyond cars.
- Optimize the model for real-time detection on video streams.
- Experiment with different YOLOv5 variants (e.g., YOLOv5m, YOLOv5l) to improve accuracy.
