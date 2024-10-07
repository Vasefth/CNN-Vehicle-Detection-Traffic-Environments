# YOLOv9 Car Object Detection with Google Drive Integration

This project demonstrates car object detection using the YOLOv9 architecture, leveraging Google Drive for dataset management. The code trains the YOLOv9 model to detect vehicles in images, normalizing bounding boxes and splitting the dataset into training and validation sets.

## Features

- **YOLOv9 Model**: The pre-trained YOLOv9 model is fine-tuned for car detection.
- **Google Drive Integration**: Dataset files are loaded, split, and managed directly from Google Drive.
- **Training & Evaluation**: The model is trained for 100 epochs, with results visualized for performance analysis.
- **Custom Dataset Handling**: Bounding boxes are normalized, and annotation files are written to Google Drive directories.

## Requirements

- Python 3.x
- Ultralytics YOLO (`pip install ultralytics`)
- Google Colab (for Drive integration)
- NumPy, Pandas, OpenCV, Matplotlib

## Setup

1. **Install Dependencies**:
   ```bash
   pip install ultralytics ray[tune] wandb
   ```

2. **Mount Google Drive**:
   The code mounts your Google Drive to load and save the dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Dataset Preparation**:
   - Images and labels are stored in `/content/drive/My Drive/working/datasets/cars/`.
   - The dataset is automatically split into training and validation sets, with annotations in YOLO format.

4. **Training**:
   The YOLOv9 model is trained using the following command:
   ```python
   model = YOLO('yolov9c.pt')
   model.train(data="dataset.yaml", epochs=100, batch=8)
   ```

5. **Inference**:
   After training, predictions can be made on test images:
   ```python
   model = YOLO('./runs/detect/train/weights/last.pt')
   ret = model("/content/drive/My Drive/data/testing_images", save=True, conf=0.2, iou=0.5)
   ```

## Visualization

Detected cars are visualized after inference by displaying the bounding boxes on images:
```python
files = glob.glob("./runs/detect/predict/*")
for i in range(0, 30, 3):
    img = Image(files[i])
    display(img)
```

## Acknowledgements

This code was inspired by the open-source implementation available at [YOLOv9 Car Object Detection](https://www.kaggle.com/code/aruaru0/yolov9-car-object-detection#setup-yaml-file). Custom adaptations have been made to integrate Google Drive for dataset handling and make the code more flexible for cloud-based workflows.
