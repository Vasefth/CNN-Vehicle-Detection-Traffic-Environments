# VGG16 Object Detection Model

This project implements an object detection system using the VGG16 model for feature extraction, combined with a custom head for predicting bounding boxes. The implementation is built using TensorFlow and Keras, and the model is trained on a dataset of images with labeled bounding boxes.

## Features

- **Model Architecture**: Uses a pre-trained VGG16 as the base model, with added layers for object detection tasks.
- **Bounding Box Prediction**: Predicts both the bounding boxes and confidence scores for objects in the images.
- **Training & Evaluation**: Tracks performance through metrics like loss, precision, recall, and F1-score. Visualizes training and validation metrics across epochs.

## Requirements

- Python 3.x
- TensorFlow (tested with TensorFlow 2.x)
- Keras
- OpenCV
- NumPy
- pandas
- matplotlib

## Setup

1. **Install Dependencies**:
   You can install the required libraries using the following command:
   ```bash
   pip install tensorflow keras numpy pandas opencv-python matplotlib
   ```

2. **Prepare Dataset**:
   - Upload your image dataset and corresponding annotation CSV file to Google Drive.
   - The CSV file should include bounding box information for each image.

3. **Paths Configuration**:
   The code expects the following paths:
   - `IMAGES_PATH`: Directory where the training images are stored.
   - `ANNOTS_PATH`: CSV file with bounding box annotations.
   - `BASE_OUTPUT`: Directory where the model and plots will be saved.
   
   Ensure these paths are properly set in the code.

## Model Training

1. **Data Preparation**:
   The dataset is loaded from the CSV file, and the images are read and preprocessed using OpenCV. The bounding boxes are extracted from the CSV and paired with the corresponding images.

2. **Model Architecture**:
   - **VGG16 Backbone**: The model uses VGG16 pre-trained on ImageNet as the base for feature extraction.
   - **Custom Head**: A custom head is added for predicting bounding boxes and object confidence scores.

3. **Training Process**:
   - The model is trained using Adam optimizer and tracks metrics like precision, recall, and F1-score.
   - Training history, including loss, precision, recall, and F1-score, is plotted over the training epochs.

4. **Saving the Model**:
   After training, the model is saved in the specified output directory as `detector.h5`.

## Evaluation

- The code evaluates the model using metrics such as loss, precision, recall, and F1-score, both on training and validation datasets.
- A series of plots are generated to visualize the training progress across epochs:
  - Loss (training vs. validation)
  - Precision (training vs. validation)
  - Recall (training vs. validation)
  - F1-score (training vs. validation)

## Visualization

- The script includes plotting functionality to display training and validation metrics over time. The plots are saved as images for later analysis.

## Running the Code

1. Mount your Google Drive to access the dataset.
2. Set the paths to the images and annotations in the code.
3. Run the script to train the model. The training process will save the model and plots in the output directory.

## Future Improvements

- Add real-time object detection support on video data.
- Experiment with different backbone models for improved performance.
- Incorporate more advanced augmentation techniques for better generalization.
