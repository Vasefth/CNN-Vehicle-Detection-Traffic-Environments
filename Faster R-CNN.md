# CorrectFasterR-CNN: Object Detection Using Faster R-CNN

This project implements object detection using the Faster R-CNN architecture with a ResNet50 backbone and Feature Pyramid Network (FPN). The implementation includes data loading, model training, evaluation, and visualization of results. The project is designed to work on a dataset of images stored in Google Drive, using PyTorch and Google Colab for training and testing.

## Requirements

- Python 3.x
- Google Colab (for running the notebook)
- PyTorch
- Torchvision
- tqdm
- pandas
- PIL (Python Imaging Library)
- matplotlib
- seaborn
- sklearn

## Setup

1. **Mount Google Drive**: The code is designed to load image data directly from Google Drive. Ensure that your dataset is uploaded in the specified directory (`/content/drive/My Drive/data/`).

2. **Install Dependencies**: Use the following commands to install the required libraries in Google Colab or your local environment:
   ```bash
   pip install torch torchvision tqdm seaborn
   ```

3. **Model Architecture**: This code uses the Faster R-CNN model pre-trained on the COCO dataset. The model is then fine-tuned for object detection using custom data.

## Key Components

### 1. **Custom Dataset Class**
   The `CustDat` class is a custom PyTorch dataset that loads images and their corresponding bounding boxes for training. Images are read from the provided dataset directory, and bounding boxes are associated with the images using a pandas DataFrame.

### 2. **Training and Evaluation**
   - The Faster R-CNN model with a ResNet50 backbone is fine-tuned for 4 epochs.
   - The training loop utilizes ground truth bounding boxes for supervision.
   - The evaluation process calculates precision and Average Precision (AP) metrics for bounding boxes.

### 3. **Data Augmentation and Preprocessing**
   - Images are transformed into tensors before being fed into the model.
   - Confidence thresholds and non-maximum suppression (NMS) are applied during the inference phase to filter overlapping bounding boxes.

### 4. **Visualization**
   The script includes functions for visualizing detected bounding boxes on images:
   - Ground truth boxes are displayed in green.
   - Predicted boxes with confidence scores are displayed in red.

### 5. **Inference on Test Data**
   The model can infer on a set of test images from a different directory (`/content/drive/My Drive/data/testing_images/`). After inference, it displays the results with bounding boxes and confidence scores.

## Running the Code

1. **Training**: The model can be trained using the provided dataset by simply running the script. Adjust the number of epochs, learning rate, or any other hyperparameters as needed.
   
2. **Evaluation**: Precision and Average Precision (AP) are calculated to evaluate the model’s performance on the validation data. Results are printed after training is completed.

3. **Inference**: Test images from a specified folder can be processed using the trained model to visualize bounding boxes and confidence scores.

## Notes

- The script assumes that training images and test images are stored in specific folders within Google Drive. Make sure that your data is structured correctly to avoid errors.
- While training on Colab, ensure that GPU acceleration is enabled for faster training and inference.

## Future Improvements

- Implement additional data augmentation techniques to improve model robustness.
- Add support for training on larger datasets by incorporating better memory management strategies.
- Extend the current model to work on video data for real-time object detection.
