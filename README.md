🧠 MNIST Digit Classifier (TensorFlow + Keras)

This project trains a simple Convolutional Neural Network (CNN) on the MNIST dataset to classify handwritten digits (0–9).
It demonstrates the core steps of a deep learning workflow — from loading data and preprocessing to training, evaluating, and visualizing results.

📚 Libraries Used
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

📂 Project Structure
├── MNIST_training.ipynb     # Colab notebook containing all code and visualizations
├── Sample_images            #images to test the model with 
├── README.md                # Project documentation

🧰 Requirements

You can install all dependencies with:

pip install numpy matplotlib seaborn opencv-python pillow tensorflow


If you’re using Google Colab, most libraries come preinstalled.

🧩 Dataset

The MNIST dataset is loaded directly from Keras:

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


Training images: 60,000

Testing images: 10,000

Image size: 28×28 pixels (grayscale)

⚙️ Model Overview

Framework: TensorFlow / Keras

Type: Convolutional Neural Network (CNN)

Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

📈 Results

The model achieves around:

Training Accuracy: ~99%

Test Accuracy: ~98%

Visualizations include:

Sample digits from the dataset

Confusion matrix heatmap

Predictions on unseen data

🚀 How to Run

Open the notebook in Google Colab:


Run all cells (Runtime → Run all).

The notebook will:

Load and preprocess the MNIST dataset

Train a CNN

Display accuracy, loss plots, and a confusion matrix

🧪 Example Output

Predicted vs Actual digit comparison

Confusion matrix of classification results

Accuracy/Loss curves

📜 License

This project is open-source under the MIT License.
