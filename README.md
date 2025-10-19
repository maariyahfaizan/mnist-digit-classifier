🧠 **MNIST Digit Classifier (TensorFlow + Keras)**

This project trains a simple Convolutional Neural Network (CNN) on the MNIST dataset to classify handwritten digits (0–9).
It demonstrates the core steps of a deep learning workflow — from loading data and preprocessing to training, evaluating, and visualizing results.

💡 **Project Motivation**

This project was created to strengthen my understanding of deep learning fundamentals and gain hands-on experience with TensorFlow and Keras.
It serves as my first step toward exploring real-world AI applications, model evaluation, and interactive deployment of trained models.

📂 **Project Structure** <br>
├── MNIST_training.ipynb     # Colab notebook containing all code and visualizations <br>
├── Sample_images            #images to test the model with <br>
├── README.md                # Project documentation<br>

🧰 **Requirements**

You can install all dependencies with:

pip install numpy matplotlib seaborn opencv-python pillow tensorflow


If you’re using Google Colab, most libraries come preinstalled.

🧩 **Dataset**

The MNIST dataset is loaded directly from Keras:

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


Training images: 60,000

Testing images: 10,000

Image size: 28×28 pixels (grayscale)

⚙️ **Model Overview**

Framework: TensorFlow / Keras

Type: Convolutional Neural Network (CNN)

Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

📈 **Results**

The model achieves around: <ul>
<li>Training Accuracy: ~99% </li>
<li>Test Accuracy: ~98% </li>
</ul>

*Visualizations include:*
<ul>
<li>Sample digits from the dataset</li>
<li>Confusion matrix heatmap</li>
<li>Predictions on unseen data</li>
</ul>

🚀**How to Run**

Open the notebook in Google Colab:<br>
Run all cells (Runtime → Run all)<br>
The notebook will:<br>
Load and preprocess the MNIST dataset<br>
Train a CNN<br>
Display accuracy and a confusion matrix<br>

📜**License**

This project is open-source under the MIT License.
