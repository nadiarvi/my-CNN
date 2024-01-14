# My-CNN Image Classification

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nadiarvi/my-CNN/blob/main/Image_Classification.ipynb)

## Introduction
This repository contains a Jupyter notebook (`Image_Classification.ipynb`) that demonstrates image classification using a Convolutional Neural Network (CNN). The code is written in Python and utilizes PyTorch for deep learning.

## Getting Started
To run the code in your local environment or Colab, simply click on the Colab badge above or navigate to the [Colab link](https://colab.research.google.com/github/nadiarvi/my-CNN/blob/main/Image_Classification.ipynb).

### Prerequisites
Make sure you have the required dependencies installed. You can install them using the following:
```bash
pip install torch torchvision matplotlib
```

## Content Overview

### 1. Packages
The notebook starts by importing necessary packages for the experiment.

### 2. Experiment Configuration
Defines hyperparameters such as the number of epochs, learning rate, batch size, and device for training.

### 3. Data Pipeline Construction
Constructs the data pipeline using the CIFAR-10 dataset.

### 4. Building Convolutional Neural Networks (CNNs)
Defines a custom convolutional layer (`My_Conv2d`) used in building the CNN architecture.

### 5. Building a CNN model based on My_Conv2d
Defines the CNN model (`MyOwnClassifier`) using the custom convolutional layer.

### 6. Initialize the network and optimizer
Initializes the CNN model and the stochastic gradient descent (SGD) optimizer.

### 7. Train the network
Trains the CNN on the CIFAR-10 dataset and prints the training and testing performance for each epoch.

### 8. Visualize the loss and accuracy
Plots the training and testing loss, as well as the training and testing accuracy over epochs.

### 9. Visualize Random Test Prediction
Displays a random test image along with its ground truth label and the predicted label.

## Results
After 50 epochs of training, the model achieved the following results:

Training Loss: 0.6349
Training Accuracy: 77.48%
Test Loss: 0.5796
Test Accuracy: 80.47%


