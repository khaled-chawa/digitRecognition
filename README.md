# Handwritten Digit Recognition with Convolutional Neural Networks

## Overview
This repository contains the code for building a convolutional neural network (CNN) to recognize handwritten digits using PyTorch. The model is trained and evaluated on the popular MNIST dataset, which consists of grayscale images of handwritten digits ranging from 0 to 9.

## Project Structure
- README.md: Overview and instructions for the repository.
- train.py: Python script for training the CNN model on the MNIST dataset.
- evaluate.py: Python script for evaluating the trained model on a separate test set.
- model.py: Definition of the CNN architecture using PyTorch.
- requirements.txt: List of dependencies required to run the code.

## Setup
1. Clone the repository to your local machine:
```git clone https://github.com/khaled-chawa/digitRecognition.git```

2. Install the required dependencies using pip:
```pip install -r requirements.txt```

## Training
To train the model on the MNIST dataset, run each block of code in the train.ipynb file. This will train the model using default hyperparameters and save the trained model weights to a file (You can skip this part as the file that will be created, model_weights.pth, has already been uploaded to github and therefore you have already downloaded it).

## Evaluation
To evaluate the trained model on a separate test set, run each block of code in the evaluate.ipynb file. This will load the trained model weights and evaluate its performance on the test set, displaying metrics such as accuracy and loss.

## Model Architecture
The CNN architecture is defined in the model.py file. It consists of convolutional layers, max-pooling layers, fully connected layers, and dropout regularization. The model architecture can be customized by adjusting the hyperparameters and layers in the model.py file.

