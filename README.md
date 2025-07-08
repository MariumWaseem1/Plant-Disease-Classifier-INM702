Air & Plant Health Classifier – INM702 Project
This repository contains two machine learning applications developed as part of the INM702 coursework by Marium Waseem and Kovarthanan Kesavan.

Task 1 predicts air quality using a neural network built from scratch in NumPy.

Task 2 classifies plant diseases using a convolutional neural network implemented in PyTorch.

Task 1: Air Quality Prediction (NumPy)
Dataset: Air Quality & Pollution Assessment (Kaggle)

Objective: Classify air quality into four classes — Good, Moderate, Poor, and Hazardous

Input Features: Temperature, Humidity, PM2.5, PM10, NO2, SO2, CO, Proximity to Industrial Areas, Population Density

Techniques Used
Preprocessing with StandardScaler and one-hot encoding

Class imbalance handled using SMOTE

Model: 3-layer dense neural network using ReLU and Softmax

Optimization: Batch Gradient Descent and Mini-batch Gradient Descent

Regularization: Dropout (rate = 0.05) using inverted scaling

Results
Accuracy with Batch Gradient Descent: 93.44%

Accuracy with Mini-batch Gradient Descent: 91.63%

Task 2: Plant Disease Classification (PyTorch)
Dataset: Plant Disease Recognition (Kaggle)

Classes: Healthy, Powdery, Rust

Image Size: 128x128 for baseline model, 224x224 for improved model

Baseline CNN Model
Architecture: 2 convolutional layers → max pooling → 2 fully connected layers

Optimizer: SGD

Result: 56% test accuracy

Improved CNN Model
Architecture: 3 convolutional layers (32, 64, 128 filters)

Dense Layer: 512 neurons

Regularization: Dropout (0.5)

Data Augmentation: Horizontal flipping and normalization

Optimizer: Adam

Hyperparameter Tuning
Parameters tuned:

Learning rates: 0.001, 0.01, 0.1

Batch sizes: 16, 32

Optimizers: SGD, Adam

Early stopping to prevent overfitting

Best configuration: LR = 0.001, Batch size = 16, Optimizer = Adam

Final validation accuracy: 95%

Future Work
Integrate live air monitoring via sensors and IoT

Apply rotation and brightness augmentation for better plant disease detection

Experiment with ResNet, DenseNet for higher performance

Deploy via web or mobile interface for public/environmental use

