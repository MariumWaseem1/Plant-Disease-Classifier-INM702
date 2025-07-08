# Plant Disease Classifier – INM702 Task 2

This project classifies plant leaf diseases using convolutional neural networks (CNNs) built with PyTorch. Developed for the INM702 coursework.

## Dataset

\- Source: Plant Disease Recognition (Kaggle)  
\- Classes: Healthy, Powdery, Rust  
\- Image Input Sizes: 128x128 (baseline) and 224x224 (improved)

## Baseline Model

\- Architecture:  
  \- 2 convolutional layers  
  \- Max pooling  
  \- 2 fully connected layers  
\- Optimizer: SGD  
\- Accuracy: 56% (signs of overfitting)

## Improved Model

\- 3 convolutional layers with 32, 64, and 128 filters  
\- Max pooling and dropout (rate = 0.5)  
\- Dense layer with 512 neurons  
\- Data Augmentation:  
  \- Horizontal flip  
  \- ImageNet normalization  
\- Optimizer: Adam  
\- Image Input Size: 224x224  
\- Validation Accuracy: 95%

## Hyperparameter Tuning

\- Grid search over:  
  \- Learning rates: 0.001, 0.01, 0.1  
  \- Batch sizes: 16, 32  
  \- Optimizers: SGD, Adam  
\- Early stopping implemented  
\- Best configuration:  
  \- Learning rate = 0.001  
  \- Batch size = 16  
  \- Optimizer = Adam

## Future Work

\- Add rotation and brightness augmentation  
\- Use transfer learning (ResNet, DenseNet)  
\- Build a user interface for real-time plant disease detection



## Authors

\- Marium Waseem  
\- Kovarthanan Kesavan
