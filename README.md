# Image Classification using Deep Learning on CIFAR-10

This repository contains the code and report for an image classification project using deep learning techniques on the CIFAR-10 dataset. The project aims to implement and evaluate the performance of different models and optimization algorithms to classify images into 10 distinct categories.

## Repository Contents
- Image_Classification_using_Deep_Learning_on_CIFAR_10.ipynb: Jupyter notebook containing the implementation of various image classification models using different optimization algorithms and loss functions.
- Report on Image_Classification.pdf: Detailed report summarizing the project, including model architecture, experimental setup, results, and conclusions.

## Project Overview
The project explores the effectiveness of various optimization algorithms (Adam, SGD, RMSProp) and loss functions (Categorical Crossentropy, Mean Squared Error) on the CIFAR-10 image classification task. The key components of the project are:

### Model Architecture:

- Convolutional Neural Network (CNN) with the following layers:
  Conv2D: 32 filters, kernel size (3, 3), ReLU activation
  MaxPooling2D: pool size (2, 2)
  Conv2D: 64 filters, kernel size (3, 3), ReLU activation
  MaxPooling2D: pool size (2, 2)
  Conv2D: 128 filters, kernel size (3, 3), ReLU activation
  Flatten
  Dense: 512 units, ReLU activation
  Dropout: 0.5
  Dense: 10 units, softmax activation

### Experimental Setup:

- Experiments conducted using three different optimization algorithms (Adam, SGD, RMSProp) and two loss functions (Categorical Crossentropy, Mean Squared Error).
- Each combination trained for 10 epochs.
- Performance evaluated using metrics: Accuracy, Precision, Recall, F1 Score, and Specificity.

### Results:

- The best-performing model used the Adam optimizer with categorical crossentropy loss, achieving an accuracy of 73.09%.
- SGD with mean squared error performed poorly, indicating a mismatch for the CIFAR-10 classification task.
- Detailed results including confusion matrices and performance metrics for each model configuration are provided in the report.

## Conclusion:

Future improvements could include deeper networks, learning rate adjustments, and data augmentation techniques to enhance model robustness.
