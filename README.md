
# Fish Image Classification with Artificial Neural Network (ANN)

This project involves building and training an Artificial Neural Network (ANN) model using PyTorch to classify different species of fish based on image data. The images are preprocessed and resized to 128x128 RGB format, and the model is trained on this dataset for classification purposes.
## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Installation](#installation)
- [Results](#results)
## Requirements
  Below is the list of libraries that are used in this project:
  ```py
    from IPython import get_ipython
    from IPython.display import display
    import kagglehub
    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from torchvision import transforms
    from sklearn.metrics import f1_score
  ```


## Project Overview
The goal of this project is to develop a deep learning model that can classify fish images into various categories. The ANN model is trained using a dataset of fish images, which have been preprocessed and labeled. The model aims to learn the underlying patterns in the images and predict the correct fish species with high accuracy
- [Click to open the project](https://colab.research.google.com/drive/1zrrzMlRB_MGcYDRkq4QGVWs7-UqFmfiM?usp=sharing)

## Dataset
This project uses a dataset of fish images, which are classified into multiple species. Each image represents a different species of fish and is labeled accordingly. The dataset is divided into two main sets: training and validation.

### Key Features of the Dataset	
- Image Format: PNG
- Image Dimensions: 128x128 pixels (resized to a uniform size)
- Color Mode: RGB
- Total Images: `(9000)`
### Class Distribution
```
  Number of label
```

| Fish Name | Count    |
| :-------- | :------- | 
| `Trout` | `1000` |
| `Shrimp` | `1000` |
| `Gilt-Head Bream	` | `1000` |
| `Red Sea Bream	` | `1000` |
| `Black Sea Sprat	` | `1000` |
| `Sea Bass	` | `1000` |
| `Striped Red Mullet	` | `1000` |
| `Hourse Mackerel	` | `1000` |
| `Red Mullet	` | `1000` |

### Data Loading and Preprocessing
1.	Filepath and Label Extraction:
The dataset is organized such that each fish species has its own folder. The fish images are stored as PNG files within these folders. The code below traverses the directory to collect the file paths of all PNG images and assigns the correct label (i.e., fish species) based on the folder name:

```python
image_dir = Path('/root/.cache/kagglehub/datasets/crowww/a-large-scale-fish-dataset/versions/2/Fish_Dataset/Fish_Dataset')

filepaths = list(image_dir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))


- Filepaths: All .png images are collected from subfolders.
- Labels: The label for each image is derived from the name of the folder it is located in. For instance, if an image is in the “Salmon” folder, it will be labeled as “Salmon”.
```
2.	Create a DataFrame:
The filepaths and labels are then combined into a Pandas DataFrame for easy manipulation and analysis:
```python
filepaths = pd.Series(filepaths, name='Filepath').astype(str)

labels = pd.Series(labels, name='Label')

image_df = pd.concat([filepaths, labels], axis=1)
```
3. In the dataset, some images might have a suffix like “GT,” which typically indicates they are ground truth images or images used for testing purposes. These are not needed for training, so the code filters out any image whose label ends with “GT”:
``` python
# Drop GT images
image_df = image_df[~image_df['Label'].str.endswith('GT')]
```
- This line removes all rows where the label ends with “GT” to ensure only the necessary images are used for training and evaluation.

You can see the full of code here : [ x ] image ekle
### Data Preparation
 In this section, we prepare the dataset for training the model by encoding the labels and splitting the data into training and validation sets.
```python
label_encoder = LabelEncoder()

image_df['Label'] = label_encoder.fit_transform(image_df['Label'])

train_df, val_df = train_test_split(image_df, test_size=0.2, random_state=42, stratify=image_df['Label'])

```
- The fit_transform method encodes the unique labels in the Label column of the image_df DataFrame. Each unique fish species is assigned an integer value, allowing the model to interpret these labels during training.
- We utilize the train_test_split function from sklearn.model_selection to divide the dataset into training and validation sets.
- The dataset is split into 80% training and 20% validation data. The stratify parameter ensures that both sets maintain the same proportion of each class (fish species), which is important for balanced training. The random_state parameter is set to 42 to ensure that the results are reproducible.

## Model Architecture
In this project, we developed a simple artificial neural network (ANN) model using PyTorch for classifying fish species. The model is designed to take images resized to 128x128 pixels as input and classify them into one of nine fish species. Below is a detailed explanation of the model architecture and its components.

![image](https://github.com/user-attachments/assets/0ce7b03d-d6eb-4518-8d29-61f6f099664e)


` Explanation of Each Component : `
```
1. **Initialization (`__init__` method)**:
   - **`input_size=3*128*128`**: 
     - This parameter defines the input size of the model. The input data consists of images resized to 128x128 pixels with 3 color channels (RGB). Therefore, each input has a total of 49,152 features (3 * 128 * 128 = 49,152).
   - **`num_classes=9`**: 
     - This parameter specifies the number of different fish species that the model will classify. The model will be trained to categorize input images into one of 9 fish species.

2. **Fully Connected Layers**:
   - **First Layer (`fc1`)**: 
     - The first fully connected layer takes the input of size 49,152 and outputs 512 features. This layer learns to map the input features to a higher-dimensional space.
   - **Second Layer (`fc2`)**: 
     - The second fully connected layer takes the output of the first layer (512 features) and reduces it to 256 features. This helps the model filter out less important information and learn more meaningful representations.
   - **Third Layer (`fc3`)**: 
     - The final fully connected layer maps the 256 features to the number of classes (9). This layer produces the output logits, which represent the model's predictions for each fish species.

3. **Dropout Layers**:
   - **Dropout (`dropout1` and `dropout2`)**: 
     - Dropout is applied after the first and second fully connected layers with a probability of 0.5. This means that during training, 50% of the neurons will be randomly deactivated. This technique helps prevent overfitting and improves the model's generalization performance.

4. **Activation Function**:
   - **ReLU Activation (`F.relu`)**: 
     - The ReLU (Rectified Linear Unit) activation function is applied after the first and second fully connected layers. This introduces non-linearity to the model, allowing it to learn more complex patterns. ReLU sets negative values to zero and passes positive values as they are.

5. **Forward Pass**:
   - **Forward Method**: 
     - The `forward` method defines how the input data flows through the model. It flattens the input tensor, passes it through the first fully connected layer with ReLU activation, applies dropout, then passes it through the second layer, applies dropout again, and finally produces the output from the third layer.
```
---
### Summary
The `SimpleANN` model is a straightforward yet effective architecture for classifying fish species based on image data. It utilizes fully connected layers, dropout for regularization, and ReLU activations to learn from the input data and make accurate predictions.

## Training & Evaluation

In this section, we describe the process of training the Simple Artificial Neural Network (ANN) model and evaluating its performance on the validation dataset.

### Training Process
In this section, we define the datasets for training and validation, and set up the DataLoader to efficiently load data during the training process.

### Creating Datasets

1. **Training Dataset**:
  - We create a training dataset using the `ImageDataset` class, which takes the training DataFrame (`train_df`) and a set of transformations (`train_transform`) as input.
  ```python
  train_dataset = ImageDataset(train_df, transform=train_transform)
  ```
2. **Validation Dataset**:
  - Similarly, we create a validation dataset using the ImageDataset class, passing the validation DataFrame (`val_df`) and the corresponding transformations (`validation_transform`).
  ```python
  validation_dataset = ImageDataset(val_df, transform=validation_transform)
  ```
3. **Training DataLoader**:
  - We set up a DataLoader for the training dataset using PyTorch’s DataLoader class. The shuffle=True parameter ensures that the data is shuffled at each epoch, which helps improve generalization.
  ```python
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  ```
4. **Validation DataLoader**:
  - We also create a DataLoader for the validation dataset. The `shuffle=False`parameter is used here, as we want to evaluate the model on the validation data in the order it is provided.
  ```python
  validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

  ```
### Model Setup
In this section, we will establish the foundational components needed for our neural network model, including model architecture, loss function, optimizer, and metric tracking for training performance.
![image](https://github.com/user-attachments/assets/4df52484-dd80-48cb-aba9-19c628eb05d6)


- *Model Creation*:
  We instantiate the `SimpleANN` model, which is designed to take input images reshaped into a flat array of pixels. The input size is calculated as `3*128*128`, where:
  - `3` represents the RGB color channels,
  - `128*128` indicates the height and width of the images.
  
  The number of output classes is determined by the total unique labels from the dataset using the label encoder.

- *Device Configuration* :
  To leverage hardware acceleration, we check if a `GPU` is   available. If so, we use `CUDA`; otherwise, we fall back to the `CPU`. This is critical for improving training speed and efficiency.

- *Loss Function* :
  We employ `Cross-Entropy` Loss, which is a standard loss function for multi-class classification problems. It measures the dissimilarity between the predicted probabilities and the actual class labels, guiding the model to improve its predictions during training.
- *optimizer* : 
  The Adam optimizer is chosen for its adaptive learning rate capabilities, making it suitable for a wide range of problems. We set a learning rate of 0.001, which determines the step size at each iteration while moving toward a minimum of the loss function.
#### *Mertric Tracking* : 
  We create lists to store the training and validation losses as well as the accuracies for each epoch. These metrics will help us monitor the model’s performance over time, allowing us to identify any signs of overfitting or underfitting.
```py
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

train_losses.clear()
val_losses.clear()
train_accuracies.clear()
val_accuracies.clear()
```
## Installation

  This document provides an overview of the `train` and `validate` functions used for training and evaluating a machine learning model using PyTorch. The code structure includes the training loop for multiple epochs and the calculation of both training and validation metrics.
  ![image](https://github.com/user-attachments/assets/196e02c3-1183-4942-8a51-18d22a38119f)


  **Parameters** :
  - dataloader: A PyTorch DataLoader that provides batches of training data.
  - model: The neural network model to be trained.
  - loss_fn: The loss function used to evaluate the model’s predictions.
  - optimizer: The optimization algorithm used to update the model’s parameters.
  ```py
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


	•	The loop iterates over batches of data.
	•	The input tensor X and target labels y are moved to the specified device (GPU or CPU).
	•	Predictions are generated by passing X through the model.
	•	The loss is computed using the model’s predictions and the target labels.
	•	Gradients are zeroed, the backward pass is executed, and the optimizer updates the model’s weights.

  total_loss += loss.item()
  correct += (pred.argmax(1) == y).sum().item()
  -	The total loss is accumulated, and the number of correct predictions is counted.

  ```
  *We prepared similar codes in the validation section, now let's see our results.*

## Results
   The dataset was divided into training and validation sets. The model was trained using the training set and evaluated using the validation set. Tracked training and validation losses and accuracies. To further evaluate the model's performance, the F1 score was calculated.You can check the project insert to further more information:
  - [Click to open the google colab](https://colab.research.google.com/drive/1zrrzMlRB_MGcYDRkq4QGVWs7-UqFmfiM?usp=sharing)

  - [Click to open the kaglelink ](https://www.kaggle.com/code/kadirallolu/fish-image-classification-with-artificial-neural-n)
   

