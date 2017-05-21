# **Traffic Sign Recognition**

### Project 2 Writeup
---
#### Project objective
- The objective of this project is to train a neural network which can classify and label German traffic signs.
- The model generated from this project can be used to recognize traffic signs from German roads.

##### Document objective
- The goal of this document is to summarize the steps, preprocessing and neural network architecture.
- TODO
---
### Pipeline architecture
#### 1. Load the data
- In this step we load the training, test and validation data into internal buffer.
- In this section, we also summarize some basic facts about data as follows:
```
Image Shape: (32, 32, 3)
Training Set:   34799 samples
Validation Set: 4410 samples
Test Set:       12630 samples
```
#### 2. Dataset Summary & Exploration
- In this step, we randomly choose a image and print out values for all 3 channels
- We also show project how the image looks.
#### 3. Design and Test a Model Architecture
- In this section, we take LeNet as basic architecture and add a few layers to improve accuracy for German Traffic signs.
- Before feeding the model with data, we pre-process the images.
- Since machine learning model works in 'Garbage In, Garbage Out' model, its important that fed data is free from any irregularities such as impossible data combinations or missing values. If not, feeding such data to model and using such model to predict images would result in misleading results. [Source: Pre-processing-Wikipedia](https://en.wikipedia.org/wiki/Data_pre-processing)
- For this project, we convert the image to gray-scale. Converting images to gray-scale helps lowering the complexity associated with dealing color images.
- These grayscale converted image are now fed into histogram equalization stage in order to enhance the contrast in each image.
- After pre-processing images, data is fed to neural network of following architecture.

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 image                                 |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 3x3x36     |
| RELU                  |                                               |
| Fully connected       | Input 324, Output 120                         |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | Input 120, Output 84                          |
| RELU                  |                                               |
| Fully connected       | Input 84, Output 43                           |

- To train this model, I used AdamOptimizer optimizer to reduce the loss.
- As choice of hyper parameters, it was observed that learning rate of 0.0007 and 100 epochs helps us achieve the target accuracy of 0.93 or better.
- Before feeding the input batch to model, data is shuffled using the shuffle module.
-