# **Traffic Sign Recognition**

### Project 2 Writeup
---
#### Project objective
- The objective of this project is to train a neural network which can classify and label German traffic signs.
- The model generated from this project can be used to recognize traffic signs from German roads.

##### Document objective
- The goal of this document is to summarize the steps, preprocessing and neural network architecture.
- This document also reflects on how the trained model works on test images found on web.
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
![Randomly chosen input image][image1]
#### 3. Design and Test a Model Architecture
- In this section, we take LeNet as basic architecture and add a few layers to improve accuracy for German Traffic signs.
- Before feeding the model with data, we pre-process the images.
- Since machine learning model works in 'Garbage In, Garbage Out' model, its important that fed data is free from any irregularities such as impossible data combinations or missing values. If not, feeding such data to model and using such model to predict images would result in misleading results. [Source: Pre-processing-Wikipedia](https://en.wikipedia.org/wiki/Data_pre-processing)
- For this project, we convert the image to grayscale. Converting images to gray-scale helps lowering the complexity associated with dealing color images.
![Grayscaled image][image2]
- These grayscale converted image are now fed into histogram equalization stage in order to enhance the contrast in each image.
![Histogram equalization][image3]
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
- As choice of hyper parameters, it was observed that learning rate of 0.0007 and 100 epochs helps us achieve the target accuracy of 0.93 or higher.
- Before feeding the input batch to model, data is shuffled using the shuffle module.
- After using above mentioned network architecture along with hyper parameters value, we are able to achive a accuracy of 0.95 on validation set.
- My final model results are:

| Data set type         | Accuracy achieved                             |
|:---------------------:|:---------------------------------------------:|
| Training set          | 0.99                                          |
| Validation set        | 0.97                                          |
| Test set              | 0.94                                          |

- This results were achieved using iterative model of LeNet with slight modifications to the CNN architecture. Without these modifications, the accuracy of model could never hit 0.90.
- These modifications are:
    - Adding a dropout layer to remove redundancies
    - Adding CNN layer of 3x3 before connecting the Fully Connected layer
    - After training, this model is saved in a file.
### Test model on New images
- In this section, we are supposed to fetch at least 5 images of German traffic signs them from web, and use them as test images for our previously trained model.
- Here are the five images that I choose for my project.
![German traffic Sign from web][image4]
![German traffic Sign from web][image5]
![German traffic Sign from web][image6]
![German traffic Sign from web][image7]
![German traffic Sign from web][image8]
- These images are then fed into same pre-processing pipeline as training data set was fed through (i.e. grayscale, and histogram equalization)
- For this, we restore the previously stored model from a file.
- Then the pre-processed test image data is fed to the model to predict softmax probabilities for each image.
- Based on the probablities generated, corresponding labels are matched.
- The expected labels for these images was hard coded into internal buffer.
- These labels were then compared against model generated labels for test images.

| Image         | Prediction                             |
|:---------------------:|:---------------------------------------------:|
| 1.png  | Yeild                                                        |
| 2.png  | Road work                                                    |
| 3.png  | Right turn ahead                                             |
| 4.png  | Traffic signal                                               |
| 5.png  | Roundabout mandatory                                         |

The model was correctly able to TODO which gives an accuracy of TODO.
- In the last section of project, we are expected to print the top 5 softmax probabilities for each of the image.
- Following is the graphical representation of softmax probabilities predicted for each image.
![Softmax probabilities for Sign 1][image9]
![Softmax probabilities for Sign 2][image10]
![Softmax probabilities for Sign 3][image11]
![Softmax probabilities for Sign 4][image12]
![Softmax probabilities for Sign 5][image13]


[//]: # (Image References)

[image1]: ./writeup_stuff/input_image.png "Randomly chosen input image"
[image2]: ./writeup_stuff/grayscale.png "Grayscaling"
[image3]: ./writeup_stuff/equalize_hist.png "Output from Equalize Histogram"
[image4]: ./writeup_stuff/test_images/1.png "Traffic Sign 1"
[image5]: ./writeup_stuff/test_images/2.png "Traffic Sign 2"
[image6]: ./writeup_stuff/test_images/3.png "Traffic Sign 3"
[image7]: ./writeup_stuff/test_images/4.png "Traffic Sign 4"
[image8]: ./writeup_stuff/test_images/5.png "Traffic Sign 5"
[image9]: ./writeup_stuff/softmax/1.png "Softmax Probabilities for Sign 1"
[image10]: ./writeup_stuff/softmax/2.png "Softmax Probabilities for Sign 2"
[image11]: ./writeup_stuff/softmax/3.png "Softmax Probabilities for Sign 3"
[image12]: ./writeup_stuff/softmax/4.png "Softmax Probabilities for Sign 4"
[image13]: ./writeup_stuff/softmax/5.png "Softmax Probabilities for Sign 5"
