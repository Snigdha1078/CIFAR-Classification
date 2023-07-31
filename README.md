# Datarock Programming Assessment
In this project, we will design and train a Convolutional Neural Network (CNN) to classify images from the CIFAR10 dataset into two classes: deer and truck. The CIFAR10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. Our objective is to build an accurate image classifier for the specific classes 'deer' and 'truck.'

## Environement Setup
Python 3.8 or higher is required for running this code.

### Dependencies Installation
1. Install the required packages using 'pip'
``pip install torch torchvision numpy matplotlib``

     OR

2. Install the required packages using requirements.txt
```pip install -r requirements.txt```

### How to Run the code
To execute each Python file, open a terminal or command prompt and run the following commands in the order:

1. For Downloading the CIFAR data 
```bash
     python codes/01-data_preparation.py
```
2. To extract images with labels deer and truck from the training and test subsets of CIFAR10 dataset
```bash
     python codes/02-preprocessing.py
```
3. To Design a CNN model with 4 convolutional layers and 2 fully connected layers
```bash
     python codes/models/cnn_model.py
```
4. To train the CNN model for 30 epochs
```bash
     python codes/04-train_model.py
```
5. To Plot the training loss/accuracy and validation loss/accuracy of the model per epoch during training
```bash
     python codes/05-plot_metrics.py
```
6. To plot two images labeled deer that the model predicts correctly and two images labeled deer that the model predicts wrongly
```bash
     python codes/06-predict_and_visualize_deer.py
```
7. To plot two images labeled truck that the model predicts correctly and two images labeled truck that the model predicts wrongly
```bash
     python codes/07-predict_and_visualize_truck.py
```

### Outputs
The 'outputs' directory will contain the following files:

- subplots.png: Plot showing the training loss/accuracy and validation loss/accuracy per epoch.
- correct_deer.png: Images of two correctly predicted 'deer' samples.
- incorrect_deer.png: Images of two incorrectly predicted 'deer' samples.
- correct_truck.png: Images of two correctly predicted 'truck' samples.
- incorrect_truck.png: Images of two incorrectly predicted 'truck' samples.
