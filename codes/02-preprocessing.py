import torch
import numpy as np
import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader

# Load the raw CIFAR10 data
trainset = CIFAR10(root='./data/raw', train=True)
testset = CIFAR10(root='./data/raw', train=False)

# CIFAR10 classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to extract 'deer' and 'truck' images
def extract_classes(train, classes):
    'Function to separate the classes in the CIFAR10 train set'
    data = list()
    targets = list()
    for i in range(len(train)):
        if train[i][1] in classes:  # If the label is in the classes
            data.append(train[i][0])
            if train[i][1] == 4:  # 'deer' in CIFAR10 is 4
                targets.append(0)  # Map 'deer' to 0
            else:  # 'truck' in CIFAR10 is 9
                targets.append(1)  # Map 'truck' to 1
    return data, targets

# Extract 'deer' and 'truck' images
train_data, train_targets = extract_classes(trainset, [4, 9])
test_data, test_targets = extract_classes(testset, [4, 9])

# Convert to PyTorch tensors
train_data = torch.stack([ToTensor()(image) for image in train_data])
train_targets = torch.tensor(train_targets)
test_data = torch.stack([ToTensor()(image) for image in test_data])
test_targets = torch.tensor(test_targets)

# Create DataLoader objects
trainset = TensorDataset(train_data, train_targets)
testset = TensorDataset(test_data, test_targets)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Create the "interim" folder if it doesn't exist
output_folder = './data/interim'
os.makedirs(output_folder, exist_ok=True)

# Save the DataLoaders to be used in the training script
torch.save(trainloader, './data/interim/trainloader.pt')
torch.save(testloader, './data/interim/testloader.pt')
