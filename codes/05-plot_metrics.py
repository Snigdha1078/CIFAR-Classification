import numpy as np
import matplotlib.pyplot as plt
import os

# Load training loss and accuracy
train_losses = np.load('./data/interim/train_losses.npy')
train_accuracy = np.load('./data/interim/train_accuracy.npy')

# Load validation loss and accuracy
valid_losses = np.load('./data/interim/valid_losses.npy')
valid_accuracy = np.load('./data/interim/valid_accuracy.npy')

# Plot training and validation loss per epoch
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy per epoch
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(valid_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Create the output folder if it doesn't exist
output_folder = './Outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the subplots as images in the output folder
plt.savefig(os.path.join(output_folder, 'subplots.png'))

# Show the plots
plt.show()
