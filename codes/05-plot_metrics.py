import numpy as np
import matplotlib.pyplot as plt

# Load training loss and accuracy
train_losses = np.load('./data/interim/train_losses.npy')
train_accuracy = np.load('./data/interim/train_accuracy.npy')

# Plot training loss per epoch
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy per epoch
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
