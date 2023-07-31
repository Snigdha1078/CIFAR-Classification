import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn_model import Net
from PIL import Image

# Load the model
net = Net()
net.load_state_dict(torch.load('./models/deer_truck_net.pth'))

# Load the test data
testloader = torch.load('./data/interim/testloader.pt')

# Get some random images from the test loader
dataiter = iter(testloader)
images, labels = next(dataiter)

# Get predictions
outputs = net(images)
_, predicted = torch.max(outputs, 1)

# Function for plotting and saving images
def imshow_and_save(img, title, save_path):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    
    # Convert to PIL Image for better resizing
    pil_img = Image.fromarray(np.uint8(npimg*255))
    
    # Resize the image using PIL's resize function with LANCZOS filter (interpolation)
    pil_img = pil_img.resize((128,128), Image.LANCZOS)
    
    plt.imshow(pil_img)
    plt.title(title)
    plt.savefig(save_path,bbox_inches='tight')  # Save the image to the specified path
    plt.show()

# Create the "Output" folder if it doesn't exist
output_folder = './Outputs'
os.makedirs(output_folder, exist_ok=True)

# Get class names
classes = ('deer', 'truck')

# Identify correctly and incorrectly classified 'deer' images
correct_deer_indices = (labels == predicted) & (labels == 0)
incorrect_deer_indices = (labels != predicted) & (labels == 0)

# Plot and save two images labeled 'deer' that the model predicts correctly
correct_deer_images = images[correct_deer_indices][:2]
correct_deer_labels = predicted[correct_deer_indices][:2]
imshow_and_save(make_grid(correct_deer_images), ' '.join('%5s' % classes[correct_deer_labels[j]] for j in range(2)), './Outputs/correct_deer.png')

# Plot and save two images labeled 'deer' that the model predicts incorrectly
incorrect_deer_images = images[incorrect_deer_indices][:2]
incorrect_deer_labels = predicted[incorrect_deer_indices][:2]
imshow_and_save(make_grid(incorrect_deer_images), ' '.join('%5s' % classes[incorrect_deer_labels[j]] for j in range(2)), './Outputs/incorrect_deer.png')
