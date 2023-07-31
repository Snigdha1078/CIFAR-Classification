import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

# Load the training data
trainloader = torch.load('./data/interim/trainloader.pt')

# Split the training data into a smaller training set and a validation set
train_size = int(0.8 * len(trainloader.dataset))  # 80% for training
valid_size = len(trainloader.dataset) - train_size  # 20% for validation
train_dataset, valid_dataset = random_split(trainloader.dataset, [train_size, valid_size])

# Create DataLoader for the validation set
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)

# Load the test data
testloader = torch.load('./data/interim/testloader.pt')

# Load the network
from models.cnn_model import Net
net = Net()

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def main():
    # Train the network
    train_losses = []
    train_accuracy = []
    valid_losses = []
    valid_accuracy = []

    for epoch in range(30):  # loop over the dataset for 30Epochs
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate accuracy and loss after each epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100. * correct / total

        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Validation
        net.eval()  # Switch to evaluation mode
        with torch.no_grad():
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(validloader, 0):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                running_loss += loss.item()
                
            valid_losses.append(running_loss / len(validloader))
            valid_accuracy.append(100. * correct / total)

        print(f'Epoch [{epoch + 1}/{30}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, Valid Loss: {valid_losses[-1]:.4f}, Valid Accuracy: {valid_accuracy[-1]:.2f}%')

    print('Finished Training')

    # Save the trained model
    torch.save(net.state_dict(), './codes/models/deer_truck_net.pth')

    # Save training and validation loss and accuracy
    np.save('./data/interim/train_losses.npy', np.array(train_losses))
    np.save('./data/interim/train_accuracy.npy', np.array(train_accuracy))
    np.save('./data/interim/valid_losses.npy', np.array(valid_losses))
    np.save('./data/interim/valid_accuracy.npy', np.array(valid_accuracy))

# This will ensure that the script runs correctly when we execute it directly
if __name__ == '__main__':
    main()
