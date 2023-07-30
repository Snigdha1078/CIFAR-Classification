import torch
import torch.nn as nn
import torch.optim as optim

# Load the training data
trainloader = torch.load('./data/interim/trainloader.pt')

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
    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    torch.save(net.state_dict(), './codes/models/deer_truck_net.pth')

# This will ensure that the script runs correctly when we execute it directly
if __name__ == '__main__':
    main()
