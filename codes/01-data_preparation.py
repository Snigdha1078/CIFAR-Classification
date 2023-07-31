import torchvision
import torchvision.transforms as transforms

# Define a transform to normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training data
trainset = torchvision.datasets.CIFAR10(root='./data/raw', train=True,
                                        download=True, transform=transform)

# Download and load the test data
testset = torchvision.datasets.CIFAR10(root='./data/raw', train=False,
                                       download=True, transform=transform)
