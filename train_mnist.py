import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary

from NeuralNet import Network

BATCH_SIZE = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

def main():
    train_loader, test_loader = create_loaders()

    model = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    summary(model, (1, 784))

    for epoch in range(1, NUM_EPOCHS + 1):
        train_model(model, train_loader, optimizer, criterion, epoch)
        acc = test_model(model, test_loader)

    torch.save(model, "mnist.pt")

def create_loaders():
    """Create DataLoader instances for training and testing neural networks
    and for creating histogram plots

    returns:
        train_loader: DataLoader instance   - loader for training set
        test_loader: DataLoader instance    - loader for test set
    """

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader


def train_model(model, train_loader, optimizer, criterion, epoch_num, log_interval=200):
    """Train neural network model in standard way (as opposed to robust training)

    params:
        model: nn.Sequential instance   - NN model to be tested
        train_loader:                   - Training data for NN
        optimizer:                      - Optimizer for NN
        criterion:                      - Loss function
        epoch_num: int                  - Number of current epoch
        log_interval: int               - interval to print output
    """

    model.train()   # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(BATCH_SIZE, -1)

        optimizer.zero_grad()   # Zero gradient buffers
        output = model(data)    # Pass data through the network

        loss = criterion(output, target)    # Calculate loss
        loss.backward()     # Backpropagate
        optimizer.step()    # Update weights

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCross-Entropy Loss: {:.6f}'.format(
                epoch_num, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test_model(model, test_loader):
    """Test neural network model in standard way (as opposed to robust training)

    params:
        model: nn.Sequential instance   - NN model to be tested
        test_loader:                    - Test data for NN

    returns:
        testing classification accuracy
    """

    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(BATCH_SIZE, -1)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)               # Increment the total count
            correct += (predicted == labels).sum()     # Increment the correct count

    test_accuracy = 100 * correct.numpy() / float(total)
    print('Test Accuracy: %.3f %%\n' % test_accuracy)

    return test_accuracy


if __name__ == '__main__':
    main()
