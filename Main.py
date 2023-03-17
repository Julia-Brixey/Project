import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Util import *
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from ResNet import ResNet, Block, Bottleneck


def train():
    transform = transforms.Compose([transforms.Resize((64, 64, 64)), transforms.ToTensor()])

    dataset = LoadDataset("path/to/dataset", transform=transform)
    labels = dataset.labels

    train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # UTO center has 3 classes
    num_classes = 3
    model = ResNet(layers=[2, 2, 2, 2], block=Block, num_classes=num_classes)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    train()
