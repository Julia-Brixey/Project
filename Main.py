import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from Util import *
from torchvision import transforms
from torchvision.datasets import ImageFolder
from ResNet import ResNet, Block, Bottleneck


def train():

    # Julia's Mac
    # dataset = LoadDataset(r"/Users/juliabrixey/Desktop/Research/Honors Thesis/Project/data/OTU")

    # Julia's PC
    dataset = LoadDataset(r"C:\Users\jkbrixey\Desktop\Honors Thesis\Project\data\OTU")

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))

    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sample = SubsetRandomSampler(train_indices)
    test_sample = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=10, sampler=train_sample)
    test_loader = DataLoader(dataset, batch_size=10, shuffle=False, sampler=test_sample)

    # UTO center has 3 classes
    num_classes = 3
    model = ResNet(layers=[2, 2, 2, 2], block=Block, num_classes=num_classes)

    # Loss Function / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train
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
