import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from Util import *
from ResNet import ResNet, Block, Bottleneck


def train():

    # OTU Dataset Labels: Healthy Controls (0), Major Depressive Disorder (1), Schizophrenia (2)

    # Julia's Mac
    # dataset = LoadDataset(r"/Users/juliabrixey/Desktop/Research/Honors Thesis/Project/data/OTU")

    # Julia's PC
    # dataset = LoadDataset(r"C:\Users\jkbrixey\Desktop\Honors Thesis\Project\data\OTU")

    # Ubuntu Remote server
    dataset = LoadDataset(r"/home/jkbrixey/Project/Project/data/OTU")

    # check if gpu is available and set device to cuda else cpu
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device available is", device)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))

    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sample = SubsetRandomSampler(train_indices)
    test_sample = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=5, sampler=train_sample)
    test_loader = DataLoader(dataset, batch_size=5, shuffle=False, sampler=test_sample)

    # UTO center has 3 classes
    num_classes = 3
    model = ResNet(layers=[2, 2, 2, 2], block=Block, num_classes=num_classes)
    model = model.to(device)

    # Loss Function / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optimizer.to(device)

    # Train
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        print("Epoch running ", epoch)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
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
