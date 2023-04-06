import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from torch.utils.data import DataLoader, SubsetRandomSampler
from Util import *
from ResNet import ResNet, Block, Bottleneck
from sklearn.metrics import confusion_matrix


def train():

    # OTU Dataset Labels: Healthy Controls (0), Major Depressive Disorder (1), Schizophrenia (2)

    # Ubuntu Remote server
    train_dataset = LoadDataset(r"data/OTU/Train")
    test_dataset = LoadDataset(r"data/OTU/Test")

    # check if gpu is available and set device to cuda else cpu
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device available is", device)

    train_loader = DataLoader(train_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # UTO center has 3 classes
    num_classes = 3
    model = ResNet(layers=[2, 2, 2, 2], block=Block, num_classes=num_classes)
    model = model.to(device)

    # Loss Function / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train
    num_epochs = 2
    epoch_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        print("Epoch running ", epoch + 1)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
        epoch_loss.append(running_loss/(i + 1))

    print(epoch_loss)
    correct = 0
    total = 0
    predicted_list = []
    labels_list = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_list.append(predicted.item())
            labels_list.append(labels.item())

    # Confusion Matrix
    classes = ('Healthy Controls', 'Major Depressive Disorder', 'Schizophrenia')
    cf_matrix = confusion_matrix(labels_list, predicted_list)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('home/jkbrixey/Project/Project/Models/OTU/confusion_matrix.png')

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    torch.save(model.state_dict(), "home/jkbrixey/Project/Project/Models/OTU/model.pth")

    # Loading epoch loss list
    with open("home/jkbrixey/Project/Project/Models/OTU/epoch_list.pkl", 'rb') as f:
        epoch_loss = pickle.load(f)
    # Saving epoch loss list as pickle file
    with open("home/jkbrixey/Project/Project/Models/OTU/epoch_list.pkl", 'wb') as f:
        pickle.dump(epoch_loss, f)


if __name__ == '__main__':
    train()
