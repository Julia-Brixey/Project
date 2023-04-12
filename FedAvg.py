import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from Util import *
from ResNet import ResNet, Block, Bottleneck
from sklearn.metrics import confusion_matrix


def fedavg():

    train_dataset = LoadDataset(r"C:\Users\jkbrixey\Desktop\Honors Thesis\Project\data\KUT\train")

    # check if gpu is available and set device to cuda else cpu
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device available is", device)

    train_loader = DataLoader(train_dataset, batch_size=2)

    x_train = []
    y_train = []

    for batch in train_loader:
        data, labels = batch

        x_train.extend(data)
        y_train.extend(labels)

    number_of_samples = 6
    learning_rate = 0.001
    numEpoch = 10
    batch_size = 2
    momentum = 0.9
    num_classes = 3

    train_amount = 12
    valid_amount = 0
    test_amount = 0
    print_amount = 3

    label_dict_train = split_and_shuffle_labels(y_data=y_train, seed=1, amount=train_amount)
    sample_dict_train = get_iid_subsamples_indices(label_dict=label_dict_train, number_of_samples=number_of_samples,
                                                   amount=train_amount)
    x_train_dict, y_train_dict = create_iid_subsamples(sample_dict=sample_dict_train, x_data=x_train, y_data=y_train,
                                                       x_name="x_train", y_name="y_train")

    main_model = ResNet(layers=[2, 2, 2, 2], block=Block, num_classes=num_classes)
    main_model.to(device)
    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=0.9)
    main_criterion = nn.CrossEntropyLoss()

    model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(number_of_samples, num_classes,
                                                                                       learning_rate, momentum)

    name_of_x_train_sets = list(x_train_dict.keys())
    name_of_y_train_sets = list(y_train_dict.keys())

    name_of_models = list(model_dict.keys())
    name_of_optimizers = list(optimizer_dict.keys())
    name_of_criterions = list(criterion_dict.keys())

    start_train_end_node_process_print_some(number_of_samples, print_amount, model_dict, name_of_models, criterion_dict,
                                            name_of_criterions, optimizer_dict, name_of_optimizers, numEpoch, x_train_dict,
                                            name_of_x_train_sets, y_train_dict, name_of_y_train_sets)


def split_and_shuffle_labels(y_data, seed, amount):
    y_data = pd.DataFrame(y_data, columns=["labels"])
    y_data["i"] = np.arange(len(y_data))
    label_dict = dict()
    for i in range(3):
        var_name = "label" + str(i)
        label_info = y_data[y_data["labels"] == i]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = label_info[0:amount]
        label_info = pd.DataFrame(label_info, columns=["labels", "i"])
        label_dict.update({var_name: label_info})
    return label_dict


def get_iid_subsamples_indices(label_dict, number_of_samples, amount):
    sample_dict = dict()
    batch_size = int(math.floor(amount/number_of_samples))
    for i in range(number_of_samples):
        sample_name = "sample"+str(i)
        dumb = pd.DataFrame()
        for j in range(3):
            label_name = str("label")+str(j)
            a = label_dict[label_name][i*batch_size:(i+1)*batch_size]
            dumb = pd.concat([dumb, a], axis=0)
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})
    return sample_dict


def create_iid_subsamples(sample_dict, x_data, y_data, x_name, y_name):
    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(len(sample_dict)):
        xname = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)

        indices = np.sort(np.array(sample_dict[sample_name]["i"]))

        x_info = [x_data[i] for i in indices]
        x_data_dict.update({xname: x_info})

        y_info = [y_data[i] for i in indices]
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


def create_model_optimizer_criterion_dict(number_of_samples, num_classes, learning_rate, momentum):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        model_info = ResNet(layers=[2, 2, 2, 2], block=Block, num_classes=num_classes)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def start_train_end_node_process_print_some(number_of_samples, print_amount, model_dict, name_of_models, criterion_dict,
                                            name_of_criterions, optimizer_dict, name_of_optimizers, numEpoch, x_train_dict,
                                            name_of_x_train_sets, y_train_dict, name_of_y_train_sets):
    for i in range(number_of_samples):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        if i < print_amount:
            print("Subset", i)

        for epoch in range(numEpoch):

            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer)
            test_loss, test_accuracy = validation(model, test_dl, criterion)

            if i < print_amount:
                print("epoch: {:3.0f}".format(epoch + 1) + " | train accuracy: {:7.5f}".format(
                    train_accuracy) + " | test accuracy: {:7.5f}".format(test_accuracy))


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def validation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return test_loss, correct


if __name__ == '__main__':
    fedavg()
