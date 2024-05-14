# Deric Shaffer
# CS487 - HW7
# Due Date - Apr. 26th, 2024

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import torch.nn as nn

# to fix the ssl verify failed error I keep running into with the mnist dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train(model, num_epochs, train_dl, valid_dl):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs

    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()

        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()
        
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0) 
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float() 
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch + 1} Accuracy: {accuracy_hist_train[epoch]:.4f}')
        print(f'Valid Accuracy: {accuracy_hist_valid[epoch]:.4f}\n')
    
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

def main():
    # load and preprocess MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    mnist = datasets.MNIST(root='./hw7_data', train=True, transform=transform, download=False)

    mnist_valid = Subset(mnist, torch.arange(10_000))
    mnist_train = Subset(mnist, torch.arange(10_000, len(mnist)))
    mnist_test = datasets.MNIST(root='./hw7_data', train=False, transform=transform, download=False)

    # construct data loader
    batch_size = 64
    torch.manual_seed(1)

    train_dl = DataLoader(mnist_train, batch_size, shuffle=True)
    valid_dl = DataLoader(mnist_valid, batch_size, shuffle=False)

    # construct model
    model = nn.Sequential()

    # initial parameters
    model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding='valid'))
    model.add_module('relu', nn.ReLU())
    model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
    model.add_module('conv2', nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=3, padding='valid'))
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool2d(kernel_size=4, stride=4))

    # modified test parameters
    # model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=4, kernel_size=7, stride=1, padding='valid'))
    # model.add_module('relu', nn.ReLU())
    # model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
    # model.add_module('conv2', nn.Conv2d(in_channels=4, out_channels=2, kernel_size=7, stride=1, padding='valid'))
    # model.add_module('relu2', nn.ReLU())
    # model.add_module('pool2', nn.MaxPool2d(kernel_size=4, stride=4))


    model.add_module('flatten', nn.Flatten())
    

    model.add_module('fc1', nn.Linear(2, 10))
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.5))

    # training
    torch.manual_seed(1)
    num_epochs = 20

    train_start = time.time()
    hist = train(model, num_epochs, train_dl, valid_dl)
    train_end = time.time()

    print(f'\nTraining Run Time: {train_end - train_start} seconds')

    # testing
    pred = model(mnist_test.data.unsqueeze(1) / 255)
    is_correct = (torch.argmax(pred, dim=1) == mnist_test.targets).float()
    print(f'\nTest Accuracy: {is_correct.mean():.4f}')

    # plot training
    x_arr = np.arange(len(hist[0])) + 1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist[0], '-o', label='Train Loss')
    ax.plot(x_arr, hist[1], '--<', label='Validation Loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist[2], '-o', label='Train Accuracy')
    ax.plot(x_arr, hist[3], '--<', label='Validation Accuracy')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    ax.legend(fontsize=15)

    plt.show()

    # plot testing
    fig = plt.figure(figsize=(12, 4))
    for i in range(12):
        ax = fig.add_subplot(2, 6, i+1)
        ax.set_xticks([])
        ax.set_yticks([])

        img = mnist_test[i][0][0, :, :]
        pred = model(img.unsqueeze(0).unsqueeze(1))
        y_pred = torch.argmax(pred)
        ax.imshow(img, cmap='gray_r')
        ax.text(0.9, 0.1, y_pred.item(), size=15, color='blue', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.show()

if __name__ == '__main__':
    main()