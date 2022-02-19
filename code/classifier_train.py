# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:28:36 2021

@author: wes_c
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

import nn_define

def train_classifier(X_train, y_train, X_test, y_test, device='cpu',
                     disc_architecture='discriminator_lstm',
                     disc_kwargs={'input_size':1000, 'hidden_states':50},
                     lr=1e-5, epochs=100, batch_size=32, beta1=0.5,
                     beta2=0.997, model_savename='ecg_clf.pt',
                     plot_savename=None):
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    if device == 'cpu':
        d = data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        d2 = data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    else:
        dx = torch.cuda.FloatTensor(X_train)
        dy = torch.cuda.FloatTensor(y_train)
        d = data.TensorDataset(dx, dy)
        d2 = data.TensorDataset(torch.cuda.FloatTensor(X_test),
                                torch.cuda.FloatTensor(y_test))
    trainloader = data.DataLoader(d, batch_size=batch_size, shuffle=True,
                                      num_workers=0)
    testloader = data.DataLoader(d2, batch_size=batch_size, shuffle=True,
                                      num_workers=0)
    clf = nn_define.NNFactory.get_nn(disc_architecture, **disc_kwargs).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(clf.parameters(), lr=lr, betas=(beta1, beta2))
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, datas in enumerate(trainloader, 0):
            inputs, labels = datas
            labels = labels.reshape((-1, 1, 1))
            optimizer.zero_grad()
            #forward, backward, then optimize
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #stats
            running_loss += loss.item()
            if i % 25 == 24:
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / 25))
                losses.append(running_loss)
                running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for datas in testloader:
            images, labels = datas
            outputs = clf(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Model accuracy: %.4f'  %(100*correct/total))
    if isinstance(model_savename, str):
        torch.save(clf.state_dict(), model_savename)
    if plot_savename:
        plt.plot(losses)
        plt.title('Classifier Loss: Training')
        plt.savefig(plot_savename, dpi=500)
    return
