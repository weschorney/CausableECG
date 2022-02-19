# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:16:56 2021

@author: wes_c
"""

from __future__ import print_function
import math
import pickle
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#import argparse
#import os
#import random
import torch
import torch.nn as nn
import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split

#REMEMBER TO SET DEVICE

def load_data():
    X = pickle.load(open('../data/x_data.pkl', 'rb'))
    y = pickle.load(open('../data/y_data.pkl', 'rb'))
    return X, y

def preprocess_data(X, y):
    #convert y to 1 (abnormal) or 0 (normal)
    y['pathological'] = y['diagnostic_superclass'].apply(lambda x: 1 - int('NORM' in x))
    #scaling the data
    X = (X - X.min())/(X.max() - X.min())
    return X, y

def sample_gan_data(X, y, first_lead=True, size=5050, test_size=0.1,
                    X_sample_name='X_sample.pkl',
                    y_sample_name='y_sample.pkl'):
    #start by taking small amount of healthy heartbeats to train GAN
    sample = np.random.choice(np.where(y['pathological'] == 0)[0], size=size, replace=False)
    #get corresponding heartbeats
    X_gan = X[sample, :, :]
    X_clf = np.delete(X, sample, axis=0)
    #reshape
    X_gan = X_gan.transpose((0, 2, 1))
    X_clf = X_clf.transpose((0, 2, 1))
    #get y corresp OFF BY ONE
    y_dropped = y['pathological'].reset_index(drop=True).values
    y_gan = y_dropped[sample]
    y_clf = np.delete(y_dropped, sample, axis=0)
    if first_lead:
        #focus on just first lead
        X_gan = X_gan[:, 0, :]
        X_clf = X_clf[:, 0, :]
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=test_size)
    #further split data to get sample data and write this
    X_train, X_sample, y_train, y_sample = train_test_split(X_train, y_train, test_size=test_size)
    #write sample data
    with open(X_sample_name, 'wb') as f:
        pickle.dump(X_sample, f)
    with open(y_sample_name, 'wb') as f:
        pickle.dump(y_sample, f)
    #finally downsample train data
    class1, class2 = y_train.sum(), y_train.shape[0] - y_train.sum()
    if class1 <= class2:
        to_sample = np.where(y_train == 0)[0]
        full = np.where(y_train == 1)[0]
        sample = np.random.choice(to_sample, size=class1)
    else:
        to_sample = np.where(y_train == 1)[0]
        full = np.where(y_train == 0)[0]
        sample = np.random.choice(to_sample, size=class2)
    down_sampled = np.concatenate((full, sample))
    X_train = X_train[down_sampled, ...]
    y_train = y_train[down_sampled]
    return X_gan, y_gan, X_train, X_test, y_train, y_test

def generate(ecg, generate_size=50, amble=25, step_size=10):
    #this function generates the pre and post-ambles for the GAN
    steps = math.floor((ecg.shape[0]-generate_size)/step_size)
    true_info = []
    ambles = []
    for step in range(steps):
        if step*step_size < amble:
            filler = np.zeros(amble - step*step_size)
            pre = ecg[:step*step_size]
            target = ecg[step*step_size:step*step_size+generate_size]
            post = ecg[step*step_size+generate_size:step*step_size+generate_size+amble]
            true_info.append(np.concatenate((filler, pre, target, post)))
            ambles.append(np.concatenate((filler, pre, post)))
        elif step*step_size > ecg.shape[0]-generate_size-amble:
            filler = np.zeros(step*step_size - ecg.shape[0]+generate_size+amble)
            pre = ecg[step*step_size-amble:step*step_size]
            target = ecg[step*step_size:step*step_size+generate_size]
            post = ecg[step*step_size+generate_size:step*step_size+generate_size+amble]
            true_info.append(np.concatenate((pre, target, post, filler)))
            ambles.append(np.concatenate((pre, post, filler)))
        else:
            pre = ecg[step*step_size-amble:step*step_size]
            target = ecg[step*step_size:step*step_size+generate_size]
            post = ecg[step*step_size+generate_size:step*step_size+generate_size+amble]
            true_info.append(np.concatenate((pre, target, post)))
            ambles.append(np.concatenate((pre, post)))
    return true_info, ambles

def get_training_data(X_gan, **kwargs):
    true_info = []
    ambles = []
    for i in range(X_gan.shape[0]):
        t, a = generate(X_gan[0, :], **kwargs)
        true_info += t
        ambles += a
    return true_info, ambles
    

def train_to_numpy(t2, a2, device='cpu'):
    t2, a2 = np.array(t2), np.array(a2)
    t2 = t2.reshape((t2.shape[0], 1, t2.shape[1]))
    a2 = a2.reshape((a2.shape[0], 1, a2.shape[1]))
    #randomly sample 10 for plotting
    fixed_eles = np.random.randint(0, t2.shape[0], size=(10))
    fixed_t2 = torch.Tensor(t2[fixed_eles, ...]).to(device)
    fixed_a = torch.Tensor(a2[fixed_eles, ...]).to(device)
    #delete sampled from t2 and a2
    t2 = np.delete(t2, fixed_eles, axis=0)
    a2 = np.delete(a2, fixed_eles, axis=0)
    return t2, a2, fixed_t2, fixed_a

def data_process(generate_size=50, amble=25, step_size=10, device='cpu'):
    X, y = preprocess_data(*load_data())
    X_gan, y_gan, X_train, X_test, y_train, y_test = sample_gan_data(X, y)
    t2, a2 = get_training_data(X_gan, generate_size=generate_size, amble=amble,
                               step_size=step_size)
    true_vals, ambles, fixed_t2, fixed_a2 = train_to_numpy(t2, a2, device=device)
    return true_vals, ambles, fixed_t2, fixed_a2, X_train, X_test, y_train, y_test

def animate(frame, ecg_list, line, axes):
    x_data = list(range(50))
    for i in range(10):
        line[i].set_data(x_data, ecg_list[frame][i,0,:])
    return line

def create_animation(ecg_list, func=animate, animation_savename=None):
    #try to create a grid animation like follows:
    fig = plt.figure(figsize=(10, 4))
    x_lim = (0, 50)
    y_lim = (0, 1)
    ax1 = plt.subplot(5,2,1)
    ax2 = plt.subplot(5,2,2)
    ax3 = plt.subplot(5,2,3)
    ax4 = plt.subplot(5,2,4)
    ax5 = plt.subplot(5,2,5)
    ax6 = plt.subplot(5,2,6)
    ax7 = plt.subplot(5,2,7)
    ax8 = plt.subplot(5,2,8)
    ax9 = plt.subplot(5,2,9)
    ax10 = plt.subplot(5,2,10)
    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    for axis in ax:
        axis.set_ylim(y_lim)
        axis.set_xlim(x_lim)
    line1, = ax1.plot([], [], lw=2)
    line2, = ax2.plot([], [], lw=2)
    line3, = ax3.plot([], [], lw=2)
    line4, = ax4.plot([], [], lw=2)
    line5, = ax5.plot([], [], lw=2)
    line6, = ax6.plot([], [], lw=2)
    line7, = ax7.plot([], [], lw=2)
    line8, = ax8.plot([], [], lw=2)
    line9, = ax9.plot([], [], lw=2)
    line10, = ax10.plot([], [], lw=2)
    line = [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10]
    ani = animation.FuncAnimation(fig, func, frames=len(ecg_list), fargs=(ecg_list, line, ax1), blit=True, repeat=False)
    if isinstance(animation_savename, str):
        writergif = animation.PillowWriter(fps=2)
        ani.save(animation_savename, writer=writergif)
    return ani

def training_loop(netD2, netG2, true_vals, ambles, fixed_a2, num_epochs=100,
                  batch_size=64, device='cpu', num_workers=0, ngpu=1,
                  learning_rate=1e-5, beta1=0.45, beta2=0.997,
                  verbose=True, fig_savename=None,
                  animation_savename=None):
    for epoch in range(num_epochs):
        ecg_list2 = []
        G_losses = []
        D_losses = []
        iters = 0
        if device == 'cpu':
            d = torch.utils.data.TensorDataset(torch.FloatTensor(true_vals), torch.FloatTensor(ambles))
            dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else:
            d = torch.utils.data.TensorDataset(torch.cuda.FloatTensor(true_vals), torch.cuda.FloatTensor(ambles))
            dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        torch.autograd.set_detect_anomaly(True)
        
        #loss function
        criterion = nn.BCELoss()
        real_label = 1.
        fake_label = 0.

        optimizerD2 = optim.Adam(netD2.parameters(), lr=learning_rate, betas=(beta1, beta2))
        optimizerG2 = optim.Adam(netG2.parameters(), lr=learning_rate, betas=(beta1, beta2))

        for i, data in enumerate(dataloader, 0):
            #Train D with all-real batch at first
            netD2.zero_grad()
            #format batch
            real = data[0].to(device)
            #real = real.reshape([128, 1, 100])
            #print(real.shape)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            #print(label.shape)
            #forward pass
            output = netD2(real).view(-1)
            #print(output.shape)
            #loss on all real
            errD_real = criterion(output, label)
            #get grads backwards
            errD_real.backward()
            D_x = output.mean().item()

            #train with all-fake batch
            noise = data[1].to(device)
            #print(noise.shape)
            #noise = noise.reshape([noise.shape[0], 1, noise.shape[1]])
            fake_middle = netG2(noise)
            #combine ambles with generated
            preamble = noise[..., :25]
            postamble = noise[..., 25:]
            fake = torch.cat((preamble, fake_middle, postamble), axis=2)
            label.fill_(fake_label)
            #classify with D

            output = netD2(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            #get grads, accumulate with prev grads
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD2.step()

            #update G
            netG2.zero_grad()
            label.fill_(real_label) #fake labels real for gen
            #pass through D
            output = netD2(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG2.step()

            #output training stats
            if i % 50 == 0 and verbose:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            #output g imgs
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake2 = netG2(fixed_a2).detach().cpu()
                ecg_list2.append(vutils.make_grid(fake2, padding=2, normalize=True))

            iters += 1
    #plot results
    if isinstance(fig_savename, str):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(fig_savename, dpi=500)

    if isinstance(animation_savename, str):
        create_animation(ecg_list2, animation_savename=animation_savename)
    return netG2, netD2
