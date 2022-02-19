# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:50:10 2021

@author: wes_c
"""

import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_size=50):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #latent vector put through 1D conv
            nn.ConvTranspose1d(in_channels, 5, 3, stride=2, bias=False), #change these later
            nn.BatchNorm1d(5),
            nn.ReLU(True),
            #state-size: 5x(n_stride*(inp-1)+kernel_size) ---> for now it's 5x101
            nn.ConvTranspose1d(5, 3, 3, stride=1, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(True),
            #state-size: 3xprev_out+ker_size ---> for now it's 3x103
            nn.ConvTranspose1d(3, 1, 3, stride=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(True),
            #state-size:1xprev_out+ker_size ---> for now it's 1x105
            nn.Linear(105, out_size),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class GeneratorLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_states=16, out_size=50):
        super(GeneratorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_states = hidden_states
        self.l1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_states, batch_first=True, bidirectional=True)
        self.l1_1 = nn.ReLU(inplace=False)
        self.l1_2 = nn.LSTM(input_size=2*self.hidden_states, hidden_size=self.hidden_states, batch_first=True, bidirectional=True)
        self.l1_3 = nn.ReLU(inplace=False)
        self.l2 = nn.Linear(2*self.hidden_states, out_size)
        self.l3 = nn.Tanh()

    def forward(self, input):
        out1, _ = self.l1(input)
        pre2 = self.l1_1(out1)
        out2, _ = self.l1_2(pre2)
        pre3 = self.l1_3(out2)
        return self.l3(self.l2(pre3))

class DiscriminatorLSTM(nn.Module):
    def __init__(self, input_size=100, hidden_states=16):
        super(DiscriminatorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_states = hidden_states
        self.l1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_states, batch_first=True, bidirectional=True)
        self.l1_1 = nn.ReLU(inplace=False)
        self.l1_2 = nn.LSTM(input_size=2*self.hidden_states, hidden_size=self.hidden_states, batch_first=True, bidirectional=True)
        self.l1_3 = nn.ReLU(inplace=False)
        self.l2 = nn.Linear(2*self.hidden_states, 1)
        self.l3 = nn.Sigmoid()

    def forward(self, input):
        out1, _ = self.l1(input)
        pre2 = self.l1_1(out1)
        out2, _ = self.l1_2(pre2)
        pre3 = self.l1_3(out2)
        return self.l3(self.l2(pre3))

class DiscriminatorConv(nn.Module):
    def __init__(self, input_size=100):
        super(DiscriminatorConv, self).__init__()
        self.input_size = input_size
        self.main = nn.Sequential(
            nn.Conv1d(1, 5, 3, bias=False),
            #nx5xin-2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(5, 10, 3, bias=False),
            #nx10xin-4
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(10, 20, 3, bias=False),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(20, 1, 3, bias=False),
            nn.Linear(input_size - 8, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class NNFactory:
    def get_nn(kind, **kwargs):
        if kind == 'generator_conv':
            nn = Generator(**kwargs)
            nn.apply(weights_init)
        elif kind == 'generator_lstm':
            nn = GeneratorLSTM(**kwargs)
        elif kind == 'discriminator_conv':
            nn = DiscriminatorConv(**kwargs)
        elif kind == 'discriminator_lstm':
            nn = DiscriminatorLSTM(**kwargs)
        else:
            raise NotImplementedError(f'Neural network {kind} has not been implemented.')
        return nn