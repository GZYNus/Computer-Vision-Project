#!/usr/bin/env python
"""
Description: create the basic architecture of DenseNet
Email: gzynus@gmail.com
Arthur: Zongyi Guo
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class bn_rl_conv(nn.Module):
    """
    :param input_channel: input channel from input tensor
    :param f: number of Convolutional filters
    :param k: kernal size
    :param s: padding stride

    :return conv(input)
    """
    def __init__(self, input_channel, f, k=1, s=1):
        super(bn_rl_conv,self).__init__()
        self.pre = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Conv2d(input_channel, f, kernel_size=k, stride=s, padding=0)


        self.pre2 = nn.Sequential(
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d(f,f//4,kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        x = self.pre(input)
        x = self.conv(x)
        x = self.pre2(x)
        x = self.conv2(x)
        return x


class dense_block(nn.Module):
    """
    :param input_channel: input channel from input tensor
    :param f: number of Convolutional filters
    :return concatenated dense

    :return tensor after passing through dense block
    """
    def __init__(self, input_channel, f, round):
        super(dense_block,self).__init__()
        self.round = round
        self.dense_cell = nn.ModuleList()
        for i in range(round):
            self.dense_cell.append(bn_rl_conv(input_channel+i*f, 4*f))

    def forward(self, input):
        for i in range(self.round):
            y = self.dense_cell[i](input)
            input = torch.cat([input, y],dim=1)  # (n,channel,width,length)
        return input


class transition_layer(nn.Module):
    """
    :param input_channel: input channel from input tensor

    :return tensor after passing through transition layer
    """
    def __init__(self, input_channel):
        super(transition_layer,self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel, input_channel//2, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, input):
        return self.transition(input)


class denseNet121(nn.Module):
    """
    Create the backbone for densenet121
    :param f: growth rate
    """
    def __init__(self, f=32):
        super(denseNet121,self).__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        block_config = [6, 12, 24, 16]
        self.dense_1 = dense_block(64,f,block_config[0])
        self.dense_2 = dense_block(128,f,block_config[1])
        self.dense_3 = dense_block(256,f,block_config[2])
        self.dense_4 = dense_block(512,f,block_config[3])

        self.trans1 = transition_layer(256)
        self.trans2 = transition_layer(512)
        self.trans3 = transition_layer(1024)

        self.glbpooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(1024, 1000)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.start_conv(input)

        x = self.dense_1(x)
        x = self.trans1(x)

        x = self.dense_2(x)
        x = self.trans2(x)

        x = self.dense_3(x)
        x = self.trans3(x)

        x = self.dense_4(x)

        x = self.glbpooling(x)

        x = x.view(x.shape[0],1024)

        out = self.softmax(x)

        return out


if __name__=='__main__':
    model = denseNet121()
    summary(model,input_size=(3,224,224))










