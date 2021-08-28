#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
import arsenal
import torch.nn.functional as F

class flowNN(nn.Module):

    def __init__(self):
        super(flowNN, self).__init__()
        self.flowNet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x1):

        x1 = self.flowNet(x1)

        return x1

class flowdecode(nn.Module):
    def __init__(self):
        super(flowdecode, self).__init__()
        self.deConv1 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.deConv2 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0, bias=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.deConv3 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, bias=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.deConv4 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0, bias=False)
        self.relu4 = nn.ReLU(inplace=False)
        self.deConv5 = nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, bias=False)
        self.relu5 = nn.ReLU(inplace=False)
        self.deConv6 = nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0, bias=False)
        self.relu6 = nn.ReLU(inplace=False)

    def forward(self, x1_0):

        x1_1 = self.relu1(self.deConv1(x1_0))
        x1_2 = self.relu2(self.deConv2(x1_1))
        x1_3 = self.relu3(self.deConv3(x1_2))
        x1_4 = self.relu4(self.deConv4(x1_3))
        x1_5 = self.relu5(self.deConv5(x1_4))

        return x1_0, x1_1, x1_2, x1_3, x1_4, x1_5


class VGG11NN(nn.Module):

    def __init__(self):
        super(VGG11NN, self).__init__()
        self.vgg11 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x2):

        x2 = self.vgg11(x2)

        return x2

class VGGdecode(nn.Module):
    def __init__(self):
        super(VGGdecode, self).__init__()
        self.deConv1 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.deConv2 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0, bias=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.deConv3 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, bias=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.deConv4 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0, bias=False)
        self.relu4 = nn.ReLU(inplace=False)
        self.deConv5 = nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, bias=False)
        self.relu5 = nn.ReLU(inplace=False)
        self.deConv6 = nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0, bias=False)
        self.relu6 = nn.ReLU(inplace=False)

    def forward(self, x3):

        x3_1 = self.relu1(self.deConv1(x3))
        x3_2 = self.relu2(self.deConv2(x3_1))
        x3_3 = self.relu3(self.deConv3(x3_2))
        x3_4 = self.relu4(self.deConv4(x3_3))
        x3_5 = self.relu5(self.deConv5(x3_4))

        return x3, x3_1, x3_2, x3_3, x3_4, x3_5

class depthNN(nn.Module):
    def __init__(self, size):
        super(depthNN, self).__init__()
        self.size = size
        self.deConv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.deConv2 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0, bias=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.deConv3 = nn.ConvTranspose2d(768, 256, 2, stride=2, padding=0, bias=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.deConv4 = nn.ConvTranspose2d(384, 128, 2, stride=2, padding=0, bias=False)
        self.relu4 = nn.ReLU(inplace=False)
        self.deConv5 = nn.ConvTranspose2d(192, 64, 2, stride=2, padding=0, bias=False)
        self.relu5 = nn.ReLU(inplace=False)
        self.deConv6 = nn.ConvTranspose2d(96, 32, 1, stride=1, padding=0, bias=False)
        self.relu6 = nn.ReLU(inplace=False)
        self.deConv7 = nn.ConvTranspose2d(32, 16, 1, stride=1, padding=0, bias=False)
        self.relu7 = nn.ReLU(inplace=False)
        self.deConv8 = nn.ConvTranspose2d(16, 1, 1, stride=1, padding=0, bias=False)
        self.relu8 = nn.ReLU(inplace=False)

    def forward(self, x1_0, x3, x1_1, x3_1, x1_2, x3_2, x1_3, x3_3, x1_4, x3_4, x1_5, x3_5):

        x4 = torch.cat((x1_0, x3), 1)
        x4 = self.relu1(self.deConv1(x4))
        x4 = torch.cat((x4, x1_1, x3_1), 1) 
        x4 = self.relu2(self.deConv2(x4))
        x4 = torch.cat((x4, x1_2, x3_2), 1)
        x4 = self.relu3(self.deConv3(x4))
        x4 = torch.cat((x4, x1_3, x3_3), 1)
        x4 = self.relu4(self.deConv4(x4))
        x4 = torch.cat((x4, x1_4, x3_4), 1)
        x4 = self.relu5(self.deConv5(x4))
        x4 = torch.cat((x4, x1_5, x3_5), 1)
        x4 = self.relu6(self.deConv6(x4))
        x4 = self.relu7(self.deConv7(x4))
        x4 = self.deConv8(x4)

        return x4

