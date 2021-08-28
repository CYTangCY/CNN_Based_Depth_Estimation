#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from NNmodule_R_test import *

class DepthAns(nn.Module):

    def __init__(self, input_size=(192, 640)):
        super().__init__()
        self.flow = flowNN()
        self.vgg11 = VGG11NN()
        self.depth = depthNN(size = input_size)
        self.vggdecode = VGGdecode()
        self.flowdecode = flowdecode()

    def forward(self, flow_input, depth_input):

        x1 = self.flow(flow_input)
        x1_0, x1_1, x1_2, x1_3, x1_4, x1_5 = self.flowdecode(x1) 
        x2 = self.vgg11(depth_input)
        x3, x3_1, x3_2, x3_3, x3_4, x3_5 = self.vggdecode(x2)
        x4 = self.depth(x1_0, x3, x1_1, x3_1, x1_2, x3_2, x1_3, x3_3, x1_4, x3_4, x1_5, x3_5)

        return x4
