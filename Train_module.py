#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob, os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import os
import time
import torch
import argparse
from predict_depth_R_test import DepthAns
import arsenal
import torch.nn.functional as F
import torch.nn

#-----Loading Data-------------------------------------------------------------

train_dir_num = 40
DatasetList = [] 
x0, x4, y0 = torch.zeros((0, 1, 192, 640)), torch.zeros((0, 1, 192, 640)), torch.zeros((0, 1, 192, 640))
for file in glob.glob("../4TB/DATASET_NEW_NEW/*.npz"):
    DatasetList.append(file)
print(len(DatasetList))
print('Data_load')
train_dir = DatasetList[0:train_dir_num]
print(len(train_dir))
wall = arsenal.makewall(80, 645.24, 640, 192)
K = np.array([[645.24, 0.0, 640/2],
			 [0.0, 645.24, 192/2],
			 [0.0, 0.0, 1.0]])

for i in range(train_dir_num):

    ImuNumber = 0
    IOF = torch.zeros((0, 1, 192, 640))
    Data = np.load(DatasetList[i])
    x0_new, x1_new, x2_new, y0_new = torch.tensor(Data['a']), torch.tensor(Data['b']), np.array(Data['c']), torch.tensor(Data['d'])
    x0_new = torch.unsqueeze(x0_new, 1)
    x2_new_new = x2_new[:, 3:6]
    x3_new_new = x2_new[:, 6:]

    for TT in range(len(x2_new)):
        Translate = x2_new_new[ImuNumber]
        tx, ty, tz = Translate[1], Translate[2], -Translate[0]
        tx, ty, tz = tx*0.05, ty*0.05, tz*0.05
        Rotate = x3_new_new[ImuNumber]
        rx, ry, rz = Rotate[1], Rotate[2], -Rotate[0]
        rx, ry, rz = rx*0.1, ry*0.1, rz*0.1
        Rh, Ry, Rx, RI = arsenal.rotationinradius(rx,ry,rz)
        translation, translation_O = arsenal.translationinmeters(tx,ty,tz,rx,ry,rz)
        translation = np.array(translation).reshape(3, 1)
        Rt_rotate = np.concatenate((Rh @ Ry @ Rx, translation_O), axis=1)
        Rt_translate = np.concatenate((RI, translation), axis=1)

        idealRotateFlow = arsenal.RYP(wall, K, Rt_rotate,192,640)
        depthIRF = idealRotateFlow
        idealRotateFlow = torch.tensor(idealRotateFlow)
        idealRotateFlow = idealRotateFlow.permute(2, 0, 1)
        idealRotateFlow = idealRotateFlow.unsqueeze(0)
        ITFVec, idealTranslationFlow = arsenal.xyz(wall, K, Rt_translate,192,640)
        idealTranslationFlow = torch.tensor(idealTranslationFlow)
        idealTranslationFlow = idealTranslationFlow.permute(2, 0, 1)
        idealTranslationFlow = idealTranslationFlow.unsqueeze(0)
        
        #FLOW
        FLOW = x1_new[ImuNumber]
        FLOW = np.array(FLOW)
        FLOW = np.transpose(FLOW, (1, 2, 0))
        NRF = FLOW-depthIRF
        depth = np.zeros((192,640))
        depth = arsenal.depthmodule(wall, 8, depth, 80, ITFVec, NRF)
        depth *= 256/80
        depth = 256 - depth
        depth[depth < 130] = 130

        IOFF = depth
        IOFF = torch.tensor(IOFF)
        IOFF = IOFF.unsqueeze(0)
        IOFF = IOFF.unsqueeze(0)
        IOF = torch.cat((IOF, IOFF))
        ImuNumber += 1

    x0, x4, y0 = torch.cat((x0, x0_new)), torch.cat((x4, IOF)), torch.cat((y0, y0_new))

batch_size = 8
'''Testing Data'''
my_train = data_utils.TensorDataset(x0, x4, y0)
train_loader = torch.utils.data.DataLoader(my_train, batch_size=batch_size)

#-----Recording Accuracy-------------------------------------------------------
num_epochs = 500
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
learning_rate = 0.001
mu = 0.85
# ---init---
device = torch.device("cuda:1")
model = DepthAns(input_size=(192, 640)).to(device)
model.train()
#-----Setting loss function and optimizer--------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# lr_decay = PolynomialLRDecay(optimizer, num_epochs, learning_rate * 1e-2)
criterion = nn.L1Loss()
sid = arsenal.SID('kitti')
#----Training------------------------------------------------------------------
print('begin')
best_acc = 0
print (batch_size,len(my_train))
lossvalue = []

for epoch in range(num_epochs):
    running_loss = 0
    correct = 0
    total = 0
    error = 0
    start_time = time.time()
    if epoch == 200:
        learning_rate = 0.0007
    if epoch == 300:
        learning_rate = 0.0005
    if epoch == 400:
        learning_rate = 0.0003
    if epoch == 500:
        learning_rate = 0.0001

    print('----------')
    print('process%d/%d'%(epoch, num_epochs))
    print('lr',learning_rate)
    model.train()
    
    for i, (x0, x4, y0) in enumerate(train_loader):

        flows = x0.float().to(device)
        depth_input = x4.float().to(device)
        target = y0.float().to(device)
        #---------------tensor board----------------------------
        
        pre_output = model(flows, depth_input)
        pre_output = torch.cuda.FloatTensor(pre_output.float())
        target = torch.flatten(target)
        pre_output = torch.flatten(pre_output)
        loss = criterion(pre_output, target)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print('E_Loss',running_loss/i)
    lossvalue.append(running_loss/i)
    correct = 0
    total = 0
    error = 0
    np.save('lossvalue_0717_math_depth2.npy',lossvalue)
    # torch.save(model, './CNNprocessModelTest_0707.pt')
    if epoch == 200:
        torch.save(model, './CNN200EpochsModelTest_0717_math_depth2.pt')
    if epoch == 400:
        torch.save(model, './CNN400EpochsModelTest_0717_math_depth2.pt')
    print('----------')
torch.save(model, './CNNfinalModelTest_0717_math_depth2.pt')

