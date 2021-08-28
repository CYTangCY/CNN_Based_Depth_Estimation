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
import cv2
import matplotlib.pyplot as plt
import cmapy
# import torch.onnx

wall = arsenal.makewall(50, 645.24, 640, 192)
K = np.array([[645.24, 0.0, 640/2],
			 [0.0, 645.24, 192/2],
			 [0.0, 0.0, 1.0]])
hsv = np.zeros((192, 640, 3))
hsv[...,1] = 255
hsv0 = np.zeros((192, 640, 3))
hsv0[...,1] = 255
device = torch.device("cuda")
model = DepthAns(input_size=(192, 640)).to(device)
model = torch.load('./CNNfinalModelTest_0717_math_depth2.pt', map_location=device)
model.eval()

test_Data_0 = np.load('B:/Code/0625_vision/0624_math_test/DATASET/cropdata_78.npz')
FrameNumber = 0
with torch.no_grad():
    imgs, test_x1, test_x2, test_y0 = np.array(test_Data_0['a']), torch.tensor(test_Data_0['b']), torch.tensor(test_Data_0['c']), torch.tensor(test_Data_0['d'])
    print(len(test_x1))
    for Frame in range(len(test_y0)):
        img = imgs[FrameNumber]
        # print(img.shape)
        flows = test_x1[FrameNumber].float()
        imus = test_x2[FrameNumber]
        target = test_y0[FrameNumber].float().to(device)
        imuT = imus[3:6]
        imuR = imus[6:]

        Translate = imuT
        tx, ty, tz = Translate[1], Translate[2], -Translate[0]
        tx, ty, tz = tx*0.05, ty*0.05, tz*0.05
        Rotate = imuR
        rx, ry, rz = Rotate[1], Rotate[2], -Rotate[0]
        rx, ry, rz = rx*0.1038, ry*0.1038, rz*0.1038
        Rh, Ry, Rx, RI = arsenal.rotationinradius(rx,ry,rz)
        translation, translation_O = arsenal.translationinmeters(tx,ty,tz,0,0,0)
        translation = np.array(translation).reshape(3, 1)
        Rt_rotate = np.concatenate((Rh @ Ry @ Rx, translation_O), axis=1)
        Rt_translate = np.concatenate((RI, translation), axis=1)

        idealRotateFlow = arsenal.RYP(wall, K, Rt_rotate,192,640)
        depthIRF = idealRotateFlow
        idealRotateFlow = torch.tensor(idealRotateFlow)
        idealRotateFlow = idealRotateFlow.permute(2, 0, 1)

        ITFVec, idealTranslationFlow = arsenal.xyz(wall, K, Rt_translate,192,640)
        ITFV = idealTranslationFlow
        idealTranslationFlow = torch.tensor(idealTranslationFlow)
        idealTranslationFlow = idealTranslationFlow.permute(2, 0, 1)


        flowsV = flows
        flows = np.array(flows)
        flows = np.transpose(flows, (1, 2, 0))
        NRF = flows-depthIRF
        depthS = np.zeros((192,640))
        depthS = arsenal.depthmodule(wall, 8, depthS, 50, ITFVec, NRF)
        depthS *= 256/80
        depthS = 256 - depthS
        depthS[depthS < 130] = 130
        IOFV = depthS
        IOF = torch.tensor(depthS)
        IOF = IOF.unsqueeze(0)
        IOF = IOF.unsqueeze(0)
        IOF = IOF.float().to(device)
        imginput = torch.tensor(img)
        imginput = imginput.unsqueeze(0)
        imginput = imginput.unsqueeze(0)
        imginput = imginput.float().to(device)
        x1 = imginput
        x2 = IOF

        input1 = (x1, x2)
        pre_output = model(x1, x2)
        pre_output = torch.cuda.FloatTensor(pre_output.float())

        pre_depth = pre_output.to('cpu')
        test_target = target.to('cpu')

        pre_depth = pre_depth.detach().numpy()
        test_target = np.array(test_target)

        test_pre_depth = pre_depth[0]
        test_target = test_target[0]
        test_pre_depth = np.resize(pre_depth, (192, 640))
        test_target = np.resize(test_target, (192,640))

        test_pre_depth = test_pre_depth
        test_pre_depth = test_pre_depth.astype(np.uint8)
        test_pre_depth = cv2.applyColorMap(test_pre_depth, cmapy.cmap('nipy_spectral'))
        test_target[test_target > 256] = 256
        test_target = test_target.astype(np.uint8)
        test_target = cv2.applyColorMap(test_target, cmapy.cmap('nipy_spectral'))

        #----------------------Flow---------------------
        flowsV = flowsV.detach().numpy()
        flowsV = np.transpose(flowsV,(1, 2, 0))
        mag, ang = cv2.cartToPolar(flowsV[...,0], flowsV[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv = hsv.astype(np.uint8)
        hsv[...,2] = 5 * mag
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('flow.',bgr)
        #----------------------Flow---------------------
        #ITFV----------------
        mag0, ang0 = cv2.cartToPolar(ITFV[...,0], ITFV[...,1])
        hsv0[...,0] = ang0*180/np.pi/2
        hsv0[...,2] = cv2.normalize(mag0,None,0,255,cv2.NORM_MINMAX)
        hsv0 = hsv0.astype(np.uint8)
        hsv0[...,2] = 50 * mag0
        bgr0 = cv2.cvtColor(hsv0,cv2.COLOR_HSV2BGR)
        cv2.imshow('ITFV.',bgr0)
        #ITFV----------------

        IOFV = IOFV.astype(np.uint8)
        IOFV = cv2.applyColorMap(IOFV, cmapy.cmap('nipy_spectral'))
        cv2.imshow('depth_ori.',IOFV)
        img = img.astype(np.uint8)
        cv2.imshow('img', img)
        

        cv2.imshow("test_pre_depth_0707", test_pre_depth)
        cv2.imshow("test_target_0707", test_target)
        cv2.waitKey(1)
        time.sleep(0.02)
        FrameNumber += 1

# torch.onnx.export(model, input1, 'modelgraph_test.onnx') 



