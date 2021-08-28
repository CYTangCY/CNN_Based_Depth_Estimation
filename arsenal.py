#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA

def conv3x3(_input, _output, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(_input, _output, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def linear_relu(_input, _output):

        return nn.Sequential(
            nn.Linear(_input, _output),
            nn.ReLU(inplace=True),
        )

def conv_relu(_input, _output, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(_input, _output, kernel_size=kernel_size, stride=stride, bias=False),
        nn.BatchNorm2d(_output),
        nn.ReLU(inplace=True),
    )

def makewall(WallDistance0, focalLength0, width0, height0):
    WallDistance = WallDistance0
    width = 640
    height = 192
    grid = 16
    focalLength = focalLength0

    K = np.array([[focalLength, 0, width/2, 0],
                  [0, focalLength, height/2, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    K_inv = LA.inv(K)

    rotation = [0, 0, 0]
    rx = rotation[0]
    ry = rotation[1]
    rz = rotation[2]
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                  [ np.sin(rz),  np.cos(rz), 0],
                  [          0,    0,          1]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [          0, 1,          0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1,          0,           0],
                  [ 0, np.cos(rx), -np.sin(rx)],
                  [ 0, np.sin(rx),  np.cos(rx)]])

    translation = [0, 0, 0]
    translation = np.array(translation).reshape(3, 1)

    Rt = np.concatenate((Rz @ Ry @ Rx, translation), axis=1)
    Rt = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)

    Rt_inv = LA.inv(Rt)
    
    frameWallPair = np.zeros(height//grid*width//grid*4*2)
    frameWallPair = frameWallPair.reshape(height//grid*width//grid, 4*2)

    for u in range(grid//2, width, grid):
        for v in range(grid//2, height, grid):
            p_frame = np.array([u, v, 1, 1]).reshape(4, 1)
            p_wall = Rt_inv @ K_inv @ p_frame
            #print(p_wall)
            p_wall = p_wall / p_wall[2] * WallDistance
            frameWallPair[u//grid*(height//grid)+v//grid] = np.concatenate((p_frame, p_wall), axis=None)

    return frameWallPair

@jit
def xyz(wall, K, Rt_translate,width,height):
    hg = 16//2
    arrayidealVectorT = np.zeros((width,height,2))
    for point in wall:	 
        x, y = int(point[0]), int(point[1])
        [ut, vt, wt] = K @ Rt_translate @ np.array([point[4], point[5], point[6],1])
        ut = ut/wt
        vt = vt/wt
        wt = 1.0
        idealVector = np.array([ut-x, vt-y]).astype(np.float32)
        mag = LA.norm(idealVector)
        for i in range(y-hg, y+hg):
            for j in range(x-hg, x+hg):
                arrayidealVectorT[i,j,0] = ut-x
                arrayidealVectorT[i,j,1] = vt-y
				
    return mag, arrayidealVectorT

@jit
def RYP(wall, K, Rt_rotate,width,height):
    hg = 16//2
    arrayidealVectorRT = np.zeros((width,height,2))
    for point in wall:	
        x, y = int(point[0]), int(point[1]) 
        [ur, vr, wr] = K @ Rt_rotate @ np.array([point[4], point[5], point[6],1])
        ur = ur/wr
        vr = vr/wr
        wr = 1.0
        for i in range(y-hg, y+hg):
            for j in range(x-hg, x+hg):
                arrayidealVectorRT[i,j,0] = ur-x
                arrayidealVectorRT[i,j,1] = vr-y

    return arrayidealVectorRT

@jit
def depthmodule(wall, hg, depth, walldistance, ITF, NRF):
    hg = 16//2
    for point in wall:
        x, y = int(point[0]), int(point[1])
        for i in range(y-hg,y+hg):
            for j in range(x-hg,x+hg):
                depth[i,j] = ITF / np.sqrt(NRF[i,j,0]**2+NRF[i,j,1]**2) * walldistance
    return depth

def rotationinradius(crx,cry,crz):

    Rz = np.array([ [ np.cos(crz), -np.sin(crz), 0],
                    [ np.sin(crz),  np.cos(crz), 0],
                    [       0,      0,      1       ]])
    Ry = np.array([[np.cos(cry), 0, np.sin(cry)],
                    [          0,      1,               0],
                    [-np.sin(cry), 0, np.cos(cry)]])
    Rx = np.array([[1,           0,            0],
                    [ 0, np.cos(crx), -np.sin(crx)],
                    [ 0, np.sin(crx),  np.cos(crx)]])
        
    RI = np.eye(3, 3)

    return Rz, Ry ,Rx, RI

def translationinmeters(ctx,cty,ctz,crx,cry,crz):
    xd, yd, zd = ctx, cty, ctz
    
    translation = np.array([xd, yd, zd]).reshape(3, 1)
    translation = R.from_euler('x', crx).as_matrix() @ R.from_euler('y', cry).as_matrix() @ R.from_euler('z', crz).as_matrix() @ translation
    translation_O = [0, 0, 0]
    translation_O = np.array(translation_O).reshape(3, 1)

    return translation, translation_O

