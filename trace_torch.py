from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10

import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from retrain.LEAStereo import LEAStereo 

from config_utils.predict_args import obtain_predict_args
from utils.colorize import get_color_map
from utils.multadds_count import count_parameters_in_MB, comp_multadds
from time import time
from struct import unpack
import matplotlib.pyplot as plt
import re
import numpy as np
import pdb
from path import Path

# opt = obtain_predict_args()
# print(opt)


device = torch.device('cpu')

weight_path = '../apollo_best.pth' # Path to pretrained weight

print('===> Building LEAStereo model')
model = LEAStereo(opt)

# An instance of your model.
# model = torchvision.models.resnet18()

model = torch.load(weight_path, map_location=device)

print('Loaded model')

global_crop_height = 500
global_crop_width = 500


def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width: 
        # padding zero 
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp    
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]  
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def test_stereo(leftname, rightname, savename):


    input1, input2, height, width = test_transform(load_data(leftname, rightname), global_crop_height, global_crop_width)
 
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():        
        prediction = model(input1, input2)
        
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= global_crop_height and width <= global_crop_width:
        temp = temp[0, global_crop_height - height: global_crop_height, global_crop_width - width: global_crop_width]
    else:
        temp = temp[0, :, :]
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # traced_script_module.save("traced_resnet_model.pt")
    traced_script_module.save("leastereo_apollo.pt")

if __name__ == "__main__":
    file_path = "test_cpp/"
    current_file = "171206_034625454_Camera_5.png"
    leftname = file_path + current_file[0: len(current_file) - 5] + '5.jpg'
    rightname = file_path + current_file[0: len(current_file) - 5] + '6.jpg'
    savename = "test_cpp/test_stereo_img.png"
    test_stereo(leftname, rightname, savename)

    # # An example input you would normally provide to your model's forward() method.
    # example = torch.rand(1, 3, 608, 608)

    # # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    # traced_script_module = torch.jit.trace(model, example)

    # # traced_script_module.save("traced_resnet_model.pt")
    # traced_script_module.save("traced_yolact_model.pt")