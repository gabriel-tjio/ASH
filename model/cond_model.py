import torch.nn as nn
import numpy as np
import torch.nn.functional as F
affine_par = True
from torch.autograd import Function
import torch

from torch.optim.optimizer import Optimizer, required

from torch import nn
from torch import Tensor
from torch.nn import Parameter

import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.autograd import Variable
torch_ver = torch.__version__[:3]



class integrated_modelreadonly(nn.Module):
    def __init__(self):
        super(integrated_modelreadonly, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale1= resize(scale)
        shift1 = resize(shift)
        fused=x * (scale1 + 1) + shift1

        return fused,scale,shift


class integrated_model(nn.Module):
    def __init__(self):
        super(integrated_model, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused


