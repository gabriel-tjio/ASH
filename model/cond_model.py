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


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class integrated_model_da(nn.Module):
    def __init__(self):
        super(integrated_model_da, self).__init__()
        # have to make sure x and y have matching dimensions
        nfilter=256
        nfilter1=512
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, nfilter, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(nfilter))
        #self.SFT_scale_conv0 = nn.Conv2d(nfilter1, nfilter2, 1)
        #self.SFT_scale_conv1 = nn.Conv2d(nfilter2, nfilter2, 1)
        #self.SFT_shift_conv0 = nn.Conv2d(nfilter1, nfilter2, 1)
        #self.SFT_shift_conv1 = nn.Conv2d(nfilter2, nfilter2, 1)
        self.cam=CAM_Module(nfilter)
        self.pam=PAM_Module(nfilter)
        self.conv51 = nn.Sequential(nn.Conv2d(nfilter, nfilter1, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(nfilter1),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(nfilter, nfilter1, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(nfilter1),
                                   nn.ReLU())
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        f_c= self.conv51(self.cam(seg_f))
        f_p= self.conv52(self.pam(seg_f))
        fuse_cp=f_c+f_p
        #scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(fuse_cp), 0.1, inplace=True))
        #shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(fuse_cp), 0.1, inplace=True))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        #scale= resize(scale)
        #shift = resize(shift)
        #fused=x * (scale + 1) + shift
        
        return fused


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# One hot matrix for GTAV  20 channels : NOTE: Synthia will also have 20 channels 
# Normalized Entropy map from Src image 1 channel
# 

class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


# https://github.com/xinntao/SFTGAN/blob/master/pytorch_test/architectures.py
class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x, y):
        # x: fea; y: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(y), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(y), 0.1, inplace=True))
        return x * (scale + 1) + shift



class integrated_modelv2(nn.Module):
    def __init__(self):
        super(integrated_modelv2, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128,affine=True), nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.1, True), nn.Conv2d(in_channels=256,  out_channels=256, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256,affine=True)
            )
        self.SFT_scale_conv0 = nn.Conv2d(in_channels=256,  out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        self.SFT_scale_conv1 = nn.Conv2d(in_channels=512,  out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        self.scale_norm     = nn.InstanceNorm2d(512,affine=True)
        self.SFT_shift_conv0 = nn.Conv2d(in_channels=256,  out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        self.SFT_shift_conv1 = nn.Conv2d(in_channels=512,  out_channels=512, kernel_size=(3, 3), padding=(1, 1))
        self.shift_norm     = nn.InstanceNorm2d(512,affine=True)
        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        scale = F.leaky_relu(self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True)))
        scale=self.scale_norm(scale)
        shift = F.leaky_relu(self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True)))
        shift =self.shift_norm(shift)
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift
        

        return fused


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


class integrated_model_groundtruth(nn.Module):
    def __init__(self):
        super(integrated_model_groundtruth, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(20, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
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


class integrated_model_noise_readonly(nn.Module):
    def __init__(self):
        super(integrated_model_noise_readonly, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.condlayer_noise = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256),nn.Conv2d(256, 512, 1),nn.InstanceNorm2d(512),nn.Sigmoid()
            )


        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        sem_noise_perturb= self.condlayer_noise(y)

        originalRandomNoise = torch.randn(x.shape)
        n_u, n_e, n_v = torch.svd(originalRandomNoise,some=True)
        n_v=n_v.reshape(x.shape) 
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        
        sem_noise_perturb=resize(sem_noise_perturb)
        x1=x+n_v*x*sem_noise_perturb
        diff=n_v*x*sem_noise_perturb




        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))
        scale= resize(scale)
        shift = resize(shift)
        fused=x1 * (scale + 1) + shift

        return fused,scale,shift,sem_noise_perturb,x1






class integrated_model_noise_type1(nn.Module):
    def __init__(self):
        super(integrated_model_noise_type1, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.condlayer_noise = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256),nn.Conv2d(256, 512, 1),nn.InstanceNorm2d(512),nn.Tanh()
            )
        def init_normal(m):
           if type(m) == nn.Conv2d:
              nn.init.normal_(m.weight,mean=0.0, std=1.0)
        self.condlayer_noise.apply(init_normal)
        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        sem_noise_perturb= self.condlayer_noise(y)
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        sem_noise_perturb=resize(sem_noise_perturb)
        x=x+x*sem_noise_perturb




        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim

        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(th.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  th.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = th.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out


class integrated_model_attn(nn.Module):
    def __init__(self):
        super(integrated_model_attn, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        self.self_attn_scale1= Self_Attn(512)
        self.self_attn_shift1= Self_Attn(512)

    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        scale = self.self_attn_scale1(self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True)))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused




class integrated_model_norm_act(nn.Module):
    def __init__(self):
        super(integrated_model_norm_act, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        self.instancenorm1= nn.InstanceNorm2d(512)
        self.instancenorm2= nn.InstanceNorm2d(512)
        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        scale = F.sigmoid(self.instancenorm1(self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))))
        shift = F.leaky_relu(self.instancenorm2(self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused



class integrated_model_norm_act2(nn.Module):
    def __init__(self):
        super(integrated_model_norm_act2, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        self.instancenorm1= nn.InstanceNorm2d(512)
        self.instancenorm2= nn.InstanceNorm2d(512)
        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        scale = F.tanh(self.instancenorm1(self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))))
        shift = F.tanh(self.instancenorm2(self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused




def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MS(nn.Module):
    def __init__(self, conv=default_conv, n_feats=64):
        super(MS, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class integrated_model_ms(nn.Module):
    def __init__(self):
        super(integrated_model_ms, self).__init__()
        # have to make sure x and y have matching dimensions
        n_feats = 64
        n_blocks = 7
        kernel_size = 3     
        self.n_blocks = n_blocks
   
        self.headlayer = nn.Sequential(
            nn.Conv2d(19, 64, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(32))

        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MS(n_feats=n_feats))

        modules_tail = [nn.InstanceNorm2d(self.n_blocks + 1),
            nn.Conv2d(n_feats * (self.n_blocks + 1), self.n_blocks + 1, 1, padding=0, stride=1),nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(self.n_blocks + 1)]

        self.SFT_scale_conv0 = nn.Conv2d((self.n_blocks + 1), 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d((self.n_blocks + 1), 512, 1)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)  
      
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        y=self.headlayer(y)
        res=y
        ms_out = []
        for i in range(self.n_blocks):
            y = self.body[i](y)
            ms_out.append(y)
        ms_out.append(res)
        res = torch.cat(ms_out,1)
        seg_f = self.tail(res)
        scale = F.sigmoid(self.SFT_scale_conv0(seg_f))
        shift = F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True)
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused




class integrated_model_ms_v2(nn.Module):
    def __init__(self):
        super(integrated_model_ms_v2, self).__init__()
        # have to make sure x and y have matching dimensions
        n_feats = 64
        n_blocks = 7
        kernel_size = 3     
        self.n_blocks = n_blocks
   
        self.headlayer = nn.Sequential(
            nn.Conv2d(19, 64, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(64))

        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MS(n_feats=n_feats))

        modules_tail = [nn.InstanceNorm2d(self.n_blocks + 1),
            nn.Conv2d(n_feats * (self.n_blocks + 1), self.n_blocks + 1, 1, padding=0, stride=1),nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(self.n_blocks + 1)]

        self.SFT_scale_conv0 = nn.Conv2d((self.n_blocks + 1), 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d((self.n_blocks + 1), 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)  
      
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        y=self.headlayer(y)
        res=y
        ms_out = []
        for i in range(self.n_blocks):
            y = self.body[i](y)
            ms_out.append(y)
        ms_out.append(res)
        res = torch.cat(ms_out,1)
        seg_f = self.tail(res)
        scale = F.sigmoid(self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True)))
        shift = F.leaky_relu(self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True)))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused



class integrated_model_ms_v3(nn.Module):
    def __init__(self):
        super(integrated_model_ms_v3, self).__init__()
        # have to make sure x and y have matching dimensions
        n_feats = 64
        n_blocks = 7
        kernel_size = 3     
        self.n_blocks = n_blocks
   
        self.headlayer = nn.Sequential(
            nn.Conv2d(19, 64, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(64))

        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MS(n_feats=n_feats))

        modules_tail = [nn.InstanceNorm2d(n_feats *(self.n_blocks + 1)),
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats * (self.n_blocks + 1), 1, padding=0, stride=1),nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(n_feats *(self.n_blocks + 1))]

        self.SFT_scale_conv0 = nn.Conv2d((self.n_blocks + 1)*n_feats, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d((self.n_blocks + 1)*n_feats, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        self.instancenorm1= nn.InstanceNorm2d(n_feats *(self.n_blocks + 1))
        self.instancenorm2= nn.InstanceNorm2d(n_feats *(self.n_blocks + 1))
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)  
      
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        y=self.headlayer(y)
        res=y
        ms_out = []
        for i in range(self.n_blocks):
            y = self.body[i](y)
            ms_out.append(y)
        ms_out.append(res)
        res = torch.cat(ms_out,1)
        seg_f = self.tail(res)
        scale = F.sigmoid(self.instancenorm1(self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))))

        shift = F.leaky_relu(self.instancenorm2(self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused


class integrated_model_ms_v5(nn.Module):
    def __init__(self):
        super(integrated_model_ms_v5, self).__init__()
        # have to make sure x and y have matching dimensions
        n_feats = 32
        n_blocks = 3
        kernel_size = 3     
        self.n_blocks = n_blocks
   
        self.headlayer = nn.Sequential(
            nn.Conv2d(19, 32, 4, 4),nn.InstanceNorm2d(32), nn.LeakyReLU(0.1, True))

        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MS(n_feats=n_feats))

        modules_tail = [nn.InstanceNorm2d(n_feats *(self.n_blocks + 1)),
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats * (self.n_blocks + 1), 1, padding=0, stride=1),nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(n_feats *(self.n_blocks + 1))]

        self.SFT_scale_conv0 = nn.Conv2d((self.n_blocks + 1)*n_feats, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d((self.n_blocks + 1)*n_feats, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        self.instancenorm1= nn.InstanceNorm2d(512)
        self.instancenorm2= nn.InstanceNorm2d(512)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)  
      
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        y=self.headlayer(y)
        res=y
        ms_out = []
        for i in range(self.n_blocks):
            y = self.body[i](y)
            ms_out.append(y)
        ms_out.append(res)
        res = torch.cat(ms_out,1)
        seg_f = self.tail(res)
        scale = F.sigmoid(self.instancenorm1(self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))))

        shift = F.leaky_relu(self.instancenorm2(self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused



class integrated_model_ht(nn.Module):
    def __init__(self):
        super(integrated_model_ht, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayerh = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0h = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1h = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0h = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1h = nn.Conv2d(512, 512, 1)

        self.condlayert = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0t = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1t = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0t = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1t = nn.Conv2d(512, 512, 1)

        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_ones=torch.ones_like(y)
        head_indices=[0,1,2,7,8,9,10,13,14,15,16]
        tail_indices=[3,4,5,6,11,12,17,18]        
        head_ones=seg_ones
        tail_ones=seg_ones
        tail_ones[:,head_indices,:,:]=0
        head_ones[:,tail_indices,:,:]=0
        head_y=y*head_ones
        tail_y=y*tail_ones

        segh_f=self.condlayerh(head_y)
        scale_h = self.SFT_scale_conv1h(F.leaky_relu(self.SFT_scale_conv0h(segh_f), 0.1, inplace=True))
        shift_h = self.SFT_shift_conv1h(F.leaky_relu(self.SFT_shift_conv0h(segh_f), 0.1, inplace=True))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale_h= resize(scale_h)
        shift_h = resize(shift_h)
        fused_h=x * (scale_h) + shift_h

        segt_f=self.condlayert(tail_y)
        scale_t = self.SFT_scale_conv1t(F.leaky_relu(self.SFT_scale_conv0t(segt_f), 0.1, inplace=True))
        shift_t = self.SFT_shift_conv1t(F.leaky_relu(self.SFT_shift_conv0t(segt_f), 0.1, inplace=True))
        scale_t= resize(scale_t)
        shift_t = resize(shift_t)
        fused_t=x * (scale_t) + shift_t
        fused = x + fused_t + fused_h
        return fused



class integrated_model_ht_v2(nn.Module):
    def __init__(self):
        super(integrated_model_ht_v2, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayerh = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0h = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1h = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0h = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1h = nn.Conv2d(512, 512, 1)



        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_ones=torch.ones_like(y)
        tail_indices=[4,5,8,17,18]        
        head_ones=seg_ones

        head_ones[:,tail_indices,:,:]=0
        head_y=y*head_ones


        segh_f=self.condlayerh(head_y)
        scale_h = F.sigmoid(self.SFT_scale_conv1h(F.leaky_relu(self.SFT_scale_conv0h(segh_f), 0.1, inplace=True)))
        shift_h = F.leaky_relu(self.SFT_shift_conv1h(F.leaky_relu(self.SFT_shift_conv0h(segh_f), 0.1, inplace=True)))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale_h= resize(scale_h)
        shift_h = resize(shift_h)
        fused_h=x * (scale_h+1) + shift_h

        return fused_h



class integrated_model_semper(nn.Module):
    def __init__(self):
        super(integrated_model_semper, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.condlayer_semper = nn.Sequential(
            nn.Conv2d(19, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True) )

        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)

        self.SFT_scale_conv0_sp = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1_sp = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0_sp = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1_sp = nn.Conv2d(512, 512, 1)


        
    def forward(self, x, y,z):
        # x: fea; y: cond (segmentation output Nx19xHxW) z: (semantic loss Nx19x1x1)
        seg_f=self.condlayer(y)
        semper=self.condlayer_semper(z)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))

        scale_sp = self.SFT_scale_conv1_sp(F.leaky_relu(self.SFT_scale_conv0_sp(semper), 0.1, inplace=True))
        shift_sp = self.SFT_shift_conv1_sp(F.leaky_relu(self.SFT_shift_conv0_sp(semper), 0.1, inplace=True))
        scale_sp = scale_sp.view(1,512,1,1).repeat(1,1,x.shape[2], x.shape[3])
        shift_sp = shift_sp.view(1,512,1,1).repeat(1,1,x.shape[2], x.shape[3])        
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift
        fused = fused*(scale_sp +1) + shift_sp
        return fused



class integrated_model_decoder(nn.Module):
    def __init__(self,decoder):
        super(integrated_model_decoder, self).__init__()
        # have to make sure x and y have matching dimensions
        self.decoder=decoder
        dec_layers = list(decoder.children())
        self.dec_1 = nn.Sequential(*dec_layers[:13])  # input -> relu1_1 512 256
        self.dec_2 = nn.Sequential(*dec_layers[13:20])  # relu1_1 -> relu2_1 256 128
        self.dec_3 = nn.Sequential(*dec_layers[20:27])  # relu2_1 -> relu3_1 128 64
        self.dec_4 = nn.Sequential(*dec_layers[27:29])  # relu3_1 -> relu4_1 64 3
        for name in ['dec_1', 'dec_2', 'dec_3', 'dec_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.condlayer1 = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        self.condlayer2 = nn.Sequential(
            nn.Conv2d(19, 64, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(64), nn.Conv2d(64, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128)
            )
        self.SFT_scale2_conv0 = nn.Conv2d(128, 256, 1)
        self.SFT_scale2_conv1 = nn.Conv2d(256, 256, 1)
        self.SFT_shift2_conv0 = nn.Conv2d(128, 256, 1)
        self.SFT_shift2_conv1 = nn.Conv2d(256, 256, 1)
        self.condlayer3 = nn.Sequential(
            nn.Conv2d(19, 32, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(32), nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(64)
            )
        self.SFT_scale3_conv0 = nn.Conv2d(64, 128, 1)
        self.SFT_scale3_conv1 = nn.Conv2d(128, 128, 1)
        self.SFT_shift3_conv0 = nn.Conv2d(64, 128, 1)
        self.SFT_shift3_conv1 = nn.Conv2d(128, 128, 1)

        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer1(y)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift # 512
        dout1=self.dec1(fused)
        seg_f2=self.condlayer2(y)
        scale2 = self.SFT_scale2_conv1(F.leaky_relu(self.SFT_scale2_conv0(seg_f2), 0.1, inplace=True))
        shift2 = self.SFT_shift2_conv1(F.leaky_relu(self.SFT_shift2_conv0(seg_f2), 0.1, inplace=True))
        resize2 =nn.Upsample(size=(dout1.shape[2], dout1.shape[3]), mode='bilinear', align_corners=True)
        scale= resize2(scale)
        shift = resize2(shift)
        fused=dout1 * (scale + 1) + shift #256
        dout2=self.dec2(fused)
        seg_f3=self.condlayer3(y)
        scale3 = self.SFT_scale3_conv1(F.leaky_relu(self.SFT_scale3_conv0(seg_f3), 0.1, inplace=True))
        shift3 = self.SFT_shift3_conv1(F.leaky_relu(self.SFT_shift3_conv0(seg_f3), 0.1, inplace=True))
        resize3 =nn.Upsample(size=(dout2.shape[2], dout2.shape[3]), mode='bilinear', align_corners=True)
        scale= resize3(scale)
        shift = resize3(shift)
        fused=dout2 * (scale + 1) + shift #128
        dout3=self.dec3(fused)
        dout4=self.dec4(dout3)
        RevGrad()

        return dout4

class integrated_model_noise(nn.Module):
    def __init__(self):
        super(integrated_model_noise, self).__init__()
        # have to make sure x and y have matching dimensions
        
        self.condlayer = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256)
            )
        self.condlayer_noise = nn.Sequential(
            nn.Conv2d(19, 128, 4, 4), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(128), nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(256, 256, 1), nn.LeakyReLU(0.1, True),nn.InstanceNorm2d(256),nn.Conv2d(256, 512, 1),nn.InstanceNorm2d(512),nn.Sigmoid()
            )


        self.SFT_scale_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_scale_conv1 = nn.Conv2d(512, 512, 1)
        self.SFT_shift_conv0 = nn.Conv2d(256, 512, 1)
        self.SFT_shift_conv1 = nn.Conv2d(512, 512, 1)
        
    def forward(self, x, y):
        # x: fea; y: cond (segmentation output Nx19xHxW)
        seg_f=self.condlayer(y)
        sem_noise_perturb= self.condlayer_noise(y)

        originalRandomNoise = torch.randn(x.shape)
        n_u, n_e, n_v = torch.svd(originalRandomNoise,some=True)
        n_v=n_v.reshape(x.shape) 
        resize =nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        
        sem_noise_perturb=resize(sem_noise_perturb)
        x=x+x*n_v.cuda(x.get_device())*sem_noise_perturb




        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg_f), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg_f), 0.1, inplace=True))
        scale= resize(scale)
        shift = resize(shift)
        fused=x * (scale + 1) + shift

        return fused

