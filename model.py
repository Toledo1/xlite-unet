
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

        
class CustomBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        gc = self.in_channels
        #config switchable block
        self.switch = torch.nn.Conv2d(
            gc,
            1,
            kernel_size=1,
            stride=1,
            bias=True)
        
        
        self.pw1 = nn.Conv2d(self.in_channels, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel = (7, 7), dilation = 1)
        self.dw3 = AxialDW(gc, mixer_kernel = (7, 7), dilation = 3)

    def forward(self, x):
        
        x = self.pw1(x)
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        out_s = self.dw1(x)
        out_l = self.dw3(x)
        
        out = switch * out_s + (1 - switch) * out_l

        return out


class SAADWEncoderAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        gc = in_channels #self.in_channels*2
        #config switchable block
        self.switch = torch.nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            stride=1,
            bias=True)
        
        

        self.dw1 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw3 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.pw1 = nn.Conv2d(gc,self.out_channels, kernel_size=1)
        

        self.bn = nn.BatchNorm2d(gc)
        self.act = nn.GELU()
        
        #Attention block
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #PWC
        self.pwc2 = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        
    def forward(self, x):

        #print(x.shape)
        #pw1 = self.pw1(x)
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        out_s = self.dw1(x)
        out_l = self.dw3(x)
        
        out = x + switch * out_s + (1 - switch) * out_l

        # post-context
        '''avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x'''
        #pwc = F.conv2d(grn,weight, None, 1,0, 1, 1)
        
        
        bn = self.bn(out)
        pw1 = self.pw1(bn)
        act = self.act(pw1)
        
        skip = bn*self.pwc2(self.avg_pool(bn))
        #print(act.shape)
        return act, skip
        
        
class LiteDecoder(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        gc = in_c+out_c
        self.bn = nn.BatchNorm2d(gc)
        self.act = nn.GELU()
        
        self.switch = torch.nn.Conv2d(
            gc,
            1,
            kernel_size=1,
            stride=1,
            bias=True)
        
        
        #self.pw1 = nn.Conv2d(self.in_channels, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw3 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.pw2 = nn.Conv2d(gc,out_c, kernel_size=1)
        

    def forward(self, x, skip):
    
        x = self.up(x)   
        x = torch.cat([x, skip], dim=1)
    
        # switch
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        out_s = self.dw1(x)
        out_l = self.dw3(x)
        
        out = x + switch * out_s + (1 - switch) * out_l
        
        #x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        x = self.act(self.pw2(self.bn(out)))
        return x


class XLITE_UNET(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(XLITE_UNET, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = nn.Conv2d(img_ch, 32, kernel_size=7, padding='same')
        self.Conv2 = SAADWEncoderAtt(in_channels=32, out_channels=64)
        self.Conv3 = SAADWEncoderAtt(in_channels=64, out_channels=128)
        self.Conv4 = SAADWEncoderAtt(in_channels=128, out_channels=256)
        self.Conv5 = SAADWEncoderAtt(in_channels=256, out_channels=512)
        
        #BottleNeck
        """Bottle Neck"""
        self.btneck = CustomBottleneck(in_channels=512,out_channels=512)
        
        self.dec5 = LiteDecoder(512, 256)
        self.dec4 = LiteDecoder(256, 128)
        self.dec3 = LiteDecoder(128, 64)
        self.dec2 = LiteDecoder(64, 32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        
        x1 = self.Conv1(x)
        
        
        x2, skip1 = self.Conv2(x1)#[0]
        x2 = self.Maxpool(x2)
        #x2 = self.leam2(x2)

        
        x3, skip2 = self.Conv3(x2)#[0]
        x3 = self.Maxpool(x3)
        #x3 = self.leam3(x3)

        
        x4, skip3 = self.Conv4(x3)#[0]
        x4 = self.Maxpool(x4)
        #x4 = self.leam4(x4)

        
        x5, skip4 = self.Conv5(x4)#[0]
        x5 = self.Maxpool(x5)
        

        # decoding + concat path
        x5 = self.btneck(x5)
        
        d5 = self.dec5(x5, skip4) 
        d4 = self.dec4(d5, skip3)
        d3 = self.dec3(d4, skip2)
        d2 = self.dec2(d3, skip1)
        

        d1 = self.Conv_1x1(d2)
        
        return d1