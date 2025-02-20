import torch.nn as nn
from torch.nn import functional as F
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import math


__all__ = ['MFDL, mfdl']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

import torch
import torch.nn as nn

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexConv2d, self).__init__()
        # define the real and imaginary parts of the convolutional kernels
        self.real_conv = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1)
        self.imag_conv = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1)

    def forward(self, x):
        # extract the real and imaginary parts of the input
        real = x.real
        imag = x.imag

        # forward the real and imaginary parts through the real and imaginary kernels
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)

        # combine the real and imaginary parts
        return torch.complex(real_out, imag_out)




class MFDL(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4], num_classes=1, zero_init_residual=False):
        super(MFDL, self).__init__()

        self.dwt = DWTForward(J=3, mode='zero', wave='db3')
        self.idwt = DWTInverse(mode='zero', wave='db3')

        # Replace nn.Parameter with nn.Conv2d
        self.conv_dw1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)

        self.complexconv1 = ComplexConv2d(64, 64)

        self.conv_dw2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)

        self.complexconv2 = ComplexConv2d(64, 64)

        self.conv_dw3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)

        self.complexconv3 = ComplexConv2d(256, 256)

        self.conv_dw4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)

        self.complexconv4 = ComplexConv2d(256, 256)

        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def SpatialFilter(self, x, scale):

        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        b,c,h,w = x.shape
        x[:,:,h//2-h//scale:h//2+h//scale,w//2-w//scale:w//2+w//scale ] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x=F.relu(x,inplace=True)
        return x

    def ChannelFilter(self, x, scale):
        x = torch.fft.fft(x, dim=1, norm="ortho")
        x = torch.fft.fftshift(x, dim=1) 
        b,c,h,w = x.shape

        x[:,c//2-c//scale:c//2+c//scale,:,:] = 0.0

        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x=F.relu(x,inplace=True)

        return x

    def dwtscale(self,yh,scale):
        for i in range(len(yh)):
            yh[i]=yh[i]*scale
        return yh
    



 
    def forward(self, x):


        # FRE
        yl,yh=self.dwt(x)
        yh=self.dwtscale(yh,2)
        dwtx=self.idwt((yl,yh))
        dwtx=self.conv_dw1(dwtx)
        dwtx=F.relu(dwtx, inplace=True)

        # FRC
        x = self.SpatialFilter(x, 4)
        x = self.conv1(x) #
        x=F.relu(x,inplace=True)
        x = self.ChannelFilter(x, 4)
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x=self.complexconv1(x)
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x=F.relu(x,inplace=True)
        x=dwtx+x
        

        #FRE

        yl,yh=self.dwt(x)
        yh=self.dwtscale(yh,2)
        dwtx=self.idwt((yl,yh))
        dwtx=self.conv_dw2(dwtx)
        dwtx=F.relu(dwtx, inplace=True)
 

        #FRC
        x = self.SpatialFilter(x, 4)
        x = self.conv2(x)
        x=F.relu(x,inplace=True)
        x = self.ChannelFilter(x, 4)
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = self.complexconv2(x)
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x=F.relu(x,inplace=True)



        x=x+dwtx

        x = self.maxpool(x)
        x = self.layer1(x)
        

        #FRE
        yl,yh=self.dwt(x)
        yh=self.dwtscale(yh,2)
        dwtx=self.idwt((yl,yh))
        dwtx=self.conv_dw3(dwtx)
        dwtx=F.relu(dwtx, inplace=True)

        #FRC
        x = self.SpatialFilter(x, 4)
        x = self.conv3(x)
        x=F.relu(x,inplace=True)
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = self.complexconv3(x)
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x=F.relu(x,inplace=True)


        x=x+dwtx



        # FRE
        yl,yh=self.dwt(x)
        yh=self.dwtscale(yh,2)
        dwtx=self.idwt((yl,yh))
        dwtx=self.conv_dw4(dwtx)
        dwtx=F.relu(dwtx, inplace=True)

        #FRC
        x = self.SpatialFilter(x, 4)
        x = self.conv4(x)
        x=F.relu(x,inplace=True)
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = self.complexconv4(x)
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x=F.relu(x,inplace=True)



        
        x=x+dwtx

        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def mfdl(**kwargs):

    return MFDL()
    
    


    


