import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from .cptconv import CPTConv2d 


class FPConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):

        super(FPConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                    stride, padding, dilation, groups, bias)
    
    def forward(self, x, actbits, wbits, gbits):
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output 

def qconv2d(qconv_type = 'fp'):
    if qconv_type == 'fp':
         return FPConv2d
    if qconv_type == 'cpt':
        return CPTConv2d
    return FPConv2d