import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


def conv(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver]) #extract values of non-None padder and convolver 
    return nn.Sequential(*layers)

     
        
class Downsample(torch.nn.Module): 
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Downsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)


class autoencodernet(torch.nn.Module):
    def __init__(self,
                 num_output_channels,
                 num_channels_up,
                 Ameas,
                 need_sigmoid=True,
                 pad='reflection',
                 upsample_mode='bilinear',
                 bn_affine = True, 
                 decodetype = 'upsample', #transposeconv
                 ):
        super(autoencodernet,self).__init__()
        self.decodetype = decodetype
        
        n_scales = len(num_channels_up)
        print('n_scales=', n_scales, 'num_channels_up=', num_channels_up)
        
        if decodetype=='upsample':
            #decoder
            self.decoder = nn.Sequential()

            for i in range(n_scales-1):
                
                #if i!=0:
                module_name = 'dconv'+str(i)    
                self.decoder.add_module(module_name,conv( num_channels_up[i], num_channels_up[i+1],  1, 1, pad=pad))    

                if i != len(num_channels_up)-1:        
                    module_name = 'drelu' + str(i)
                    self.decoder.add_module(module_name,nn.ReLU())   
                    module_name = 'dbn' + str(i)
                    self.decoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))        

                module_name = 'dups' + str(i)
                self.decoder.add_module(module_name,nn.Upsample(scale_factor=2, mode=upsample_mode))   

            module_name = 'dconv' + str(i+1)
            self.decoder.add_module(module_name,conv(num_channels_up[-1], num_output_channels, 1, pad=pad))                                   
        
            if need_sigmoid:
                self.decoder.add_module('sig',nn.Sigmoid())
                
        #encoder
        self.encoder = nn.Sequential()
        module_name = 'uconv'+str(n_scales-1)   
        self.encoder.add_module(module_name,conv(64,num_channels_up[-1], 1, pad=pad))
        
        for i in range(n_scales-2,-1,-1):
            
            if i != len(num_channels_up)-1:  
                module_name = 'urelu' + str(i)
                self.encoder.add_module(module_name,nn.ReLU())
                module_name = 'ubn' + str(i)
                self.encoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))     
                
            module_name = 'uconv'+str(i)
            self.encoder.add_module(module_name,conv(num_channels_up[i+1], num_channels_up[i],  1, 1, pad=pad))    
            module_name = 'udns'+str(i)
            self.encoder.add_module(module_name,Downsample(scale_factor=0.5, mode=upsample_mode))       

        if decodetype=='transposeconv':
            #convolutional decoder
            self.convdecoder = nn.Sequential()
            
            for i in range(n_scales-1):
                module_name = 'cdconv'+str(i) 
                
                if i==0:
                    self.convdecoder.add_module(module_name,conv( num_channels_up[i], num_channels_up[i+1],  1, 1, pad=pad)) 
                else:
                    self.convdecoder.add_module(module_name,nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i+1],2,2)) 

                if i != len(num_channels_up)-1:        
                    module_name = 'cdrelu' + str(i)
                    self.convdecoder.add_module(module_name,nn.ReLU())   
                    module_name = 'cdbn' + str(i)
                    self.convdecoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))        

            module_name = 'cdconv' + str(i+2)
            self.convdecoder.add_module(module_name,nn.ConvTranspose2d(num_channels_up[-1], num_output_channels, 2, 2)) 
            
            if need_sigmoid:
                self.convdecoder.add_module('sig',nn.Sigmoid())
        
    def forward(self,x):
        if self.decodetype=='upsample':
            x = self.decoder(x)
        elif self.decodetype=='transposeconv':
            x = self.convdecoder(x)
        return x