import torch 
import torch.nn as nn
import torchvision
import sys
import numpy as np
from PIL import Image
import PIL
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Variable

def np_to_tensor(img_np):
    #Converts image in numpy.array to torch.Tensor from C x W x H [0..1] to  C x W x H [0..1]
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    #Converts image in numpy.array to torch.Variable from C x W x H [0..1] to  1 x C x W x H [0..1]
    return Variable(np_to_tensor(img_np)[None, :])

def pil_to_np(img_PIL):
    #Converts image in PIL format to np.array from W x H x C [0...255] to C x W x H [0..1]
    ar = np.array(img_PIL)
    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255.

def mse(x_hat,x_true,maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse = np.mean(np.square(x_hat-x_true))
    energy = np.mean(np.square(x_true))   
    return mse/energy

def psnr(x_hat,x_true,maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse=np.mean(np.square(x_hat-x_true))
    psnr_ = 10.*np.log(maxv**2/mse)/np.log(10.)
    return psnr_

def rgb2gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.array([gray])
    
def myimgshow(plt,img):
    plt.imshow(np.clip(img[0],0,1),interpolation='nearest')    

def num_param(net):
    s = sum([np.prod(list(p.size())) for p in net.parameters()]);
    return s

def ComputeInitErr(nettype,netintype,lvls):
    if lvls>0:
        W1 = list(nettype.parameters())[0].detach().cpu().numpy()
        W1_0 = list(netintype.parameters())[0].detach().cpu().numpy()
        del1 = np.linalg.norm(W1-W1_0)/np.linalg.norm(W1)
        print(del1)
    if lvls>1:
        W2 = list(nettype.parameters())[3].detach().cpu().numpy()
        W2_0 = list(netintype.parameters())[3].detach().cpu().numpy()
        del2 = np.linalg.norm(W2-W2_0)/np.linalg.norm(W2)
        print(del2)
    if lvls>2:
        W3 = list(nettype.parameters())[6].detach().cpu().numpy()
        W3_0 = list(netintype.parameters())[6].detach().cpu().numpy()
        del3 = np.linalg.norm(W3-W3_0)/np.linalg.norm(W3)
        print(del3)
    return None  

def gamma_correction(image, gamma):
    corrected_image = image**(gamma)
    return corrected_image

def add_noise(image, SNR):
    noise = np.random.randn(image.shape[0],image.shape[1],image.shape[2])
    sigma = image.max()/(10**(SNR/20))
    noise = sigma*noise
    noisy_image = image + noise
    return noisy_image