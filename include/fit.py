from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np
from scipy.linalg import hadamard
from scipy.stats import ortho_group
from .helpers import *
if torch.cuda.device_count()==0:
    dtype = torch.FloatTensor
    device = 'cpu'
else:
    dtype = torch.cuda.FloatTensor
    device = 'cuda'
import time

def apply_f(x,Ameas,model):
    if model==1:
        y = x
    elif model==2: #CS
        y = torch.matmul(Ameas,x.reshape(x.numel(),1))
    elif model==3: #PR
        y = torch.matmul(Ameas,x.reshape(x.numel(),1))
        y = torch.abs(y)
    else:
        print('incorrect model')
    return y

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500, factor=0.5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (factor**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('\nLR is set to {}'.format(lr))
        print('\n')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def WeightMask(weights,total_layers):
    wavelet_lvls = list(range(1,total_layers[0]+1,3))
    wavelet_mid = list(np.array(wavelet_lvls)-2)
    mask = np.zeros(weights.shape)
    if weights.shape[0] in wavelet_mid:
        mask[0,:2] = np.array([1,1]).reshape(2,1,1)
        mask[1,2:4] = np.array([1,1]).reshape(2,1,1)
        mask[2:,4:] = np.identity(weights.shape[0]-2).reshape(weights.shape[0]-2,weights.shape[0]-2,1,1)
    elif weights.shape[0] in wavelet_lvls:
        mask[0,:2] = np.array([1,1]).reshape(2,1,1)
        mask[1:,2:] = np.identity(weights.shape[0]-1).reshape(weights.shape[0]-1,weights.shape[0]-1,1,1)
    else:
        raise ValueError('Require channel structure to match wavelet decomposition')
    mask = torch.from_numpy(mask).to(torch.float)
    maskedW = mask.to(device)*weights
    return maskedW

def fit(net,
        num_channels,
        img_clean_var,
        num_iter = 5000,
        LR = 0.01,
        OPTIMIZER='adam',
        optimizer2='SGD',
        opt_input = False,
        reg_noise_std = 0,
        reg_noise_decayevery = 100000,
        mask_var = None,
        lr_decay_epoch = 0,
        net_input = None,
        net_input_gen = "random",
        find_best=False,
        weight_decay=0,
        Ameas = 1,
        model = 1,
        LR_LS = 0.02,
        code = 'uniform',
        num_iters_inner = 100,
        decodetype='upsample', #'upsample','transposeconv'
        optim = 'gd', #gd or pgd
        print_inner = False,
        numit_inner = 20,
        decay_every = 500,
        out_channels=1,
       ):

    if net_input is not None:
        print("input provided")
    else:
        # feed uniform noise into the network 
        totalupsample = 2**(len(num_channels)-1)
        
        #if running as decoder/compressor
        if len(img_clean_var.shape)==4:
            width = int(img_clean_var.data.shape[2]/(totalupsample))
            height = int(img_clean_var.data.shape[3]/(totalupsample))
        #if running compressive imaging    
        elif len(img_clean_var.shape)==2:
            w = np.sqrt(int(Ameas.shape[1]/out_channels))
            width = int(w/(totalupsample))
            height = int(w/(totalupsample))
            
        shape = [1,num_channels[0], width, height]  
        print("shape of latent code B1: ", shape)

        print("initializing latent code B1...")
        net_input = Variable(torch.zeros(shape))
        if code== 'uniform':
            net_input.data.uniform_()
        elif code== 'gaussian':
            net_input.data.normal_()
        elif code== 'hadamard':
            B = Variable(torch.tensor(hadamard(width*height,dtype=float)))
            idx = np.random.choice(width*height,num_channels[0])
            net_input.data = B[list(idx),:].view(-1,num_channels[0],width,height)
        elif code== 'identity':
            B = Variable(torch.tensor(np.identity(width*height,dtype=float)))
            idx = np.random.choice(width*height,num_channels[0])
            net_input.data = B[list(idx),:].view(-1,num_channels[0],width,height)
        elif code=='xavier':
            torch.nn.init.xavier_uniform(net_input.data)
        
        net_input.data *= 1./10
        
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    if decodetype=='upsample':
        p = [x for x in net.decoder.parameters() ] #list of all weigths
    elif decodetype=='transposeconv':
        p = [x for x in net.convdecoder.parameters() ] #list of all weigths

    if(opt_input == True): # optimizer over the input as well
        net_input.requires_grad = True
        print('optimizing over latent code Z1')
        p += [net_input]
    else:
        print('not optimizing over latent code Z1')

    mse_wrt_truth = np.zeros(num_iter)
    mse_outer = np.zeros(num_iter)
    
    if OPTIMIZER == 'SGD':
        print("optimize decoder with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9,weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        print("optimize decoder with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)

    mse = torch.nn.MSELoss() 
    
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    if optim == 'pgd':
        print('optimizing with projected gradient descent...')
        if model==2 or model==3:
            x = Variable(torch.zeros([out_channels,int(w),int(w)]))
        elif model==1:
            x = Variable(torch.zeros(img_clean_var.shape))
        x = x.to(device)
        
        x.data = net(net_input.type(dtype))
        x_in = x.data.clone()
        
        print("optimize least squares loss with SGD")
        x.requires_grad=True
        x.retain_grad()
        xvar = [x]
        if optimizer2 == 'SGD':
            optimizer_LS = torch.optim.SGD(xvar,lr=LR_LS,momentum=0.9,weight_decay=weight_decay)
        elif optimizer2 == 'adam':
            optimizer_LS = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)
    if optim=='gd':    
        print('optimizing with gradient descent...')
        x_in = net(net_input.type(dtype)).data.clone()
        for i in range(num_iter):

            #################
            if lr_decay_epoch is not 0:
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch,factor=0.7)
                
            #################
            def closure():
                optimizer.zero_grad()           
                outp = net(net_input.type(dtype))
                loss = mse(apply_f(outp,Ameas,model), img_clean_var)
                loss.backward()
                mse_wrt_truth[i] = loss.data.cpu().numpy()
                return loss
            
            loss = optimizer.step(closure) 
                  
            print ('Iteration %05d   Train loss %f ' % (i, loss.detach().cpu().numpy()), '\r', end='')

            if find_best:
                # if training loss improves by at least one percent, we found a new best net
                if best_mse > 1.01*loss.detach().cpu().numpy():
                    best_mse = loss.detach().cpu().numpy()
                    best_net = copy.deepcopy(net)

        if find_best:
            net = best_net
    
    elif optim=='pgd':
        for i in range(num_iter):

            ################
            if lr_decay_epoch is not 0:
                optimizer_LS = exp_lr_scheduler(optimizer_LS, i, init_lr=LR_LS, lr_decay_epoch=lr_decay_epoch,factor=0.7) 
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch,factor=0.9) 
            if i % decay_every == 0:
                numit_inner = round(1.2*numit_inner)
                print('max iters for inner loop set to', numit_inner,'\n')
            #################
            #gradient step for least squares problem
            with torch.no_grad():
                loss_pre = mse(apply_f(x,Ameas,model), img_clean_var)
            optimizer_LS.zero_grad()
            output = apply_f(x,Ameas,model)
            loss_LS = mse(output,img_clean_var)
            loss_LS.backward()           
            optimizer_LS.step()
            mse_wrt_truth[i] = loss_LS.item()
            print ('Iteration %05d   Train loss %f ' % (i, mse_wrt_truth[i]), '\r', end='')
            
            for j in range(numit_inner):
                optimizer.zero_grad()          
                out = net(net_input.type(dtype))
                loss_inner = mse(out,x)
                loss_inner.backward()
                optimizer.step()
                if print_inner:
                    print ('Inner iteration %05d  Train loss %f' % (j, loss_inner.detach().cpu().numpy()))
                    
            #project on learned network
            x.data = net(net_input.type(dtype))
            loss_updated = mse(apply_f(Variable(x.data, requires_grad=True),Ameas,model), img_clean_var)
            if find_best:
                # if training loss improves by at least one percent, we found a new best net
                if best_mse > 1.01*loss_updated.item():
                    best_mse = loss_updated.item()
                    best_net = copy.deepcopy(net)
                    
        if find_best:
            net = best_net
        
    return mse_wrt_truth,net_input_saved, net, net_input, x_in