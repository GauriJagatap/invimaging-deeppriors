import pywt
import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct

def construct_W(d=64,wave_name='db1',channels=1):
    X = np.identity(d)
    c1,c2 = pywt.dwt(np.ones(d),wave_name)
    W = np.zeros((2*c1.size,d)) # zeros(d,d+1); for db2
    d2 = c1.size
    for i in range(d):
        a,b = pywt.dwt(X[:,i],wave_name)
        W[:d2,i] = a
        W[d2:,i] = b
    
    W_lin = np.kron(W,W)
    W_lin = np.kron(np.identity(channels),W_lin)
    return W_lin

def construct_Winv(dout=64,wave_name='db2',channels=1):
    c1,c2 = pywt.dwt(np.ones(dout),wave_name)
    d = 2*c1.size
    X = np.identity(d)
    d2 = int(np.ceil(d/2))
    c3 = pywt.idwt(np.ones(d2),np.ones(d2),wave_name)
    W = np.zeros((c3.size,d)) # zeros(d,d+1); for db2
    for i in range(d):
        a = pywt.idwt(X[:d2,i],X[d2:,i],wave_name)
        W[:,i] = a
    
    W_lin = np.kron(W,W)
    W_lin = np.kron(np.identity(channels),W_lin)
    return W_lin

def construct_Wm(d=64,wave_name='db1',channels=1,lvls=2):
    X = np.identity(d)
    coeffs = pywt.wavedec(np.ones(d),wave_name,level=lvls)
    size_list = [coeffs[i].size for i in range(len(coeffs))]
    d_w = sum(size_list)
    W = np.zeros((d_w,d)) # zeros(d,d+1); for db2
    for i in range(d):
            coeffs = pywt.wavedec(X[:,i],wave_name,level=lvls)
            offset = 0
            for j in range(len(coeffs)):
                leng = coeffs[j].size
                W[offset:offset+leng,i] = coeffs[j]
                offset = offset + leng
    W_lin = np.kron(W,W)
    W_lin = np.kron(np.identity(channels),W_lin)
    return W_lin

def construct_DCT2mat(d=64):
    X = np.identity(d)
    D = np.zeros((d,d))
    for i in range(d):
        D[:,i] = dct(X[:,i])

    D_lin = np.kron(D,D)/(2*d)
    return D_lin

def construct_IDCT2mat(d=64):
    X = np.identity(d)
    iD = np.zeros((d,d))
    for i in range(d):
        iD[:,i] = idct(X[:,i])

    iD_lin = np.kron(iD,iD)/(2*d)
    return iD_lin

def construct_Wminv(d=8,wave_name='db1'):
    """generate the basis"""
    x = np.zeros((d, d))
    coefs = pywt.wavedec2(x, wave_name)
    n_levels = len(coefs)
    basis = []
    for i in range(n_levels):
        coefs[i] = list(coefs[i])
        n_filters = len(coefs[i])
        for j in range(n_filters):
            for m in range(coefs[i][j].shape[0]):
                try:
                    for n in range(coefs[i][j].shape[1]):
                        coefs[i][j][m][n] = 1
                        temp_basis = pywt.waverec2(coefs, wave_name)
                        basis.append(temp_basis)
                        coefs[i][j][m][n] = 0
                except IndexError:
                    coefs[i][j][m] = 1
                    temp_basis = pywt.waverec2(coefs, wave_name)
                    basis.append(temp_basis)
                    coefs[i][j][m] = 0   
                    
    W_ = np.array(basis)
    dnew = W_.shape[0]
    W_ = W_.reshape(( d*d,dnew))
    return W_

'''
def construct_Wminv(d=64,wave_name='db2'):
    """generate the basis"""
    x = np.zeros((d, d))
    coefs = pywt.wavedec2(x, wave_name)
    n_levels = len(coefs)
    print(n_levels)
    basis = []
    for i in range(n_levels):
        coefs[i] = list(coefs[i])
        n_filters = len(coefs[i])
        for j in range(n_filters):
            for m in range(coefs[i][j].shape[0]):
                try:
                    for n in range(coefs[i][j].shape[1]):
                        coefs[i][j][m][n] = 1
                        temp_basis = pywt.waverec2(coefs, wave_name)
                        
                        basis.append(temp_basis)
                        coefs[i][j][m][n] = 0
                except IndexError:
                    coefs[i][j][m] = 1
                    temp_basis = pywt.waverec2(coefs, wave_name)
                    basis.append(temp_basis)
                    coefs[i][j][m] = 0                
    basis = np.array(basis)
    return basis
'''