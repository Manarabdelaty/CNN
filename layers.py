import numpy as np
from utils import *


class ConvLayer():
    
    def __init__(self):
        pass
    
    def forward(self, X, kernel, bias, padding = 1, stride = 1):
        N,C,H,W = X.shape
        fn,filter_c, filter_H, filter_W, = kernel.shape
    
        X = np.pad(X, ((0,0),(0,0),(padding, padding), (padding, padding)), mode='constant')      # Zero padding input volume
        assert (W+2*padding-filter_W) % stride == 0
        assert (H+2*padding-filter_H) % stride == 0
    
        out_W = int ((W+2*padding-filter_W) / stride + 1)
        out_H = int ((H+2*padding-filter_H) / stride + 1)
        out_d = fn                                       # Output depth : number of filters
        out = np.zeros((N,out_d,out_H, out_W))
    
        for i in range(out_d):     # compute activation maps
            for j in range(out_H):
                for k in range(out_W):
                    #kernel_repeat = np.repeat(np.array(kernel[i,:,:,:])[None, :],N,axis = 0)
                    out[:,i,j,k] = np.sum(X[:,:,j*stride:j*stride+filter_H, k*stride:k*stride+filter_W] * kernel[i,:,:,:] , axis =(1,2,3)) + bias[i]
        
        cache = (X, kernel, bias, padding, stride)
        return out 
    
    def backward(self ,dH,X,kernel,bias, padding=1 , stride =1):
       # X, kernel , bias, padding,stride = cache
        N,C,H,W = X.shape
        fn,C,fH,fW = kernel.shape
    
        Ed,Ec, Eh, Ew =dH.shape
    
        dx = np.zeros(X.shape)
        dw = np.zeros(kernel.shape)
        db = np.zeros(bias.shape)
    
        # calculate dw : convlution of padded x and error
        X_pad = np.pad(X ,((0,0),(0,0),(padding, padding), (padding, padding)), mode='constant')    
        dx_pad = np.zeros(X_pad.shape)

        for n in range(N):    # loop over all batch
            for i in range(fn):
                for j in range(Eh):
                    for k in range(Ew):
                        dw[i,:,:,:] += X_pad[n,:,j*stride:j*stride+fH, k*stride:k*stride+fW ] * dH[n,i,j,k]
                        dx_pad[n, :, j * stride:j * stride + fH, k * stride:k * stride + fW] += kernel[i] * dH[n, i, j, k]
        #calculate dx
        dx = dx_pad[:,:,padding:padding+H, padding: padding+W]
    
    
        # calculate db
        for i in range(fn):
            db[i] = np.sum(dH[:,i,:,:])
    
        return dx,dw,db
    
    def forward_fast(self,X, w,b,padding,stride):
        N,C,H,W = X.shape
        fn,filter_c, filter_H, filter_W, = w.shape
        assert (W+2*padding-filter_W) % stride == 0
        assert (H+2*padding-filter_H) % stride == 0
    
        out_W = int ((W+2*padding-filter_W) / stride + 1)
        out_H = int ((H+2*padding-filter_H) / stride + 1)
        out_d = fn                                       # Output depth : number of filters
        out = np.zeros((N,out_d,out_H, out_W))    
    
        X_flattend = im2col_indices(X,filter_H, filter_W,padding, stride)
   
        out = np.dot(w.reshape(fn , -1) , X_flattend) + b.reshape(fn,1)
    
        out = out.reshape(out_d,out_H,out_W,N).transpose(3,0,1,2)
        return out , X_flattend
    

    def backward_fast(self,dH,X,X_shape,w,bias, padding=1 , stride =1):
        fn,C,fH,fW = w.shape
    
        Ed,Ec, Eh, Ew =dH.shape
        
        dw = np.zeros(w.shape)
        db = np.zeros(bias.shape)
        # dw : convlution of padded X and error
        dH_f = dH.transpose(1,2,3,0).reshape(fn, -1)            # flatten the error matriz
    
        dw = np.dot(dH_f, X.T).reshape(w.shape)                       

        dx_cols =  np.dot(w.reshape(fn,-1).T,dH.transpose(1,2,3,0).reshape(fn, -1))
  
        dx = col2im_indices(dx_cols, X_shape , fH, fW, padding, stride)
        db = np.sum(dH, axis=(0,2,3))            # sum the acitvation maps error

        return dx,dw,db

class PoolLayer():
    
    def __init(self):
        pass
    
    def forward(self, x,F = 2, S = 2):
        N,C,H,W = x.shape
        out_W = int ((W-F)/S +1)
        out_H = int((H-F)/S +1)
        
        assert (W-F) % S == 0           
        assert (H-F) % S == 0
    
        out_D = C
        out = np.zeros((N,out_D,out_H,out_W))
        self.masks = np.zeros(x.shape , dtype = int)
        
        for i in range(out_D):     # compute activation maps
            for j in range(out_H):
                for k in range(out_W):
                    out[:,i,j,k] = np.amax(x[:,i,j*S:j*S+F, k*S:k*S+F] , axis =(1,2))
                    self.masks[:,i,j*S:j*S+F, k*S:k*S+F] = (x[:,i,j*S:j*S+F, k*S:k*S+F] == np.amax(x[:,i,j*S:j*S+F, k*S:k*S+F], axis=(1,2)).reshape((N,1,1)))
        return out
    
    def backward(self, dout, x, F=2, S =2):
        N,C,H,W = x.shape
        N,C,out_H, out_W = dout.shape
        
        dx = np.zeros(x.shape)
    
        for i in range(C):
            for j in range(out_H):
                for k in range(out_W):                       
                    dx[:,i,j*S:j*S+F, k*S:k*S+F] = self.masks[:,i,j*S:j*S+F, k*S:k*S+F]*dout[:, i,j,k].reshape((N,1,1))
        return dx
    
    def max_pool_forward_reshape(self,x, F = 2, S = 2):
        
        N, C, H, W = x.shape
        pool_height, pool_width = F,F
        stride = S
        assert pool_height == pool_width == stride, 'Invalid pool params'
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = x.reshape(N, C, H // pool_height, pool_height,
                           W // pool_width, pool_width)
        out = x_reshaped.max(axis=3).max(axis=4)

        cache = (x, x_reshaped, out)
        return out, cache
    

    def forward_fast(self, x, F = 2, S = 2):
       
        N, C, H, W = x.shape
        pool_height, pool_width = F,F
        stride = S

        same_size = pool_height == pool_width == stride
        tiles = H % pool_height == 0 and W % pool_width == 0
        if same_size and tiles:
            out, reshape_cache = self.max_pool_forward_reshape(x,F = 2, S = 2 )
            cache = ('reshape', reshape_cache)
        else:
            out, im2col_cache = self.max_pool_forward_im2col(x, F = 2, S = 2)
            cache = ('im2col', im2col_cache)
        return out, cache
    
  

    def backward_fast(self,dout,cache):
        method, real_cache = cache
        if method == 'reshape':
            return self.max_pool_backward_reshape(dout, real_cache)
        elif method == 'im2col':
            return self.max_pool_backward_im2col(dout, real_cache)
        else:
            raise ValueError('Unrecognized method "%s"' % method)

    def max_pool_backward_reshape(self,dout, cache):
  
        x, x_reshaped, out = cache

        dx_reshaped = np.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_reshaped == out_newaxis)
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(x.shape)

        return dx


    def max_pool_forward_im2col(self,x, F = 2, S = 2):
   
        N, C, H, W = x.shape
        pool_height, pool_width =F,F
        stride = S

        assert (H - pool_height) % stride == 0, 'Invalid height'
        assert (W - pool_width) % stride == 0, 'Invalid width'

        out_height = (H - pool_height) // stride + 1
        out_width = (W - pool_width) // stride + 1

        x_split = x.reshape(N * C, 1, H, W)
        x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

        cache = (x, x_cols, x_cols_argmax, pool_param)
        return out, cache


    def max_pool_backward_im2col(self,dout, cache):

        x, x_cols, x_cols_argmax, pool_param = cache
        
        N, C, H, W = x.shape
        pool_height, pool_width = F,F
        stride =S

        dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
        dx_cols = np.zeros_like(x_cols)
        dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
        dx = col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width,
                padding=0, stride=stride)
        dx = dx.reshape(x.shape)

        return dx
    
class ReLU():
    
    def __init__(self):
        pass
    
    def forward(self,X):
        return np.maximum(0,X)
    
    def backward(self,X, dout):   
        dx = np.ones(X.shape)
        dx[X<=0] = 0
        return (dx * dout)

    
class FClayer():
    
    def __init__(self):
        pass
    
    def forward(self,X, W, b):
        
        out = np.dot(X.reshape((X.shape[0], -1)),W)+b       
        cache = (X,W,b)
        return out,cache
    
    def backward(self,dout,cache):
        x,w,b = cache
        
        db = np.sum(dout, axis = 0, keepdims = True)
        dw = np.dot(np.transpose(x.reshape(x.shape[0],-1)),dout)
        dx = np.dot(dout, np.transpose(w)).reshape(x.shape)
        
        return dx,dw,db
        
class dropoutLayer():
    def __init__(self, p=0.5):
        self.p = p
        
    def forward(self, h , mode = 'train'):
        if (mode is 'train'):
            self.U = (np.random.rand(*h.shape) < self.p ) / self.p
            out = self.U * h
            return out
        else:
            return h
        
    def backward(self, dh, dropout, mode = 'train'):
            return dh * self.U
         
        
        
        
        