import numpy as np
import layers as ly


class ConvReLU():            #  conv layer followed by a relu activation unit
    
    def __init__(self):    
        self.convlayer = ly.ConvLayer()
        self.relu      = ly.ReLU()
        
    def forward(self, x, w, bias, padding, stride ):
        convout,X_flat  = self.convlayer.forward_fast(x,w,bias,padding, stride)
        relu_out = self.relu.forward(convout)
        relu_cache = (convout)
        conv_cache = (X_flat, x.shape,w,bias, padding,stride)
        cache = (conv_cache, relu_cache)
        return relu_out , cache
    
    def backward(self, dout, cache):
        conv_cache, relu_cache = cache
        convout = relu_cache
        x,x_shape,w,b,p,s = conv_cache
        dh = self.relu.backward(convout, dout)
        dx, dw,db = self.convlayer.backward_fast(dh, x ,x_shape, w,b, p, s)
        return dx,dw,db
    
class ConvReLUPool():
    
    def __init__(self, poolF, poolS):
        self.convRelu = ConvReLU()
        self.pool = ly.PoolLayer()
        self.poolF = poolF
        self.poolS = poolS
        
    def forward(self,x,w, bias,padding, stride):
        convRelu_out, convRelu_cache = self.convRelu.forward(x,w,bias,padding,stride)
        pool_out,pool_cache = self.pool.forward_fast(convRelu_out, self.poolF, self.poolS)
        #pool_cache = (convRelu_out)
        cache = (convRelu_cache, pool_cache)
        return pool_out , cache
    
    def backward(self, dout ,cache ):
        convRelu_cache , pool_cache = cache
        #pool_x = pool_cache
        dout_convRelu = self.pool.backward_fast(dout, pool_cache)
        dx,dw,db = self.convRelu.backward(dout_convRelu, convRelu_cache)
        return dx,dw,db
    
class FCReLU():
    
    def __init__(self):
        self.fc = ly.FClayer()
        self.relu = ly.ReLU()
    
    def forward(self,X,W,b):
        fc_out, fc_cache = self.fc.forward(X,W,b)
        relu_out = self.relu.forward(fc_out)
        relu_cache = fc_out
        cache = (fc_cache, relu_cache)
        return relu_out, cache
    
    def backward(self, dout, cache):
        fc_cache, relu_cache = cache
        relu_x = relu_cache
        dout_relu = self.relu.backward(relu_x,dout)
        dx,dw,db = self.fc.backward(dout_relu, fc_cache)
        return dx,dw,db
    