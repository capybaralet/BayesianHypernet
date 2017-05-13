# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:29:14 2017

@author: Chin-Wei
"""

import theano
import theano.tensor as T
import numpy as np
floatX = theano.config.floatX
import matplotlib.pyplot as plt
import lasagne
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import get_output

conv = lasagne.theano_extensions.conv


delta = 0.001

class CoupledDenseLayer(lasagne.layers.base.Layer):    
    def __init__(self, incoming, num_units, W=init.Normal(0.01),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(CoupledDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1]/2))

        self.W1 = self.add_param(W, (num_inputs, num_units), name="W1")
        self.W21 = self.add_param(W, (num_units, num_inputs), name="W21")
        self.W22 = self.add_param(W, (num_units, num_inputs), name="W22")
        if b is None:
            self.b1 = None
            self.b21 = None
            self.b22 = None
        else:
            self.b1 = self.add_param(b, (num_units,), name="b1",
                                     regularizable=False)
            self.b21 = self.add_param(b, (num_inputs,), name="b21",
                                      regularizable=False)
            self.b22 = self.add_param(b, (num_inputs,), name="b22",
                                      regularizable=False)
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        num_inputs = input.shape[1]
        input1 = input[:,:num_inputs/2]
        input2 = input[:,num_inputs/2:]
        output1 = input1
        
        a = T.dot(input1,self.W1)
        if self.b1 is not None:
            a = a + self.b1
        h = self.nonlinearity(a)
        
        s_ = T.dot(h,self.W21)
        if self.b21 is not None:
            s_ = s_ + self.b21
        s = T.nnet.softplus(s_) + 0.001
        ls = T.log(s)
        
        m = T.dot(h,self.W22)
        if self.b22 is not None:
            m = m + self.b22
            
        output2 = s * input2 + m
        output = T.concatenate([output1,output2],1)
        
        return output, ls.sum(1)


class CoupledConv1DLayer(lasagne.layers.base.Layer):
    """
    shape[1] should be even number
    """
    def __init__(self, incoming, num_units, filter_size,
                 W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=True, convolution=conv.conv1d_mc0, **kwargs):
        super(CoupledConv1DLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.filter_size = filter_size
        self.num_units = num_units
        self.flip_filters = flip_filters
        self.convolution = convolution

        
        W1_shape = (num_units,1,filter_size)
        W21_shape = (1,num_units,filter_size)
        W22_shape = (1,num_units,filter_size)
        
        self.W1 = self.add_param(W, W1_shape, name="W1")
        self.W21 = self.add_param(W, W21_shape, name="W21")
        self.W22 = self.add_param(W, W22_shape, name="W22")
        if b is None:
            self.b1 = None
            self.b21 = None
            self.b22 = None
        else:
            self.b1 = self.add_param(b, (num_units,), name="b1",
                                     regularizable=False)
            self.b21 = self.add_param(b, (1,), name="b21",
                                      regularizable=False)
            self.b22 = self.add_param(b, (1,), name="b22",
                                      regularizable=False)
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        border_mode = 'half'
        
        num_units = self.num_units
        filter_size = self.filter_size
        W1_shape = (num_units,1,filter_size)
        W21_shape = (1,num_units,filter_size)
        W22_shape = (1,num_units,filter_size)
        num_inputs = self.input_shape[1]
        input1 = input[:,:num_inputs/2]
        input2 = input[:,num_inputs/2:]
        output1 = input1
        
        input_shape = self.input_shape 
        input1_shape = (input_shape[0], 1, num_inputs/2)
        h_shape = (input_shape[0], num_units, num_inputs/2)
        conved = self.convolution(input1.reshape((-1,1,num_inputs/2)), self.W1,
                                  input1_shape, W1_shape,
                                  subsample=(1,),
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)

        if self.b1 is not None:
            a = conved + self.b1.dimshuffle('x',0,'x')
        h = self.nonlinearity(a)
        
        conved = self.convolution(h, self.W21,
                                  h_shape, W21_shape,
                                  subsample=(1,),
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        
        if self.b21 is not None:
            s_ = conved + self.b21.dimshuffle('x',0,'x')
        s = T.nnet.softplus(s_).reshape((-1,num_inputs/2)) + delta
        ls = T.log(s)
        
        conved = self.convolution(h, self.W22,
                                  h_shape, W22_shape,
                                  subsample=(1,),
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        
        if self.b22 is not None:
            m = conved + self.b22.dimshuffle('x',0,'x')
        m = m.reshape((-1,num_inputs/2))
        
        output2 = s * input2 + m
        output = T.concatenate([output1,output2],1)

        return output, ls.sum(1)


class LinearFlowLayer(lasagne.layers.base.Layer):    
    def __init__(self, incoming, W=init.Normal(0.01,-3),
                 b=init.Normal(0.01,0),
                 **kwargs):
        super(LinearFlowLayer, self).__init__(incoming, **kwargs)
        
        num_inputs = int(np.prod(self.input_shape[1]))

        self.W = self.add_param(W, (num_inputs,), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_inputs,), name="b",
                                    regularizable=False)
            
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        s = T.exp(self.W) + delta
        output = input * s
        if self.b is not None:
            output = output + self.b
        
        return output, (T.ones_like(input)*T.log(s)).sum(1)


class IndexLayer(lasagne.layers.Layer):
    
    def __init__(self, incoming, index, **kwargs):
        super(IndexLayer, self).__init__(incoming, **kwargs)
        self.index = index

    def get_output_for(self, input, **kwargs):
        return input[self.index] 


class PermuteLayer(lasagne.layers.Layer):
    
    def __init__(self, incoming, num_units, axis=-1, **kwargs):
        super(PermuteLayer, self).__init__(incoming, **kwargs)
        indices = np.random.permutation(np.arange(num_units))
        while np.all(indices == np.arange(num_units)):
            indices = np.random.permutation(np.arange(num_units))
        self.indices = indices
        self.axis = axis
        
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self,input, **kwargs):
        
        slc = [slice(None)] * input.ndim
        slc[self.axis] = self.indices
        return input[slc]


if __name__ == '__main__':
    
    """
    an example of using invertible transformation to fit a complicated 
    density function that is hard to sample from
    """
    
    def U(Z):
        """ Toroid """
        z1 = Z[:, 0]
        z2 = Z[:, 1]
        R = 5.0
        return - 2*(R-(z1**2+.5*z2**2)**0.5)**2 
        
    
    print 'building model'
    logdets_layers = []
    layer = lasagne.layers.InputLayer([None,2])
    
    layer_temp = LinearFlowLayer(layer)
    layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    layer_temp = CoupledDenseLayer(layer,100)
    layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    layer = PermuteLayer(layer,2)
    
    layer_temp = CoupledDenseLayer(layer,100)
    layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    
    ep = T.matrix('ep')
    z = lasagne.layers.get_output(layer,ep)
    logdets = sum([get_output(logdet,ep) for logdet in logdets_layers])
    
    logq = - logdets
    logp = U(z)
    losses = logq - logp
    loss = losses.mean()
    
    params = lasagne.layers.get_all_params(layer)
    updates = lasagne.updates.adam(loss,params,0.001)
    
    train = theano.function([ep],loss,updates=updates)
    
    z0 = (ep - params[1]) / T.exp(params[0])
    logq_ = sum([get_output(logdet,z0) for logdet in logdets_layers])
    samples = theano.function([ep],z)

    print 'starting training'
    for i in range(20000):
        spl = np.random.randn(128,2).astype(floatX)
        l = train(spl)
    
        if i%1000==0:
            print l
    
    print "\nvisualizing"
    prior_noise = T.matrix('prior_noise')
    density = U(prior_noise)
    f0 = theano.function([prior_noise],density)
    
    fig = plt.figure()
    
    ax = fig.add_subplot(1,2,1)
    x = np.linspace(-10,10,1000)
    y = np.linspace(-10,10,1000)
    xx,yy = np.meshgrid(x,y)
    X = np.concatenate((xx.reshape(1000000,1),yy.reshape(1000000,1)),1)
    X = X.astype(floatX)
    Z = f0(X).reshape(1000,1000)
    ax.pcolormesh(xx,yy,np.exp(Z))
    ax.axis('off')
    
    ax = fig.add_subplot(1,2,2)
    Z0 = spl = np.random.randn(100000,2).astype(floatX)
    Zs = samples(Z0)
    XX = Zs[:,0]
    YY = Zs[:,1]
    plot = ax.hist2d(XX,YY,100)
    plt.xlim((-10,10))
    plt.ylim((-10,10))
    plt.axis('off')
    
    
    plt.savefig('autoregressive_ex_toroid.jpg')






