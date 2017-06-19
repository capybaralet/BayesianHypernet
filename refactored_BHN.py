#!/usr/bin/env python
import os
import time

import numpy
np = numpy
from scipy.stats import mode

import theano
floatX = theano.config.floatX
import lasagne

#from dk_hyperCNN import MCdropoutCNN
from riashat_cnn import RiashatCNN as MCdropoutCNN
from BHNs import HyperCNN

from helpers import *
from layers import *


"""

I'm trying to understand the training of BHNs.

Questions:
    1. how long do they need to be trained for?  
    2. is there *any* concern of overfitting?
    3. what is happening at the beginning (when the accuracy doesn't go up for 100+ steps)
        bad init?
    4. how does it compare to dropout?
        we should do better on tiny datasets???

"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:58:58 2017

@author: David Krueger, Chin-Wei
"""

# TODO: we should have a function for the core hypernet architecture (agnostic of whether we do WN/CNN/full Hnet)

import numpy
np = numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
RSSV = T.shared_randomstreams.RandomStateSharedVariable
floatX = theano.config.floatX

import lasagne
from lasagne import nonlinearities
rectify = nonlinearities.rectify
softmax = nonlinearities.softmax
from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy as cc
from lasagne.objectives import squared_error as se

from helpers import flatten_list
from helpers import log_normal
from helpers import SaveLoadMIXIN

from layers import LinearFlowLayer, IndexLayer, PermuteLayer, SplitLayer, ReverseLayer
from layers import CoupledDenseLayer, ConvexBiasLayer, CoupledWNDenseLayer, \
                    stochasticDenseLayer2, stochasticConv2DLayer, \
                    stochastic_weight_norm
from layers import *


"""
I would like to just have an easy way of getting the invertible transformation, etc...
"""



# TODO: the model should only define the graph!
# TODO: just fucking DECLASSIFY this shit!
class DK_BHN(SaveLoadMIXIN):
    """
    I would like to write this in one of two ways:
        1. take a primary-net and convert it to a hypernet (PREFERED)
        2. take a description of pnet, and build the hnet based on that...
        
    """
    
    max_norm = 10
    clip_grad = 5
    
    def __init__(self,
                 arch=None,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 opt='adam',
                 coupling=4,
                 pad='same',
                 stride=2,
                 pool=None,
                 uncoupled_init=False,
                 extra_linear=0):
                                  
        if arch == 'Riashat':
            kernel_width = 3
            self.kernel_width = kernel_width
            stride=1
            self.stride = stride
            pad = 'valid'
            self.pad = pad
            self.weight_shapes = [(32,1,kernel_width,kernel_width),        # -> (None, 16, 14, 14)
                                  (32,32,kernel_width,kernel_width)]       # -> (None, 16,  7,  7)
            self.args = [[32,kernel_width,stride,pad, rectify, 'none'],
                         [32,kernel_width,stride,pad, rectify, 'max']]
            self.pool_size = 5
        else:
            self.pool_size = 2
                                  

        self.n_kernels = np.array(self.weight_shapes)[:,1].sum()
        self.kernel_shape = self.weight_shapes[0][:1]+self.weight_shapes[0][2:]
        print "kernel_shape", self.kernel_shape
        self.kernel_size = np.prod(self.weight_shapes[0])
    
    
        self.num_classes = 10
        if arch == 'Riashat':
            self.num_hids = 256
        else:
            self.num_hids = 128
        self.num_mlp_layers = 1
        self.num_mlp_params = self.num_classes + \
                              self.num_hids * self.num_mlp_layers
        self.num_cnn_params = np.sum(np.array(self.weight_shapes)[:,0])
        self.num_params = self.num_mlp_params + self.num_cnn_params
        self.coupling = coupling
        self.extra_l2 = 0
        self.extra_linear = extra_linear
    
    #def __init__(self,

        self.lbda = lbda
        self.perdatapoint = perdatapoint
        self.srng = srng
        self.prior = prior
        self.__dict__.update(locals())
        
        if perdatapoint:
            self.wd1 = self.input_var.shape[0]
        else:
            self.wd1 = 1
        
    #def _get_theano_variables(self):
        self.input_var = T.matrix('input_var')
        self.input_var = T.tensor4('input_var')# <-- for CNN
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate') 
        
    #def _get_hyper_net(self):
        # inition random noise
        print self.num_params
        ep = self.srng.normal(size=(self.wd1,
                                    self.num_params),dtype=floatX)
        logdets_layers = []
        h_net = lasagne.layers.InputLayer([None,self.num_params])
        
        # mean and variation of the initial noise
        layer_temp = LinearFlowLayer(h_net)
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        if self.coupling:
            layer_temp = CoupledWNDenseLayer(h_net,200, uncoupled_init=uncoupled_init)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
            for c in range(self.coupling-1):
                h_net = PermuteLayer(h_net,self.num_params)
                
                layer_temp = CoupledWNDenseLayer(h_net,200, uncoupled_init=uncoupled_init)
                h_net = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))
        
        if self.extra_linear:
            layer_temp = ConvexBiasLayer(h_net, upweight_primary=0)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    #def _get_primary_net(self):
        
        t = np.cast['int32'](0)
        if 1:#self.dataset == 'mnist':
            p_net = lasagne.layers.InputLayer([None,1,28,28])
        print p_net.output_shape
        inputs = {p_net:self.input_var}

        #logpw = np.float32(0.)
        
        for ws, args in zip(self.weight_shapes,self.args):

            num_filters = ws[0]
            
            # TO-DO: generalize to have multiple samples?
            weight = self.weights[0,t:t+num_filters].dimshuffle(0,'x','x','x')

            num_filters = args[0]
            filter_size = args[1]
            stride = args[2]
            pad = args[3]
            nonl = args[4]
            p_net = lasagne.layers.Conv2DLayer(p_net,num_filters,
                                               filter_size,stride,pad,
                                               nonlinearity=nonl)
            p_net = stochastic_weight_norm(p_net,weight)
            
            if args[5] == 'max':
                p_net = lasagne.layers.MaxPool2DLayer(p_net,self.pool_size)
            #print p_net.output_shape
            t += num_filters

            
        for layer in range(self.num_mlp_layers):
            weight = self.weights[:,t:t+self.num_hids].reshape((self.wd1,
                                                                self.num_hids))
            p_net = lasagne.layers.DenseLayer(p_net,self.num_hids,
                                              nonlinearity=rectify)
            p_net = stochastic_weight_norm(p_net,weight)
            if self.extra_l2:
                self.l2_penalty = lasagne.regularization.regularize_layer_params_weighted({p_net: 3.5 / 128},
                        lasagne.regularization.l2)
            t += self.num_hids


        weight = self.weights[:,t:t+self.num_classes].reshape((self.wd1,self.num_classes))

        p_net = lasagne.layers.DenseLayer(p_net,self.num_classes,
                                          nonlinearity=nonlinearities.softmax)
        p_net = stochastic_weight_norm(p_net,weight)

        y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
        
        self.p_net = p_net
        self.y = y
        

    #def _get_params(self):
        
        params = lasagne.layers.get_all_params([self.h_net,self.p_net])
        self.params = list()
        for param in params:
            if type(param) is not RSSV:
                self.params.append(param)
    
        params0 = lasagne.layers.get_all_param_values([self.h_net,self.p_net])
        params = lasagne.layers.get_all_params([self.h_net,self.p_net])
        updates = {p:p0 for p, p0 in zip(params,params0)}
        self.reset = theano.function([],None,
                                      updates=updates)
        self.add_reset('init')
    
    #def _get_elbo(self):

        logdets = self.logdets
        self.logqw = - logdets
        self.logpw = self.prior(self.weights,0.,-T.log(self.lbda)).sum(1)
        self.kl = (self.logqw - self.logpw).mean()
        self.kl_term = self.kl/T.cast(self.dataset_size,floatX)
        self.logpyx = - cc(self.y,self.target_var).mean()
        self.loss = - self.logpyx + self.kl_term

        # DK - extra monitoring (TODO)
        params = self.params
        ds = self.dataset_size
        self.logpyx_grad = flatten_list(T.grad(-self.logpyx, params, disconnected_inputs='warn')).norm(2)
        self.logpw_grad = flatten_list(T.grad(-self.logpw.mean() / ds, params, disconnected_inputs='warn')).norm(2)
        self.logqw_grad = flatten_list(T.grad(self.logqw.mean() / ds, params, disconnected_inputs='warn')).norm(2)
        self.monitored = [self.logpyx, self.logpw, self.logqw,
                          self.logpyx_grad, self.logpw_grad, self.logqw_grad]
        

    #def _get_grads(self):
        grads = T.grad(self.loss, self.params)
        mgrads = lasagne.updates.total_norm_constraint(grads,
                                                       max_norm=self.max_norm)
        cgrads = [T.clip(g, -self.clip_grad, self.clip_grad) for g in mgrads]
        if self.opt == 'adam':
            self.updates = lasagne.updates.adam(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
        elif self.opt == 'momentum':
            self.updates = lasagne.updates.nesterov_momentum(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
        elif self.opt == 'sgd':
            self.updates = lasagne.updates.sgd(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
                                    
    #def _get_train_func(self):
        train = theano.function([self.input_var,
                                 self.target_var,
                                 self.dataset_size,
                                 self.learning_rate],
                                self.loss,updates=self.updates)
        self.train_func = train
        # DK - putting this here, because is doesn't get overwritten by subclasses
        self.monitor_func = theano.function([self.input_var,
                                 self.target_var,
                                 self.dataset_size,
                                 self.learning_rate],
                                self.monitored,
                                on_unused_input='warn')


    #def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))       
        

    def sample_predictions(self, inputs, nsamples=100):
        # nsamples, nexamples, nout
        return np.array([self.predict_proba(inputs) for _ in range(nsamples)])

    def get_acc(self, inputs, targets, nsamples=100):
        return (self.sample_predictions(inputs, nsamples).mean(axis=0).argmax(-1) == targets.argmax(-1)).mean()
        

if 1:#__name__ == '__main__':
    
    import argparse
    import os
    import sys
    import numpy
    
    parser = argparse.ArgumentParser()
    
    # boolean: 1 -> True ; 0 -> False
    parser.add_argument('--arch',default='Riashat',type=str)
    parser.add_argument('--bs',default=128,type=int)  
    parser.add_argument('--convex_combination',default=0,type=int)  
    parser.add_argument('--coupling',default=4,type=int)  
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--lrdecay',default=0,type=int)  
    parser.add_argument('--lr0',default=0.001,type=float)  
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--nonlinearity',default='rectify',type=str)  
    parser.add_argument('--num_train',default=100, type=int)
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--uncoupled_init',default=0, type=int) # actually, CW was right; it is decoupled when this is 0!  scale = 1 and shift = 0...
    #
    #parser.add_argument('--save_path',default=None,type=str)  
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', type=int, default=1)


    # --------------------------------------------
    # PARSE ARGS and SET-UP SAVING and RANDOM SEED
    args = parser.parse_args()
    args_dict = args.__dict__

    # save_path = filename + PROVIDED parser arguments
    flags = [flag.lstrip('--') for flag in sys.argv[1:]]
    flags = [ff for ff in flags if not ff.startswith('save_dir')]
    save_dir = args_dict.pop('save_dir')
    save_path = os.path.join(save_dir, os.path.basename(__file__) + '___' + '_'.join(flags))
    args_dict['save_path'] = save_path

    if args_dict['save']:
        # make directory for results, save ALL parser arguments
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open (os.path.join(save_path,'exp_settings.txt'), 'w') as f:
            for key in sorted(args_dict):
                f.write(key+'\t'+str(args_dict[key])+'\n')
        print( save_path)
        #assert False

    locals().update(args_dict)
    assert arch == 'Riashat'
    extra_linear = convex_combination

    if nonlinearity == 'rectify':
        nonlinearity = lasagne.nonlinearities.rectify
    elif nonlinearity == 'gelu':
        nonlinearity = gelu
    
    lbda = np.cast[floatX](args.lbda)
    clip_grad = 100
    max_norm = 100

    # SET RANDOM SEED (TODO: rng vs. random.seed)
    if seed is None:
        rng = np.random.randint(2**32 - 1)
    np.random.seed(seed)  # for reproducibility
    rng = numpy.random.RandomState(seed)
    random.seed(seed)

    # --------------------------------------------
    print "\n\n\n-----------------------------------------------------------------------\n\n\n"
    print args
    

    locals().update(args.__dict__) 

    if not os.path.exists(save_dir):
         os.makedirs(save_dir)
    

    coupling = args.coupling
    perdatapoint = args.perdatapoint
    lrdecay = args.lrdecay
    lr0 = args.lr0
    lbda = np.cast['float32'](args.lbda)
    bs = args.bs
    epochs = args.epochs
    if args.prior=='log_normal':
        prior = log_normal
    elif args.prior=='log_laplace':
        prior = log_laplace
    

    #main()

    t0 = time.time()

    # get data
    filename = '../../mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    train_x = train_x.reshape(50000,1,28,28)
    valid_x = valid_x.reshape(10000,1,28,28)
    test_x = test_x.reshape(10000,1,28,28)
    #train_x, train_y, pool_x, pool_y = split_train_pool_data(train_x, train_y)
    #train_y_multiclass = train_y.argmax(1)
    #train_x, train_y = get_initial_training_data(train_x, train_y_multiclass)
    train_y = train_y.astype('float32')
    print "Initial Training Data", train_x.shape

    # select model
    model = DK_BHN(lbda=lbda,
                        arch=arch,
                         perdatapoint=perdatapoint,
                         prior=prior,
                         coupling=coupling,
                         pad='valid',
                         stride=1,
                         uncoupled_init=uncoupled_init,
                         extra_linear=extra_linear)
                         #dataset=dataset)
    model.acc = T.eq(model.y.argmax(-1), model.target_var.argmax(-1)).mean()
    model.kl_term = model.kl/T.cast(model.dataset_size,floatX)
    model.nll = - model.logpyx
    #model.loss = - self.logpyx + self.kl_term
    train_fn = theano.function([model.input_var,
                                 model.target_var,
                                 model.dataset_size,
                                 model.learning_rate],
                                [model.loss, model.acc, model.nll, model.kl_term],updates=model.updates)
if 1:
    test_fn = theano.function([model.input_var,
                                 model.target_var,
                                 model.dataset_size],
                                [model.loss, model.acc, model.nll, model.kl_term],updates={})


    X = train_x[:num_train]
    Y = train_y[:num_train]
    Xva = valid_x[:200]
    Yva = valid_y[:200]

    import matplotlib.pyplot as plt
    plt.ion()

    loss, nll, kl, acc = [], [], [], []
    val_loss, val_nll, val_kl, val_acc = [], [], [], []

    plot_every = 50
    for step in range(100000):
        loss_, acc_, nll_, kl_ = train_fn(X, Y, X.shape[0], lr0)
        loss.append(loss_)
        nll.append(nll_)
        kl.append(kl_)
        acc.append(acc_)
        print loss_, acc_
        if step % plot_every == 0:
            # TODO: multiple samples
            val_loss_, val_acc_, val_nll_, val_kl_ = test_fn(Xva, Yva, Xva.shape[0])
            val_loss.append(val_loss_)
            val_nll.append(val_nll_)
            val_kl.append(val_kl_)
            val_acc.append( model.get_acc(Xva, Yva))
            #val_acc.append(val_acc_)
            print "                                                       ", val_loss_, val_acc[-1]
            plt.clf()
            plt.subplot(121)
            plt.plot(loss[::plot_every], 'm', label='loss')
            plt.plot(val_loss, 'm--')
            plt.plot(nll[::plot_every], 'r', label='nll')
            plt.plot(val_nll, 'r--')
            plt.plot(kl[::plot_every], 'b', label='kl')
            plt.plot(val_kl, 'b--')
            plt.legend(loc='upper right')
            #
            plt.subplot(122)
            plt.plot(acc[::plot_every], 'm')
            plt.plot(val_acc, 'm--')


