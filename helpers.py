#!/usr/bin/env python



##################################################################
##################################################################
# AL_helpers.py
#!/usr/bin/env python
#from ops import load_mnist
#from utils import log_normal, log_laplace
import numpy
np = numpy
import random
random.seed(5001)
from lasagne import layers
from scipy.stats import mode

import time
import os


def riashat_to_categorical(y):
    num_classes=10
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1

    return categorical



def split_train_pool_data(X_train, y_train):

    X_train_All = X_train
    y_train_All = y_train

    random_split = np.asarray(random.sample(range(0,X_train_All.shape[0]), X_train_All.shape[0]))

    X_train_All = X_train_All[random_split, :, :, :]
    y_train_All = y_train_All[random_split]

    X_pool = X_train_All[10000:50000, :, :, :]
    y_pool = y_train_All[10000:50000]


    X_train = X_train_All[0:10000, :, :, :]
    y_train = y_train_All[0:10000]


    return X_train, y_train, X_pool, y_pool

def get_initial_training_data(X_train_All, y_train_All):
    #training data to have equal distribution of classes
    idx_0 = np.array( np.where(y_train_All==0)  ).T
    idx_0 = idx_0[0:2,0]
    X_0 = X_train_All[idx_0, :, :, :]
    y_0 = y_train_All[idx_0]

    idx_1 = np.array( np.where(y_train_All==1)  ).T
    idx_1 = idx_1[0:2,0]
    X_1 = X_train_All[idx_1, :, :, :]
    y_1 = y_train_All[idx_1]

    idx_2 = np.array( np.where(y_train_All==2)  ).T
    idx_2 = idx_2[0:2,0]
    X_2 = X_train_All[idx_2, :, :, :]
    y_2 = y_train_All[idx_2]

    idx_3 = np.array( np.where(y_train_All==3)  ).T
    idx_3 = idx_3[0:2,0]
    X_3 = X_train_All[idx_3, :, :, :]
    y_3 = y_train_All[idx_3]

    idx_4 = np.array( np.where(y_train_All==4)  ).T
    idx_4 = idx_4[0:2,0]
    X_4 = X_train_All[idx_4, :, :, :]
    y_4 = y_train_All[idx_4]

    idx_5 = np.array( np.where(y_train_All==5)  ).T
    idx_5 = idx_5[0:2,0]
    X_5 = X_train_All[idx_5, :, :, :]
    y_5 = y_train_All[idx_5]

    idx_6 = np.array( np.where(y_train_All==6)  ).T
    idx_6 = idx_6[0:2,0]
    X_6 = X_train_All[idx_6, :, :, :]
    y_6 = y_train_All[idx_6]

    idx_7 = np.array( np.where(y_train_All==7)  ).T
    idx_7 = idx_7[0:2,0]
    X_7 = X_train_All[idx_7, :, :, :]
    y_7 = y_train_All[idx_7]

    idx_8 = np.array( np.where(y_train_All==8)  ).T
    idx_8 = idx_8[0:2,0]
    X_8 = X_train_All[idx_8, :, :, :]
    y_8 = y_train_All[idx_8]

    idx_9 = np.array( np.where(y_train_All==9)  ).T
    idx_9 = idx_9[0:2,0]
    X_9 = X_train_All[idx_9, :, :, :]
    y_9 = y_train_All[idx_9]

    X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )
    
    y_train = riashat_to_categorical(y_train)

    return X_train, y_train





##################################################################
##################################################################
# helpers.py


import theano
import theano.tensor as T
import numpy
np = numpy

# from Hendrycks
def gelu_fast(x):
    return 0.5 * x * (1 + T.tanh(T.sqrt(2 / np.pi) * (x + 0.044715 * T.pow(x, 3))))
gelu = gelu_fast


  
######################
class SaveLoadMIXIN(object):
    """
    These could use set/get _all_param_values, if we're willing to use self.layer instead of self.params...
    (just based on https://github.com/Lasagne/Lasagne/blob/06e4ad666873bf9e5a0f914386a7f0bd80bb341a/lasagne/layers/helper.py)
    """
    def save(self, save_path):
        np.save(save_path, [p.get_value() for p in self.params])

    def load(self, save_path):
        # LOAD lasagne.layers.set_all_param_values([h_layer, layer], np.load(save_path + '_params_best.npy'))
        values = np.load(save_path)

        if len(self.params) != len(values):
            raise ValueError("mismatch: got %d values to set %d parameters" %
                             (len(values), len(self.params)))

        for p, v in zip(self.params, values):
            if p.get_value().shape != v.shape:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" %
                                 (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

    # instead of saving/loading to disk, it may be faster to keep the reset params as attributes
    def add_reset(self, name):
        """
        store current params in self.reset_dict using name as a key
        """
        if not 'reset_dict' in self.__dict__.keys():
            self.reset_dict = {}
        current_params = [p.get_value() for p in self.params]#lasagne.layers.get_all_param_values(self.layer)
        updates = {p:p0 for p, p0 in zip(self.params,current_params)}
        reset_fn = theano.function([],None, updates=updates)
        # 
        self.reset_dict[name] = reset_fn

    def call_reset(self, name):
        self.reset_dict[name]()
         
  
######################

def flatten_list(plist):
    return T.concatenate([p.flatten() for p in plist])


def plot_dict(dd):
    from pylab import *
    figure()
    for kk, vv in dd.items():
        plot(vv, label=kk)
    legend()

######################

def get_mushrooms():
    from mushroom_data import X,Y
    from lasagne.objectives import squared_error
    return X, Y, None, squared_error

def get_mnist():
    pass

def get_task(task_name):
    """
    returns:
        X, Y, output_function, loss_function, {other}
    """
    pass


######################
# load_cifar10
# code repurposed from the tf-learn library
import sys
import os
import pickle
import numpy as np
from six.moves import urllib
import tarfile

def to_categorical(y, nb_classes):
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

# load training and testing data
def load_data10(randomize=True, return_val=False, one_hot=False, dirname="cifar-10-batches-py", mnistify=False):

    def load_batch(fpath):
        with open(fpath, 'rb') as f:
            #d = pickle.load(f, encoding='latin1')
            d = pickle.load(f)
        data = d["data"]
        labels = d["labels"]
        return data, labels


    def maybe_download(filename, source_url, work_directory):
        if not os.path.exists(work_directory):
            os.mkdir(work_directory)
        filepath = os.path.join(work_directory, filename)
        if not os.path.exists(filepath):
            print("Downloading CIFAR 10...")
            filepath, _ = urllib.request.urlretrieve(source_url + filename,
                                                     filepath)
            statinfo = os.stat(filepath)
            print(('CIFAR 10 downloaded', filename, statinfo.st_size, 'bytes.'))
            untar(filepath)
        return filepath


    def untar(fname):
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname)
            tar.extractall()
            tar.close()
            print("File Extracted in Current Directory")
        else:
            print("Not a tar.gz file: '%s '" % sys.argv[0])

    tarpath = maybe_download("cifar-10-python.tar.gz",
                             "http://www.cs.toronto.edu/~kriz/", dirname)
    X_train = []
    Y_train = []

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)

    X_test, Y_test = load_batch(os.path.join(dirname, 'test_batch'))

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                         X_train[:, 2048:])) / 255.
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                        X_test[:, 2048:])) / 255.
    X_test = np.reshape(X_test, [-1, 32, 32, 3])

    if randomize is True:
        test_perm = np.array(np.random.permutation(X_test.shape[0]))
        X_test = X_test[test_perm]
        Y_test = np.asarray(Y_test)
        Y_test = Y_test[test_perm]

        perm = np.array(np.random.permutation(X_train.shape[0]))
        X_train = X_train[perm]
        Y_train = np.asarray(Y_train)
        Y_train = Y_train[perm]
    if return_val:
        X_train, X_val = np.split(X_train, [45000])     # 45000 for training, 5000 for validation
        Y_train, Y_val = np.split(Y_train, [45000])

        if one_hot:
            Y_train, Y_val, Y_test = to_categorical(Y_train, 10), to_categorical(Y_val, 10), to_categorical(Y_test, 10)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
        else:
            return X_train, Y_train, X_val, Y_val, X_test, Y_test
    else:
        if one_hot:
            Y_train, Y_test = to_categorical(Y_train, 10), to_categorical(Y_test, 10)
            return X_train, Y_train, X_test, Y_test
        else:
            return X_train, Y_train, X_test, Y_test


##################################################################
##################################################################
# ops.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:54:50 2017

@author: Chin-Wei
"""


import cPickle as pickle
import gzip
from sklearn.preprocessing import OneHotEncoder
floatX = 'float32'

def load_mnist(filename):
    try:
        tr,va,te = pickle.load(gzip.open('data/mnist.pkl.gz','r'))
    except:
        tr,va,te = pickle.load(gzip.open(filename,'r'))
    tr_x,tr_y = tr
    va_x,va_y = va
    te_x,te_y = te
    # doesn't work on hades :/ 
    enc = OneHotEncoder(10)
    tr_y = enc.fit_transform(tr_y.reshape((-1,1))).toarray().reshape(50000,10).astype(int)
    va_y = enc.fit_transform(va_y.reshape((-1,1))).toarray().reshape(10000,10).astype(int)    
    te_y = enc.fit_transform(te_y.reshape((-1,1))).toarray().reshape(10000,10).astype(int)
    f = lambda d:d.astype(floatX) 
    return (f(d) for d in [tr_x, tr_y, va_x, va_y, te_x, te_y])
    

def load_cifar10(filename):
    tr_x, tr_y, te_x, te_y = pickle.load(open(filename,'r'))
    enc = OneHotEncoder(10)
    tr_y = enc.fit_transform(tr_y).toarray().reshape(50000,10).astype(int)
    te_y = enc.fit_transform(te_y).toarray().reshape(10000,10).astype(int)
    f = lambda d:d.astype(floatX) 
    return (f(d) for d in [tr_x, tr_y, te_x, te_y])


##################################################################
##################################################################
# utils.py
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:39:54 2017

@author: Chin-Wei
"""

import theano.tensor as T
import numpy as np

from lasagne.init import Normal
from lasagne.init import Initializer, Orthogonal

c = - 0.5 * T.log(2*np.pi)

def log_sum_exp(A, axis=None, sum_op=T.sum):

    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(sum_op(T.exp(A - A_max), axis=axis, keepdims=True)) + A_max

    if axis is None:
        return B.dimshuffle(())  # collapse to scalar
    else:
        if not hasattr(axis, '__iter__'): axis = [axis]
        return B.dimshuffle([d for d in range(B.ndim) if d not in axis])  
        # drop summed axes

def log_mean_exp(A, axis=None,weights=None):
    if weights:
        return log_sum_exp(A, axis, sum_op=weighted_sum(weights))
    else:
        return log_sum_exp(A, axis, sum_op=T.mean)


def weighted_sum(weights):
    return lambda A,axis,keepdims: T.sum(A*weights,axis=axis,keepdims=keepdims)    


def log_stdnormal(x):
    return c - 0.5 * x**2 


def log_normal(x,mean,log_var,eps=0.0):
    return c - log_var/2. - (x - mean)**2 / (2. * T.exp(log_var) + eps)


def log_laplace(x,mean,inv_scale,epsilon=1e-7):
    return - T.log(2*(inv_scale+epsilon)) - T.abs_(x-mean)/(inv_scale+epsilon)


def log_scale_mixture_normal(x,m,log_var1,log_var2,p1,p2):
    axis = x.ndim
    log_n1 = T.log(p1)+log_normal(x,m,log_var1)
    log_n2 = T.log(p2)+log_normal(x,m,log_var2)
    log_n_ = T.stack([log_n1,log_n2],axis=axis)
    log_n = log_sum_exp(log_n_,-1)
    return log_n.sum(-1)


def softmax(x,axis=1):
    x_max = T.max(x, axis=axis, keepdims=True)
    exp = T.exp(x-x_max)
    return exp / T.sum(exp, axis=axis, keepdims=True)



# inds : the indices of the examples you wish to evaluate
#   these should probably be ALL of the inds, OR be randomly sampled
def MCpred(X, predict_probs_fn=None, num_samples=100, inds=None, returns='preds', num_classes=10):
    if inds is None:
        inds = range(len(X))
    rval = np.empty((num_samples, len(inds), num_classes))
    for ind in range(num_samples):
        rval[ind] = predict_probs_fn(X[inds])
    if returns == 'samples':
        return rval
    elif returns == 'probs':
        return rval.mean(0)
    elif returns == 'preds':
        return rval.mean(0).argmax(-1)


    
    
# TODO
class DanNormal(Initializer):
    def __init__(self, initializer=Normal, nonlinearity='relu', c01b=False, dropout_p=0.):
        if nonlinearity == 'relu':
            g1 = g2 = .5
        elif nonlinearity == 'gelu':
            g1 = .425
            g2 = .444

        p = 1 - dropout_p
        self.denominator = (g1 / p + p * g2)**.5

        self.__dict__.update(locals())

    def sample(self, shape):
        if self.c01b:
            assert False
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            n1, n2 = shape[0], shape[3]
            receptive_field_size = shape[1] * shape[2]
        else:
            if len(shape) < 2:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

            n1, n2 = shape[:2]
            receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        # TODO: orthogonal
        return self.initializer(std=std).sample(shape)


##################################################################
##################################################################
# utils_dqn.py

import os
import numpy as np


def get_last_folder_id(folder_path):
    t = 0
    for fn in os.listdir('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/gym_examples/DQN_Experiments'):
        t = max(t, int(fn))
    return t


def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
