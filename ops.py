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
    tr,va,te = pickle.load(gzip.open(filename,'r'))
    tr_x,tr_y = tr
    va_x,va_y = va
    te_x,te_y = te
    enc = OneHotEncoder(10)
    tr_y = enc.fit_transform(tr_y).toarray().reshape(50000,10).astype(int)
    va_y = enc.fit_transform(va_y).toarray().reshape(10000,10).astype(int)    
    te_y = enc.fit_transform(te_y).toarray().reshape(10000,10).astype(int)
    f = lambda d:d.astype(floatX) 
    return (f(d) for d in [tr_x, tr_y, va_x, va_y, te_x, te_y])