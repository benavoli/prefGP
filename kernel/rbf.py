#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:10:46 2022

@author: benavoli
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag

def RBF(X1,X2,params,diag_=False):
    lengthscale=params['lengthscale']['value']
    variance   =params['variance']['value']
    if diag_==False:
        diffs = cdist(np.atleast_2d(X1)/ lengthscale, np.atleast_2d(X2) / lengthscale, metric='sqeuclidean')
    else:
        diffs = np.sum((np.atleast_2d(X1)/ lengthscale-np.atleast_2d(X2)/ lengthscale)*(np.atleast_2d(X1)/ lengthscale-np.atleast_2d(X2)/ lengthscale),axis=1)
    return variance*np.exp(-0.5 * diffs)


def BlockRBF(X1,X2,params,diag_=False):
    numblocks = int((len(params)-1)/2)
    n1 = int(X1.shape[0]/numblocks)
    n2 = int(X2.shape[0]/numblocks)
    
    B1 = RBF(X1[0:n1,:],X2[0:n2,:],dict((k,params[k+"_0"]) for k in ('lengthscale','variance')),diag_=False)
    for i in range(1,numblocks):
        pp = dict((k,params[k+"_"+str(i)]) for k in ('lengthscale','variance'))
        B2 = RBF(X1[n1*(i):n1*(i+1),:],X2[n2*(i):n2*(i+1),:],pp,diag_=False)
        B1 = block_diag(B1,B2)

    return B1