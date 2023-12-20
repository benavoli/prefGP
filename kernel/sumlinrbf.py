#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:10:46 2022

@author: benavoli
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag

def LinearRBF(X1,X2,params,diag_=False):
    lengthscale=params['rbf_lengthscale']['value']
    variance   =params['rbf_variance']['value']
    rv   =np.sqrt(params['lin_variance']['value'])
    if diag_==False:
        cov1 = (np.atleast_2d(X1)*rv)@(np.atleast_2d(X2)*rv).T
        diffs = cdist(np.atleast_2d(X1)/ lengthscale, np.atleast_2d(X2) / lengthscale, metric='sqeuclidean')
        cov2 = variance*np.exp(-0.5 * diffs)
        return cov1+cov2
    else:
        return "error"
   

