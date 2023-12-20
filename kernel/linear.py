#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:10:46 2022

@author: benavoli
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag

def Linear(X1,X2,params,diag_=False):
    rv   =np.sqrt(params['variance']['value'])
    if diag_==False:
        return (np.atleast_2d(X1)*rv)@(np.atleast_2d(X2)*rv).T
    else:
        return "error"
   


def BlockLinear(X1,X2,params,diag_=False):
    #print(params)
    numblocks = np.sum(['variance' in  c for c in list(params.keys())])
    n1 = int(X1.shape[0]/numblocks)
    n2 = int(X2.shape[0]/numblocks)
    
    B1 = Linear(X1[0:n1,:],X2[0:n2,:],dict((k,params[k+"_0"]) for k in ('variance')),diag_=False)
    for i in range(1,numblocks):
        pp = dict((k,params[k+"_"+str(i)]) for k in ('variance'))
        B2 = Linear(X1[n1*(i):n1*(i+1),:],X2[n2*(i):n2*(i+1),:],pp,diag_=False)
        B1 = block_diag(B1,B2)

    return B1