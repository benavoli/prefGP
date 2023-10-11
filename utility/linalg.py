#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun  18 14:24:17 2021

@author: benavoli
"""
import numpy as np

from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse




def build_sparse_prefM(Pairs, nX, latent_dim):
    """
    Arguments:
        Pairs: (i, j) prefrence pairs over the rows of X
        nX: X.shape[0]
        latent_dim: number of latent dimensions of the utility
    Returns:
        (value,(row,col)) sparse matrix 
    """
    row=[]
    col=[]
    value=[]
    r=0
    for p in Pairs:
        for d in range(latent_dim):
            row.append(r)
            col.append(d*nX+p[0])
            value.append(1)
            row.append(r)
            col.append(d*nX+p[1])
            value.append(-1)
        r=r+1
    return (np.array(value), (np.array(row),np.array(col)))


def sparse_cholesky(A,typed="natural"):  
    """
    Arguments:
        A: The input matrix A must be a sparse symmetric positive-definite. 
    Returns:
        Lower sparse cholesky matrix computed using LU decomposition
    """
    #LU = splinalg.splu(A,diag_pivot_thresh=0,permc_spec="NATURAL") # sparse LU decomposition
    #return (LU.L.dot( sparse.diags(LU.U.diagonal()**0.5,format="csc") ))
    from sksparse.cholmod import cholesky
    factor = cholesky(A,ordering_method=typed)
    return factor#.L()
