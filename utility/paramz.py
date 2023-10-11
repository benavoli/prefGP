#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  18 14:24:17 2020

@author: benavoli
"""
import numpy as np


# Transformations
class logexp:
    def __init__(self):
        self.name='logexp'
    def transform(self,x):
        return np.log(x)
    def inverse_transform(self,x):
        return np.exp(x)
    
class identity:
    def __init__(self):
        self.name='identity'
    def transform(self,x):
        return x
    def inverse_transform(self,x):
        return x

# Params dictionary        
class DictVectorizer():
    def __init__(self):
        self.Name=[]
        self.Size=[]
        self.Bounds=[]
        self.Transforms=[]

    def fit_transform(self,x):
        self.Name=[]
        self.Size=[]
        Vec=np.array([])
        TBounds=np.empty((0,2), float)
        self.Transforms=[]
        for f, v in x.items():
            self.Name.append(f)
            self.Size.append(v['value'].shape)
            assert v['value'].size==v['range'].shape[0]
            value=np.clip(v['value'].flatten(),np.vstack(v['range'])[:,0],np.vstack(v['range'])[:,1])#clip values outside bounds
            transformed=v['transform'].transform(value)#apply transformation to parameters
            Vec = np.hstack([Vec,transformed]) #clip values outside bounds
            TBounds = np.vstack([TBounds,v['transform'].transform(v['range'])])  
            self.Transforms.append(v['transform'])       
        return Vec, TBounds
    
    def inverse_transform(self,Vec,Bounds):
        pp={}
        prev = 0
        for i in range(len(self.Size)):
            if len(self.Size[i])==1:
                value=Vec[prev:prev+self.Size[i][0]].reshape(self.Size[i])
                pp[self.Name[i]]={'value':self.Transforms[i].inverse_transform(value),
                                  'range':self.Transforms[i].inverse_transform(Bounds[prev:prev+self.Size[i][0]]),
                                  'transform':self.Transforms[i]}
                prev = prev+self.Size[i][0]
            else:
                value=Vec[prev:prev+self.Size[i][0]*self.Size[i][1]].reshape(self.Size[i])
                pp[self.Name[i]]={'value':self.Transforms[i].inverse_transform(value),
                                  'range':self.Transforms[i].inverse_transform(Bounds[prev:prev+self.Size[i][0]*self.Size[i][1]]),
                                  'transform':self.Transforms[i]}
                prev = prev+self.Size[i][0]*self.Size[i][1]
        return pp



    