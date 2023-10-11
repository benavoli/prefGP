#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 2022

@author: benavoli
"""
from jax.config import config
config.update("jax_enable_x64", True)
#config.update('jax_platform_name', 'cpu')
from jax import jit, grad, vmap
from jax import random
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.scipy.stats.norm as norm
import jax.scipy.stats.multivariate_normal as multivariate_normal
from functools import partial

from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.sparse import csc_matrix
import pypardiso
from scipy.sparse import identity
import numpy as np
from utility.paramz import DictVectorizer
from utility.linalg import sparse_cholesky
from scipy.optimize import minimize
from tqdm import tqdm
#from scipy.stats import multivariate_normal
import jax

class inference_advi:
    """
    Class GP_Laplace
    """
    def __init__(self, data, Kernel, params, log_likelihood,  
                 jitter=1e-6):
        self._Kernel = Kernel #function
        self.params=params
         #data
        self.data=data
        self.X=data["X"]
        self.fMAP=[]
        self.LambdaMAP=[]
        self.jitter=jitter
        self.log_likelihood=log_likelihood
        
                  
    

    def advi(self, loglike,n_iter=8000,progress=True,kernel_hypers_fixed=False):
        #f = f0 #np.zeros((K.shape[0],1))
        #I = identity(K.shape[0])
       

        #L = sparse_cholesky(K+I*self.jitter,typed="natural").L()
        #L_inv = pypardiso.spsolve(L.T,I)
        #Ki = L_inv@L_inv.T
        #logdet_L= np.sum(np.log(L.diagonal()))
        #d = -0.5*np.linalg.slogdet(6.28*K.toarray())[1]
        
        #set problem 
        rng = random.PRNGKey(1)
        dic = DictVectorizer()       
        init_params_kernel,bounds_hyper=dic.fit_transform(self.params)
        loglike =self._log_like()
        K = self._Kernel(self.X,self.X,jnp.exp(init_params_kernel))
        f0= jax.random.normal(rng,(K.shape[0],))*0.1
        #index traingular matrix for ADVI
        #idx = np.tril_indices(K.shape[0],  m=K.shape[0])
        #meanf = jnp.zeros(K.shape[0])
        
        
        #define likelihood and prior
        def _log_likelihood(f):
            return loglike(f[:,None]) 
        
        #def calculate_prior(f,kernel_hypers):
        #    #f = theta[:,None]
        #    K = self._Kernel(self.X,self.X,jnp.exp(kernel_hypers))
        #    M = K+self.jitter*np.eye(K.shape[0])
        #    return multivariate_normal.logpdf(f,meanf,M)
        #-0.5*f.T@Ki@f+logdet_L
        
        log_lik_fun = jit(_log_likelihood)
        #log_prior_fun = jit(calculate_prior)
        
        # define ADVI stuff
        @jit
        def target_log_density(f):
            #unnormalized target density
            return jnp.sum(log_lik_fun(f))#+jnp.sum(log_prior_fun(params,kernel_hypers))
        
      
        #def gaussian_logpdf(x, mean, Sigma):
        #    # Evaluate a single point on a diagonal multivariate Gaussian.            
        #    return jnp.sum(multivariate_normal.logpdf(x,mean,Sigma))
        
        def gaussian_sample(rng, mean, Sigma):
            # Take a single sample from a diagonal multivariate Gaussian.
            L= jax.numpy.linalg.cholesky(Sigma)
            return mean + L @ random.normal(rng, mean.shape)
        
        def kl_div(f,Sigma,M):
            f = f[:,None]
            t1 = jax.numpy.linalg.slogdet(Sigma)[1]+Sigma.shape[0]-jax.numpy.linalg.slogdet(M)[1]#
            t2 = -jax.numpy.trace(jax.numpy.linalg.solve(M, Sigma))
            t3 = -f.T@jax.numpy.linalg.solve(M, f)
            kl = 0.5*(t1 + t2 + t3)
            return jax.numpy.sum(kl)
        
        def elbo(logprob, rng, mean, Sigma):
            # Single-sample Monte Carlo estimate of the variational lower bound.
            sample = gaussian_sample(rng, mean, Sigma)
            return logprob(sample) #- gaussian_logpdf(sample, mean, L)
        
        
        def batch_elbo(target_log_density, rng, params, num_samples):
            # Average over a batch of random samples.
            rngs = random.split(rng, num_samples)
            #L = jnp.zeros((K.shape[0],K.shape[0]))
            #L = L.at[idx].set(params[1])
            kernel_hypers = params[2]
            D = params[1]
            A = jax.numpy.diag(1/(D*D)) #+self.jitter))
            if kernel_hypers_fixed==False:
                pp = kernel_hypers
            else:
                pp = init_params_kernel
            K = self._Kernel(self.X,self.X,jnp.exp(pp)) 
            M = K+self.jitter*np.eye(K.shape[0])
            Sigma=A - A@jax.numpy.linalg.solve(M+A, A) +self.jitter*jnp.eye(A.shape[0])
            vectorized_elbo = vmap(partial(elbo, target_log_density), in_axes=(0, None, None))
            expected_log_like = jnp.nanmean(vectorized_elbo(rngs, params[0], Sigma))
            return expected_log_like+kl_div(params[0],Sigma,M)#-params[2]**2
        
        num_samples = 600
        @jit
        def objective(params, t):
            rng = random.PRNGKey(t)
            return jnp.sum(-batch_elbo(target_log_density, rng, params, num_samples))
        
        def callback(params, t):
            print(f"Iteration {t} lower bound {objective(params, t)}")
            print(params[2])

 
        # Set up optimizer.
        D = K.shape[0]
        init_mean = f0 #jnp.zeros(D)
        #init_std  = np.tril(np.diag(jax.random.normal(rng,(D,)))*0.5)[idx]
        log_init_std = jax.numpy.ones((D,))
        init_params = (init_mean, log_init_std, init_params_kernel)
        opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
        opt_state = opt_init(init_params)

        
        @jit
        def update(i, opt_state):
            params = get_params(opt_state)            
            gradient = grad(objective)(params, i)
            return opt_update(i, gradient, opt_state)
        
        
        #Optimizing variational parameters
        pbar = tqdm(total = n_iter, disable=1-progress, position=0, leave=True)
        for t in range(n_iter):
            opt_state = update(t, opt_state)
            params = get_params(opt_state)
            pbar.set_description(f" lower bound {objective(params, t)}", refresh=True)
            pbar.update(1)
            #callback(params, t)
            if np.isnan(objective(params, t)):
                break
        pbar.close()
            #
        #print([np.sum(log_lik_fun(params[0])),log_prior_fun(params[0])])
        #lla = -objective(params, t)#ELBO
        
        f = params[0]
        D = params[1]
        log_kernel_hypers = params[2]
        A = jax.numpy.diag(1/(D*D+self.jitter))
        K = self._Kernel(self.X,self.X,jnp.exp(log_kernel_hypers)) 
        M = K#+self.jitter*np.eye(K.shape[0])
        Sigma = A - A@jax.numpy.linalg.solve(M+A, A) +self.jitter*jnp.eye(A.shape[0])
        
            
       
        return f, Sigma, log_kernel_hypers, params
    
    def _log_like(self):
        def loglike(f):
            return self.log_likelihood(f,self.data,self.params)
        return loglike
    
    def optimize(self,n_iter,progress=True,
                 kernel_hypers_fixed=False):
        #dic = DictVectorizer()
        
        loglike =self._log_like()
        f, Sigma, log_kernel_hypers, advi_params = self.advi(loglike,
                                                             kernel_hypers_fixed=kernel_hypers_fixed,
                                                             n_iter=n_iter)
        self.meanVI = f[:,None]
        self.SigmaVI = Sigma
        self.log_kernel_hypers = log_kernel_hypers
        self.advi_params=advi_params
       
        
    
    def predict(self, Xpred):
        Kxx = self._Kernel(self.X,self.X,jnp.exp(self.log_kernel_hypers))
        Kxz = self._Kernel(self.X,Xpred,jnp.exp(self.log_kernel_hypers))
        Kzz = self._Kernel(Xpred,Xpred,jnp.exp(self.log_kernel_hypers))
        
        L = np.linalg.cholesky(Kxx+self.jitter*np.eye(Kxx.shape[0]))
        L_inv = np.solve(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T

        M=np.linalg.dot(IKxx-IKxx@self.SigmaI@IKxx,Kxz)
        return Kxz.T@IKxx@self.meanVI,  Kzz -Kxz.T@M #()@Kxz


    
    def predict_noiseless(self,Xpred,full_cov=True):
        return self.predict(Xpred)
    
    
        