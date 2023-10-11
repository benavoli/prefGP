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
from functools import partial
import numpy as np
from utility.paramz import DictVectorizer
from tqdm import tqdm
import jax
from functools import partial

class inference_advi:
    """
    Class GP_Laplace
    """
    def __init__(self, data, Kernel, params, log_likelihood,  
                 jitter=1e-4):
        self._Kernel = Kernel #function
        self.params=params #dictionary of kernel parameters
        #data
        self.data=data
        self.X=data["X"]
        
        self.jitter=jitter
        self.log_likelihood=log_likelihood
        self.meanVI = []#mean ADVI
        self.SigmaVI = [] #cov ADVI
        self.log_kernel_hypers = [] # lof of kernel hypers
        self.advi_params = [] # advi_params
        
                  
    

    def advi(self, niterations, loglike,init_params_kernel,progress=True,
             kernel_hypers_fixed=False,init_f=[]):
    
        #set problem 
        
        Klist = self._Kernel(self.X,self.X,jnp.exp(init_params_kernel))
        K = jax.scipy.linalg.block_diag(
            *Klist)
            
        L = np.linalg.cholesky(K+1e-4*np.eye(K.shape[0]))
        L_inv = np.linalg.solve(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T
        
        num_latent = len(Klist)
        #init mean for ADVI
        

        
        
        #define likelihood and prior
        def _log_likelihood(f):
            return loglike(f[:,None]) 
        log_lik_fun = jit(_log_likelihood)
        
        from jax.scipy.optimize import minimize        
        def logjoint(f):
            return -log_lik_fun(f)+f.T@IKxx@f
       
        #find MAP
        if init_f==[]:
            best_f=np.inf
            best_x=[]
            for rnd in range(10):
                rng = random.PRNGKey(rnd)
                f0= jax.random.normal(rng,(K.shape[0],))*0.5
                res=minimize(jit(logjoint),f0,
                     #bounds=[[-0.1,0.1]]*len(f0),
                     options={'maxiter': 10000}, method="BFGS")
                print(res.fun)
                if res.fun<best_f:
                    best_f=res.fun
                    best_x=res.x
            f0 = best_x
            self.MAP = f0
            print(best_f)
        else:
            f0=init_f
            print(logjoint(f0))
            res=minimize(jit(logjoint),f0,
                         #bounds=[[-0.1,0.1]]*len(f0),
                         options={'maxiter': 10000}, method="BFGS")
            print(res.fun)
            self.MAP = res.x
            f0=res.x
        
        
        
        # define ADVI functions
        @jit
        def target_log_density(f):
            #unnormalized log-likelihood density
            return jnp.sum(log_lik_fun(f))
        
        
        def gaussian_sample(rng, mean, Sigma):
            # Take a single sample from a  multivariate Gaussian.
            L= jax.numpy.linalg.cholesky(Sigma)
            return mean + L @ random.normal(rng, mean.shape)
        
        '''
        def kl_div(f,Sigma,M):
            #kl divergence from GP prior and ADI multivariate Normal
            f = f[:,None]
            t1 = jax.numpy.linalg.slogdet(Sigma)[1]+Sigma.shape[0]-jax.numpy.linalg.slogdet(M)[1]#
            t2 = -jax.numpy.trace(jax.numpy.linalg.solve(M, Sigma))
            t3 = -f.T@jax.numpy.linalg.solve(M, f)
            kl = 0.5*(t1 + t2 + t3)
            return jax.numpy.sum(kl)
        '''
        
        def exp_likelihood(logprob, rng, mean, Sigma):
            # Single-sample Monte Carlo estimate of the variational lower bound.
            sample = gaussian_sample(rng, mean, Sigma)
            return logprob(sample) 
        
        
        def batch_elbo(target_log_density, rng, params, num_samples):
            # Average over a batch of random samples.
            rngs = random.split(rng, num_samples)
            #L = jnp.zeros((K.shape[0],K.shape[0]))
            #L = L.at[idx].set(params[1])
            kernel_hypers = params[2]
            D = params[1]
            
            if kernel_hypers_fixed==False:
                pp = kernel_hypers
            else:
                pp = init_params_kernel
            Klist = self._Kernel(self.X,self.X,jnp.exp(pp)) 
            f = params[0][:,None]
            logdetM = 0.0
            quadratic_term = 0.0
            IM1 = []
            IM2 = []
            for i in range(num_latent):
                R = Klist[i]
                E = D[i*R.shape[0]:(i+1)*R.shape[0]]
                A1 = jax.numpy.diag(1/(E*E+self.jitter))                
                M = R+self.jitter*np.eye(R.shape[0])
                logdetM = logdetM + jax.numpy.linalg.slogdet(M)[1]
                fp = f[i*R.shape[0]:(i+1)*R.shape[0]]
                quadratic_term = quadratic_term+fp.T@jax.numpy.linalg.solve(M, fp)
                IM1.append(jax.numpy.linalg.solve(M+A1, A1))
                IM2.append(jax.numpy.linalg.solve(M, jnp.eye(R.shape[0])))
                
                
            #IM, _ = jax.lax.scan(step, [], np.arange(2))
            IM_A = jax.scipy.linalg.block_diag(*IM1)
            A = jax.numpy.diag(1/(D*D+self.jitter))
            #M = jax.scipy.linalg.block_diag(*Klist) +self.jitter*np.eye(A.shape[0])
            Sigma=A - A@IM_A +self.jitter*jnp.eye(A.shape[0])
            vectorized_elbo = vmap(partial(exp_likelihood, target_log_density), in_axes=(0, None, None))
            expected_log_like = jnp.nanmean(vectorized_elbo(rngs, params[0], Sigma))
            
            #KL divegergence
            t1 = jax.numpy.linalg.slogdet(Sigma)[1]+Sigma.shape[0]-logdetM #jax.numpy.linalg.slogdet(M)[1]#
            t2 = -jnp.einsum("ij,ji", jax.scipy.linalg.block_diag(*IM2), Sigma)#jax.scipy.linalg.block_diag(*IM2)
            t3 = -quadratic_term
            kl = 0.5*(t1 + t2 + t3)
            
            
            elbo =  expected_log_like+kl #kl_div(params[0],Sigma,M,quadratic_term,logdetM)
            return elbo
        
        num_samples = 300
       
        
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
        pbar = tqdm(total = niterations, disable=1-progress, position=0, leave=True)
        for t in range(niterations):
            opt_state = update(t, opt_state)
            params = get_params(opt_state)
            #pbar.set_description(f" lower bound {objective(params, t)}", refresh=True)
            pbar.update(1)
            #callback(params, t)
            if np.isnan(objective(params, t)):
                break
        pbar.close()
        
        #results 
        f = params[0]
        D = params[1]
        log_kernel_hypers = params[2]
        
        K = jax.scipy.linalg.block_diag(
            *self._Kernel(self.X,self.X,jnp.exp(log_kernel_hypers))
            )
        
        A = jax.numpy.diag(1/(D*D+self.jitter))
        M = K+self.jitter*np.eye(K.shape[0])
        Sigma = A - A@jax.numpy.linalg.solve(M+A, A) +self.jitter*jnp.eye(A.shape[0])        
              
        return f, Sigma, log_kernel_hypers, params
    
    def _log_like(self):
        def loglike(f):
            return self.log_likelihood(f,self.data,self.params)
        return loglike
    
    def optimize(self,niterations,progress=True,
                 kernel_hypers_fixed=False,
                 init_f=[]):
        #print(self.params)
        dic = DictVectorizer()       
        init_params_kernel,bounds_hyper=dic.fit_transform(self.params)
        #print(init_params_kernel)
        loglike =self._log_like()
        f, Sigma, log_kernel_hypers, advi_params = self.advi(niterations,loglike,
                                                             init_params_kernel,
                                                             kernel_hypers_fixed=kernel_hypers_fixed,
                                                             init_f=init_f)
        self.meanVI = f[:,None]
        self.SigmaVI = Sigma
        self.log_kernel_hypers = log_kernel_hypers
        self.advi_params=advi_params
        
        self.params = dic.inverse_transform(log_kernel_hypers, bounds_hyper)
       
        
    
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
    
    
        
