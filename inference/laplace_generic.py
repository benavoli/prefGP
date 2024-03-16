#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from jax.config import config
config.update("jax_enable_x64", True)
#config.update('jax_platform_name', 'cpu')
#import jax
import numpy as np
from jax import grad,  jit, jacfwd, jacrev
from utility.paramz import DictVectorizer
from scipy.optimize import minimize
from scipy.linalg import solve, solve_triangular, cholesky
from scipy import optimize
from tqdm import tqdm_notebook as tqdm


        
class inference_laplace:
    """
    Class GP_Laplace
    """
    def __init__(self, data, Kernel, params, log_likelihood,  grad_loglike=[], hess_loglike=[], jitter=1e-8):
        self._Kernel = Kernel #function
        self.params=params
         #data
        self.data=data
        self.X=data["X"]
        self.fMAP=[]
        self.LambdaMAP=[]
        self.jitter=jitter
        self.log_likelihood=log_likelihood
        self.grad_loglike=grad_loglike
        self.hess_loglike=hess_loglike
        self.max_iter  = 30
        self.tolerance = 1e-4
        
    def _compute_B_matrix(self,K, W0):
        W = W0+np.eye(W0.shape[0])*1e-4
        W_12 = cholesky(W,lower=False)  #
        B = np.eye(K.shape[0]) + W_12@K@W_12.T
        L = cholesky(B,lower=True)
        LiW12 = solve_triangular(L, W_12, lower=True) 
        K_Wi_i = LiW12.T@LiW12 
        logdet_L= np.sum(np.log(np.diag(L)))
        return K_Wi_i, logdet_L
    
   
    def laplace(self,K,loglike,grad_loglike, hess_loglike):
        f = np.zeros((K.shape[0],1))
        I = np.eye(K.shape[0])
        Ki_f = solve(K+self.jitter*I,f)
        #define the objective function (to be maximised)
        def obj(Ki_f, f):
            llv = -0.5*np.sum(np.dot(Ki_f.T, f)) + np.sum(loglike(f))
            if np.isnan(llv):
                return -np.inf
            else:
                return llv

        difference = np.inf
        iters = 0
        while (difference > self.tolerance and iters < self.max_iter):
            if self.hess_loglike==[]:
                W = -hess_loglike(f)[:,0,:,0]
            else:
                W = -hess_loglike(f)
            if np.any(np.isnan(W)):
                raise ValueError('One or more element(s) of W is NaN')
            grad = grad_loglike(f)
            if np.any(np.isnan(grad)):
                raise ValueError('One or more element(s) of grad is NaN')
            W_f = W@f
            b = W_f + grad 
            T, logdet = self._compute_B_matrix(K, W)
            Tb = T@(K@b)
            full_step_Ki_f = b - Tb #
            dKi_f = full_step_Ki_f - Ki_f
            #define  objective for the line search (minimize this one)
            def inner_obj(step_size):
                Ki_f_trial = Ki_f + step_size*dKi_f
                f_trial = K@Ki_f_trial
                val = -obj(Ki_f_trial, f_trial)
                return val
            #use scipy for the line search, the compute new values of f, Ki_f
            step = optimize.brent(inner_obj, tol=1e-4, maxiter=12)
            Ki_f_new = Ki_f + step*dKi_f
            f_new = K@Ki_f_new
            old_obj = obj(Ki_f, f)
            new_obj = obj(Ki_f_new, f_new)
            if new_obj < old_obj:
                #print(new_obj,old_obj)
                raise ValueError("Error Brent optimization failing")
            difference = np.abs(new_obj - old_obj)
            Ki_f = Ki_f_new
            f = f_new
            iters += 1

        lla=obj(Ki_f, f)-logdet
        return f, lla
    
    def _log_like_grad_Hessian(self):
        
        def loglike(f):
            return self.log_likelihood(f,self.data,self.params)
        
        if self.grad_loglike==[]:
             grad_loglike = jit(grad(loglike))
        else:
             def grad_loglike(f):
                 return self.grad_loglike(f,self.data,self.params)
             
        if self.hess_loglike==[]:
             hess_loglike = jit(jacfwd(jacrev(loglike)))
        else:
             def hess_loglike(f):
                 return self.hess_loglike(f,self.data,self.params)


        return loglike,grad_loglike,hess_loglike
    
    def optimize(self,recompute_grad_hessian_at_params_change,
                 num_restarts=1,max_iters=200,method='l-bfgs-b'):
        dic = DictVectorizer()
       
        init_params,bounds=dic.fit_transform(self.params)
        loglike,grad_loglike,hess_loglike=self._log_like_grad_Hessian()

        def objective(params_flatten,loglike,grad_loglike,hess_loglike):
            self.params=dic.inverse_transform(params_flatten,bounds)
            #print(self.params)
            if recompute_grad_hessian_at_params_change==True:
                loglike,grad_loglike,hess_loglike=self._log_like_grad_Hessian()
        
            #print(self.params['lengthscale']['value'],self.params['variance']['value'])
            K = self._Kernel(self.X,self.X,self.params)

            f,lla = self.laplace(K,loglike,grad_loglike,hess_loglike)
            val = np.real(-lla)#minimize
            #print(val)
            pbar.set_description(f" Objective {val}", refresh=True)
            return val
        
        def progress_callback(xk):
            # This callback function will be called after each iteration of the optimization
            pbar.update(1)  # Update the progress bar
            
        def tmp_fun():
            global pbar
            pbar = tqdm(total=max_iters,  position=0, leave=True)
            res = minimize(objective, init_params, 
                              bounds=bounds,
                              args=(loglike,grad_loglike,hess_loglike),
                              callback=progress_callback,
                              method=method,options={'maxiter': max_iters, 
                                                     #'ftol':1e-9,
                                                     'disp': False},
                              )
            return res


        optml=np.inf
        for i in range(num_restarts):
            
            res = tmp_fun()
            if res.fun<optml:
                params_best=res.x #init_params 
                optml=res.fun
            init_params=bounds[:,0]+(bounds[:,1]-bounds[:,0])*np.random.rand(len(bounds[:,0]))
            print("Iteration "+str(i)+" ",-res.fun)
            
        self.params=dic.inverse_transform(params_best,bounds)
        
        loglike,grad_loglike,hess_loglike=self._log_like_grad_Hessian()
        Kxx=self._Kernel(self.X,self.X,self.params) 
        Kxx = Kxx + self.jitter * np.eye(Kxx.shape[0])  
        self.fMAP,lla = self.laplace(Kxx,loglike,grad_loglike,hess_loglike)
        
    
    def predict(self, Xpred):
        #recompute Laplace
        loglike,grad_loglike,hess_loglike=self._log_like_grad_Hessian()
        Kxx=self._Kernel(self.X,self.X,self.params) 
        Kxx = Kxx + self.jitter * np.eye(Kxx.shape[0])  
        self.fMAP,lla = self.laplace(Kxx,loglike,grad_loglike,hess_loglike)
        L = cholesky(Kxx+np.eye(Kxx.shape[0])*self.jitter,lower=True)
        L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T
        if self.hess_loglike==[]:
            H = -hess_loglike( self.fMAP)[:,0,:,0]
        else:
            H = -hess_loglike( self.fMAP)
        self.LambdaMAP=H+IKxx
    
    
        #predict mean and covariance
        #Kxx = self._Kernel(self.X,self.X,self.params)
        #Kxx = Kxx +self.jitter * np.eye(Kxx.shape[0])        
        Kxz = self._Kernel(self.X,Xpred,self.params)
        Kzz = self._Kernel(Xpred,Xpred,self.params)
        
        #L = cholesky(Kxx,lower=True)
        #L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        #IKxx = L_inv@L_inv.T
        
        L = cholesky(self.LambdaMAP+self.jitter*np.eye(self.LambdaMAP.shape[0]),lower=True)
        L_inv = solve_triangular(L.T,np.eye(L.shape[0]))
        ILambdaMAP = L_inv@L_inv.T

        M=np.linalg.solve(ILambdaMAP+Kxx,Kxz)
        return Kxz.T@IKxx@self.fMAP,  Kzz -Kxz.T@M #()@Kxz
    
    def predict_noiseless(self,Xpred,full_cov=True):
        return self.predict(Xpred)
    
    
        
