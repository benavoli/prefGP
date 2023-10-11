from .abstractModel import abstractModelFull
from scipy.linalg import cholesky, block_diag, solve_triangular, cho_solve
import numpy as np
from scipy.optimize import linprog
from inference import slice_sampler
from utility.linalg import build_sparse_prefM
import scipy.sparse as sparse
from scipy.stats import norm, multivariate_normal

class erroneousPreference(abstractModelFull):
    
    def __init__(self,data,Kernel,params,inf_method='Laplace'):
        """
        The erroneousPreference model assumes preferences may be wrong due to . The likelihood 
        is probit.
        
        :param data: data["X"] objects, data['Pairs"] preference pairs
        :type data: dictionary
        :param Kernel: Kernel function
        :type  Kernel: function Kernel(X1,X2,params)
        :param params: parameters of the Kernels, bounds and transformations
        :type  params: dictionary
        """
        self.data   = data
        self.X = data["X"]
        self.params = params
  
        self.Kernel = Kernel
        
        self.samples = [] # posterior samples
        self.jitter = 1e-6
        self.inf_method = inf_method
        
        #build sparse pref Matrix
        self.PrefM = build_sparse_prefM(self.data["Pairs"],
                                        self.X.shape[0],
                                        1)
        self.PrefM = sparse.coo_matrix(self.PrefM, shape=(len(self.data["Pairs"]),self.X.shape[0]))
        
        #this is used internally to estimate hyperparams via Laplace approximation
        def log_likelihood(f,data=self.data,params=self.params):
            W = self.PrefM
            z = W@f
            return np.sum(norm.logcdf(z))

        def grad_log_like(f,data=self.data,params=self.params):
            W = self.PrefM
            z=W@f
            r = np.exp(norm.logpdf(z)-norm.logcdf(z))
            val = (W.T).multiply(r.T)
            return np.sum(val,axis=1)

        def hess_log_like(f,data=self.data,params=self.params):
            W = self.PrefM
            z = W@f
            r = np.exp(norm.logpdf(z)-norm.logcdf(z))
            D = (W.multiply((np.multiply(r,z)+r**2)))    
            H = -np.sum( D.toarray()[...,None] * W.toarray()[:,None],axis=0)
            return H
        
        self._log_likelihood = log_likelihood       
        self._grad_loglikelihood = grad_log_like
        self._hess_loglikelihood = hess_log_like
        self.recompute_grad_hessian_at_params_change=False#log_likelihood does not depend on params
        
    
        
    def _compute_SkewGP_posterior_params(self): #params,X,W0,Z0 
        '''
        Computes the parameters of the SkewGP posterior
        '''
        W=self.PrefM
        Kxx = self.Kernel(self.X,self.X,self.params)
        Ω  = Kxx+np.eye(Kxx.shape[0])*self.jitter
        iω = np.diag(1/np.sqrt(np.diag(Ω)))
        xi = np.zeros((Ω.shape[0],1))+0.0
        #computed posterior parameters
        xip = xi
        Ωp = Ω
        Δp = iω@Ω@W.T
        γp = W@xi
        Γp = W@Ω@W.T+np.eye(W.shape[0])
        return xip, Ωp, Δp, γp, Γp
                
    def sample(self, nsamples = 2000, tune=6000, disable=False):
        """
        Compute the posterior samples using liness, that is we sample
        from a skew multivariate Normal distributions
        
        :param nsamples: number of samples
        :type nsamples: integer
        :returns nx x nsamples array stored in self.samples
        """
        #define parameters Skew Normal posterior
        xi, Ω, Δ, γ, Γ = self._compute_SkewGP_posterior_params()
        iω = np.diag(1/np.sqrt(np.diag(Ω)))
        Ω_c = np.linalg.multi_dot([iω  , Ω , iω])  #correlation matrix
        L = cholesky(Γ+self.jitter*np.eye(Γ.shape[0]),lower=True)
        M = cho_solve(( L,True),Δ.T).T
        M1 = Ω_c-np.linalg.multi_dot([M,Δ.T])

        del Ω_c
        M1=0.5*(M1+M1.T)+self.jitter*np.identity(M1.shape[0])

        L=cholesky(M1,lower=True)
        rv1=multivariate_normal(np.zeros(M1.shape[0]),np.identity(M1.shape[0]))
        del M1
        points1 = np.dot(L,rv1.rvs(nsamples).T)
        # liness
        A = np.eye(Γ.shape[0])
        b = -γ
        res = linprog(np.zeros(A.shape[1]),A_ub=-A,b_ub=-b, bounds=[[0.,1]]*A.shape[1],method='interior-point')
        x0=res.x[:,None]#init point
        points2 = (slice_sampler.liness_step(x0, A, b, np.zeros(Γ.shape[0]), 
                                            Γ, nsamples= nsamples,  tune=tune, 
                                            progress=1-disable).T)
        self.samples = xi+ np.dot(np.diag(np.sqrt(np.diag(Ω))), points1 + M@points2)


        
                

        
