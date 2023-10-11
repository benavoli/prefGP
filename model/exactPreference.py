from .abstractModel import abstractModelFull
from scipy.linalg import cholesky
import numpy as np
from scipy.optimize import linprog
from inference import slice_sampler
from utility.linalg import build_sparse_prefM
import scipy.sparse as sparse
from scipy.stats import norm

class exactPreference(abstractModelFull):
    
    def __init__(self,data,Kernel,params,inf_method='laplace'):
        """
        The ExactPreference model assumes preferences are coherent strict preferences. The likelihood 
        is therefore an indicator function.
        
        :param data: data["X"] objects, data['Pairs"] preference pairs
        :type data: dictionary
        :param Kernel: Kernel function
        :type Kernel: function Kernel(X1,X2,params)
        :param params: parameters of the Kernels, bounds and transformations
        :type params: dictionary
        """
        self.data   = data
        self.X = data["X"]
        self.Kernel = Kernel
        self.params = params
        self.samples = [] # posterior samples
        self.jitter = 1e-4
        self._scale = 0.01
        self.inf_method=inf_method
        
        #build sparse pref Matrix
        self.PrefM = build_sparse_prefM(self.data["Pairs"],
                                        self.X.shape[0],
                                        1)
        self.PrefM = sparse.coo_matrix(self.PrefM, shape=(len(self.data["Pairs"]),self.X.shape[0]))
        
        #this is only used internally to estimate hyperparams via Laplace approximation
        '''
        import jax.numpy as jnp
        from jax.scipy.stats import norm
        def log_likelihood(f,data=self.data,params=self.params):
            z=data['PrefM']@f
            return jnp.sum(norm.logcdf(z/self._scale)) # we approximate the indicator with the normal CDF
        self._log_likelihood = log_likelihood
        '''
        def log_likelihood(f,data=self.data,params=self.params):
            W = self.PrefM/self._scale
            z = W@f
            return np.sum(norm.logcdf(z))

        def grad_log_like(f,data=self.data,params=self.params):
            W = self.PrefM/self._scale
            z=W@f
            r = np.exp(norm.logpdf(z)-norm.logcdf(z))
            val = (W.T).multiply(r.T)
            return np.sum(val,axis=1)

        def hess_log_like(f,data=self.data,params=self.params):
            W = self.PrefM/self._scale
            z = W@f
            r = np.exp(norm.logpdf(z)-norm.logcdf(z))
            D = (W.multiply((np.multiply(r,z)+r**2)))    
            H = -np.sum( D.toarray()[...,None] * W.toarray()[:,None],axis=0)
            return H
        
        self._log_likelihood = log_likelihood       
        self._grad_loglikelihood = grad_log_like
        self._hess_loglikelihood = hess_log_like
        self.recompute_grad_hessian_at_params_change=False#log_likelihood does not depend on params
        
        
        
    def sample(self, nsamples = 2000, tune=2000, disable=False):
        """
        Compute the posterior samples using liness, that is we sample
        from a truncated multivariate Normal distributions
        
        :param nsamples: number of samples
        :type nsamples: integer
        :returns (nx x nsamples) array stored in self.samples
        """
        Kxx = self.Kernel(self.X,self.X,self.params)+np.eye(self.X.shape[0])*self.jitter
        A = self.PrefM .toarray()
        b = np.zeros((A.shape[0],1))
        # find initial feasible 
        res = linprog(np.zeros(A.shape[1]),A_ub=-A,b_ub=-b, bounds=[[-1e-4,1]]*A.shape[1],method='interior-point' )
        x0 = res.x[:,None]#init point
        self.samples = slice_sampler.liness_step(x0, A, b, np.zeros(Kxx.shape[0]), 
                                            Kxx, nsamples= nsamples,  tune=tune, 
                                            progress=1-disable).T
       

        
