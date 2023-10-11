from .abstractModel import abstractModelFull
import numpy as np
from scipy.optimize import linprog
from inference import slice_sampler
from utility.linalg import build_sparse_prefM
import scipy.sparse as sparse
from scipy.stats import norm

class LuceJND(abstractModelFull):
    
    def __init__(self,data,Kernel,params, inf_method='Laplace'):
        """
        The LuceJND is a model for preference learning which accounts for the limit of discernibility
        of the subject. The likelihood coincides with Luce's just noticeable difference
        model.
        
        :param data: data["X"] objects, data['Pairs"] preference pairs, data['Indisc"] indiscirnible pairs
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
        Data=np.vstack([np.hstack([np.vstack(data["Pairs"])[:,[1]],np.vstack(data["Pairs"])[:,[0]]]),
                        self.data["Indisc"],
                        np.hstack([np.vstack(data["Indisc"])[:,[1]],np.vstack(data["Indisc"])[:,[0]]])])
        self.PrefM = build_sparse_prefM(Data,
                                        self.X.shape[0],
                                        1)
        
        #build sparse pref Matrix
        self.ones = np.vstack([-np.ones((self.data["Pairs"].shape[0],1)),np.ones((2*self.data["Indisc"].shape[0],1))])
        self.PrefM = sparse.coo_matrix(self.PrefM, shape=(self.ones.shape[0],self.X.shape[0]))
        
        #this is only used internally to estimate hyperparams via Laplace approximation
        def log_likelihood(f,data=self.data,params=self.params):
            z = -(self.PrefM@f-self.ones)/self._scale
            return np.sum(norm.logcdf(z))

        def grad_log_like(f,data=self.data,params=self.params):
            W = -self.PrefM/self._scale
            z = W@f+(self.ones)/self._scale
            r = np.exp(norm.logpdf(z)-norm.logcdf(z))
            val = (W.T).multiply(r.T)
            return np.sum(val,axis=1)

        def hess_log_like(f,data=self.data,params=self.params):
            W = -self.PrefM/self._scale
            z = W@f+(self.ones)/self._scale
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
        A = self.PrefM.toarray()
        b = self.ones
        # find initial feasible 
        res = linprog(np.zeros(A.shape[1]),A_ub=A,b_ub=b,method='interior-point' ) #bounds=[[-1e-4,3]]*A.shape[1]
        x0 = res.x[:,None]#init point
        #self.x0=x0
        #print(res)
        self.samples = slice_sampler.liness_step(x0, -A, -b, np.zeros(Kxx.shape[0]), 
                                            Kxx, nsamples= nsamples,  tune=tune, 
                                            progress=1-disable).T
       

        
