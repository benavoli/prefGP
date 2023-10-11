from .abstractModel import abstractModelSparse
from scipy.sparse  import block_diag
from scipy.sparse.linalg import spsolve_triangular
from utility.linalg import sparse_cholesky
import numpy as np
from scipy.optimize import linprog
from inference import slice_sampler
from utility.linalg import build_sparse_prefM
import scipy.sparse as sparse
from scipy.stats import norm

class gaussianNoisePreference(abstractModelSparse):
    
    def __init__(self,data,Kernel,params,inf_method="laplace"):
        """
        The GussianNoisePreference model assumes preferences are corrupted by Gaussian Noise. The likelihood 
        is an indicator function, but the GP prior is augmented to include independent noise.
        
        :param data: data["X"] objects, data['prefM"] preference matrix
        :type data: dictionary
        :param Kernel: Kernel function
        :type Kernel: function Kernel(X1,X2,params)
        :param params: parameters of the Kernels, bounds and transformations
        :type params: dictionary
        """
        self.data   = data
        self.X = data["X"]
        self.params = params
        self._scale = 0.05
        self.inf_method=inf_method
        
        #build sparse pref Matrix
        self.PrefM = build_sparse_prefM(self.data["Pairs"],
                                        self.X.shape[0],
                                        1)
        self.PrefM = sparse.coo_matrix(self.PrefM, shape=(len(self.data["Pairs"]),self.X.shape[0]))
        self.augPrefM = sparse.hstack([self.PrefM ,self.PrefM ])
        
        #build full pref Matrix
        #self.PrefM = self.PrefM.toarray()
        #self.augPrefM =  self.augPrefM.toarray()
        
        
        # block diagonal Kernel augmented with noise
        def augmKernel(X1,X2,params):
            Kxx=Kernel(X1,X2,params)
            sigma2 = params["noise_variance"]["value"]
            return block_diag([Kxx,sigma2*np.eye(self.X.shape[0])])
        self.originalKernel = Kernel
        self.Kernel = augmKernel
        
        self.samples = [] # posterior samples
        self.jitter = 1e-6
        #augmented preference matrix
        
        
        #this is only used internally to estimate hyperparams via Laplace approximation
        '''
        import jax.numpy as jnp
        from jax.scipy.stats import norm
        def log_likelihood(f,data=self.data,params=self.params):
            z=(self.augPrefM@f)/self._scale
            return jnp.sum(norm.logcdf(z)) # we approximate the indicator with the normal CDF
        self._log_likelihood = log_likelihood
        self._grad_loglikelihood = []
        self._hess_loglikelihood = []
        '''
        def log_likelihood(f,data=self.data,params=self.params):
            W = self.augPrefM/self._scale
            z = W@f
            return np.sum(norm.logcdf(z))

        def grad_log_like(f,data=self.data,params=self.params):
            W = self.augPrefM/self._scale
            z=W@f
            r = np.exp(norm.logpdf(z)-norm.logcdf(z))
            val = (W.T).multiply(r.T)
            return np.sum(val,axis=1)
       
        def hess_log_like(f,data=self.data,params=self.params):
            W = self.augPrefM/self._scale
            w = W.tocsr()[[0],:]
            z = w@f
            r = np.exp(norm.logpdf(z)-norm.logcdf(z))
            d = (w.multiply((r*z+r**2))).T    
            H = - d@w
            for i in range(1,W.shape[0]):
                w = W.tocsr()[[i],:]
                z = w@f
                r = np.exp(norm.logpdf(z)-norm.logcdf(z))
                d = (w.multiply((r*z+r**2))).T
                H = H - d@w
            return H.tocsc()
        
        self._log_likelihood = log_likelihood       
        self._grad_loglikelihood = grad_log_like
        self._hess_loglikelihood = hess_log_like
        
        
        self.recompute_grad_hessian_at_params_change=False # log_likelihood does not depend on params
        
        
    
        
        
                
    def sample(self, nsamples = 2000, tune=6000, disable=False):
        """
        Compute the posterior samples using liness, that is we sample
        from a truncated multivariate Normal distributions
        
        :param nsamples: number of samples
        :type nsamples: integer
        :returns nx x nsamples array stored in self.samples
        """
        Kxx = self.Kernel(self.X,self.X,self.params)
        Kxx = Kxx+np.eye(Kxx.shape[0])*self.jitter

        A = self.augPrefM#.toarray() #self.data["augPrefM"]
        b = np.zeros((A.shape[0],1))
        res = linprog(np.zeros(A.shape[1]),A_ub=-A,b_ub=-b, bounds=[[0.,1]]*A.shape[1],method='interior-point' )
        x0=res.x[:,None]#init point
        self.samples = (slice_sampler.liness_step(x0, A, b, np.zeros(Kxx.shape[0]), 
                                            Kxx, nsamples= nsamples,  tune=tune, 
                                            progress=1-disable).T)[0:self.X.shape[0],:]
        
                
    def predict(self, Xpred):
        """
        Compute the posterior predictions for Xpred 
        
        :param Xpred: test points
        :type Xpred: nD-array (ntestpoints x dimX)
        :returns (ntestpoints x nsamples) array  
        """
        Kxx = self.originalKernel(self.X,self.X,self.params)+np.eye(self.X.shape[0])*self.jitter
        L = sparse_cholesky(Kxx)
        L_inv = spsolve_triangular(L.T, np.eye(L.shape[0]))
        Kxz = self.originalKernel(self.X,Xpred,self.params)
        IKxx = L_inv@L_inv.T
        return Kxz.T@IKxx@self.samples

        
