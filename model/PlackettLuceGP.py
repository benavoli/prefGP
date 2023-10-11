from .abstractModel import abstractModelFull
import numpy as np
from inference import slice_sampler
import jax

class PlackettLuceGP(abstractModelFull):
    
    def __init__(self,data,Kernel,params,latent_dim,delta=0, jitter=1e-6,inf_method='advi',ARD=True):
        """
        The PlackettLuceGP model for modelling label ranking preferences. The likelihood 
        is an extension of the Plackett-Luce model. 
        The GP prior consists of latent_dim indpendent GP priors.
        
        :param data: data["X"] objects, data['Ranking"] ranked labels
        :type data: dictionary
        :param Kernel: Kernel function
        :type Kernel: function Kernel(X1,X2,params)
        :param params: parameters of the Kernels, bounds and transformations
        :type params:  dictionary
        :latent_dim: latent dimension
        :type latent_dim: integer
        """
        self.data   = data
        self.X = data["X"]
        self.params = params
        self.inf_method=inf_method 
        #define the number of latent dimensions
        self.latent_dim = latent_dim #len(np.unique(data['Ranking']))
        
        self.meanVI = []
        self.SigmaVI = []
        self.log_kernel_hypers = []
        self._MAP = []
        
        
        
        self.samples = [] # posterior samples
        self.jitter = jitter
        self._scale = 1.65
        self.delta=delta
        
        def build_M(pref):
            M={}
            lenp = np.max([len(p) for p in pref])
            #initialise
            for i in range(2,lenp+1):
                M[i]=[]
            for p in pref:
                for d in np.arange(len(p),1,-1):
                    M[d].append(p[-d:])
            for k in  M.keys():
                M[k]=np.vstack(M[k])    
            return M
        #build a dictionary of partial ranking
        dict_M=build_M(data['Ranking'])    
        


        self.dict_M = dict_M
        # block diagonal Kernel augmented with noise
        '''
        def augmKernelSparse(X1,X2,params):
            #print(self.latent_dim,params)
            d = X1.shape[1]
            if ARD==True:
            	Kxx = [Kernel(X1,X2,params[i*(d+1):(i+1)*(d+1)],ARD=ARD) for i in range(self.latent_dim)]
            else:
                Kxx = [Kernel(X1,X2,params[i*2:(i+1)*2],ARD=ARD) for i in range(self.latent_dim)]
            
            return Kxx #jax.scipy.linalg.block_diag(*Kxx)
        '''
        def augmKernel(X1,X2,params):
            d = X1.shape[1]
            if ARD==True:
            	Kxx = [Kernel(X1,X2,params[i*(d+1):(i+1)*(d+1)],ARD=ARD) for i in range(self.latent_dim)]
            else:
            	Kxx = [Kernel(X1,X2,params[i:(i+2)],ARD=ARD) for i in range(self.latent_dim)]
            return jax.scipy.linalg.block_diag(*Kxx)
        
        self.originalKernel = Kernel
        self.Kernel = augmKernel #augmKernelSparse #
        self.augmKernel = augmKernel

        
        #eps=1e-10
        #nu = jax.numpy.arange(self.X.shape[0])
        def loglike_M(U0,M):    
            '''
            U0: nx x nlatent utility matrix
            M: preference matrix of indexes
            '''
            #print(U0.shape)
            #print(M[:,0])
            #d2 = jax.numpy.log1p(jax.numpy.sum(jax.numpy.exp(U0),axis=1))
            V1 = jax.numpy.take_along_axis(U0, M, axis=1) 
            V2 = jax.numpy.take_along_axis(U0, M[:,0][:,None], axis=1) 
            d2 = jax.numpy.log(jax.numpy.sum(jax.numpy.exp(V1-V2),axis=1))
            #d1 = 0#U0[:,M[:,0]]
            return jax.numpy.sum(-d2)
            '''
                        d2 = jax.numpy.log(jax.numpy.sum(jax.numpy.exp(U0),axis=1))
            d1 = U0[nu,M[:,0]]
            return jax.numpy.sum(d1-d2)
            '''
        
        def log_likelihood(f,data=[],params=[]):
            U = jax.numpy.reshape(f,(self.latent_dim,self.X.shape[0])).T
            v = 0.0
            for k in  dict_M.keys():
                v=v+loglike_M(U,jax.numpy.array(self.dict_M[k]))
            return v
                
        self._log_likelihood = log_likelihood       
        self.recompute_grad_hessian_at_params_change=False
   
        
        
    
        
        
                
    def sample(self, nsamples = 2000, tune=6000, disable=False):
        """
        Compute the posterior samples using liness, that is we sample
        from a truncated multivariate Normal distributions
        
        :param nsamples: number of samples
        :type nsamples: integer
        :returns nx x nsamples array stored in self.samples
        """
        Kxx = self.augmKernel(self.X,self.X,np.exp(self.log_kernel_hypers))
        L = np.linalg.cholesky(Kxx+self.jitter*np.eye(Kxx.shape[0]))
        def log_like(f):
            f= f[:,None]
            return np.array(self._log_likelihood(f))
        ess = slice_sampler.ess(self.meanVI[:,0],L,log_like)
        self.samples = ess.sample(nsamples,tune=tune).T
        
        
    def predict_VI(self, Xpred):
           

        Kxx = self.augmKernel(self.X,self.X,np.exp(self.log_kernel_hypers))
        Kxz = self.augmKernel(self.X,Xpred,np.exp(self.log_kernel_hypers))
        Kzz = self.augmKernel(Xpred,Xpred,np.exp(self.log_kernel_hypers))
        
        L = np.linalg.cholesky(Kxx+self.jitter*np.eye(Kxx.shape[0]))
        L_inv = np.linalg.solve(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T

        M=np.dot(IKxx-IKxx@self.SigmaVI@IKxx,Kxz)
        V =  Kzz -Kxz.T@M
        V = (V+V.T)/2
        return Kxz.T@IKxx@self.meanVI, V
        
    def predict(self, Xpred):
        """
        Compute the posterior predictions for Xpred 
        
        :param Xpred: test points
        :type Xpred: nD-array (ntestpoints x dimX)
        :returns (ntestpoints x nsamples) array  
        """
        Kxx = self.augmKernel(self.X,self.X,np.exp(self.log_kernel_hypers))
        L = np.linalg.cholesky(Kxx+self.jitter*np.eye(Kxx.shape[0]))
        L_inv = np.linalg.solve(L.T,np.eye(L.shape[0]))
        IKxx = L_inv@L_inv.T
        Kxz = self.augmKernel(self.X,Xpred,np.exp(self.log_kernel_hypers))

        return Kxz.T@IKxx@self.samples

        
