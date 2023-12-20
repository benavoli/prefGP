from .abstractModel import abstractModelFull
import numpy as np
from inference import slice_sampler
import jax
from itertools import combinations

class erroneousChoice(abstractModelFull):
    
    def __init__(self,data,Kernel,params,latent_dim,typeR="rational",scale=1.65, jitter=1e-6,inf_method='advi',ARD=True):
        """
        The erroneousChoice model for modelling choice functions. The likelihood 
        uses a normal-cdf to smooth the indicators to model the errors. 
        The GP prior consists of latent_dim indpendent GP priors.
        
        :param data: data["X"] objects, data['CA"], data['RA"] choice and rejection
        :type data: dictionary
        :param Kernel: Kernel function
        :type Kernel: function Kernel(X1,X2,params)
        :param params: parameters of the Kernels, bounds and transformations
        :type params:  dictionary
        :param latent_dim: number of utility functions
        :type latent_dim: int
        """
        self.data   = data
        self.X = data["X"]
        self.params = params
        self.latent_dim = latent_dim
        self.inf_method=inf_method 
        
        self.meanVI = []
        self.SigmaVI = []
        self.log_kernel_hypers = []
        self._MAP = []
        
        
        
        self.samples = [] # posterior samples
        self.jitter = jitter
        self._scale = scale
     
        
        CA = data["CA"]
        RA = data["RA"]
        dimA = data["dimA"]
        #process CA, RA sets
        from itertools import combinations
        CAr=[]   
        GroupCA=[]
        for i in range(len(CA)):
            if len(CA[i])>1:
                ttl = list(combinations(CA[i],2))
                CAr.append(ttl)
                GroupCA.append(i*np.ones(len(ttl)))
            else:
                GroupCA.append(np.nan*np.ones(1))
        if len(CAr)>0:
            CAr = np.vstack(CAr).astype(int)
        self.GroupCA=GroupCA
        RAr=[]
        GroupRA=[]
        for r in range(len(RA)):
            ttRA=[]
            for j in range(len(RA[r])):
                tmp = -np.ones(dimA)
                tmp[-1] = RA[r][j]
                d = len(CA[r])
                tmp[0:d]=CA[r]
                #tmp = np.vstack([[CA[r][i],RA[r][j]] for i in range(len(CA[r]))])
                RAr.append(tmp)
                ttRA.append(r)
            GroupRA.append(ttRA)
        if len(RAr)>0:
            RAr = np.vstack(RAr).astype(int)
        self.GroupRA=GroupRA
        self.CAr = CAr
        self.RAr = RAr
        
        
       
        #make indices for fucc  covariance of ADVI
        def make_indices(CA,RA):
            val = CA+RA
            left=val.copy()
            right=val.copy()
            res=list(combinations(val,2))
            if len(res)>0:
                r=np.vstack(res)
                left=left+r[:,0].tolist()
                right=right+r[:,1].tolist()
            return left,right
        
        leftInd=[]
        rightInd=[]
        for i in range(len(CA)):
            c,r = CA[i],RA[i]
            lefti,righti=make_indices(c,r)
            leftInd=leftInd+lefti
            rightInd=rightInd+righti
        LeftRight=np.unique(np.vstack([leftInd,rightInd]).T,axis=0)
        self._leftindices =LeftRight[:,0]
        self._rightindices=LeftRight[:,1]
        

        def augmKernel(X1,X2,params):
            d = X1.shape[1]
            if ARD==True:
            	Kxx = [Kernel(X1,X2,params[i*(d+1):(i+1)*(d+1)],ARD=ARD) for i in range(self.latent_dim)]
            else:
            	Kxx = [Kernel(X1,X2,params[i:(i+2)],ARD=ARD) for i in range(self.latent_dim)]
            return jax.scipy.linalg.block_diag(*Kxx)
        
        self.originalKernel = Kernel
        self.Kernel = augmKernel #
        self.augmKernel = augmKernel

        
        eps=1e-10
        def loglike_CA(U0,CAr):
            '''
            U0: nx x nlatent utility matrix
            '''
            #v = jax.scipy.stats.norm.cdf(U0[CAr[:,0]]-U0[CAr[:,1]])
            if len(CAr)>0:
                x = (U0[CAr[:,0]]-U0[CAr[:,1]])
                v = 0.5 * (jax.numpy.tanh(x * self._scale / 2) + 1)
                q = -jax.numpy.prod(v,axis=1)-jax.numpy.prod(1-v,axis=1)
                return jax.numpy.sum(
                      jax.numpy.log1p(eps+q))
            else:
                return jax.numpy.array(0.0)
        
        def loglike_RA_rational(U0, RAr):
            '''
            U0: nx x nlatent utility matrix
            '''
            #A=jax.scipy.stats.norm.cdf(U0[RAr[:,0:-1],:]-U0[RAr[:,[-1]],:])
            if len(RAr)>0:
                x = (U0[RAr[:,0:-1],:]-U0[RAr[:,[-1]],:])
                A = 0.5 * (jax.numpy.tanh(x * self._scale / 2) + 1)
                q = -jax.numpy.prod(1-jax.numpy.prod(A,axis=2),axis=1)
                return jax.numpy.sum(jax.numpy.log1p(eps+q))
            else:
                return jax.numpy.array(0.0)
            
        def loglike_RA_pseudo(U0, RAr):
            '''
            U0: nx x nlatent utility matrix
            '''
            #A=jax.scipy.stats.norm.cdf(U0[RAr[:,0:-1],:]-U0[RAr[:,[-1]],:])
            if len(RAr)>0:
                x = (-U0[RAr[:,0:-1],:]+U0[RAr[:,[-1]],:])
                A = 0.5 * (jax.numpy.tanh(x * self._scale / 2) + 1)
                q = jax.numpy.prod(1-jax.numpy.prod(A,axis=1),axis=1)
                return jax.numpy.sum(jax.numpy.log(eps+q))
            else:
                return jax.numpy.array(0.0)
        
        if typeR=='rational':
            loglike_RA = loglike_RA_rational
            def log_likelihood(f,data=[],params=[]):
                #print(CAr)
                U = jax.numpy.reshape(f,(self.latent_dim,self.X.shape[0])).T
                #add worst element
                U = jax.numpy.vstack([U,-np.ones((1,self.latent_dim))*np.inf])
                return loglike_CA(U,CAr)+loglike_RA_rational(U,RAr)  
        elif typeR=='pseudo':
            loglike_RA = loglike_RA_pseudo
            def log_likelihood(f,data=[],params=[]):
                #print(CAr)
                U = jax.numpy.reshape(f,(self.latent_dim,self.X.shape[0])).T
                #add worst element
                U = jax.numpy.vstack([U,-np.ones((1,self.latent_dim))*np.inf])
                return loglike_CA(U,CAr)+loglike_RA_pseudo(U,RAr)  #+anchor(U)
                
        self._loglike_CA = loglike_CA 
        self._loglike_RA = loglike_RA 
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

        
