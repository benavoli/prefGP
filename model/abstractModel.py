
from abc import ABC, abstractmethod
from inference import laplace_generic as LAP
from inference import laplace_sparse  as LAPsp
from inference import advi_sparse as ADVIS
from inference import advi_full as ADVIF
import numpy as np

class abstractModelFull(ABC):
    
    @abstractmethod
    def sample(self, nsamples):
        pass
    
    def predict(self, Xpred):
        from scipy.linalg import cholesky, solve_triangular
        Kxx = self.Kernel(self.X,self.X,self.params)
        Kxx = Kxx +np.eye(Kxx.shape[0])*self.jitter
        L = cholesky(Kxx, lower=True)
        L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        Kxz = self.Kernel(self.X,Xpred,self.params)
        #Kzz = self.Kernel(Xpred,Xpred,self.params)
        IKxx = L_inv@L_inv.T
        return Kxz.T@IKxx@self.samples
         
    
    def optimize_hyperparams(self,num_restarts=1,
                             niterations=2000,
                             kernel_hypers_fixed=False,
                             init_f=[]):

        if self.inf_method=='laplace':
            infer = LAP.inference_laplace(self.data, 
                                          self.Kernel, 
                                          self.params,
                                          self._log_likelihood,
                                          self._grad_loglikelihood,
                                          self._hess_loglikelihood)
            infer.optimize(self.recompute_grad_hessian_at_params_change,
                           num_restarts=num_restarts)
            self.params = infer.params
        elif self.inf_method=='advi':
            infer = ADVIF.inference_advi(self.data, 
                                          self.Kernel, 
                                          self.params,
                                          self._log_likelihood)
            infer.optimize(niterations,
                           kernel_hypers_fixed=kernel_hypers_fixed,
                           init_f=init_f)
            self.meanVI = infer.meanVI
            self.SigmaVI = infer.SigmaVI
            self.log_kernel_hypers = infer.log_kernel_hypers
            self.advi_params = infer.advi_params
            self.params = infer.params
            self._MAP = infer.MAP
        
        
class abstractModelSparse(ABC):
    
    @abstractmethod
    def sample(self, nsamples):
        pass
    
    def predict(self, Xpred):
        from scipy.sparse.linalg import spsolve_triangular
        from utility.linalg import sparse_cholesky
        Kxx = self.Kernel(self.X,self.X,self.params)
        Kxx = Kxx +np.eye(Kxx.shape[0])*self.jitter
        L = sparse_cholesky(Kxx, lower=True)
        L_inv = spsolve_triangular(L.T, np.eye(L.shape[0]))
        Kxz = self.Kernel(self.X,Xpred,self.params)
        #Kzz = self.Kernel(Xpred,Xpred,self.params)
        IKxx = L_inv@L_inv.T
        return Kxz.T@IKxx@self.samples
         
    
    def optimize_hyperparams(self,num_restarts=1,
                             niterations=2000,
                             kernel_hypers_fixed=False,
                             init_f=[]):
        if self.inf_method=='laplace':
            infer = LAPsp.inference_laplace(self.data, 
                                          self.Kernel, 
                                          self.params,
                                          self._log_likelihood,
                                          self._grad_loglikelihood,
                                          self._hess_loglikelihood)
            infer.optimize(self.recompute_grad_hessian_at_params_change,
                           num_restarts=num_restarts)
            self.params = infer.params
        elif self.inf_method=='advi':
            infer = ADVIS.inference_advi(self.data, 
                                          self.Kernel, 
                                          self.params,
                                          self._log_likelihood)
            infer.optimize(niterations,
                           kernel_hypers_fixed=kernel_hypers_fixed,
                           init_f=init_f)
            self.meanVI = infer.meanVI
            self.SigmaVI = infer.SigmaVI
            self.log_kernel_hypers = infer.log_kernel_hypers
            self.advi_params = infer.advi_params
            self.params = infer.params
            self._MAP = infer.MAP
            #print(infer.MAP)

    