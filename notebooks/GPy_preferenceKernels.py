import GPy as GPy
from GPy.kern import Kern
import numpy as np
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from sklearn.metrics import pairwise_distances
from GPy.util.linalg import tdot
from GPy import util

class ProdRBFKernel(Kern):
    
    def __init__(self,input_dim,variance=1.,lengthscale=1.,active_dims=None, ARD=True):
        #the input dim should be equal to the dimension of the pair (u,v)
        super(ProdRBFKernel, self).__init__(input_dim, active_dims, 'ProdRBFKernel')
        self.variance    = Param('variance', variance)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.link_parameters(self.variance, self.lengthscale)
        self.ARD= ARD
        self.input_dim=input_dim
        
    def parameters_changed(self):
        # nothing todo here
        pass
    
    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        #X, = self._slice_X(X)
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return r2
        else:
            #X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
            r2 = np.clip(r2, 0, np.inf)
            return r2
        
    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards

        """
        if self.ARD:
            if X2 is not None:
                X2 = X2 / self.lengthscale
            return self._unscaled_dist(X/self.lengthscale, X2)
        else:
            return self._unscaled_dist(X, X2)/self.lengthscale**2
        
    
    def K(self,X,X2):
        #kernel: note the lengthscale is shared between u and v
        if X2 is None: X2 = X
        d=int(self.input_dim/2)
        dist2 = self._scaled_dist(X[:, 0:d],X2[:, 0:d])+self._scaled_dist(X[:, d:],X2[:, d:])
        return self.variance*np.exp(-dist2/2.)
    
    def Kdiag(self,X):
        return self.variance*np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X
        d=int(self.input_dim/2)
        #distance
        dist2 = self._scaled_dist(X[:, 0:d],X2[:, 0:d])+self._scaled_dist(X[:, d:],X2[:, d:])
        #derivative variance
        dvar = np.exp(-dist2/2.)
        #derivative lengthscale
        grad=[]
        if self.ARD:
            grad=[]
            for i in range(len(self.lengthscale)):
                lengthscale = self.lengthscale.copy()
                #lengthscale[i] = lengthscale[i]**1.5 # derivaitve is */l^3

                tmp = (
                       self._unscaled_dist(X[:, [i]], X2[:, [i]])/(lengthscale[i]**3)
                       +self._unscaled_dist(X[:, [d+i]], X2[:, [d+i]])/(lengthscale[i]**3)
                      )
                #print(tmp)
                dl= self.variance * dvar*tmp
                grad.append(np.sum(dl*dL_dK))
                    
        else:
            
            tmp=self._unscaled_dist(X[:, 0:d], X2[:, 0:d])+self._unscaled_dist(X[:, d:], X2[:, d:])
            #print(tmp)
            dl = self.variance * dvar*tmp/self.lengthscale**3
            grad.append(np.sum(dl*dL_dK))
        
        
        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale.gradient = np.hstack(grad)
      

class PreferenceKern(Kern):

    def __init__(
        self,
        input_dim,
        variance=1.0,
        lengthscale=1.0,
        active_dims=None,
        name="PreferenceKern",
        ARD=True,
    ):
        super(PreferenceKern, self).__init__(input_dim, active_dims, name)
        dim = int(input_dim / 2)# the input is a pair (u,v), so dim=dim(u).
        self.name = name
        if name=="PreferenceKern":
            self._kernel = GPy.kern.RBF(
            dim,
            variance   =variance,
            lengthscale=lengthscale,
            active_dims=self.active_dims[:dim],
            ARD=ARD,
        )
            
        elif name=="GeneralPreferenceKern":
            # the kernel is a product RBF*RBF sharing same hyperparameters
            self._kernel = ProdRBFKernel(input_dim,
                                         variance=variance,
                                         lengthscale=lengthscale,
                                         ARD=ARD)
  
            
        self.variance    = Param("variance", variance, Logexp())
        self.lengthscale = Param("lengthscale", lengthscale, Logexp())
        self.link_parameters(self.variance, self.lengthscale)

    def K(self, X, X2=None):
        # print(X2)
        d = int(X.shape[1] / 2)#input (u,v), d=dim(u)
        u = X[:, 0:d]
        v = X[:, d:]
        if X2 is None:
            u2 = X[:, 0:d]
            v2 = X[:, d:]
        else:
            u2 = X2[:, 0:d]
            v2 = X2[:, d:]
        self._kernel.variance    = self.variance
        self._kernel.lengthscale = self.lengthscale
        if self.name=="PreferenceKern":
            return (
                self._kernel.K(u, u2)
                + self._kernel.K(v, v2)
                - self._kernel.K(u, v2)
                - self._kernel.K(v, u2)
            )  
        elif self.name=="GeneralPreferenceKern":
            U = np.hstack([u, v])
            V = np.hstack([u2,v2])
            K1 = self._kernel.K(U, V)# it takes the pair as input
            U = np.hstack([u, v])
            V = np.hstack([v2,u2])
            K2 = self._kernel.K(U, V)
            return  (K1-K2) 

    def Kdiag(self, X):
        return np.diag(self.K(X))  

    def update_gradients_full(self, dL_dK, X, X2):
        
        d = int(X.shape[1] / 2)#input (u,v), d=dim(u)
        u = X[:, 0:d]
        v = X[:, d:]
        if X2 is None:
            u2 = X[:, 0:d]
            v2 = X[:, d:]
        else:
            u2 = X2[:, 0:d]
            v2 = X2[:, d:]
        self._kernel.variance    = self.variance
        self._kernel.lengthscale = self.lengthscale

        if self.name=="PreferenceKern":
            self._kernel.update_gradients_full(dL_dK, u, u2)       
            vargrad = self._kernel.variance.gradient+0.0
            lgrad = self._kernel.lengthscale.gradient+0.0
            self._kernel.update_gradients_full(dL_dK, v, v2)
            vargrad = vargrad + self._kernel.variance.gradient
            lgrad = (
                lgrad+ self._kernel.lengthscale.gradient
            )
            self._kernel.update_gradients_full(dL_dK, u, v2)
            vargrad = vargrad - self._kernel.variance.gradient
            lgrad = (
                lgrad - self._kernel.lengthscale.gradient
            )
            self._kernel.update_gradients_full(dL_dK, v, u2)
            vargrad = vargrad - self._kernel.variance.gradient
            lgrad = (
                lgrad - self._kernel.lengthscale.gradient
            )
        elif self.name=="GeneralPreferenceKern":
            U = np.hstack([u,v])
            V = np.hstack([u2,v2])
            self._kernel.update_gradients_full(dL_dK, U, V)
            vargrad    = self._kernel.variance.gradient + 0.0
            lgrad = self._kernel.lengthscale.gradient  + 0.0
            V = np.hstack([v2,u2])
            self._kernel.update_gradients_full(dL_dK, U, V)
            vargrad =  vargrad - self._kernel.variance.gradient
            lgrad =  lgrad - self._kernel.lengthscale.gradient

        self.variance.gradient = vargrad
        self.lengthscale.gradient = lgrad
            

