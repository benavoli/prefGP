
import numpy as np
import sys
sys.path.append('../')
from model.erroneousChoice_full import  erroneousChoice
from kernel import jaxrbf
from utility import  paramz
import matplotlib.pyplot as plt  

np.random.seed(42)
bounds=[[-0.25,1.25]]
bounds=[[-4.0,4.0]]
def fun(x,noise=0):
    u1 = np.cos(2*x) #np.exp(-(x-0.25)**2*15)
    u2 = -np.sin(2*x) #np.exp(-(x-0.75)**2*15)
    return np.hstack([u1,u2])+noise
Xpred=np.linspace(bounds[0][0],bounds[0][1],365)[:,None]

output = fun(Xpred)
plt.plot(Xpred[:,0],output[:,0],color='C0',label="u$_1$")
plt.plot(Xpred[:,0],output[:,1],color='C1',label="u$_2$")
plt.xlabel("x",fontsize=16)
plt.ylabel("$u_i(x)$",fontsize=16);
plt.grid()
plt.legend(fontsize=16)
#plt.savefig("ChoiceCupcake.pdf")

import torch
from botorch.utils.multi_objective import is_non_dominated

def is_pareto(X):
    return is_non_dominated(torch.from_numpy(X),deduplicate=False)

#generate CA RA sets
def make_CA_RA(x, y, rows=[]):
    if len(rows)==0:
        rows=np.arange(x.shape[0])
    acc = rows[is_pareto(y)]
    rej = rows[~ is_pareto(y)]
    return acc, rej

def make_observations(X, fun, nA, dimA):
    CA=[]
    RA=[]   
    ix = 0
    for i in range(nA):
        rows = np.random.permutation(np.arange(X.shape[0]))[0:dimA]
        x=X[rows,:]
        y=fun(x)
        acc,rej=make_CA_RA(x, y, rows)
        if len(acc)>0:
            CA.append(acc)
        else:
            CA.append([])
        if len(acc)<dimA:
            RA.append(rej)
        else:
            RA.append([])
        ix = ix+1
    return CA, RA


#generate data
np.random.seed(1)

#bounds=[[-0.25,1.25]]

# we randomly generate objects
n = 50 # number of objects
X = np.sort(np.vstack(bounds)[:,0]+np.random.rand(n,1)*(np.vstack(bounds)[:,1]-np.vstack(bounds)[:,0]),axis=0)

# we randomly generate choice data
nA = 100 # number of choice sets
dimA = 3 # dimension of each choice set
CA, RA = make_observations(X, fun, nA, dimA)

# We use this for prediction (plotting)
Xpred=np.linspace(bounds[0][0],bounds[0][1],200)[:,None]

#choice data
data={'X': X,#objects
      'CA': CA,#choiced objects
      'RA': RA,#rejected objects
      'dimA':dimA# dimension of the choice set
          }

# number of latent utilities
latent_dim=2

# define kernel 
Kernel = jaxrbf.RBF
#hyperparameters of the kernel
params = {'lengthscale_0': {'value':0.7*np.ones(data["X"].shape[1],float), 
                            'range':np.vstack([[0.1, 3.0]]*data["X"].shape[1]),
                            'transform': paramz.logexp()},
                 'variance_0': {'value':np.array([50.0]), 
                            'range':np.vstack([[1.0, 100.0]]),
                            'transform': paramz.logexp()},
          'lengthscale_1': {'value':0.7*np.ones(data["X"].shape[1],float), 
                            'range':np.vstack([[0.1, 3.0]]*data["X"].shape[1]),
                            'transform': paramz.logexp()},
                 'variance_1': {'value':np.array([50.0]), 
                            'range':np.vstack([[1.0, 100.0]]),
                            'transform': paramz.logexp()}
              }

# define choice model 
model = erroneousChoice(data,Kernel,params,latent_dim,typeR="pseudo")

model._log_likelihood(fun(X).T.flatten(),data)
# compute variational inference and estimate hyperparameters
model.optimize_hyperparams(niterations=1000,kernel_hypers_fixed=False, diagonal=False)
print(model.params)
# predicted samples
predictions = model.predict_VI(Xpred)
#it returns the joint mean (predictions[0]) and joint covariance matrix (predictions[1]) for the latent utilities. They have