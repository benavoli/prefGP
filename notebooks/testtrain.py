
import sys,os
sys.path.append('../')   
import numpy as np
import pandas as pd
from model.erroneousPreference import erroneousPreference
from kernel import RBF
from utility import  paramz
from sklearn.model_selection import train_test_split
# for plotting
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import arviz as az

df = pd.read_csv("datasets/train.csv",index_col=0)
df.reset_index(drop=True,inplace=True)
df_train,df_test = train_test_split(df, test_size=0.9, shuffle=True, random_state=1)

def compute_pair(df):
    Xa = np.unique(df[['price_A', 'time_A', 'change_A',
       'comfort_A']],axis=0)
    Xb = np.unique(df[['price_B', 'time_B', 'change_B',
           'comfort_B']],axis=0)
    X = np.unique(np.vstack([Xa,Xb]),axis=0)# objects
    Pair = []
    for el in range(0,df.shape[0]):
        rowi=df.iloc[el][['price_A', 'time_A', 'change_A',
           'comfort_A']]
        rowj=df.iloc[el][['price_B', 'time_B', 'change_B',
           'comfort_B']]
        i = np.where( np.isin(X, rowi).all(axis=1))[0][0]
        j = np.where( np.isin(X, rowj).all(axis=1))[0][0]
        if df.iloc[el].choice=="A":
            Pair.append([i,j])
        else:
            Pair.append([j,i])
    Pair=np.vstack(Pair)
    return Pair,X


Pair_tr,X_tr = compute_pair(df_train)
#Pair_te,X_te = compute_pair(df_test)


scalerx = StandardScaler().fit(X_tr)
X_tr_n = scalerx.transform(X_tr)
#X_te_n = scalerx.transform(X_te)

# data dictionary
data = {}
data["Pairs"] = Pair_tr
data["X"] = X_tr_n

# define kernel and hyperparams
Kernel = RBF

# kernel parameter dictionary
params={}
params['lengthscale']={'value':np.ones((1,data["X"].shape[1])) ,
                            'range':np.vstack([[0.1, 50.0]]*data["X"].shape[1]),
                            'transform': paramz.logexp()}
params['variance']={'value':np.array([1]), 
                            'range':np.vstack([[0.1, 200.0]]),
                            'transform': paramz.logexp()}



# define preference model 
model = erroneousPreference(data,Kernel,params,inf_method="laplace")
# compute hyperparameters
model.optimize_hyperparams(num_restarts=1) 
print(model.params)
# sample from posterior
model.sample(nsamples=5000, tune=500)
