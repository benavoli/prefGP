#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gpytorch
from gpytorch.models import ApproximateGP
import torch
import os
import pickle
from typing import Any, Dict
from gpytorch.distributions import MultivariateNormal, Distribution
from gpytorch.likelihoods import _OneDimensionalLikelihood
import torch
from torch import Tensor


class ChoiceLikelihood(_OneDimensionalLikelihood):
    
    def __init__(self, dim_a: int, num_choices: int, num_gp = 1, use_batches = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim_a = dim_a
        self.num_gp = num_gp
        self.eps = 1e-10
        self.use_batches = use_batches
        self.num_choices = num_choices
        self.num_comb = len(torch.combinations(torch.ones(dim_a), 2))
        
        self.CAr = None
        self.RAr = None
    
    def make_CAr_RAr(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        CA = observations[0]
        RA = observations[1]
        
        CAr = []
        for ca in CA:
            comb = torch.combinations(ca, 2)
            mask = (comb != -1).prod(dim=1, dtype=torch.bool)
            CAr.append(comb[mask])
        CAr = torch.vstack(CAr) if len(CAr) > 0 else CAr
        
        RAr = []
        for r in range(len(RA)):
            msk_c = CA[r] > -1
            msk_r = RA[r] > -1
            
            len_c = len(CA[r][msk_c])
            len_r = len(RA[r][msk_r])
            
            for j in range(len_r):
                for c in range(len_c):
                    tmp = -torch.ones(2, dtype=torch.int)
                    
                    tmp[0] = CA[r][msk_c][c]
                    tmp[1] = RA[r][msk_r][j]
                    
                    RAr.append(tmp)

        RAr = torch.vstack(RAr) if len(RAr) > 0 else []
        
        return CAr, RAr
        
    def loglike_CA(self, U0: Tensor, CAr: Tensor, _scale = 1.65) -> Tensor:
        '''
        U0: nx x nlatent utility matrix
        '''
        
        if len(CAr) == 0:
            return 0
        
        x = (U0[..., CAr[:, 0]] - U0[..., CAr[:, 1]])
        
        v = 0.5 * (torch.tanh(x * _scale / 2) + 1)
        
        prod_tmp = -torch.prod(v, dim=1)-torch.prod(1-v, dim=1)
        # this is made to prevent -inf as result, instead you put a 0
        prod_tmp[prod_tmp <= -1] = 0.
        
        # this is replaces the nan from -inf+inf in x
        full_prod_tens = torch.zeros(torch.Size([U0.shape[0]]) + torch.Size([self.dim_a*self.num_choices]), dtype=prod_tmp.dtype)
        full_prod_tens[..., torch.arange(0, CAr.shape[0])] = prod_tmp
        
        # we reshape to do the product over each choice set
        full_prod_tens = torch.reshape(full_prod_tens, (full_prod_tens.shape[0], self.num_choices, self.dim_a))

        out_tensor = torch.sum(torch.log1p(full_prod_tens+self.eps), axis=-1)
        
        return out_tensor
        
    def loglike_RA(self, U0: Tensor, RAr: Tensor, _scale = 1.65) -> Tensor:
        '''
        U0: nx x nlatent utility matrix
        '''
        
        if len(RAr) == 0:
            return 0
        
        x = (U0[..., RAr[:, 0]] - U0[..., RAr[:, 1]])
        
        A = 0.5 * (torch.tanh(x * _scale / 2) + 1)
        A = torch.prod(A, dim=1)
        
        # this is replaces the nan from -inf+inf in x
        full_A = torch.ones(torch.Size([U0.shape[0]]) + torch.Size([self.dim_a*self.num_choices]), dtype=A.dtype)
        full_A[..., torch.arange(0, RAr.shape[0])] = 1-A
        
        # reshape to be able to product over the choice set
        full_A = torch.reshape(full_A, (full_A.shape[0], self.num_choices, self.dim_a))

        out_1 = torch.prod(full_A, axis=-1)
        
        out_2 = torch.zeros_like(out_1)
        out_2[out_1 != 1] = out_1[out_1 != 1]

        out_tensor = torch.log1p(-out_2+self.eps)
        
        return out_tensor
        
    def log_prob_fun(self, function_samples: Tensor, CAr: Tensor, RAr: Tensor) -> Tensor:
        
        if function_samples.dim() > 2:
            U = function_samples.transpose(1,2) if self.num_gp == function_samples.shape[2] else function_samples
        else:
            #asser num_gp == 1 ??
            U = function_samples
            U = U.reshape(U.shape[0], 1, U.shape[1])
        
        return self.loglike_CA(U, CAr)+self.loglike_RA(U, RAr)
    
    def forward(self, function_samples: Tensor, *args: Any, data: Dict[str, Tensor] = ..., **kwargs: Any) -> Distribution:
        return super().forward(function_samples, *args, data=data, **kwargs)
    
    def expected_log_prob(self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> Tensor:
        
        if self.use_batches:
            self.CAr, self.RAr = self.make_CAr_RAr(observations)
        else:
            if self.CAr is None or self.RAr is None:
                self.CAr, self.RAr = self.make_CAr_RAr(observations)
        
        lambda_fun = lambda function_samples: self.log_prob_fun(function_samples, self.CAr, self.RAr)
        
        log_prob = self.quadrature(lambda_fun, function_dist)
        
        return log_prob

class PlackettLuceLikelihood(_OneDimensionalLikelihood):
    
    def __init__(self, use_batches=False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_batches = use_batches
        self.dict_m = None
    
    def build_m(self, pref):
        M={}
        lenp = torch.max(torch.tensor([len(p) for p in pref]))

        for i in range(2, lenp+1):
            M[i]=[]
        for p in pref:
            for d in torch.arange(len(p), 1, -1):
                M[d.item()].append(p[-d:])
        for k in M.keys():
            M[k] = torch.vstack(M[k])    
        return M
    
    def loglike_m(self, U0: Tensor, M) -> Tensor:
        
        V1 = torch.stack([torch.take_along_dim(u, M, dim=1) for u in U0])
        V2 = torch.stack([torch.take_along_dim(u, M[:, [0]], dim=1) for u in U0])
        # d2 = torch.log(torch.sum(torch.exp(V1-V2), dim=2))
        v1_tmp = torch.log(torch.sum(torch.exp(V1), dim=2, keepdim=True))
        d2 = V2-v1_tmp
        
        return d2.squeeze(2)
    
    def log_prob_fun(self, function_samples):
        U = function_samples
        v = 0
        
        for k in self.dict_m.keys():
            v += self.loglike_m(U, self.dict_m.get(k))
        
        return v
    
    def forward(self, function_samples: Tensor, *args: Any, data: Dict[str, Tensor] = ..., **kwargs: Any) -> Distribution:
        return super().forward(function_samples, *args, data=data, **kwargs)
    
    def expected_log_prob(self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> Tensor:
        
        if self.use_batches:
            self.dict_m = self.build_m(observations)
        else:
            if self.dict_m is None:
                self.dict_m = self.build_m(observations)
        
        lambda_fun = lambda function_samples: self.log_prob_fun(function_samples)
        
        log_prob = self.quadrature(lambda_fun, function_dist)
        
        return log_prob

class MultiChoiceGP(ApproximateGP):
    def __init__(self, num_tasks, inducing_points, learn_inducing_locations=False):

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, 
                learn_inducing_locations=learn_inducing_locations
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class AbstractPrefGPtorch:

    def __init(self,num_gp,inducing_points, learn_inducing_locations=False, use_batches=False):
        # dimA, num_gp, num_choices, inducing_points, learn_inducing_locations=False, use_batches=False
        # num_gp, num_pref, inducing_points, learn_inducing_locations=False, use_batches=False
        self.num_gp = num_gp
        self.inducing_points = inducing_points
        self.learn_inducing_locations = learn_inducing_locations
        self.use_batches = use_batches
        self.device = 'cpu'
        self.num_target = None

        self.innermodel = None
        self.likelihood = None
    
    def optimize(self, x_train, observations, num_iterations=1000, lr=0.01):
        self.innermodel.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.innermodel.parameters(), lr=lr)
        
        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the number of training datapoints
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.innermodel, self.num_target)
        
        for i in range(num_iterations):
            optimizer.zero_grad()

            output = self.innermodel(x_train)

            loss = -mll(output, observations)
            loss.backward()
            
            print(f'Iter {i+1}/{num_iterations} - Loss: {loss}')
            
            optimizer.step()
    
    def optimize_batches(self, dataloader, num_iterations=1000, lr=0.01):
        self.innermodel.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.innermodel.parameters(), lr=lr)
        
        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the number of training datapoints
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.innermodel, self.num_target)
        
        for i in range(num_iterations):
            step_loss = 0
            for data, target in dataloader:
                optimizer.zero_grad()
                # Get predictive output
                output = self.innermodel(data)
                # Calc loss and backprop gradients
                loss = -mll(output, target)
                loss.backward()
                step_loss += loss
                optimizer.step()
        
            print(f'Iter {i+1}/{num_iterations} - Loss: {step_loss/len(dataloader)}')
    
    def predict(self, x_pred,covariance=True):
        self.innermodel.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            y_preds = self.innermodel(x_pred)
            lower, upper = y_preds.confidence_region()
        if covariance:    
            return y_preds.mean.numpy(), y_preds.covariance_matrix.numpy(), lower.numpy(), upper.numpy()
        else:
            return y_preds.mean.numpy(), y_preds.variance.numpy(), lower.numpy(), upper.numpy()
    
    def _load(self, location):

        with open(location+"/param_dict.pkl", 'rb') as handle:
            self.params_dict = pickle.load(handle)

    def _save(self, location):
        if not os.path.exists(location):
            print("Creating directory: "+location)
            os.mkdir(location)
        else:
            print("Directory "+location+" already exists. Overwriting the files.")

        # write the params dict to pickle
        with open(location+"/param_dict.pkl", 'wb') as handle:
            pickle.dump(self.params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, location):

        self._save(location)

        # inner model state dict to save
        state_dict_to_save = self.innermodel.state_dict()

        # write inducing points to a pickle file
        with open(location+"/ind_pts.pkl", 'wb') as handle:
            pickle.dump(state_dict_to_save['variational_strategy.base_variational_strategy.inducing_points'], handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save gpytorch model
        torch.save(state_dict_to_save,location+"/gpytorch_model.pth")

    

class ChoiceGPtorch(AbstractPrefGPtorch):
    
    def __init__(self, dimA, num_gp, num_choices, inducing_points, learn_inducing_locations=False, use_batches=False):

        self.num_gp = num_gp
        self.num_target = num_choices
        self.learn_inducing_locations = learn_inducing_locations
        self.use_batches = use_batches
        self.dimA = dimA
        
        self.params_dict = {
            "num_gp": self.num_gp,
            "dimA": self.dimA,
            "learn_inducing_locations": self.learn_inducing_locations,
            "use_batches": self.use_batches,
            "num_target": self.num_target
        }  

        self.innermodel = MultiChoiceGP(num_tasks=self.num_gp, inducing_points=inducing_points, 
                                        learn_inducing_locations=self.learn_inducing_locations)
        self.likelihood = ChoiceLikelihood(dim_a=dimA, num_choices=self.num_target, num_gp=self.num_gp, use_batches=self.use_batches)

    

    def load(self, location):

        # Load and initialize parameters
        self._load(location)

        self.learn_inducing_locations = self.params_dict['learn_inducing_locations']
        self.num_gp = self.params_dict['num_gp']
        self.dimA = self.params_dict['dimA']
        self.num_target = self.params_dict['num_target']
        self.use_batches = self.params_dict['use_batches']


        # Load and initialize inducing points
        with open(location+"/ind_pts.pkl", 'rb') as handle:
            inducing_points = pickle.load(handle)

            
        self.innermodel = MultiChoiceGP(num_tasks=self.num_gp, inducing_points=inducing_points, 
                                        learn_inducing_locations=self.learn_inducing_locations)
        self.likelihood = ChoiceLikelihood(dim_a=self.dimA, num_choices=self.num_target, 
                                           num_gp=self.num_gp, use_batches=self.use_batches)

        
        state_dict = torch.load(location+"/gpytorch_model.pth")
        self.innermodel.load_state_dict(state_dict)
  
    
    
class PreferenceGPtorch(AbstractPrefGPtorch):
    def __init__(self, num_gp, num_pref, inducing_points, learn_inducing_locations=False, use_batches=False):

        self.num_gp = num_gp
        self.num_target = num_pref
        self.learn_inducing_locations = learn_inducing_locations
        self.use_batches = use_batches

        self.innermodel = MultiChoiceGP(num_tasks=self.num_gp, inducing_points=inducing_points, 
                                        learn_inducing_locations=self.learn_inducing_locations)
        self.likelihood = PlackettLuceLikelihood(use_batches=self.use_batches)

        self.params_dict = {
            "num_gp": self.num_gp,
            "learn_inducing_locations": self.learn_inducing_locations,
            "use_batches": self.use_batches,
            "num_target": self.num_target
        }

    def load(self, location):

        # Load and initialize parameters
        self._load(location)

        self.learn_inducing_locations = self.params_dict['learn_inducing_locations']
        self.num_gp = self.params_dict['num_gp']
        self.num_target = self.params_dict['num_target']
        self.use_batches = self.params_dict['use_batches']


        # Load and initialize inducing points
        with open(location+"/ind_pts.pkl", 'rb') as handle:
            inducing_points = pickle.load(handle)


        self.innermodel = MultiChoiceGP(num_tasks=self.num_gp, inducing_points=inducing_points, 
                                        learn_inducing_locations=self.learn_inducing_locations)
        self.likelihood = PlackettLuceLikelihood(use_batches=self.use_batches)

        
        state_dict = torch.load(location+"/gpytorch_model.pth")
        self.innermodel.load_state_dict(state_dict)
        