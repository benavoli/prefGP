#!/usr/bin/env python3

import torch

class ChoiceDataset(torch.utils.data.Dataset):
    def __init__(self, data, CA, RA):
        
        self.data = data
        self.CA = CA
        self.RA = RA
        self.dim_a = len(self.CA[0])
        self.num_choices = len(self.CA)
    
    def __len__(self):
        return len(self.CA)
    
    def __getitem__(self, idx):
        return self.CA[idx], self.RA[idx]
    
    def get_data(self, ca, ra):
        idx_ca = ca.flatten()[ ca.flatten() > -1]
        idx_ra = ra.flatten()[ ra.flatten() > -1]
        idxs = torch.cat((idx_ca, idx_ra))
                
        return self.data[idxs.unique()], idxs.unique()
    
    def collate_fn(self, batch):
        ca = torch.vstack([ b[0] for b in batch])
        ra = torch.vstack([ b[1] for b in batch])
        
        data, idx_list = self.get_data(ca, ra)
        
        map_idx = {}.fromkeys(idx_list[idx_list > -1].numpy())
        for i, k in enumerate(map_idx.keys()):
            map_idx[k] = i
        
        map_idx[-1] = -1
        
        ca = ca.apply_(map_idx.get)
        ra = ra.apply_(map_idx.get)
        
        target = torch.stack((ca, ra))
        
        return data, target
        