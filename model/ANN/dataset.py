#!/usr/bin/env python
from torch.utils.data import Dataset

class mlb_dataset(Dataset):
    def __init__(self,features,targets):
        super(mlb_dataset,self).__init__()
        self.features=features
        self.targets=targets
  
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self,idx):
        return self.features[idx],self.targets[idx]
