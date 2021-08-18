#!/usr/bin/env python
from torch.utils.data import Dataset

class mlb_dataset(Dataset):
    def __init__(self,targets,features):
        super(mlb_dataset,self).__init__()
        self.targets=targets
        self.features=features
    
    def __len__(self):
        return len(targets)
    
    def __getitem__(self,idx):
        return self.targets[idx],self.features[idx]
