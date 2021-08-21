#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.nn.functional import relu

class mlp_block(nn.Module):
    def __init__(self,input_dim,output_dim,dropout=None):
        super(mlp_block,self).__init__()
        self.norm=nn.BatchNorm1d(input_dim,affine=False)
        if dropout:
            self.dropout=nn.Dropout(dropout)
        else:
            self.dropout=None
        self.fc=nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        x=self.norm(x)
        if self.dropout:
            x=self.dropout(x)
        x=self.fc(x)
        return x
    
class mlp_model(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(mlp_model,self).__init__()
        self.blk1=mlp_block(input_dim,hidden_dim,dropout=None)
        self.blk2=mlp_block(hidden_dim,hidden_dim,dropout=0.5)
        self.blk3=mlp_block(hidden_dim,output_dim,dropout=0.5)
    
    def forward(self,x):
        x=relu(self.blk1(x))
        x=relu(self.blk2(x))
        x=self.blk3(x)

        return x



