#!/usr/bin/env python
import torch
import torch.nn as nn
from torch import optim

def eval_ann(model,metric,val_dataloader,device):
    val_loss=0.0
    for x,y in val_dataloader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            output=model(x)
            loss=metric(output,y)
            val_loss+=loss.item()
    return val_loss
    

def train_ann(model,metric,train_dataloader,val_dataloader,epochs,lr,device):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)

    min_loss,best_epoch=1000000.0,0
    for epoch in range(epochs):
        total_loss=0.0
        for x,y in train_dataloader:
            x,y=x.to(device),y.to(device)
            output=m(x)
            loss=metric(output,y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        
        val_loss=eval_ann(m,metric,val_dataloader,device)
        if val_loss<min_loss:
            min_loss=val_loss
            best_epoch=epoch
            torch.save(m.state_dict(),'./train_out/bm.ckpt')
        print('epoch {},training loss:{}'.format(epoch,total_loss)+' validation loss:{}'.format(val_loss))
    
    print('Done!\n')
        


