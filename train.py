#!/usr/bin/env python
import torch
import torch.nn as nn
from torch import optim

def val_ann(model,metric,val_dataloader,device):
    val_loss=0.0
    for step,(x,y) in enumerate(val_dataloader):
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            output=model(x)
            loss=metric(output,y)
            val_loss+=loss.item()
    
    return val_loss/(step+1)
    

def train_ann(model,metric,train_dataloader,val_dataloader,epochs,lr,device):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)

    min_loss,best_epoch=1000000.0,0
    for epoch in range(epochs):
        total_loss=0.0
        for step,(x,y) in enumerate(train_dataloader):
            x,y=x.to(device),y.to(device)
            output=m(x)
            loss=metric(output,y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_loss=total_loss/(step+1)
        
        val_loss=val_ann(m,metric,val_dataloader,device)
        if val_loss<min_loss:
            min_loss=val_loss
            best_epoch=epoch
            torch.save(m.state_dict(),'./train_out/bm.ckpt')
        print('epoch {},training loss:{}'.format(epoch,avg_loss)+' validation loss:{}'.format(val_loss))
    
    print('Done!\n')
        


