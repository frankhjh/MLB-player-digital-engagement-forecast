#!/usr/bin/env python
import torch

def pred_ann(model,metric,test_dataloader,device):
    test_loss=0.0
    for x,y in test_dataloader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            output=model(x)
            loss=metric(output,y)
            test_loss+=loss.item()
    return test_loss