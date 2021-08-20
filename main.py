#!/usr/bin/env python
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
from json_load import *
from data_process import feat_build,tar_feat_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from model.ANN.ann import mlp_model
from model.ANN.dataset import mlb_dataset
from train import train_ann,val_ann
from predict import pred_ann

def raw_data_process(path):
    # load raw data
    raw_data=pd.read_csv(path)
    # fill na with 0 for some cols
    raw_data.games=raw_data.games.fillna(0)
    raw_data.playerBoxScores=raw_data.playerBoxScores.fillna(0)
    raw_data.teamBoxScores=raw_data.teamBoxScores.fillna(0)
    raw_data.standings=raw_data.standings.fillna(0)

    # extract target df
    targets_df=json_load_engagement(raw_data[['nextDayPlayerEngagement']])
    # extract games df
    games_df=json_load_games_info(raw_data[['date','games']])
    # extract rosters
    rosters_df=json_load_roster_info(raw_data[['date','rosters']])
    # extract pbs df (playerBoxScores)
    missing_days=list(raw_data[(raw_data.games!=0) & (raw_data.playerBoxScores==0)].date)
    pbs_df=json_load_pBS_info(raw_data[['date','playerBoxScores']],missing_days)
    # extract tbs df (teamBoxScores)
    missing_days2=list(raw_data[(raw_data.games!=0) & (raw_data.teamBoxScores==0)].date)
    tbs_df=json_load_tBS_info(raw_data[['date','teamBoxScores']],missing_days2)
    # extract sd df
    sd_df=json_load_sd_info(raw_data[['date','standings']])
    # combine them together
    final_df=feat_build(targets_df,rosters_df,games_df,sd_df,pbs_df,tbs_df)

    return final_df

def train_val_test_split(df,train_size,val_size):
    # split the target and feature
    tar_feat_dict=tar_feat_split(final_df)
    
    # train/val/test split
    train_x,train_y=[],[]
    val_x,val_y=[],[]
    test_x,test_y=[],[]
    
    for i,(k,v) in enumerate(tar_feat_dict.items()):
        if i<train_size:
            train_x.append(v['features'])
            train_y.append(v['targets'])
        elif i>=train_size and i<train_size+val_size:
            val_x.append(v['features'])
            val_y.append(v['targets'])
        else:
            test_x.append(v['features'])
            test_y.append(v['targets'])
    
    train_x,train_y=torch.Tensor(train_x),torch.Tensor(train_y)
    val_x,val_y=torch.Tensor(val_x),torch.Tensor(val_y)
    test_x,test_y=torch.Tensor(test_x),torch.Tensor(test_y)

    train_data=mlb_dataset(train_y,train_x)
    val_data=mlb_dataset(val_y,val_x)
    test_data=mlb_dataset(test_y,test_x)

    train_data_loader=DataLoader(train_data,batch_size=128,shuffle=True)
    val_data_loader=DataLoader(val_data,batch_size=128,shuffle=False)
    test_data_loader=DataLoader(test_data,batch_size=1,shuffle=False)

    return train_data_loader,val_data_loader,test_data_loader


def Main(use_ann=True,raw_path,train_size,val_size):

    df=raw_data_process(raw_path)
    
    if use_ann:
        train_data_loader,val_data_loader,test_data_loader=train_val_test_split(df,train_size,val_size)
        # load model
        model=mlp_model
        # metric
        metric=nn.MSELoss()
        # epoch
        epochs=30
        # learning rate
        lr=1e-2
        # device
        device=torch.device('cpu')
        # train the model
        train_ann(model,metric,train_data_loader,val_data_loader,epochs,lr,device)

        # use the saved model to make prediction for test set
        best_model=model.load_state_dict(torch.load('./train_out/bm.ckpt'))
        # check the performance of bm on test set
        test_loss=pred_ann(best_model,metric,test_data_loader,device)

if __name__=='__main__':
    Main(use_ann=True,raw_path='./data/train_updated.csv',train_size=15000,val_size=3000)




    
    
    




