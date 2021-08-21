#!/usr/bin/env python
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
from sklearn.preprocessing import MinMaxScaler
from json_load import *
from data_process import feat_build,tar_feat_split,normalization
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch import optim
from torch.utils.data import DataLoader
from model.ann import mlp_model
from model.xgb import xgb_model
from utils.dataset import mlb_dataset
from utils.mse_loss import mse
from utils.target_split import tar_split
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
    print('raw data process done!')
    return final_df

def train_val_test_split(df,train_size,val_size):
    # split the target and feature
    tar_feat_dict=tar_feat_split(df)

    x,y=[],[]
    for k,v in tar_feat_dict.items():
        x.append(v['features'])
        y.append(v['targets'])
    print('target/feature split done!')

    # do the normalization for feature sets
    x=normalization(x)
    print('normalization done!')

    # train/val/test split
    train_x=x[:train_size]
    train_y=y[:train_size]
    val_x=x[train_size:train_size+val_size]
    val_y=y[train_size:train_size+val_size]
    test_x=x[train_size+val_size:]
    test_y=y[train_size+val_size:]
    print('train/val/test split done!')
    
    return train_x,train_y,val_x,val_y,test_x,test_y


def Main(raw_path,train_size,val_size,model):
    # process raw data
    df=raw_data_process(raw_path) 
    # split train/val/test data
    train_x,train_y,val_x,val_y,test_x,test_y=train_val_test_split(df,train_size,val_size)
    
    if model=='ann':
        train_x,train_y=torch.Tensor(train_x),torch.Tensor(train_y)
        val_x,val_y=torch.Tensor(val_x),torch.Tensor(val_y)
        test_x,test_y=torch.Tensor(test_x),torch.Tensor(test_y)

        train_data=mlb_dataset(train_x,train_y)
        val_data=mlb_dataset(val_x,val_y)
        test_data=mlb_dataset(test_x,test_y)

        train_data_loader=DataLoader(train_data,batch_size=128,shuffle=True)
        val_data_loader=DataLoader(val_data,batch_size=128,shuffle=False)
        test_data_loader=DataLoader(test_data,batch_size=64,shuffle=False)
        # load model
        input_dim=train_x.size(1)
        model=mlp_model(input_dim=input_dim,hidden_dim=64,output_dim=4)
        # metric
        metric=nn.MSELoss()
        # epoch
        epochs=30
        # learning rate
        lr=1e-2
        # device
        device=torch.device('cpu')
        # train the model
        print('start training...')
        train_ann(model,metric,train_data_loader,val_data_loader,epochs,lr,device)

        # use the saved model to make prediction for test set
        model.load_state_dict(torch.load('./train_out/bm.ckpt'))
        # check the performance of bm on test set
        print('start predicting...')
        test_loss=pred_ann(model,metric,test_data_loader,device)
        print(f'test loss:{test_loss}')
    
    if model=='xgboost':
        test_loss=0.0
        for idx in tqdm(range(4)): # we have 4 targets
            train_y=tar_split(train_y,idx)
            test_y=tar_split(test_y,idx)
            print(f'{idx+1}th model start training...')
            xgb_model.fit(train_x,train_y)
            print('train done!')
            pred=xgb_model.predict(test_x)
            
            single_mse_loss=mse(pred,test_y)
            test_loss+=0.25*single_mse_loss
        print(f'test loss:{test_loss}')



if __name__=='__main__':
    Main(raw_path='./data/train_updated.csv',train_size=150000,val_size=30000,model='xgboost')




    
    
    




