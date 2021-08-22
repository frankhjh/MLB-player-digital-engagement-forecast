#!/usr/bin/env python
import pandas as pd
import numpy as np
import gc
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
from model.GBDT import gbdt
from utils.dataset import mlb_dataset
from utils.mse_loss import mse
from utils.target_split import tar_split
from train import train_ann,val_ann
from predict import pred_ann
import argparse
parser=argparse.ArgumentParser(description='Parameters for the data prepare and model selection')
parser.add_argument('--data_path',type=str)
parser.add_argument('--train_size',type=int)
parser.add_argument('--val_size',type=int)
parser.add_argument('--model',type=str)
args=parser.parse_args()


if __name__=='__main__':
    ######### parameters ############
    path=args.data_path
    train_size=args.train_size
    val_size=args.val_size
    model=args.model
    ################################
    
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
    
    del raw_data,targets_df,rosters_df,games_df,sd_df,pbs_df,tbs_df
    gc.collect()
 

    #split target and features
    tar_feat_dict=tar_feat_split(final_df)
    del final_df
    gc.collect()

    x,y=[],[]
    for k,v in tar_feat_dict.items():
        x.append(v['features'])
        y.append(v['targets'])
    print('target/feature split done!')

    del tar_feat_dict
    gc.collect()
    
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
    
    # modeling
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
        lr=1e-3
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

    if model=='gbdt':
        test_loss=0.0
        for idx in tqdm(range(4)): # we have 4 targets

            train_x_arr=np.array(train_x)
            train_y_arr=np.array(tar_split(train_y,idx))

            test_x_arr=np.array(test_x)
            test_y_arr=np.array(tar_split(test_y,idx))
            
            print(f'{idx+1}th model start training...')
            gbdt.fit(train_x_arr,train_y_arr)
            print(f'{idx+1}th model training done!')
            pred=gbdt.predict(test_x_arr)
            
            single_mse_loss=mse(pred,test_y_arr)
            test_loss+=0.25*single_mse_loss
        print(f'test loss:{test_loss}')







    
    
    




