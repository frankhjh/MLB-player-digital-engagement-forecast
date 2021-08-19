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
from torch.utils.data import Dataset,DataLoader
from model.ANN.ann import mlp_model
from model.ANN.dataset import mlb_dataset
from train import train_ann,eval_ann
from predict import pred_ann


def main(use_ann=True,train_size,val_size):
    
    ##########DATA PROCESS!############
    # load raw data
    raw_data=pd.read_csv('./data/train_updated.csv')
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
    
    


