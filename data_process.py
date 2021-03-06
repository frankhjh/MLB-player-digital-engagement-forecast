#!/usr/bin/env python
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

# combination all information contained in json together
def feat_combination(tar_out,rosters_out,games_out,sd_out,pbs_out,tbs_out):
    #merge the rosters
    meg_rosters=pd.merge(tar_out,rosters_out,on=['date','playerId'])
    #merge the games
    meg_games=pd.merge(meg_rosters,games_out,on=['date'],how='left')
    #merge the standings
    meg_standings=pd.merge(meg_games,sd_out,on=['date','teamId'])
    
    #divide the pbs_out into 2 parts and merge them separately
    pbs1,pbs2=pbs_out[pbs_out.teamId!=0],pbs_out[pbs_out.teamId==0]
    meg_pbs1=pd.merge(meg_standings,pbs1.dropna(axis=0),on=['date','teamId','playerId'])

    #remove the teamId/playerId in pbs2 (all 0)
    pbs2_col=pbs_out.columns
    pbs2_col=[list(pbs2_col)[0]]+list(pbs2_col)[3:]
    pbs2=pbs2[pbs2_col]
    meg_pbs2=pd.merge(meg_standings,pbs2,on=['date'])
    
    meg_pbs=meg_pbs1.append(meg_pbs2)

    #divide the tbs_out into 2 parts and merge them separately
    tbs1,tbs2=tbs_out[tbs_out.teamId!=0],tbs_out[tbs_out.teamId==0]
    meg_tbs1=pd.merge(meg_pbs,tbs1,on=['date','teamId'])

    #remove the teamId in tbs2 (all 0)
    tbs2_col=tbs_out.columns
    tbs2_col=[list(tbs2_col)[0]]+list(tbs2_col)[2:]
    tbs2=tbs2[tbs2_col]
    meg_tbs2=pd.merge(meg_pbs,tbs2,on=['date'])

    meg_tbs=meg_tbs1.append(meg_tbs2)

    return meg_tbs

# create features
def feat_build(tar_out,rosters_out,games_out,sd_out,pbs_out,tbs_out):
    df_combination=feat_combination(tar_out,rosters_out,games_out,sd_out,pbs_out,tbs_out)
    df_combination=df_combination.reset_index(drop=True)
    
    #first transform the date into features
    df_combination['year']=df_combination.date.apply(lambda x:x.year)
    df_combination['month']=df_combination.date.apply(lambda x:x.month)
    df_combination['day']=df_combination.date.apply(lambda x:x.day)

    #one-hot encoding of the teamId
    teams=np.sort(df_combination.teamId.unique())
    for team in teams:
        df_combination[f'team{team}']=df_combination.teamId.apply(lambda x:x==team).astype(int)

    #one-hot encoding of the status Code
    stats=df_combination.statusCode.unique()
    for stat in stats:
        df_combination[f'{stat}']=df_combination.statusCode.apply(lambda x:x==stat).astype(int)
    
    # game result 
    #init
    df_combination['gamer']=0
    df_combination['winner']=0
    df_combination['losser']=0
    df_combination['drawer']=0
    df_combination['home_losser']=0
    df_combination['away_winner']=0

    for i in tqdm(range(len(df_combination))):
        if df_combination['teamId'][i] in df_combination['draw_team_id'][i]+df_combination['win_team_id'][i]+df_combination['loss_team_id'][i]:
            df_combination['gamer'][i]=1
        if df_combination['teamId'][i] in df_combination['win_team_id'][i]:
            df_combination['winner'][i]=1
        if df_combination['teamId'][i] in df_combination['loss_team_id'][i]:
            df_combination['losser'][i]=1
        if df_combination['teamId'][i] in df_combination['home_loss_id'][i]:
            df_combination['home_losser'][i]=1
        if df_combination['teamId'][i] in df_combination['away_win_id'][i]:
            df_combination['away_winner'][i]=1
    
    # drop out unnecessary columns
    df_combination.drop('Metrics_Date',axis=1,inplace=True)
    df_combination.drop('teamId',axis=1,inplace=True)
    df_combination.drop('statusCode',axis=1,inplace=True)
    df_combination.drop('has_game',axis=1,inplace=True)
    df_combination.drop('draw_team_id',axis=1,inplace=True)
    df_combination.drop('win_team_id',axis=1,inplace=True)
    df_combination.drop('loss_team_id',axis=1,inplace=True)
    df_combination.drop('home_loss_id',axis=1,inplace=True)
    df_combination.drop('away_win_id',axis=1,inplace=True)
    df_combination.drop('positionCode',axis=1,inplace=True)

    df_combination.sort_values(['date'],inplace=True)
    df_combination.reset_index(drop=True,inplace=True)

    # drop the numerical features with low variance(index before 66)
    nfeats=df_combination.columns[6:66]
    for f in nfeats:
        if df_combination[df_combination[f]==0].shape[0]/df_combination.shape[0]>0.95:
            df_combination.drop(f,axis=1,inplace=True)
    
    return df_combination

# do the normalization for features
def normalization(x): # x should be in form of [[...],[...],[...]], that is list of lists
    scaler=MinMaxScaler()
    return scaler.fit_transform(x)

# separate the target and features
def tar_feat_split(df):
    tar_feat_dict=defaultdict(dict)

    max_idx=df.shape[1]
    for i in tqdm(range(df.shape[0])):
        tar_feat_dict[str(df.playerId[i])+'-'+str(df.date[i])[:10]]['targets']=[float(df.iloc[i,j]) for j in range(1,5)]
        tar_feat_dict[str(df.playerId[i])+'-'+str(df.date[i])[:10]]['features']=[float(df.iloc[i,j]) for j in range(6,max_idx)]
    
    return tar_feat_dict

    



