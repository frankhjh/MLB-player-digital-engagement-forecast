#!/usr/bin/env python
import pandas as pd

def feat_combination(tar_out,rosters_out,games_out,sd_out,pbs_out,tbs_out):
    #merge the rosters
    meg_rosters=pd.merge(tar_out,rosters_out,on=['date','playerId'])
    #merge the games
    meg_games=pd.merge(meg_rosters,games_out,on=['date'],how='left')
    #merge the standings
    meg_standings=pd.merge(meg_games,sd_out,on=['date','teamId'])
    
    #divide the pbs_out into 2 parts and merge them separately
    pbs1,pbs2=pbs_out[pbs_out.teamId!=0],pbs_out[pbs_out.teamId==0]
    meg_pbs1=pd.merge(meg_standings,pbs1,on=['date','teamId','playerId'])

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

if __name__=='__main__':
    pass

