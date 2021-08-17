#!/usr/bin/env python

import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict

def json_load_engagement(target_df):
    size=target_df.shape[0]
    
    output=pd.DataFrame()
    for i in tqdm(range(size)):
        target_day=json.loads(target_df.nextDayPlayerEngagement[i])
        
        # create a dictionary to sotre sub result
        target_day_dict=defaultdict(list)
        
        # extract date
        target_day_dict['Metrics_Date']=pd.to_datetime([target_day[_]['engagementMetricsDate'] for _ in range(len(target_day))])
        # extract playerId
        target_day_dict['playerId']=[target_day[_]['playerId'] for _ in range(len(target_day))]
        # extract target 1-4
        for index in range(1,5):
            target_day_dict[f'target{index}']=[target_day[_][f'target{index}'] for _ in range(len(target_day))]
        
        sub_output=pd.DataFrame(target_day_dict)
        output=output.append(sub_output)
    output['date']=output['Metrics_Date'].apply(lambda x:x-pd.Timedelta(1,unit='d'))
    return output.reset_index(drop=True)
            

def json_load_games_info(games_df):
    games_df['games']=games_df['games'].fillna(0) # fill nan with 0
    
    # store result
    info_dict=defaultdict(list)
    
    size=games_df.shape[0]
    for i in tqdm(range(size)):
        info_dict['date'].append(games_df['date'][i])
        
        if games_df['games'][i]==0:
            info_dict['has_game'].append(0)
            info_dict['draw_team_id'].append([])
            info_dict['win_team_id'].append([])
            info_dict['loss_team_id'].append([])
            info_dict['home_loss_id'].append([])
            info_dict['away_win_id'].append([])
        else:
            games_info=json.loads(games_df['games'][i])
            info_dict['has_game'].append(1)
           
            win_list,loss_list,draw_list,home_loss,away_win=[],[],[],[],[]
            for game in games_info:
                if game['homeWinner']==game['awayWinner']:
                    draw_list+=[game['homeId'],game['awayId']]
                elif game['homeWinner'] and not game['awayWinner']:
                    win_list.append(game['homeId'])
                    loss_list.append(game['awayId'])
                else:
                    win_list.append(game['awayId'])
                    loss_list.append(game['homeId'])
                    home_loss.append(game['homeId'])
                    away_win.append(game['awayId'])
            info_dict['draw_team_id'].append(draw_list)
            info_dict['win_team_id'].append(win_list)
            info_dict['loss_team_id'].append(loss_list)
            info_dict['home_loss_id'].append(home_loss)
            info_dict['away_win_id'].append(away_win)
                    
    output=pd.DataFrame(info_dict)
    output['date']=pd.to_datetime(output['date'].apply(str).apply(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}'))
   
    return output
    

def json_load_roster_info(rosters_df):
    
    output=pd.DataFrame()
    
    size=rosters_df.shape[0]
    for i in tqdm(range(size)):
        info_dict_day=defaultdict(list)
        if type(rosters_df['rosters'][i])==str:
            roster_day=json.loads(rosters_df['rosters'][i])
            info_dict_day['date']=pd.to_datetime([roster_day[_]['gameDate'] for _ in range(len(roster_day))])
            info_dict_day['playerId']=[roster_day[_]['playerId'] for _ in range(len(roster_day))]
            info_dict_day['teamId']=[roster_day[_]['teamId'] for _ in range(len(roster_day))]
            info_dict_day['statusCode']=[roster_day[_]['statusCode'] for _ in range(len(roster_day))]

            sub_output=pd.DataFrame(info_dict_day)
            output=output.append(sub_output)
        else:
            print(f'The {i}th day has exception!')
   
    return output.reset_index(drop=True)


def json_load_pBS_info(pBS_df,missing_days): 
    # Data shows that for some days having games, the pBS lack information
    # so for these days, i will not include them in the train set
    pBS_dict=defaultdict(list)
    
    size=pBS_df.shape[0]
    for i in tqdm(range(size)):
        if pBS_df.date[i] not in missing_days:
            if pBS_df.playerBoxScores[i]==0:
                pBS_dict['date'].append(pBS_df.date[i])
                pBS_dict['teamId'].append(0)
                pBS_dict['playerId'].append(0)
                pBS_dict['positionCode'].append(0)
                pBS_dict['gamesPlayedBatting'].append(0) #?
                pBS_dict['flyOuts'].append(0)
                pBS_dict['groundOuts'].append(0)
                pBS_dict['runsScored'].append(0)
                pBS_dict['doubles'].append(0)
                pBS_dict['triples'].append(0)
                pBS_dict['homeRuns'].append(0)
                pBS_dict['strikeOuts'].append(0)
                pBS_dict['baseOnBalls'].append(0)
                pBS_dict['intentionalWalks'].append(0)
                pBS_dict['hits'].append(0)
                pBS_dict['hitByPitch'].append(0)
                pBS_dict['atBats'].append(0)
                pBS_dict['caughtStealing'].append(0)
                pBS_dict['stolenBases'].append(0)
                pBS_dict['groundIntoDoublePlay'].append(0)
                pBS_dict['groundIntoTriplePlay'].append(0)
                pBS_dict['plateAppearances'].append(0)
                pBS_dict['totalBases'].append(0)
                pBS_dict['rbi'].append(0)
                pBS_dict['leftOnBase'].append(0)
                pBS_dict['sacBunts'].append(0)
                pBS_dict['sacFlies'].append(0)
                pBS_dict['catchersInterference'].append(0)
                pBS_dict['pickoffs'].append(0)
            else:
                pBS_day=json.loads(pBS_df['playerBoxScores'][i])
                day=pBS_df['date'][i]
                for j in range(len(pBS_day)):
                    pBS_dict['date'].append(day)
                    pBS_dict['teamId'].append(pBS_day[j]['teamId'])
                    pBS_dict['playerId'].append(pBS_day[j]['playerId'])
                    pBS_dict['positionCode'].append(pBS_day[j]['positionCode'])
                    pBS_dict['gamesPlayedBatting'].append(pBS_day[j]['gamesPlayedBatting']) 
                    pBS_dict['flyOuts'].append(pBS_day[j]['flyOuts'])
                    pBS_dict['groundOuts'].append(pBS_day[j]['groundOuts'])
                    pBS_dict['runsScored'].append(pBS_day[j]['runsScored'])
                    pBS_dict['doubles'].append(pBS_day[j]['doubles'])
                    pBS_dict['triples'].append(pBS_day[j]['triples'])
                    pBS_dict['homeRuns'].append(pBS_day[j]['homeRuns'])
                    pBS_dict['strikeOuts'].append(pBS_day[j]['strikeOuts'])
                    pBS_dict['baseOnBalls'].append(pBS_day[j]['baseOnBalls'])
                    pBS_dict['intentionalWalks'].append(pBS_day[j]['intentionalWalks'])
                    pBS_dict['hits'].append(pBS_day[j]['hits'])
                    pBS_dict['hitByPitch'].append(pBS_day[j]['hitByPitch'])
                    pBS_dict['atBats'].append(pBS_day[j]['atBats'])
                    pBS_dict['caughtStealing'].append(pBS_day[j]['caughtStealing'])
                    pBS_dict['stolenBases'].append(pBS_day[j]['stolenBases'])
                    pBS_dict['groundIntoDoublePlay'].append(pBS_day[j]['groundIntoDoublePlay'])
                    pBS_dict['groundIntoTriplePlay'].append(pBS_day[j]['groundIntoTriplePlay'])
                    pBS_dict['plateAppearances'].append(pBS_day[j]['plateAppearances'])
                    pBS_dict['totalBases'].append(pBS_day[j]['totalBases'])
                    pBS_dict['rbi'].append(pBS_day[j]['rbi'])
                    pBS_dict['leftOnBase'].append(pBS_day[j]['leftOnBase'])
                    pBS_dict['sacBunts'].append(pBS_day[j]['sacBunts'])
                    pBS_dict['sacFlies'].append(pBS_day[j]['sacFlies'])
                    pBS_dict['catchersInterference'].append(pBS_day[j]['catchersInterference'])
                    pBS_dict['pickoffs'].append(pBS_day[j]['pickoffs'])
            
                    
    output=pd.DataFrame(pBS_dict)
    output['date']=pd.to_datetime(output['date'].apply(str).apply(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}'))
    return output
                
                

def json_load_tBS_info(tBS_df,missing_days):
    tBS_dict=defaultdict(list)
    
    size=tBS_df.shape[0]
    for i in tqdm(range(size)):
        if tBS_df.date[i] not in missing_days:
            if tBS_df.teamBoxScores[i]==0:
                tBS_dict['date'].append(tBS_df.date[i])
                tBS_dict['teamId'].append(0)
                tBS_dict['t_flyOuts'].append(0)
                tBS_dict['t_groundOuts'].append(0)
                tBS_dict['t_runsScored'].append(0)
                tBS_dict['t_doubles'].append(0)
                tBS_dict['t_triples'].append(0)
                tBS_dict['t_homeRuns'].append(0)
                tBS_dict['t_strikeOuts'].append(0)
                tBS_dict['t_baseOnBalls'].append(0)
                tBS_dict['t_intentionalWalks'].append(0)
                tBS_dict['t_hits'].append(0)
                tBS_dict['t_hitByPitch'].append(0)
                tBS_dict['t_atBats'].append(0)
                tBS_dict['t_caughtStealing'].append(0)
                tBS_dict['t_stolenBases'].append(0)
                tBS_dict['t_groundIntoDoublePlay'].append(0)
                tBS_dict['t_groundIntoTriplePlay'].append(0)
                tBS_dict['t_plateAppearances'].append(0)
                tBS_dict['t_totalBases'].append(0)
                tBS_dict['t_rbi'].append(0)
                tBS_dict['t_leftOnBase'].append(0)
                tBS_dict['t_sacBunts'].append(0)
                tBS_dict['t_sacFlies'].append(0)
                tBS_dict['t_catchersInterference'].append(0)
                tBS_dict['t_pickoffs'].append(0)
            else:
                tBS_day=json.loads(tBS_df['teamBoxScores'][i])
                day=tBS_df['date'][i]
                for j in range(len(tBS_day)):
                    tBS_dict['date'].append(day)
                    tBS_dict['teamId'].append(tBS_day[j]['teamId'])
                    tBS_dict['t_flyOuts'].append(tBS_day[j]['flyOuts'])
                    tBS_dict['t_groundOuts'].append(tBS_day[j]['groundOuts'])
                    tBS_dict['t_runsScored'].append(tBS_day[j]['runsScored'])
                    tBS_dict['t_doubles'].append(tBS_day[j]['doubles'])
                    tBS_dict['t_triples'].append(tBS_day[j]['triples'])
                    tBS_dict['t_homeRuns'].append(tBS_day[j]['homeRuns'])
                    tBS_dict['t_strikeOuts'].append(tBS_day[j]['strikeOuts'])
                    tBS_dict['t_baseOnBalls'].append(tBS_day[j]['baseOnBalls'])
                    tBS_dict['t_intentionalWalks'].append(tBS_day[j]['intentionalWalks'])
                    tBS_dict['t_hits'].append(tBS_day[j]['hits'])
                    tBS_dict['t_hitByPitch'].append(tBS_day[j]['hitByPitch'])
                    tBS_dict['t_atBats'].append(tBS_day[j]['atBats'])
                    tBS_dict['t_caughtStealing'].append(tBS_day[j]['caughtStealing'])
                    tBS_dict['t_stolenBases'].append(tBS_day[j]['stolenBases'])
                    tBS_dict['t_groundIntoDoublePlay'].append(tBS_day[j]['groundIntoDoublePlay'])
                    tBS_dict['t_groundIntoTriplePlay'].append(tBS_day[j]['groundIntoTriplePlay'])
                    tBS_dict['t_plateAppearances'].append(tBS_day[j]['plateAppearances'])
                    tBS_dict['t_totalBases'].append(tBS_day[j]['totalBases'])
                    tBS_dict['t_rbi'].append(tBS_day[j]['rbi'])
                    tBS_dict['t_leftOnBase'].append(tBS_day[j]['leftOnBase'])
                    tBS_dict['t_sacBunts'].append(tBS_day[j]['sacBunts'])
                    tBS_dict['t_sacFlies'].append(tBS_day[j]['sacFlies'])
                    tBS_dict['t_catchersInterference'].append(tBS_day[j]['catchersInterference'])
                    tBS_dict['t_pickoffs'].append(tBS_day[j]['pickoffs'])
            
                    
    output=pd.DataFrame(tBS_dict)
    output['date']=pd.to_datetime(output['date'].apply(str).apply(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}'))
    return output
    

def json_load_sd_info(sd_df):
    sd_dict=defaultdict(list)
    
    size=sd_df.shape[0]
    for i in tqdm(range(size)):
        if sd_df['standings'][i]!=0:
            sd_day=json.loads(sd_df['standings'][i])
            for _ in range(len(sd_day)):
                sd_dict['date'].append(sd_df['date'][i])
                sd_dict['teamId'].append(sd_day[_]['teamId'])
                sd_dict['divisionRank'].append(sd_day[_]['divisionRank'])
                sd_dict['leagueRank'].append(sd_day[_]['leagueRank'])
                sd_dict['pct'].append(sd_day[_]['pct'])
                sd_dict['runsAllowed'].append(sd_day[_]['runsAllowed'])
                #sd_dict['runsScored'].append(sd_day[_]['runsScored'])
                homepct=0.0 if sd_day[_]['homeWins']+sd_day[_]['homeLosses']==0 else sd_day[_]['homeWins']/(sd_day[_]['homeWins']+sd_day[_]['homeLosses'])
                sd_dict['homepct'].append(homepct)
                awaypct=0.0 if sd_day[_]['awayWins']+sd_day[_]['awayLosses']==0 else sd_day[_]['awayWins']/(sd_day[_]['awayWins']+sd_day[_]['awayLosses'])
                sd_dict['awaypct'].append(awaypct)
                last10pct=0.0 if sd_day[_]['lastTenWins']+sd_day[_]['lastTenLosses']==0 else sd_day[_]['lastTenWins']/(sd_day[_]['lastTenWins']+sd_day[_]['lastTenLosses'])
                sd_dict['last10pct'].append(last10pct)
                extraInningpct=0.0 if sd_day[_]['extraInningWins']+sd_day[_]['extraInningLosses']==0 else sd_day[_]['extraInningWins']/(sd_day[_]['extraInningWins']+sd_day[_]['extraInningLosses'])
                sd_dict['extraInningpct'].append(extraInningpct)
                onerunpct=0.0 if sd_day[_]['oneRunWins']+sd_day[_]['oneRunLosses']==0 else sd_day[_]['oneRunWins']/(sd_day[_]['oneRunWins']+sd_day[_]['oneRunLosses'])
                sd_dict['onerunpct'].append(onerunpct)
                daypct=0.0 if sd_day[_]['dayWins']+sd_day[_]['dayLosses']==0 else sd_day[_]['dayWins']/(sd_day[_]['dayWins']+sd_day[_]['dayLosses'])
                sd_dict['daypct'].append(daypct)
                nightpct=0.0 if sd_day[_]['nightWins']+sd_day[_]['nightLosses']==0 else sd_day[_]['nightWins']/(sd_day[_]['nightWins']+sd_day[_]['nightLosses'])
                sd_dict['nightpct'].append(nightpct)
    output=pd.DataFrame(sd_dict)
    output['date']=pd.to_datetime(output['date'].apply(str).apply(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}'))
    return output
    
    