
import os
from typing import ValuesView
import numpy as np
from collections import defaultdict

import spyglass as nd
import datajoint as dj
from spyglass.utils import SpyglassMixin

import pandas as pd
import statescriptparse as ssp
import datetime

# Import relevant schema/tables and helper functions
from spyglass.common import (Session, StateScriptFile, 
                                  DIOEvents, 
                                  IntervalList,                                   
                                  TaskEpoch,
                                  Nwbfile, AnalysisNwbfile)
                                #   NwbfileKachery, AnalysisNwbfileKachery) # not needed here for export
from spyglass.utils.dj_helper_fn import fetch_nwb #, get_nwb_file, moved to nwb helper fxns not dj helper fn
from spyglass.utils.nwb_helper_fn import get_nwb_file


def get_sc_file(key):
    #need a param that is dict of unique file name and epoch to get a unique entry in statscriptfile()
    sc = StateScriptFile.fetch_nwb(StateScriptFile & key)[0]['file']
    #sc = (StateScriptFile() & key).fetch_nwb()[0]['file']
    #sc_contents = sc.content
    f = sc.content.split('\n')
    return f

def get_printed_trial_info(sc_file, key):
    sc_trim = [line for line in sc_file if '#' not in line and '~~~' not in line and (len(line) != 0)]
    sc_printouts = [line for line in sc_trim if '=' in line]
    sc_printouts_split = [line.split() for line in sc_printouts]
    sc_printouts_info = [line for line in sc_printouts_split if len(line)==4]

    info_to_collect = ['contingency', 'trialThresh', 'timeMaxOut', 'totalPokes','totalRewards','leafProbs1','leafProbs2','leafProbs3','leafProbs4','leafProbs5','leafProbs6',
                       'decayAmnt','leafResetProbs1','leafResetProbs2','leafResetProbs3','leafResetProbs4','leafResetProbs5','leafResetProbs6',
                      'countPokes1','countPokes2','countPokes3','countPokes4','countPokes5','countPokes6']
    parsed_printed_info = defaultdict(list)
    for line in sc_printouts_info:
        info_name = line[1]
        info_value = int(line[3])
        if info_name in info_to_collect:
            parsed_printed_info[info_name].append(info_value)
    
    # Deal with no decay info about peanut and senor decay days
    # test if the nwb is supposedly a decay day based on the metadata file
    if (Session & {'nwb_file_name': key['nwb_file_name']}).fetch1('subject_id') in ['Senor', 'peanut']:
        if (Session & {'nwb_file_name': key['nwb_file_name']}).fetch1('session_description') == 'Spatial bandit task (depletion)':
            print("Parsing specially for peanut and Senor decay sc file!")
            #decay_variables = ['decayAmnt','leafResetProbs1','leafResetProbs2','leafResetProbs3','leafResetProbs4','leafResetProbs5','leafResetProbs6']
            #for i,v in enumerate(decay_variables):
            #   parsed_printed_info_decay = {}
            if 'decayAmnt' not in parsed_printed_info.keys():
                print(f"No decayAmnt info found in parsed printed sc info for {key['nwb_file_name']}, adding it now.")
                # try to fill in all the vars, not just decayAmnt
                sc_printouts_info_decay_catch = [line for line in sc_printouts_split if len(line)==5]
                assert len(sc_printouts_info_decay_catch) == 7, "there should be one line of len 5 in a sc log that separately prints decay amnt and reset probs once at start all 7 vars only"
                for line_caught in sc_printouts_info_decay_catch:
                    parsed_printed_info[line_caught[2]] = (np.ones(len(parsed_printed_info['contingency']))*(int(line_caught[4]))).astype(int).tolist()
                    print(f"Added {line_caught[2]} to parsed printed sc info for {key['nwb_file_name']}.")

    return pd.DataFrame.from_dict(parsed_printed_info)

def add_rew_and_leaf_to_printed_info(trial_info):
    trial_info['reward'] = ~((trial_info['totalRewards']==trial_info['totalRewards'].shift(1)) | (trial_info['totalRewards']==0))

    leaves = [1,2,3,4,5,6]
    leaf_poke_bools = {}
    total_true_so_far = 0
    for leaf in leaves:
        info_name = 'countPokes' + str(leaf)
        poke_here = ~((trial_info[info_name]==trial_info[info_name].shift(1)) | (trial_info[info_name]==0))
        leaf_poke_bools[leaf] = list(poke_here.values)
        total_true_so_far = total_true_so_far+sum(poke_here.values)
    #print('total poke: ', total_true_so_far)

    #check that every single trial sums to 1 across all the series for each location
    leafpokesdf = pd.DataFrame.from_dict(leaf_poke_bools)
    assert len(np.unique(leafpokesdf.sum(axis=1).values)) == 1, 'not detecting unique pokes across leaves properly'

    #multiply to get leaf id
    for leaf in leaves:
        leafpokesdf[leaf] = leaf*leafpokesdf[leaf]

    #print(leafpokesdf.sum(axis=1).values)
    trial_info['leaf_poke'] = leafpokesdf.sum(axis=1).values
    
    return trial_info

# def get_all_printed_info(sc_file, key):
#     trial_info = add_rew_and_leaf_to_printed_info( get_printed_trial_info(sc_file, key) )
#     return trial_info
def get_all_printed_info(sc_file, key): # modified this on Jul 18 2024 to debug senor20201110
    printed_trial_info = get_printed_trial_info(sc_file, key)
    if (key['nwb_file_name'] == "senor20201110_.nwb") and (key['epoch'] == 2):
        printed_trial_info = printed_trial_info[:161] # before dio breakout board malfunction
    trial_info = add_rew_and_leaf_to_printed_info(printed_trial_info)
    return trial_info

def parse_sc_with_time(f, _ecu_in, _ecu_out, key={}):
    df = ssp.todataframe(f, _ecu_in, _ecu_out) #make df of any dio signal with a timestamp from sc log
    #focus in on light on rows and poke in rows and poke columns
    # modified May 12, 2024 to solve the bug in parsing DIOs.
    if key['nwb_file_name'] == "senor20201110_.nwb":
        if key['epoch'] == 2:
            df = df.rename(columns = {'light3_r1': 'light3', 'poke3_r1': 'poke3'})
        else:
            df = df.rename(columns = {'light3_r2-6': 'light3', 'poke3_r2-6': 'poke3'})
        
    lightOnRows = df[(df['light1']==1.0) | (df['light2']==1.0) |
               (df['light3']==1.0) | (df['light4']==1.0) |
               (df['light5']==1.0) | (df['light6']==1.0)    ]        
    allPokeInRows = df[(df['poke1']==1.0) | (df['poke2']==1.0) |
                       (df['poke3']==1.0) | (df['poke4']==1.0) |
                       (df['poke5']==1.0) | (df['poke6']==1.0)    ]
    pokeCols = allPokeInRows[['poke1','poke2', 'poke3', 'poke4', 'poke5', 'poke6']]

    # lightOnRows = df[(df['light1']==1.0) | (df['light2']==1.0) |
    #                    (df['light3']==1.0) | (df['light4']==1.0) |
    #                    (df['light5']==1.0) | (df['light6']==1.0)    ]
    # allPokeInRows = df[(df['poke1']==1.0) | (df['poke2']==1.0) |
    #                    (df['poke3']==1.0) | (df['poke4']==1.0) |
    #                    (df['poke5']==1.0) | (df['poke6']==1.0)    ]
    # pokeCols = allPokeInRows[['poke1','poke2', 'poke3', 'poke4', 'poke5', 'poke6']]

    #when diff to find the changes, get rid of first column
    #so save the important things from the first column - the well and the timestamp of poking into the well
    firstPokeIn_well = pokeCols[pokeCols.isin([1.0]).any(axis=1)].idxmax(axis=1).values[0][4]
    firstPokeIn_ts = pokeCols.index[0]

    #now diff the changes to see when animal poked in somewhere and out of somewhere else
    pokeColsDiff = pokeCols.diff()
    pokeColsDiff = pokeColsDiff.drop(pokeColsDiff.index[[0]])

    #now find the location poked into and the timestamp of that pokeIn
    newPokesIn_wells = pokeColsDiff[pokeColsDiff.isin([1.0]).any(axis=1)].idxmax(axis=1).values
    newPokesIn_ts = pokeColsDiff[pokeColsDiff.isin([1.0]).any(axis=1)].idxmax(axis=1).index

    #keep track of all pokeIn wells and timestamps
    new_leaf_in_well = [int(firstPokeIn_well)]
    new_leaf_in_ts = [firstPokeIn_ts]
    for i in range(len(newPokesIn_wells)):
        new_leaf_in_well.append(int(newPokesIn_wells[i][4]))
        new_leaf_in_ts.append(newPokesIn_ts[i])
    #print(len(new_leaf_in_well), len(new_leaf_in_ts))
    #print(new_leaf_in_well, new_leaf_in_ts)
    
    #now add a pokeOUT timestamp for each of these
    dfDiff = df.diff()
    dfDiff = dfDiff.drop(dfDiff.index[[0]])
    dfDiff = dfDiff[['poke1','poke2', 'poke3', 'poke4', 'poke5', 'poke6']]

    #find the rows with unpokes 
    unpokeCols = dfDiff[dfDiff.isin([-1.0]).any(axis=1)]

    #similar strategy as with pokes, but now with unpokes
    new_leaf_out_well = []
    new_leaf_out_ts = []
    for i in range(len(new_leaf_in_well)):
        if i > 0:
            priorUnpokes = unpokeCols[unpokeCols.index < new_leaf_in_ts[i]]
            previousUnpoke_well = priorUnpokes.iloc[[-1]].idxmin(axis=1).values[0][4]
            previousUnpoke_ts = priorUnpokes.iloc[[-1]].idxmin(axis=1).index[0]
            new_leaf_out_well.append(int(previousUnpoke_well))
            new_leaf_out_ts.append(previousUnpoke_ts)
    lastPokeOut_well = pokeCols[pokeCols.isin([1.0]).any(axis=1)].idxmax(axis=1).values[len(pokeCols.index)-1][4]
    lastPokeOut_ts = pokeCols.index[len(pokeCols.index)-1]
    new_leaf_out_well.append(int(lastPokeOut_well))
    new_leaf_out_ts.append(lastPokeOut_ts)

    #trim off additional trials beyond the max trial number.
    # Jan 16 2024. Will make this specific to certain nwb files, this is for senor 1109 rn
    # max_trial = 300
    # new_leaf_out_well = new_leaf_out_well[:max_trial]
    # new_leaf_out_ts = new_leaf_out_ts[:max_trial]
    # new_leaf_in_well = new_leaf_in_well[:max_trial]
    # new_leaf_in_ts = new_leaf_in_ts[:max_trial]
    max_trial = 300
    if key['nwb_file_name'] == "senor20201112_.nwb" and key['epoch'] == 3:
        max_trial = 214
    if key['nwb_file_name'] == "senor20201110_.nwb" and key['epoch'] == 2:
        max_trial = 162
    if len(new_leaf_out_well) > max_trial:
        new_leaf_out_well = new_leaf_out_well[:max_trial]
        new_leaf_out_ts = new_leaf_out_ts[:max_trial]
        new_leaf_in_well = new_leaf_in_well[:max_trial]
        new_leaf_in_ts = new_leaf_in_ts[:max_trial]

    #test that things line up as expected
    for i in range(len(new_leaf_in_well)):
        assert new_leaf_out_well[i] == new_leaf_in_well[i], 'unpoke location at new leaf on trial index '+str(i)+'is NOT equal to poke location at new leaf'
        assert new_leaf_out_ts[i] >= new_leaf_in_ts[i], 'trial index '+str(i)+': unpoke time at new leaf is NOT greater than poke time at new leaf'
        if (new_leaf_out_ts[i] == new_leaf_in_ts[i]):
            print('WARNING new leaf out ts is equal to new leaf in ts on trial index '+str(i))
            
    if key['nwb_file_name'] == "senor20201112_.nwb" and key['epoch'] == 3: # Get rid of the entry that has a super big trodes time.
        df = df[df.index <= df.index[-1]] #specific debugging  july 2024 implemented oct 8 24
    dfPumpDiff = df[['pump1','pump2', 'pump3', 'pump4', 'pump5', 'pump6']].diff()

    #first cycle through each new leaf poke in
    reward_start_ts = []
    reward_end_ts = []

    for i,v in enumerate(new_leaf_in_well):
        # #get all pump values after a poke in
        # dfAfterPokeIn = dfPumpDiff[dfPumpDiff.index >= new_leaf_in_ts[i]] # removed oct 8 24 debugging
        #get all pump values after a poke in and before its corresponding poke out, modified  on Jul 2, 2024 implemented oct 8 24, for debugging senor20201112
        dfAfterPokeIn = dfPumpDiff[np.logical_and(dfPumpDiff.index >= new_leaf_in_ts[i],dfPumpDiff.index <= new_leaf_out_ts[i])] 
        if (len(dfAfterPokeIn[dfAfterPokeIn.isin([1.0]).any(axis=1)]) > 0):
            #look for the next pump turning on time
            #assess handling of IF NO MORE REWARDS can it still get pumpOn_ts
            pumpOn_ts = dfAfterPokeIn[dfAfterPokeIn.isin([1.0]).any(axis=1)].index[0]
            whichPumpIsOn = int(dfAfterPokeIn[dfAfterPokeIn.isin([1.0]).any(axis=1)].idxmax(axis=1).values[0][4])
            #for all except final trial
            if i<(len(new_leaf_in_well)-1):
                if (pumpOn_ts < new_leaf_in_ts[i+1]):
                    rewarded = 1
                    if (whichPumpIsOn != new_leaf_in_well[i]):
                        print('wrong pump rewarded? uh oh!')
                    reward_start_ts.append(pumpOn_ts)
                    pumpOff_ts = dfAfterPokeIn[dfAfterPokeIn.isin([-1.0]).any(axis=1)].index[0]
                    reward_end_ts.append(pumpOff_ts)
                else:
                    rewarded = 0
                    reward_start_ts.append('NaN')
                    reward_end_ts.append('NaN')
            elif i==(len(new_leaf_in_well)-1):
                if np.logical_and(key['nwb_file_name'] == 'peanut20201207_.nwb', key['epoch'] == 6):
                    rewarded = 0
                    reward_start_ts.append('NaN')
                    reward_end_ts.append('NaN')
                elif pumpOn_ts:
                    rewarded = 1
                    reward_start_ts.append(pumpOn_ts)
                    pumpOff_ts = dfAfterPokeIn[dfAfterPokeIn.isin([-1.0]).any(axis=1)].index[0]
                    reward_end_ts.append(pumpOff_ts)            
        else:
            #no more rewards in remaining trials! including trial i
            reward_start_ts.append('NaN')
            reward_end_ts.append('NaN')
    #make big df
    alltimestamps_df = pd.DataFrame(list(zip(new_leaf_in_well, new_leaf_in_ts, reward_start_ts, reward_end_ts, new_leaf_out_ts)),
                   columns =['leaf', 'pokeINtime', 'rewardONtime', 'rewardOFFtime', 'pokeOUTtime']) 
    alltimestamps_df = alltimestamps_df.replace('NaN', np.nan) #confirm  NaN handling throughout
    return alltimestamps_df 

# this currently uses unadjusted first nosepoke leaf location rather than printed trial info first nosepoke leaf location
def adj_sc_trodes_to_ptp_time(key, unadj_sc):
    adjusted_sc = unadj_sc.copy()

    #find trodes time and leaf location of first pokeINt in unadj sc log df
    first_poke_leaf_location = unadj_sc['leaf'][0]
    first_poke_unadj_trodes_timestamp = unadj_sc['pokeINtime'][0]

    #find ptp time of the same first dio nosepoke at the first leaf of the epoch
    #first, get the dio times associated with the first poke leaf location
    dio_name_for_first_leaf_poke = 'Poke'+str(first_poke_leaf_location)
    dio_poke_times, dio_poke_data = get_some_dio_events(key['nwb_file_name'], [dio_name_for_first_leaf_poke])
    #then, find the first dio time after the epoch starts at that leaf
    #interval_list_name = (IntervalList() & key).fetch('interval_list_name')[key['epoch']-1] #this  assumes the interval lists are in positions by epoch number
    ## updating 20240213 instead of ^ try to take advantage of where epoch number came from originally
    interval_list_name = (TaskEpoch() & key).fetch1('interval_list_name')
    #epoch_start_ptp_time = datetime.datetime.fromtimestamp((IntervalList & {'nwb_file_name':key['nwb_file_name'], 'interval_list_name':interval_list_name}).fetch1('valid_times')[0][0])
    epoch_start_ptp_time = (IntervalList & {'nwb_file_name':key['nwb_file_name'], 'interval_list_name':interval_list_name}).fetch1('valid_times')[0][0]
    mask = (dio_poke_data[dio_name_for_first_leaf_poke]==1) & (dio_poke_times[dio_name_for_first_leaf_poke]>epoch_start_ptp_time)
    first_poke_ptp_dio_timestamp = dio_poke_times[dio_name_for_first_leaf_poke][mask][0]

    #adjust
    for column in adjusted_sc.keys()[1:5]:
        #20220117 not using timedelta/datetime anymore, just leaving in s
        #adjusted_sc[column] = first_poke_ptp_dio_timestamp + pd.to_timedelta(unadj_sc[column]-first_poke_unadj_trodes_timestamp, unit="ms")
        adjusted_sc[column] = first_poke_ptp_dio_timestamp + (unadj_sc[column]-first_poke_unadj_trodes_timestamp)/1000 #ms to s before adding

    return adjusted_sc
    
def add_rew_to_adjusted_sc(adjusted_sc):
    #add whether rewarded or unrewarded to adjusted time parsed sc df
    adjusted_sc['reward_by_pumptime'] = ~pd.isnull(adjusted_sc['rewardONtime'])
    return adjusted_sc
    
def combine_time_and_printed_info(adj_sc_plus, printed_info):
    #check that first pokes are the same in printed info and nosepoke info
    #turn this into a try statement and handle exceptions appropriately
    assert adj_sc_plus['leaf'][0] == printed_info['leaf_poke'][0], "first leaf poke locations do NOT match across sc printed info and sc timing info!"

    #cut off excess pokes from nosepoke info
    n_trials_between_printed_and_timed_dfs = len(adj_sc_plus)  - len(printed_info)
    if n_trials_between_printed_and_timed_dfs > 0: 
        adj_sc_plus_trimmed = adj_sc_plus[:-n_trials_between_printed_and_timed_dfs]
    elif n_trials_between_printed_and_timed_dfs < 0:
        print('printed info found MORE trials than found in timed info!? what is going on?')
        adj_sc_plus_trimmed = adj_sc_plus
    else:
        print('same trial count in printed and timed sc info')
        adj_sc_plus_trimmed = adj_sc_plus
    assert len(adj_sc_plus_trimmed) == len(printed_info), "printed and timed sc info are NOT the same length and have different numbers of trials even after trailing nosepoke correction"

    #check that leaves and rewards match up still across dfs
    assert adj_sc_plus_trimmed['leaf'].equals(printed_info['leaf_poke']), 'leaves are NOT consistent across printed and timed sc info'
    assert adj_sc_plus_trimmed['reward_by_pumptime'].equals(printed_info['reward']), 'rewards are NOT consistent across printed and timed sc info'

    #combine dfs into one
    combined_sc_df_all = adj_sc_plus_trimmed.join(printed_info)

    combined_sc_df = combined_sc_df_all.drop(['reward_by_pumptime', 'leaf_poke','totalPokes','totalRewards',
                         'countPokes1','countPokes2','countPokes3','countPokes4','countPokes5','countPokes6'], axis=1)

    #make rew column bool int instead of bool T/F
    combined_sc_df['reward'] = 1*combined_sc_df['reward']
    
    #add stem info
    leaf_to_stem_map = {1:'A',2:'A',3:'B',4:'B',5:'C',6:'C'}
    combined_sc_df['stem'] = combined_sc_df['leaf'].map(leaf_to_stem_map)

    #make column ordering more human readable
    column_names = list(combined_sc_df.columns.values)
    column_names.insert(1,column_names.pop(column_names.index('stem')))
    column_names.insert(2,column_names.pop(column_names.index('reward')))
    combined_sc = combined_sc_df[column_names]
    combined_sc

    #improve naming of columns
    new_name_dict = {'pokeINtime':'poke_in_ts','pokeOUTtime':'poke_out_ts','rewardONtime':'reward_on_ts','rewardOFFtime':'reward_off_ts',
                    'trialThresh':'trials_thresh_per_conting','timeMaxOut':'minutes_thresh_per_conting','contingency':'contingency_count',
                    'leafProbs1':'pRew_leaf1','leafProbs2':'pRew_leaf2','leafProbs3':'pRew_leaf3',
                     'leafProbs4':'pRew_leaf4','leafProbs5':'pRew_leaf5','leafProbs6':'pRew_leaf6',
                    'leafResetProbs1':'pRew_reset_leaf1','leafResetProbs2':'pRew_reset_leaf2','leafResetProbs3':'pRew_reset_leaf3',
                     'leafResetProbs4':'pRew_reset_leaf4','leafResetProbs5':'pRew_reset_leaf5','leafResetProbs6':'pRew_reset_leaf6',
                    'decayAmnt':'decay_percent'
                    }
    combined_sc_full = combined_sc.rename(new_name_dict, axis='columns')
    
    return combined_sc_full

def get_dio_mapping(nwb_file_name):
    nwbf = get_nwb_file(Nwbfile().get_abs_path(nwb_file_name))
    _ecu_in = {}
    _ecu_out = {}
    for dio_name, dios in nwbf.fields["processing"]["behavior"]["behavioral_events"].fields["time_series"].items(): 
        dio_id = dios.fields['description']
        if dio_id[1] == "i":
            dio_id_minus_one = int(dio_id.split('n')[1])-1
            if len(dio_name) < 7:
                _ecu_in[dio_id_minus_one] = dio_name.lower()
            # also ok here when dio port went bad
            elif (len(dio_name) < 11) and (nwb_file_name == 'senor20201110_.nwb'):
                _ecu_in[dio_id_minus_one] = dio_name.lower()
        elif dio_id[1] == "o":
            dio_id_minus_one = int(dio_id.split('t')[1])-1
            _ecu_out[dio_id_minus_one] = dio_name.lower()
    return _ecu_in, _ecu_out

def get_some_dio_events(nwb_file_name2, list_of_dio_names):
    #param that is dict of unique file name and the funciton will get all the events from that table
    dio_times = {}
    dio_data = {}
    for dio_event in list_of_dio_names:
        dio_nwb_inside = DIOEvents.fetch_nwb(DIOEvents() & {'nwb_file_name':nwb_file_name2, 'dio_event_name':dio_event})[0]['dio']
        #dio_nwb_inside = (DIOEvents() & {'nwb_file_name':nwb_file_name2, 'dio_event_name':dio_event}).fetch_nwb()[0]['nwb']
        #20220117 commenting out to leave in ts seconds
        #event_times = np.array([datetime.datetime.fromtimestamp(t) for t in dio_nwb_inside.timestamps[:]])
        event_times = dio_nwb_inside.timestamps[:]
        event_data = dio_nwb_inside.data[:]
        #print(dio_event, len(event_times), len(event_data))
        #add to dictionaries
        dio_times[dio_event] = event_times
        dio_data[dio_event] = event_data
    return dio_times, dio_data

schema = dj.schema('alison_behav')

@schema
class StateScriptTrials(SpyglassMixin, dj.Computed):
    # Stores the poke and reward timing, as well as other printed metadata, 
    # for all trials in each run epoch's statescript log
    definition="""
    # Stores trial timing and info from all trials of each epoch's sc log
    -> StateScriptFile
    ---
    trial : blob #array of trial count (int) within a run epoch, same len as blobs of all other keys
    leaf : blob #array of leaves visited (int) in order of trials, 1-6
    stem : blob #array of stems visited (char) in order of trials, 'A','B','C'
    reward : blob #array of reward outcomes (bool) in order of trials
    poke_in_ts : blob #array of times in seconds
    reward_on_ts : blob #array of times in seconds
    reward_off_ts : blob #array of times in seconds
    poke_out_ts : blob #array of times in seconds
    contingency_count : blob #increments when conting changes
    trials_thresh_per_conting : blob #is constant for each run epoch
    minutes_thresh_per_conting : blob #is constant for each run epoch
    p_rew_leaf1 : blob #nominal p(Reward) as percent
    p_rew_leaf2 : blob #nominal p(Reward) as percent
    p_rew_leaf3 : blob #nominal p(Reward) as percent
    p_rew_leaf4 : blob #nominal p(Reward) as percent
    p_rew_leaf5 : blob #nominal p(Reward) as percent
    p_rew_leaf6 : blob #nominal p(Reward) as percent
    """
    class DecayTrialsInfo(SpyglassMixin, dj.Part):
        definition = """
        # Stores info from decaying epochs only for all trials of each epoch's sc log
        ->StateScriptTrials
        ---
        decay_percent : blob #constant for each epoch
        p_rew_reset_leaf1 : blob #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf2 : blob #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf3 : blob #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf4 : blob #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf5 : blob #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf6 : blob #nominal p(Reward) resets after decaying + stem switch
        """
        
    def make(self,key):
        """
        parse sc logs one day+epoch at a time to a df with trials as indices, then store
        df columns as arrays/blobs in table directly (instead of writing to analysisnwbfile)
        """
        
        task_name = (TaskEpoch() & key).fetch1('task_name')
        interval_list_name = (TaskEpoch() & key).fetch1('interval_list_name')
        print('starting epoch: ', key['epoch'], '. task_name: ', task_name, '. interval_list_name: ', interval_list_name)
    
        sc_file = get_sc_file(key)
    
        _ecu_in, _ecu_out = get_dio_mapping(key['nwb_file_name'])
        if ~np.logical_and(key['nwb_file_name'] == 'peanut_20201207_.nwb', key['epoch']==6):
            unadj_sc = parse_sc_with_time(sc_file, _ecu_in, _ecu_out, key)
        else:
            print('Special parsing for peanut 20201207 epoch 6!')
            unadj_sc = parse_sc_with_time(sc_file, _ecu_in, _ecu_out, key)
        print('got unadj sc')
        adj_sc = adj_sc_trodes_to_ptp_time(key, unadj_sc)
        print('adj trodes to ptp time')
        adj_sc_plus = add_rew_to_adjusted_sc(adj_sc)
        print('added rew info to adjusted sc')
        printed_info = get_all_printed_info(sc_file, key)
        print('got all printed info')
        df_for_an_epoch = combine_time_and_printed_info(adj_sc_plus, printed_info)
        print('made df for an epoch combining time and printed info')

        self.insert1({**key, **{'trial' : df_for_an_epoch.index.values,
                                'leaf' : df_for_an_epoch['leaf'].values,
                                'stem' : df_for_an_epoch['stem'].values,
                                'reward' : df_for_an_epoch['reward'].values,
                                'poke_in_ts' : df_for_an_epoch['poke_in_ts'].values,
                                'reward_on_ts' : df_for_an_epoch['reward_on_ts'].values,
                                'reward_off_ts' : df_for_an_epoch['reward_off_ts'].values,
                                'poke_out_ts' : df_for_an_epoch['poke_out_ts'].values,
                                'contingency_count' : df_for_an_epoch['contingency_count'].values,
                                'trials_thresh_per_conting' : df_for_an_epoch['trials_thresh_per_conting'].values,
                                'minutes_thresh_per_conting' : df_for_an_epoch['minutes_thresh_per_conting'].values,
                                'p_rew_leaf1' : df_for_an_epoch['pRew_leaf1'].values,
                                'p_rew_leaf2' : df_for_an_epoch['pRew_leaf2'].values,
                                'p_rew_leaf3' : df_for_an_epoch['pRew_leaf3'].values,
                                'p_rew_leaf4' : df_for_an_epoch['pRew_leaf4'].values,
                                'p_rew_leaf5' : df_for_an_epoch['pRew_leaf5'].values,
                                'p_rew_leaf6' : df_for_an_epoch['pRew_leaf6'].values}})
 

        print('\nPopulated computed table StateScriptTrials for nwb_file_name={nwb_file_name} on epoch={epoch}'.format(**key))
        
        #insert into parts table
        if ('decay_percent' in df_for_an_epoch):
            StateScriptTrials.DecayTrialsInfo.insert1({**key, **{'decay_percent' : df_for_an_epoch['decay_percent'].values,
                                                                 'p_rew_reset_leaf1' : df_for_an_epoch['pRew_reset_leaf1'].values,
                                                                 'p_rew_reset_leaf2' : df_for_an_epoch['pRew_reset_leaf2'].values,
                                                                 'p_rew_reset_leaf3' : df_for_an_epoch['pRew_reset_leaf3'].values,
                                                                 'p_rew_reset_leaf4' : df_for_an_epoch['pRew_reset_leaf4'].values,
                                                                 'p_rew_reset_leaf5' : df_for_an_epoch['pRew_reset_leaf5'].values,
                                                                 'p_rew_reset_leaf6' : df_for_an_epoch['pRew_reset_leaf6'].values}})

            print('\nPopulated parts table StateScriptTrials.DecayTrialsInfo for nwb_file_name={nwb_file_name} on epoch={epoch}'.format(**key))
        else:
            print('\nNot decaying, so nothing to populate in StateScriptTrials.DecayTrialsInfo for nwb_file_name={nwb_file_name} on epoch={epoch}'.format(**key))

@schema
class TrialsInfoByEpoch(SpyglassMixin, dj.Computed):
    definition="""
    # Stores trial timing and info from all trials, incl Decay info, of each epoch's sc log
    -> StateScriptTrials
    ---
    trial : blob #array of trial count (int) within a run epoch, same len as blobs of all other keys
    leaf : blob #array of leaves visited (int) in order of trials, 1-6
    stem : blob #array of stems visited (char) in order of trials, 'A','B','C'
    reward : blob #array of reward outcomes (bool) in order of trials
    poke_in_ts : blob #array of times in seconds
    reward_on_ts : blob #array of times in seconds
    reward_off_ts : blob #array of times in seconds
    poke_out_ts : blob #array of times in seconds
    contingency_count : blob #increments when conting changes
    trials_thresh_per_conting : blob #is constant for each run epoch
    minutes_thresh_per_conting : blob #is constant for each run epoch
    p_rew_leaf1 : blob #nominal p(Reward) as percent
    p_rew_leaf2 : blob #nominal p(Reward) as percent
    p_rew_leaf3 : blob #nominal p(Reward) as percent
    p_rew_leaf4 : blob #nominal p(Reward) as percent
    p_rew_leaf5 : blob #nominal p(Reward) as percent
    p_rew_leaf6 : blob #nominal p(Reward) as percent
    decay_percent = NULL : blob #constant for each epoch
    p_rew_reset_leaf1 = NULL : blob #nominal p(Reward) resets after decaying + stem switch
    p_rew_reset_leaf2 = NULL : blob #nominal p(Reward) resets after decaying + stem switch
    p_rew_reset_leaf3 = NULL : blob #nominal p(Reward) resets after decaying + stem switch
    p_rew_reset_leaf4 = NULL : blob #nominal p(Reward) resets after decaying + stem switch
    p_rew_reset_leaf5 = NULL : blob #nominal p(Reward) resets after decaying + stem switch
    p_rew_reset_leaf6 = NULL : blob #nominal p(Reward) resets after decaying + stem switch
    """
    class ByTrial(SpyglassMixin, dj.Part):
        definition = """
        # Stores info trial by trial (instead of by epoch)
        ->TrialsInfoByEpoch
        trial_number_by_epoch : int #trial count within the epoch, 0 based
        ---
        leaf : int #leaf visited, 1-6
        stem : varchar(40) #stem visited, 'A','B','C'
        reward : bool #reward outcome
        poke_in_ts : double #time in seconds
        reward_on_ts = NULL : double #time in seconds
        reward_off_ts = NULL : double #time in seconds
        poke_out_ts : double #time in seconds
        contingency_count : int #increments when conting changes
        trials_thresh_per_conting : int #is constant for each run epoch
        minutes_thresh_per_conting : int #is constant for each run epoch
        p_rew_leaf1 : int #nominal p(Reward) as percent
        p_rew_leaf2 : int #nominal p(Reward) as percent
        p_rew_leaf3 : int #nominal p(Reward) as percent
        p_rew_leaf4 : int #nominal p(Reward) as percent
        p_rew_leaf5 : int #nominal p(Reward) as percent
        p_rew_leaf6 : int #nominal p(Reward) as percent
        decay_percent = NULL : int #constant for each epoch
        p_rew_reset_leaf1 = NULL : int #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf2 = NULL : int #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf3 = NULL : int #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf4 = NULL : int #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf5 = NULL : int #nominal p(Reward) resets after decaying + stem switch
        p_rew_reset_leaf6 = NULL : int #nominal p(Reward) resets after decaying + stem switch
        """
        
    def make(self,key):
        """
        make new table that is essentially union of the parent tables
        """        
        all_trial_info = ((StateScriptTrials & key) + (StateScriptTrials.DecayTrialsInfo & key)).fetch(as_dict=True)[0]
        all_trial_info.pop('nwb_file_name')
        all_trial_info.pop('epoch')

        self.insert1({**key,**all_trial_info})
         
        #fill in nans for stable task - rather than separating decaying info into another parts table 
        trial_counts = all_trial_info.pop('trial')
        total_trials_by_epoch = len(trial_counts)
        
        if all_trial_info['decay_percent'] is None:
            for decay_related_data in ['decay_percent','p_rew_reset_leaf1','p_rew_reset_leaf2','p_rew_reset_leaf3','p_rew_reset_leaf4','p_rew_reset_leaf5','p_rew_reset_leaf6']:
                all_trial_info[decay_related_data] = np.full(total_trials_by_epoch, np.nan)
        
        #insert into parts table
        for trial in trial_counts:
            TrialsInfoByEpoch.ByTrial.insert1({**key,**{'trial_number_by_epoch': trial,
                                                        'leaf': all_trial_info['leaf'][trial],
                                                        'stem': all_trial_info['stem'][trial],
                                                        'reward': all_trial_info['reward'][trial],
                                                        'poke_in_ts': all_trial_info['poke_in_ts'][trial],
                                                        'reward_on_ts': all_trial_info['reward_on_ts'][trial],
                                                        'reward_off_ts': all_trial_info['reward_off_ts'][trial],
                                                        'poke_out_ts': all_trial_info['poke_out_ts'][trial],
                                                        'contingency_count': all_trial_info['contingency_count'][trial],
                                                        'trials_thresh_per_conting': all_trial_info['trials_thresh_per_conting'][trial],
                                                        'minutes_thresh_per_conting': all_trial_info['minutes_thresh_per_conting'][trial],
                                                        'p_rew_leaf1': all_trial_info['p_rew_leaf1'][trial],
                                                        'p_rew_leaf2': all_trial_info['p_rew_leaf2'][trial],
                                                        'p_rew_leaf3': all_trial_info['p_rew_leaf3'][trial],
                                                        'p_rew_leaf4': all_trial_info['p_rew_leaf4'][trial],
                                                        'p_rew_leaf5': all_trial_info['p_rew_leaf5'][trial],
                                                        'p_rew_leaf6': all_trial_info['p_rew_leaf6'][trial],
                                                        'decay_percent': all_trial_info['decay_percent'][trial],
                                                        'p_rew_reset_leaf1': all_trial_info['p_rew_reset_leaf1'][trial],
                                                        'p_rew_reset_leaf2': all_trial_info['p_rew_reset_leaf2'][trial],
                                                        'p_rew_reset_leaf3': all_trial_info['p_rew_reset_leaf3'][trial],
                                                        'p_rew_reset_leaf4': all_trial_info['p_rew_reset_leaf4'][trial],
                                                        'p_rew_reset_leaf5': all_trial_info['p_rew_reset_leaf5'][trial],
                                                        'p_rew_reset_leaf6': all_trial_info['p_rew_reset_leaf6'][trial]}})

        print('\nPopulated computed + parts tables TrialsInfoByEpoch and TrialsInfoByEpoch.ByTrial for nwb_file_name={nwb_file_name} on epoch={epoch}'.format(**key))

            
