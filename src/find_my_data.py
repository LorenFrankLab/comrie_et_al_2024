import spyglass as nd
import datajoint as dj
import os

from spyglass.common import (Session, IntervalList, TaskEpoch, Nwbfile)
# from spyglass.utils.dj_helper_fn import get_nwb_file # moved
from spyglass.utils.nwb_helper_fn import get_nwb_file

from alison_position import PosValidTimesToEpoch


def is_nwb_mine(nwb_file_name):
    '''
    test T/F whether one nwb file name is from spatial bandit experiment
    '''
    my_subject_ids = ["Senor","chimi","wilbur","peanut","j16"]
    try:
        subject_id = (Session & {'nwb_file_name': nwb_file_name} & {'session_description LIKE "Spatial bandit%"'}).fetch1('subject_id')      
        if subject_id in my_subject_ids: # limit to relevant ones rn
            is_nwb_mine = True
        else:
            is_nwb_mine = False
    except Exception as e:
        #print(e)
        is_nwb_mine = False
    return is_nwb_mine   

def spatial_bandit_nwb_file_names():
    '''
    return list of all nwb file names from spatial bandit task, regardless of rat id
    '''
    spatial_bandit_nwbs = (Session & {'session_description LIKE "Spatial bandit%"'}).fetch('nwb_file_name')
    return spatial_bandit_nwbs

def spatial_bandit_nwb_file_names_by_rat(rat_list=["Senor","chimi","wilbur","peanut","j16"]):
    '''
    return dict of rats and their nwbs from spatial bandit experiment
    '''
    assert type(rat_list) == list, f"rat_list needs to be a list, not {type(rat_list)}"
    spatial_bandit_nwbs = {}
    for rat in rat_list:
        spatial_bandit_nwbs[rat] = list( (Session & {'session_description LIKE "Spatial bandit%"'} & {"subject_id": rat}).fetch('nwb_file_name') )
    return spatial_bandit_nwbs

def spatial_bandit_query_by_rat(rat_list=["Senor","chimi","wilbur","peanut","j16"]):
    '''
    to restrict a table to your data, e.g.: LFP & spatial_bandit_query_by_rat
    '''
    assert type(rat_list) == list, f"rat_list needs to be a list, not {type(rat_list)}"
    condition_collection = []
    for rat in rat_list:
        condition_collection.append(f'subject_id = "{rat}"')
    query = Session & {'session_description LIKE "Spatial bandit%"'} & condition_collection
    return query

def run_only_pos_interval_names(nwb_file_name):
    if is_nwb_mine(nwb_file_name):
        #get only the 'pos EP# valid times' intervals from interval list. 
        nwb_pos_intervals = [name for name in (IntervalList() & {'nwb_file_name': nwb_file_name}).
                             fetch('interval_list_name') if (name.split()[0]=='pos' and len(name.split())==4)]
        #only get runs
        nwb_pos_intervals_runs = [interval for interval in nwb_pos_intervals
                                  if ((TaskEpoch*PosValidTimesToEpoch) & {'nwb_file_name': nwb_file_name, 'pos_interval_list_name': interval})
                                  .fetch('task_name') != "sleep"]
        nwb_pos_intervals_runs.sort(key = lambda x: int(x.split()[1])) #sort by epch order / as int not str
        return nwb_pos_intervals_runs

def get_epoch_interval_names(nwb_file_name):
    '''Return run and sleep interval names in lists to make it easy to iterate over them.
    Also return the run noPrePostTrialTimes epoch names in a list.'''
    nwbf = get_nwb_file(Nwbfile().get_abs_path(nwb_file_name))
    epochs = nwbf.epochs.to_dataframe()
    epoch_dict = dict()
    epoch_dict['nwb_file_name'] = nwb_file_name
    
    #after finding epoch names, find associated intervals
    interval_list_names_runs = []
    interval_list_names_sleeps = []
    for e in epochs.iterrows():
        if e[1].tags[0][3] == "r":
            epoch_as_int = int(e[1].tags[0].split("_")[0])
            interval_list_names_runs.append(e[1].tags[0]) #such as 02_r1
        elif e[1].tags[0][3] == "s":
            epoch_as_int = int(e[1].tags[0].split("_")[0])
            interval_list_names_sleeps.append(e[1].tags[0]) #such as 03_s2
    
    interval_list_names_runs_nopreposttrialtimes = []
    for i in interval_list_names_runs:
        potential_interval_list_name = f"{i} noPrePostTrialTimes"
        interval_list_entries_len = len(IntervalList() & {"nwb_file_name": nwb_file_name, "interval_list_name": potential_interval_list_name})
        if interval_list_entries_len == 1:
            interval_list_names_runs_nopreposttrialtimes.append(potential_interval_list_name)
        elif interval_list_entries_len == 0:
            print(f"No interval list named {potential_interval_list_name} in IntervalList corresponding to epoch {i}.")
        elif interval_list_entries_len > 1:
            raise Exception(f"ERROR: More than one entry in IntervalList for potential interval list name {potential_interval_list_name}.")
    
    return interval_list_names_runs, interval_list_names_sleeps, interval_list_names_runs_nopreposttrialtimes