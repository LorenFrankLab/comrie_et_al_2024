import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import s
from scipy import stats
import seaborn as sns
import warnings

import datajoint as dj
import spyglass as sg

# Custom schema

from alison_position import PosValidTimesToEpoch
from alison_decoding import (ClusterlessAcausalResultsSummary, ClusterlessResults, _retrieve_clusterless_results_data)
from plot_decode import trial_to_time_slice
from alison_behav import TrialsInfoByEpoch

# Import this  last b/c changes dir
from alison_rlmodel import BehaviorModelResults

## For scatterplots by trial of ahbeh as fxn of dv

def get_ahebeh_summary_stats_by_trials(
    rl_results_ep, clusterless_results_ep, 
    nwb_file_name, results, epoch_number, 
    run=True, well=False, pre_sec=-.2, post_sec=-.1
    ):
    '''
    return a dataframe with trial rows and ahbeh summary statistic columns
    operates on epoch at a time
    final df has same number of rows as initial input data (may have nans though)
    
    Inputs:
    rl_results_ep: df for one epoch of BehaviorModelResults.ByDay
    clusterless_results_ep: df for one epoch of Clusterless(A)causalResultsSummary()
    nwb_file_name: string
    results: xarray, based on decoding results
    epoch_number: int
    run: bool #include run times
    well: bool #include well times
    pre_sec: float #seconds to add or subtract from beginning of trial
    post_sec: float #seconds to add or subtract from end of trial
    
    Returns:
    ahbeh_by_trial: df of one epoch of trial rows and ahbeh summary statistic columns
    '''
    ahbeh_by_trial = pd.DataFrame(index=rl_results_ep.trial_number_by_epoch.values)

    ahbeh_summary_stats_by_trial = {'mean':{}, 'median':{}, 'max':{}, 'min':{}, 'var':{}, 'std':{}} 

    ahead_behind_distance = clusterless_results_ep['ahead_behind_distance'].values

    for trial_number in rl_results_ep.trial_number_by_epoch:
        # skip 0th trial because dont have a previous unpoke time for it
        if np.logical_and(trial_number >= 1, trial_number <= np.max(rl_results_ep.trial_number_by_epoch)):
            try:
                time_slice = trial_to_time_slice(nwb_file_name, results, epoch_number, trial_number, run=run, well=well, pre_sec=pre_sec, post_sec=post_sec)
                ahbeh = ahead_behind_distance[time_slice]
                # keep track of summary stats in dict
                ahbeh_summary_stats_by_trial['mean'][trial_number] = np.mean(ahbeh)
                ahbeh_summary_stats_by_trial['median'][trial_number] = np.median(ahbeh)
                ahbeh_summary_stats_by_trial['max'][trial_number] = np.max(ahbeh)
                ahbeh_summary_stats_by_trial['min'][trial_number] = np.min(ahbeh)
                ahbeh_summary_stats_by_trial['var'][trial_number] = np.var(ahbeh)
                ahbeh_summary_stats_by_trial['std'][trial_number] = np.std(ahbeh)          
            except Exception as e:
                print(f'\nskipped {trial_number} and just included nans, with exception: {e}')
                for summary_stat in ahbeh_summary_stats_by_trial.keys():
                    ahbeh_summary_stats_by_trial[summary_stat][trial_number] = np.nan
        else: #for first trial
            for summary_stat in ahbeh_summary_stats_by_trial.keys():
                ahbeh_summary_stats_by_trial[summary_stat][trial_number] = np.nan
    #turn the dict into a df
    for summary_stat in ahbeh_summary_stats_by_trial.keys():
        ahbeh_by_trial[f'ahbeh_{summary_stat}'] = ahbeh_summary_stats_by_trial[summary_stat].values()
    
    return ahbeh_by_trial

def get_ahbeh_summary_stats_by_trials_by_day(
    nwb_file_name,
    run=True, well=False, pre_sec=-.2, post_sec=-.1):
    '''
    loop through all epochs of a day to make one df of ahbeh summary stats for a day of data

    inputs:
    nwb_file_name
    usual flags for trial time slicing

    returns:
    ahbeh_by_trial_day: df of ahbeh summary stats for one day of data
    rl_results: just returns it because it adds some columns - should prob split that into a separate fxn though rather than in here
    '''
    # get a day worth of data
    rl_results = rl_results = (BehaviorModelResults.ByDay() & {'nwb_file_name':nwb_file_name}).fetch1_dataframe()    
    
    # order the pos intervals 
    intervals = (ClusterlessAcausalResultsSummary() & {'nwb_file_name':nwb_file_name}).fetch('interval_list_name')
    intervals = sorted(intervals, key=lambda x: int(x.split()[1]))
    
    # only work with intervals where  have decode data
    epochs = [(PosValidTimesToEpoch() & {'nwb_file_name':nwb_file_name, 'pos_interval_list_name':i}).fetch1('epoch') for i in intervals]
    
    # loop through each epoch, then concatenate epoch dfs into full day df, containing the ahbeh summary stats for each trial
    ahbeh_by_trial_ep_dfs_list = []
    for epoch_number, pos_x_valid_time_interval_name in zip(epochs, intervals): 
        print(f'getting ahbeh summay stats for epoch {epoch_number} and interval {pos_x_valid_time_interval_name}')
        # limit rl results to epoch
        rl_results_ep = rl_results[rl_results['epoch']==epoch_number]

        # clusterless results are really stored in epochs not days so work day at a time       
        clusterless_results_ep = (ClusterlessAcausalResultsSummary & {'nwb_file_name':nwb_file_name, 'interval_list_name':pos_x_valid_time_interval_name}).fetch1_dataframe()
        cr_key = (ClusterlessResults & {'nwb_file_name':nwb_file_name, 'interval_list_name':pos_x_valid_time_interval_name}).fetch1('KEY')
        environment, track_graph, results = _retrieve_clusterless_results_data(cr_key)

        # for each epoch, make a df of the summary stats as the columsn, trials as rows
        ahbeh_by_trial_ep = get_ahebeh_summary_stats_by_trials(rl_results_ep, clusterless_results_ep, 
            nwb_file_name, results, epoch_number, 
            run=run, well=well, pre_sec=pre_sec, post_sec=post_sec)
        ahbeh_by_trial_ep_dfs_list.append(ahbeh_by_trial_ep)
        # concatenate the dfs vertically
        ahbeh_by_trial_day = pd.concat(ahbeh_by_trial_ep_dfs_list)

    return ahbeh_by_trial_day # return the full day, so a separate fxn can plot it

def add_rl_results_analysis_columns(rl_results):
    """ Add a few columns collapsing some variables from many leaves/stems to the chosen or previous leaf/stem.

    Parameters
    ----------
    rl_results
        pd DataFrame from BehaviorModelResults()

    Returns
    -------
    rl_results
        pd DataFrame with new columns added
    """
    # can input rl results df for an epoch or day, but if a day, will treat as if one long epoch (won't do anything special at epoch transitions)
    Q_leaf_t_series = pd.concat([rl_results[rl_results['leaf']==leaf][f'Q{leaf}'] for leaf in [1,2,3,4,5,6]] , axis=0) # dont need to .sort_index()
    Q_stem_t_series = pd.concat([rl_results[rl_results['stem']==stem[0]][f'Qstem{stem[1]}'] for stem in {'A':1,'B':2,'C':3}.items()] , axis=0) # dont need to .sort_index() # str needs Qstem{stem[0]} not Q{stem[0]}, fixed 20240115
    rl_results = rl_results.assign(Q_leaf_t = Q_leaf_t_series)
    rl_results = rl_results.assign(Q_stem_t = Q_stem_t_series)
    rl_results['Q_leaf_tminus1'] = rl_results['Q_leaf_t'].shift()
    rl_results['Q_stem_tminus1'] = rl_results['Q_stem_t'].shift()
    return rl_results

def plot_ahbeh_summary_stats_by_trials_by_day_dv(dv, 
    nwb_file_name,
    run=True, well=False, pre_sec=-.2, post_sec=-.1,
    figsize=[10,5],limit_y=None,
    fit_reg=True,show_reg=True,linreg_p_thresh=.01,
    sns_ci=95,
    switch_only=False):
    '''
    plots a scatterplot of ahbeh summary statistic vs dv with a point for each trial for a day of data
    currently not excluding the first trial of each epoch data in a buggy way.. sometimes it doesnt make sense to plot, fyi

    inputs:
    dv: string of column name from rl_results df to say which decision variable is the one of interest
    nwb_file_name
    trial time slice params
    plotting params
    switch_only: bool that should be yoked to the dv in case the dv only applies to switch trials. Or, can be used if only want to plot switch trials for any dv.

    dv options currently include:
    ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Qstem1', 'Qstem2', 'Qstem3', #first row would be kind of weird to plot use last row instead
            'state_entropy', 'reward_entropy', 'stem_stay_p', 'stem_go_turn_p',
           'stem_turn_alone_p',  'stem_stay_var', 'stem_go_turn_var',
           'stem_turn_alone_var',  'stem_choice_p',
           'leaf_choice_p', 'stem_choice_var', 'leaf_choice_var',
           'Q_leaf_t', 'Q_stem_t', 'Q_stem_tminus1', 'Q_leaf_tminus1']
    not options/ not implemented well into one column yet:
    ['Qleaf1', 'Qleaf2', 'leaf_1_p', 'leaf_2_p','leaf_1_var', 'leaf_2_var',
            'stem_1_p', 'stem_2_p', 'stem_3_p', 'stem_1_var', 'stem_2_var', 'stem_3_var']
    '''
    
    # get ahbeh by trial df for full day
    ahbeh_by_trial_day = get_ahbeh_summary_stats_by_trials_by_day(
        nwb_file_name, 
        run=run, well=well, pre_sec=pre_sec, post_sec=post_sec)
    
    rl_results = (BehaviorModelResults.ByDay() & {'nwb_file_name':nwb_file_name}).fetch1_dataframe()    
    rl_results = add_rl_results_analysis_columns(rl_results)

    # some dvs need to be stem only, so override user param in just those cases
    if dv in ['leaf_choice_p', 'leaf_choice_var', 
             'stem_go_turn_p', 'stem_go_turn_var', 
             'stem_turn_alone_p', 'stem_turn_alone_var']: 
        switch_only = True

    ahbeh_summary_stats_by_trial = {'mean':[],'median':[],'min':[],'max':[],'std':[],'var':[]}
    for summary_stat in ahbeh_summary_stats_by_trial.keys():
        if switch_only:
            switch_only_mask = rl_results['stemswitch'].values #already bool, but dif dfs have dif inds, so make list not series
            dv_data = rl_results[dv][switch_only_mask]
            summary_stat_data = ahbeh_by_trial_day[f'ahbeh_{summary_stat}'][switch_only_mask]
        else:
            dv_data = rl_results[dv]            
            summary_stat_data = ahbeh_by_trial_day[f'ahbeh_{summary_stat}']
            #if True: #if dv[-1] in ['p','r']: # 
            #    first_trial_of_epoch_mask = [rl_results['trial_number_by_epoch']!=0]
            #    dv_data = rl_results[dv][first_trial_of_epoch_mask]         
            #    summary_stat_data = ahbeh_by_trial_day[f'ahbeh_{summary_stat}'][first_trial_of_epoch_mask]

        # set up lin reg to plot over scatterplot
        nan_mask = ~np.isnan(dv_data.values) & ~np.isnan(summary_stat_data.values) #just for linreg calculation, everythign else handles nans ok
        if fit_reg:
            linreg_result = stats.linregress(dv_data[nan_mask],summary_stat_data[nan_mask])
            linreg_results_str = (f'r-squared: {linreg_result.rvalue**2:.5f}\n'
                  f'r-value: {linreg_result.rvalue:.5f}\n'
                  f'p-value: {linreg_result.pvalue:.5f}\n'
                  f'slope: {linreg_result.slope:.5f}\n'
                  f'intercept: {linreg_result.intercept:.5f}\n')
            reg_color='lightblue'
            if linreg_result.pvalue < linreg_p_thresh:
                reg_color = 'darkblue'
        
        # look at a dv vs ah beh sum stat per trial
        plt.figure(figsize=figsize)
        plt.scatter(dv_data,summary_stat_data,alpha=.6,zorder=2,color='black')
        if show_reg:
            plt.plot(dv_data, linreg_result.intercept+linreg_result.slope*(dv_data),color=reg_color,zorder=1) #cant handle nans
            sns.regplot(x=dv_data,y=summary_stat_data, ci=sns_ci, scatter_kws={'alpha':0}, line_kws={'zorder':0})
        if limit_y is not None:
            plt.ylim(limit_y)
        plt.xlabel(f"{dv}")
        plt.ylabel(f"{summary_stat} ahbeh")
        plt.title(f"nwb_file_name: {nwb_file_name}, ntrials: {len(dv_data)}\n run: {run}, well: {well}, pre_sec: {pre_sec}, post_sec: {post_sec}")
        axes = plt.gca()
        text_y = axes.get_ylim()[0]+(axes.get_ylim()[1]-axes.get_ylim()[0])/2
        text_x = axes.get_xlim()[1]
        plt.text(text_x, text_y, linreg_results_str)
        sns.despine()
        plt.show()


## Plots looking at ahead behind vs dvs over trials around switches


def dict_to_df_by_trials_from_switch(dict_by_trials_from_switch, summary_stat):
    """ Turns dict into melted df of ahbeh stats by trials from switch
    Parameters
    ----------
    dict_by_trials_from_switch : dict
        Dictionary of dictionaries for ahbeh summary statistics around stem switches
    summary_stat : string 
        options are from AHBEH_SUMMARY_STATS=['mean','median','min','max','std','var','all']
    
    Returns
    -------
    df_by_trials_from_switch : pd DataFrame
        dataframe with trials from switch repeating in one column and the summary statistic in the second column - melted format
        useful for plotting with seaborn, like in plot_ahbeh_and_dv_by_trials_from_switch
    """
    df_by_trials_from_switch = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v, in dict_by_trials_from_switch[f'ahbeh_{summary_stat}'].items() ])).melt()
    df_by_trials_from_switch = df_by_trials_from_switch.rename(columns={"variable":"trials_from_switch","value":"ahbeh_statistic"})
    return df_by_trials_from_switch

def dv_to_df_by_trials_from_switch(dv_by_trial):
    """ Turns dict into melted df of rl results dvs by trials from switch
    Parameters
    ----------
    dv_by_trial : dict
        Dictionary of an rl results decision variable from BehaviorModelResults based on trials from switches
    
    Returns
    -------
    df_by_trials_from_switch : 
        Melted df with dv values in one column and trials from switch in another column
        useful for plotting with seaborn, like in plot_ahbeh_and_dv_by_trials_from_switch
    """
    df_by_trials_from_switch = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v, in dv_by_trial.items() ])).melt()
    df_by_trials_from_switch = df_by_trials_from_switch.rename(columns={"variable":"trials_from_switch","value":"dv"})
    return df_by_trials_from_switch


def get_ahbeh_stats_by_trials_from_switch_full(nwb_file_name, trials_surrounding_switch=7, ahbeh_inclusion='all',
                                            remove_unconcentrated_timepoints=True, hpd_percent=50, hpd_thresh=50,
                                            remove_unconcentrated_trials=True, hpd_min_prop_concentrated = .75,
                                            run=True, well=False, pre_sec=-.2, post_sec=-.1):
    """ Enables collecting ahbeh data or summary stats based on how many trials from a stem switch trial
    Collects info for all potential summary stats (mean median min max std var or raw ahbeh values)
    Can look at ahead and behind, ahead only, behind only, or abs(ahead behind)
    Can also filter timepoints by HPD or select trials by HPD

    Parameters
    ----------
    nwb_file_name
    trials_surrounding_switch : int
    ahbeh_inclusion : str
        ahbeh_inclusion can be ['all', 'ahead', 'behind', 'abs']
    remove_unconcentrated_timepoints : bool
    hpd_percent : int
        either 50 or 95
    hpd_thresh : int
        number of cm
    remove_unconcentrated_trials : bool
    hpd_min_prop_concentrated : float
        min proportion of included trial datapoints that must pass hpd criteria to be included in the analysis
    run : bool
    well : bool
    pre_sec : float
    post_sec : float

    Returns
    -------
    dict_by_trials_from_switch : dict
        the ahbeh summary stat data around switches
    len(interval_list_names) : int
        n epochs included - only returning this and the next two so it's easy to look at
    total_stem_switch_trials : int
        n stem switches in this day of data (not necessarily the included trials)
    concentration_proportion_per_trial : list
        list of the # points that are concentrated enough / total # points, which may have been thresholded
    """
    warnings.simplefilter('ignore', category=DeprecationWarning)
    warnings.simplefilter('ignore', category=ResourceWarning)
    
    AHBEH_SUMMARY_STATS = ['mean','median','min','max','std','var','all']
    
    # Set up empty dict to fill in with ahbeh summary stats per trial from stem switch
    dict_by_trials_from_switch = {}
    for summary_stat in AHBEH_SUMMARY_STATS:
        dict_by_trials_from_switch[f'ahbeh_{summary_stat}'] = {}
        for trials_from_switch in range(-trials_surrounding_switch, trials_surrounding_switch+1):
            dict_by_trials_from_switch[f'ahbeh_{summary_stat}'][trials_from_switch] = []
    
    concentration_proportion_per_trial = []
    total_stem_switch_trials = 0
    
    interval_list_names = np.unique((ClusterlessAcausalResultsSummary() & {'nwb_file_name':nwb_file_name}).fetch('interval_list_name'))
    interval_list_names = sorted(interval_list_names, key=lambda x: int(x.split()[1]))
    for i in interval_list_names: 
        print(f'Calculating ahbeh stats for interval list name: {i}')
        
        # Get clusterless results, ahbeh, and stem switches
        results_df_ep = (ClusterlessAcausalResultsSummary() & {'nwb_file_name':nwb_file_name, 'interval_list_name':i}).fetch1_dataframe() 
        cr_key = (ClusterlessResults & {'nwb_file_name':nwb_file_name, 'interval_list_name':i}).fetch1('KEY')
        results = _retrieve_clusterless_results_data(cr_key)[2]

        #ahead_behind_distance = results_df_ep['ahead_behind_distance'].values

        epoch_number = results_df_ep['epoch_number'].values[0] 
        trial_info = pd.DataFrame(TrialsInfoByEpoch().ByTrial() & {'nwb_file_name':nwb_file_name,'epoch':epoch_number})
        stem_switch_trials = [trial_info.index[i] for i in range(1,len(trial_info)) if trial_info["stem"][i] != trial_info["stem"][i-1]]
        total_stem_switch_trials += len(stem_switch_trials)
        #print(f'\nstem switch trials include: {stem_switch_trials}\n')
        #print(f'number of mean ahead behinds for keys -7 to 7: {[len(value) for value in dict_by_trials_from_switch.values()]}')
                
        if len(stem_switch_trials) > 0:
            for stem_switch_trial in stem_switch_trials:
                for trials_from_switch in range(-trials_surrounding_switch, trials_surrounding_switch+1):
                    trial_number = stem_switch_trial + trials_from_switch
                    if np.logical_and(trial_number >= 1, trial_number < len(trial_info)):
                        try:
                            time_slice = trial_to_time_slice(nwb_file_name, results, epoch_number, trial_number, run=run, well=well, pre_sec=pre_sec, post_sec=post_sec)

                            results_df_trial = results_df_ep.iloc[time_slice]
                            if remove_unconcentrated_timepoints:
                                results_df_trial_concentrated = results_df_trial[results_df_trial[f'spatial_coverage_{hpd_percent}_hpd']<hpd_thresh]
                                ahbeh = results_df_trial_concentrated['ahead_behind_distance'].values
                            else:
                                ahbeh = results_df_trial['ahead_behind_distance'].values
                            
                            if ahbeh_inclusion == 'abs':
                                ahbeh = np.abs(ahbeh)
                            elif ahbeh_inclusion == 'ahead':
                                ahbeh = ahbeh[ahbeh>=0]
                            elif ahbeh_inclusion == 'behind':
                                ahbeh = ahbeh[ahbeh<0]
                            elif ahbeh_inclusion != 'all':
                                print(f'Warning: invalid ahbeh_inclusion argument {ahbeh_inclusion}, using \'all\' ahbeh instead')
                            proportion_concentrated = len(results_df_trial_concentrated)/len(results_df_trial)
                            concentration_proportion_per_trial.append(proportion_concentrated)
                            if remove_unconcentrated_trials:
                                assert proportion_concentrated > hpd_min_prop_concentrated, f'proportion concentrated {proportion_concentrated} is <{hpd_min_prop_concentrated}% of trimmed trial time'
                            
                            dict_by_trials_from_switch[f'ahbeh_mean'][trials_from_switch].append(np.mean(ahbeh))
                            dict_by_trials_from_switch[f'ahbeh_median'][trials_from_switch].append(np.median(ahbeh))
                            dict_by_trials_from_switch[f'ahbeh_min'][trials_from_switch].append(np.min(ahbeh))
                            dict_by_trials_from_switch[f'ahbeh_max'][trials_from_switch].append(np.max(ahbeh))
                            dict_by_trials_from_switch[f'ahbeh_std'][trials_from_switch].append(np.std(ahbeh))
                            dict_by_trials_from_switch[f'ahbeh_var'][trials_from_switch].append(np.var(ahbeh))
                            dict_by_trials_from_switch[f'ahbeh_all'][trials_from_switch].extend(ahbeh)
                        except Exception as e:
                            print(f'\nskipped {trial_number} when doing stem switch trial {stem_switch_trial}, with exception: {e}')
    return dict_by_trials_from_switch, len(interval_list_names), total_stem_switch_trials, concentration_proportion_per_trial

def plot_hist_prop_of_datapoints_concentrated_per_trial(concentration_proportion_per_trial, hpd_percent, hpd_threshold, hpd_min_prop_concentrated, nwb_file_name, run, well, pre_sec, post_sec, fontsize=16, save_fig=False):
    """Visualize distribution of how much of each trial passes concentration thresholds
    Assumes your inputs all go together form previously used parameter sets! Isn't actually calculating anything new - just plotting
    """
    plt.figure(figsize=(10,5))
    plt.hist(concentration_proportion_per_trial,bins=30)
    plt.xlabel(f'proportion of concentrated trial time\nwith hpd_{hpd_percent} < {hpd_threshold} cm', fontsize=fontsize)
    plt.ylabel('trials', fontsize=fontsize)
    plt.gca().tick_params(labelsize=fontsize-2)
    plt.vlines(x=hpd_min_prop_concentrated,ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1],color='red', alpha=.5)
    plt.title(f'{nwb_file_name}\n run:{run}, well:{well}, pre_sec:{pre_sec}, post_sec:{post_sec}', fontsize=fontsize)
    sns.despine()
    if save_fig:
            save_path = f'/stelmo/alison/ahbeh_and_dv_around_stem_switch_figs/hist_prop_of_datapoinst_concentrated_per_trial_{nwb_file_name}_hpd{hpd_percent}_{hpd_threshold}cm_minpropconc{hpd_min_prop_concentrated}_run{run}_well{well}_presec{pre_sec}_postsec{post_sec}.pdf'
            plt.savefig(save_path, bbox_inches="tight")
            print(f'saved pdf at: {save_path}')
    plt.show()

def get_dv_around_switch_by_trial(rl_results, decision_var, nwb_file_names,
                                  trials_surrounding_switch,
                                  dv_specific_leaf=None, dv_specific_stem=None):
    """
    Parameters
    ----------
    rl_results : pd DataFrame
        from BehaviorModelResults
    decision_var : string
        the name of a column in rl_results df
    trials_surrounding_switch : int
    dv_specific_leaf : int (1,2,3,4,5,6)
        if dv is specific to one leaf
    dv_specific_stem : int (1,2,3)
        if dv is specific to one stem

    Returns
    -------
    dv_by_trial : dict
        dictionary of dv values based on trials around switches
    """
    # this is not for a specific leaf, that shoudl be provided in teh dv
    dv_by_trial = {}
    for trials_from_switch in range(-trials_surrounding_switch,trials_surrounding_switch+1):
        dv_by_trial[trials_from_switch] = []
    for nwb_file_name in nwb_file_names:
        if dv_specific_leaf is None and dv_specific_stem is None:
            switches = np.array(rl_results['stemswitch'][(rl_results['stemswitch'] == True)].index)
        elif dv_specific_leaf in [1,2,3,4,5,6]:
            switches = np.array(rl_results['stemswitch'][(rl_results['stemswitch'] == True) &
                                                        (rl_results['leaf'] )== dv_specific_leaf].index)
        elif dv_specific_stem in [1,2,3]:
            switches = np.array(rl_results['stemswitch'][(rl_results['stemswitch'] == True) &
                                                        (rl_results['leaf'] )== dv_specific_stem].index)
        else:
            print('arguments for specific leaf/stem are incompatible')
    #print(f'switches for day {nwb} include indices {switches}')
    #for each switch in a day, find the X trials before and after, and plot the dv    
    for stem_switch_trial in switches:
         for trials_from_switch in range(-trials_surrounding_switch, trials_surrounding_switch+1):
            trial_number = stem_switch_trial+trials_from_switch
            if trial_number>rl_results.index.min():
                try:
                    dv = rl_results[decision_var][trial_number]
                    dv_by_trial[trials_from_switch].append(dv)
                except Exception as e:
                    #print(f'skipped trial {trial_number} with exception {e}')
                    pass
    return dv_by_trial

def plot_ahbeh_and_dv_by_trials_from_switch(ahbeh_dict_by_trials_from_switch, rl_results, decision_var,
                                    nwb_file_name,
                                    summary_stats=['mean','median','min','max','std','var','all'],
                                    n_epochs=None,n_switches=None,
                                    run=True, well=False, pre_sec=-.2, post_sec=-.1,
                                    fontsize=16,figsize=(20,16), save_fig=False): #set_ylim=None):
    """ Plots violinplot and stripplot (a jittered grouped scatter of raw datapts) for ahbeh, again for decision_var
    Also plots subplots with mean/median of ahbeh and decision_var so can zoom in on just centers rather than full distribution

    Note: trials aren't paired up right now across the ahbeh data and dv data - a trial excluded by hpd for ex from ahbeh would still be included in dv

    Parameters
    ----------
    ahbeh_dict_by_trials_from_switch : dict
        from get_ahbeh_stats_by_trials_from_switch_full
    rl_results : pd DataFrame
        from BehaviorModelResults
    decision_var : str
        a column name from rl_results, determines which dv to plot
    nwb_file_name : str
    summary_stats : list of strings
        any subset or all of ['mean','median','min','max','std','var','all']
        determines which ones to actually plot
    n_epochs : int
        from get_ahbeh_stats_by_trials_from_switch_full
    n_switches : int
        from get_ahbeh_stats_by_trials_from_switch_full
    run : bool
    well : bool
    pre_sec : float
    post_sec : float
    fontsize : int
    figsize : tuple (width,height)
    save_fig : bool
        path is hardcoded on stelmo rn, includes params, doesnt include HPD info or ah/beh/ahbeh/abs info!!

    Returns
    -------
    Just shows plots, doesn't return
    """
    # get extra rl results columns
    rl_results = add_rl_results_analysis_columns(rl_results)
    # get rl results for just dv of interest
    trials_surrounding_switch=int((len(ahbeh_dict_by_trials_from_switch['ahbeh_all'])-1)/2)
    dv_by_trial = get_dv_around_switch_by_trial(rl_results, decision_var=decision_var, nwb_file_names=[nwb_file_name],  trials_surrounding_switch=trials_surrounding_switch)
    # turn that dv info into df
    dv_by_trials_from_switch = dv_to_df_by_trials_from_switch(dv_by_trial)
    
    for summary_stat in summary_stats:
        # Turn dictionary into long df for sns plotting
        df_by_trials_from_switch = dict_to_df_by_trials_from_switch(ahbeh_dict_by_trials_from_switch, summary_stat)

        # Make figure with violinplot and stripplot
        fig, axes = plt.subplots(nrows=6, sharey=False, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [1, 1, 1, 1, .5, .5]})
        sns.violinplot(x="trials_from_switch", y="ahbeh_statistic", data=df_by_trials_from_switch, ax=axes[0], 
                       inner="quartiles", scale="count", cut=0, color="lightgreen", linewidth=.5)
        # make median more obvious, leave the other quartile lines
        for l in axes[0].lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('green')
        sns.stripplot(x="trials_from_switch",y="ahbeh_statistic",data=df_by_trials_from_switch, ax=axes[1],
                      color='green', linewidth=.5, edgecolor='green',alpha=.4, size=5, jitter=1)
        
        # Below the ahbeh plots, add the DV plots
        sns.violinplot(x="trials_from_switch", y="dv", data=dv_by_trials_from_switch, ax=axes[2], 
                       inner="quartiles", scale="count", cut=0, color="lightblue", linewidth=.5)
        # make median more obvious, leave the other quartile lines
        for l in axes[2].lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('blue')
        sns.stripplot(x="trials_from_switch",y="ahbeh_statistic",data=df_by_trials_from_switch, ax=axes[1],
                      color='green', linewidth=.5, edgecolor='green',alpha=.4, size=5, jitter=1)
        sns.stripplot(x="trials_from_switch",y="dv",data=dv_by_trials_from_switch, ax=axes[3],
                      color='blue', linewidth=.5, edgecolor='blue',alpha=.4, size=5, jitter=1)
        
        trials_from_switch = df_by_trials_from_switch.groupby('trials_from_switch').mean().ahbeh_statistic.index + np.max(df_by_trials_from_switch['trials_from_switch'])
        mean_only_ahbeh = df_by_trials_from_switch.groupby('trials_from_switch').mean().ahbeh_statistic.values
        median_only_ahbeh = df_by_trials_from_switch.groupby('trials_from_switch').median().ahbeh_statistic.values
        mean_only_dv = dv_by_trials_from_switch.groupby('trials_from_switch').mean().dv.values
        median_only_dv = dv_by_trials_from_switch.groupby('trials_from_switch').median().dv.values
        axes[4].scatter(trials_from_switch, mean_only_ahbeh, color='lightgreen', label='ahbeh mean')
        axes[4].scatter(trials_from_switch, median_only_ahbeh, color='green', label='ahbeh median')
        axes[4].plot(trials_from_switch, mean_only_ahbeh, color='lightgreen', alpha=.3)
        axes[4].plot(trials_from_switch, median_only_ahbeh, color='green', alpha=.3)
        axes[4].set_ylabel('ahbeh_statistic')
        axes[5].scatter(trials_from_switch, mean_only_dv, color='lightblue', label='dv mean')
        axes[5].scatter(trials_from_switch, median_only_dv, color='blue', label='dv median')
        axes[5].plot(trials_from_switch, mean_only_dv, color='lightblue', alpha=.3)
        axes[5].plot(trials_from_switch, median_only_dv, color='blue', alpha=.3)
        axes[5].set_ylabel('dv')
        axes[5].set_xlabel('trials_from_switch')
        
        # some fontsize formatting for readability
        for ax in [0,1,2,3,4,5]:
            axes[ax].xaxis.get_label().set_fontsize(fontsize)
            axes[ax].yaxis.get_label().set_fontsize(fontsize)
            axes[ax].tick_params(labelsize=fontsize-4)
            axes[ax].vlines(df_by_trials_from_switch["trials_from_switch"].values[-1],axes[ax].get_ylim()[0],axes[ax].get_ylim()[1],
                       color="pink", alpha=.5, zorder=0,label="changed stem")
        
        for ax in [0,1,2]:
            axes[ax].set(xlabel=None)
        axes[4].legend(loc = 'upper right', framealpha=0.2)
        axes[5].legend(loc = 'upper right', framealpha=0.2)

        # title
        plt.suptitle(f'ahbeh {summary_stat} and {decision_var}\n{nwb_file_name}, n_switches={n_switches}, n_epochs={n_epochs}\nrun: {run}, well: {well}, pre_sec: {pre_sec}, post_sec: {post_sec}', fontsize=fontsize)
        #if set_ylim is not None:
        #    plt.ylim(set_ylim)
        fig.tight_layout()
        sns.despine()

        if save_fig:
            save_path = f'/stelmo/alison/ahbeh_and_dv_around_stem_switch_figs/ahbeh_and_dv_by_trials_from_switch_{nwb_file_name}_{summary_stat}_{decision_var}_{trials_surrounding_switch}_run{run}_well{well}_presec{pre_sec}_postsec{post_sec}.pdf'
            plt.savefig(save_path, bbox_inches="tight")
            print(f'saved pdf at: {save_path}')
        #showing fig afterwards creates new fig so important to have this after the savefig line
        plt.show()

## Plots to look just at ahbeh or dv, rather than both at the same time

def plot_ahbeh_by_trials_from_switch(ahbeh_dict_by_trials_from_switch,
                                    nwb_file_name,
                                    summary_stats=['mean','median','min','max','std','var','all'],
                                    n_epochs=None,n_switches=None,
                                    run=True, well=False, pre_sec=-.2, post_sec=-.1,
                                    fontsize=16,figsize=(20,10), save_fig=False): #set_ylim=None):
    """ Plots violinplot and stripplot (a jittered grouped scatter of raw datapts) for ahbeh data from a day
    Also plots subplot with mean/median of ahbeh so can zoom in on just centers rather than full distribution

    Parameters
    ----------
    ahbeh_dict_by_trials_from_switch : dict
        from get_ahbeh_stats_by_trials_from_switch_full
    nwb_file_name : str
    summary_stats : list of strings
        any subset or all of ['mean','median','min','max','std','var','all']
        determines which ones to actually plot
    n_epochs : int
        from get_ahbeh_stats_by_trials_from_switch_full
    n_switches : int
        from get_ahbeh_stats_by_trials_from_switch_full
    run : bool
    well : bool
    pre_sec : float
    post_sec : float
    fontsize : int
    figsize : tuple (width,height)
    save_fig : bool
        path is hardcoded on stelmo rn, includes params, doesnt include HPD info or ah/beh/ahbeh/abs info!!

    Returns
    -------
    Just shows plots, doesn't return
    """

    trials_surrounding_switch=int((len(ahbeh_dict_by_trials_from_switch['ahbeh_all'])-1)/2)
    
    for summary_stat in summary_stats:
        # Turn dictionary into long df for sns plotting
        df_by_trials_from_switch = dict_to_df_by_trials_from_switch(ahbeh_dict_by_trials_from_switch, summary_stat)

        # Make figure with violinplot and stripplot
        fig, axes = plt.subplots(nrows=3, sharey=False, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [1, 1, 1]})
        sns.violinplot(x="trials_from_switch", y="ahbeh_statistic", data=df_by_trials_from_switch, ax=axes[0], 
                       inner="quartiles", scale="count", cut=0, color="lightgreen", linewidth=.5)
        # make median more obvious, leave the other quartile lines
        for l in axes[0].lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('green')
        sns.stripplot(x="trials_from_switch",y="ahbeh_statistic",data=df_by_trials_from_switch, ax=axes[1],
                      color='green', linewidth=.5, edgecolor='green',alpha=.4, size=5, jitter=1)
        
        trials_from_switch = df_by_trials_from_switch.groupby('trials_from_switch').mean().ahbeh_statistic.index + np.max(df_by_trials_from_switch['trials_from_switch'])
        mean_only_ahbeh = df_by_trials_from_switch.groupby('trials_from_switch').mean().ahbeh_statistic.values
        median_only_ahbeh = df_by_trials_from_switch.groupby('trials_from_switch').median().ahbeh_statistic.values
        
        axes[2].scatter(trials_from_switch, mean_only_ahbeh, color='lightgreen', label='ahbeh mean')
        axes[2].scatter(trials_from_switch, median_only_ahbeh, color='green', label='ahbeh median')
        axes[2].plot(trials_from_switch, mean_only_ahbeh, color='lightgreen', alpha=.3)
        axes[2].plot(trials_from_switch, median_only_ahbeh, color='green', alpha=.3)
        axes[2].set_ylabel('ahbeh_statistic')
        axes[2].set_xlabel('trials_from_switch')
        
        # some fontsize formatting for readability
        for ax in [0,1,2]:
            axes[ax].xaxis.get_label().set_fontsize(fontsize)
            axes[ax].yaxis.get_label().set_fontsize(fontsize)
            axes[ax].tick_params(labelsize=fontsize-4)
            axes[ax].vlines(df_by_trials_from_switch["trials_from_switch"].values[-1],axes[ax].get_ylim()[0],axes[ax].get_ylim()[1],
                       color="pink", alpha=.5, zorder=0,label="changed stem")
        
        for ax in [0,1]:
            axes[ax].set(xlabel=None)
        axes[2].legend(loc = 'upper right', framealpha=0.2)

        # title
        plt.suptitle(f'ahbeh {summary_stat}\n{nwb_file_name}, n_switches={n_switches}, n_epochs={n_epochs}\nrun: {run}, well: {well}, pre_sec: {pre_sec}, post_sec: {post_sec}', fontsize=fontsize)
        #if set_ylim is not None:
        #    plt.ylim(set_ylim)
        fig.tight_layout()
        sns.despine()
        
        if save_fig:
            save_path = f'/stelmo/alison/ahbeh_and_dv_around_stem_switch_figs/ahbeh_by_trials_from_switch_{nwb_file_name}_{summary_stat}_{trials_surrounding_switch}_run{run}_well{well}_presec{pre_sec}_postsec{post_sec}.pdf'
            plt.savefig(save_path, bbox_inches="tight")
            print(f'saved pdf at: {save_path}')
        plt.show()

def plot_dv_by_trials_from_switch(rl_results, decision_vars, trials_surrounding_switch,
                                    nwb_file_name,
                                    fontsize=16,figsize=(20,10), save_fig=False): #set_ylim=None):
    """ Plots violinplot and stripplot (a jittered grouped scatter of raw datapts) for any number of decision vars, again for decision_var
    Also plots subplots with mean/median of ahbeh and decision_var so can zoom in on just centers rather than full distribution

    Note: trials aren't paired up right now across the ahbeh data and dv data - a trial excluded by hpd for ex from ahbeh would still be included in dv

    Parameters
    ----------
    rl_results : pd DataFrame
        from BehaviorModelResults
    decision_vars : list of strings
        list of column names from rl_results, determines which dvs to plot
        easy ones that exist on all (not just switch) trials include:  ['state_entropy','reward_entropy','Q_leaf_t','Q_stem_t','stem_choice_p','stem_choice_var','stem_stay_p','stem_stay_var']
    trials_surrounding_switch: int
        need to enter this!! nto getting it from ahbeh data because just plotting for the dv
    nwb_file_name : str
    fontsize : int
    figsize : tuple (width,height)
    save_fig : bool
        path is hardcoded on stelmo rn, includes params, doesnt include HPD info or ah/beh/ahbeh/abs info!!

    Returns
    -------
    Just shows plots, doesn't return
    """
    # get extra rl results columns
    rl_results = add_rl_results_analysis_columns(rl_results)
    
    for decision_var in decision_vars:
        # get rl results for just dv of interest
        dv_by_trial = get_dv_around_switch_by_trial(rl_results, decision_var=decision_var, nwb_file_names=[nwb_file_name],  trials_surrounding_switch=trials_surrounding_switch)
        # turn that dv info into df
        dv_by_trials_from_switch = dv_to_df_by_trials_from_switch(dv_by_trial)

        # Make figure with violinplot and stripplot
        fig, axes = plt.subplots(nrows=3, sharey=False, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [1, 1, 1]})
        
        # Below the ahbeh plots, add the DV plots
        sns.violinplot(x="trials_from_switch", y="dv", data=dv_by_trials_from_switch, ax=axes[0], 
                       inner="quartiles", scale="count", cut=0, color="lightblue", linewidth=.5)
        # make median more obvious, leave the other quartile lines
        for l in axes[0].lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('blue')
        sns.stripplot(x="trials_from_switch",y="dv",data=dv_by_trials_from_switch, ax=axes[1],
                      color='blue', linewidth=.5, edgecolor='blue',alpha=.4, size=5, jitter=1)
        
        trials_from_switch = dv_by_trials_from_switch.groupby('trials_from_switch').mean().dv.index + np.max(dv_by_trials_from_switch['trials_from_switch'])
        mean_only_dv = dv_by_trials_from_switch.groupby('trials_from_switch').mean().dv.values
        median_only_dv = dv_by_trials_from_switch.groupby('trials_from_switch').median().dv.values
        axes[2].scatter(trials_from_switch, mean_only_dv, color='lightblue', label='dv mean')
        axes[2].scatter(trials_from_switch, median_only_dv, color='blue', label='dv median')
        axes[2].plot(trials_from_switch, mean_only_dv, color='lightblue', alpha=.3)
        axes[2].plot(trials_from_switch, median_only_dv, color='blue', alpha=.3)
        axes[2].set_ylabel('dv')
        axes[2].set_xlabel('trials_from_switch')
        
        # some fontsize formatting for readability
        for ax in [0,1,2]:
            axes[ax].xaxis.get_label().set_fontsize(fontsize)
            axes[ax].yaxis.get_label().set_fontsize(fontsize)
            axes[ax].tick_params(labelsize=fontsize-4)
            axes[ax].vlines(dv_by_trials_from_switch["trials_from_switch"].values[-1],axes[ax].get_ylim()[0],axes[ax].get_ylim()[1],
                       color="pink", alpha=.5, zorder=0,label="changed stem")
        
        for ax in [0,1]:
            axes[ax].set(xlabel=None)
        axes[2].legend(loc = 'upper right', framealpha=0.2)

        # title
        plt.suptitle(f'{decision_var}\n{nwb_file_name} n_switches={len(dv_by_trial[0])}', fontsize=fontsize)
        #if set_ylim is not None:
        #    plt.ylim(set_ylim)
        fig.tight_layout()
        sns.despine()

        if save_fig:
            save_path = f'/stelmo/alison/ahbeh_and_dv_around_stem_switch_figs/dv_by_trials_from_switch_{nwb_file_name}_{decision_var}_{trials_surrounding_switch}.pdf'
            plt.savefig(save_path, bbox_inches="tight")
            print(f'saved pdf at: {save_path}')
        plt.show()