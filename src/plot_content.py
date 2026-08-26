from base64 import decode
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from spyglass.common import IntervalLinearizedPosition
from alison_rlmodel import BehaviorModelResults
from alison_decoding import ClusterlessAcausalResultsSummary, add_trial_info_to_clusterless_results_withpokes, ClusterlessResults
from alison_position import PosValidTimesToEpoch
from plot_rlmodel import add_rl_results_analysis_columns
from rl_helpers import create_hmm_plus, rescale_hmm_plus, add_rpe_deltaq_to_hmm_plus, make_full_beta_bernoulli_dv_df


# hardcoded
segment_id_to_renumbered_id = {
    0: 1,
    1: 2,
    2: 4,
    3: 5,
    4: 7,
    5: 8,
    6: 0,
    7: 3,
    8: 6,
}
segment_id_to_label = {
    0: 'A.1',
    1: 'A.2',
    2: 'B.1',
    3: 'B.2',
    4: 'C.1',
    5: 'C.2',
    6: 'A',
    7: 'B',
    8: 'C',
}
stem_segs_mapped = [0,3,6]
leaf_segs_mapped = [1,2,4,5,7,8]
# Translate to labels for plots
segment_renumbered_and_relabeled = [i[1] for i in segment_id_to_label.items()]
segment_renumbered_and_relabeled.sort()

def get_clusterless_trial_info_clean_and_filter(nwb_file_names,
                                                remove_hpd_timepoints = True,
                                                hpd_percent = 50,
                                                hpd_threshold = 50,
                                                require_nonlocal_by_segment = False,
                                                remove_low_speed_timepoints = True,
                                                head_speed_threshold = 10):
    '''Create large dataframe across days and add some new columns and apply some filters to only include subset of the data.
    Initially meant to be used to assess run time decodes by segment.
    Removed 0th trials and any others where first and last seg identities are for some reason equal.
    Maps segment ids to a more intuitive ordering.

    Parameters
    ----------
    nwb_file_names: list
        nwb file name strings, presumed to be one rat at a time
    remove_hpd_timepoints : bool
        whether or not to filter by hpd
    hpd_percent : int
        50 or 95 % hpd value to threshold
    hpd_threshold : int
        cm over which hpd percent is distributed. Anything < thresh is included.
    require_nonlocal_by_segment : bool
        only include data where mental and actual segments disagree
    remove_low_speed_timepoints : bool
        whether to remove stationary times or not
    head_speed_thrsehold : int
        minimum head speed threshold to include in analyses
    
    Returns
    -------
    decode_and_trial_info_manydays : pd.DataFrame
        large df concatenated over days of data and filtered
    '''
    # Get decode info concatenated across all trials of the whole list of eps/nwbs
    for n, nwb_file_name in enumerate(nwb_file_names):
        # Iterate through intervals
        interval_list_names = np.unique((ClusterlessAcausalResultsSummary() & {'nwb_file_name':nwb_file_name}).fetch('interval_list_name'))
        interval_list_names = sorted(interval_list_names, key=lambda x: int(x.split()[1]))
        for i,interval_list_name in enumerate(interval_list_names):
            print(f'starting with interval list name: {interval_list_name}')
            decode_and_trial_info = add_trial_info_to_clusterless_results_withpokes(nwb_file_name,interval_list_name)
            if i == 0:
                decode_and_trial_info_day = decode_and_trial_info
            else:
                decode_and_trial_info_day = pd.concat([decode_and_trial_info_day, decode_and_trial_info])
            print(len(decode_and_trial_info))
        if n == 0:
            decode_and_trial_info_manydays = decode_and_trial_info_day
        else:
            decode_and_trial_info_manydays = pd.concat([decode_and_trial_info_manydays, decode_and_trial_info_day])

    # Now have all the data, so can do some adding of extra columns, etc BEFORE FILTERING

    # create segment count within trial
    seg_count = decode_and_trial_info_manydays.groupby(['nwb_file_name','epoch_number','trial_number_by_epoch']).apply(lambda x_df: (x_df['actual_segment'] != x_df['actual_segment'].shift()).cumsum()-1)
    # add seg count within trial as column
    decode_and_trial_info_manydays['segment_count_within_trial'] = seg_count.values

    # add first segment bool column
    decode_and_trial_info_manydays['is_first_seg_of_trial'] = (decode_and_trial_info_manydays['segment_count_within_trial'] == 0)

    # create bool for final segment per trial
    final_seg_bool = decode_and_trial_info_manydays.groupby(['nwb_file_name','epoch_number','trial_number_by_epoch']).apply(lambda x_df: x_df['segment_count_within_trial'] == x_df['segment_count_within_trial'].max())
    # add a column for last seg bool (occasionaly have segment changes - 0 so only 1 seg in a trial due to parsing
    decode_and_trial_info_manydays['is_last_seg_of_trial'] = final_seg_bool.values

    # make a new df that has the mapping of seg ids to make it plot-able
    decode_and_trial_info_manydays_mapped = decode_and_trial_info_manydays[['actual_segment','mental_segment']].applymap(segment_id_to_renumbered_id.get)
    # add these straight to the main df too
    decode_and_trial_info_manydays['actual_segment_mapped'] = decode_and_trial_info_manydays_mapped['actual_segment']
    decode_and_trial_info_manydays['mental_segment_mapped'] = decode_and_trial_info_manydays_mapped['mental_segment']
    
    decode_and_trial_info_manydays = decode_and_trial_info_manydays[decode_and_trial_info_manydays['trial_number_by_epoch'] != 0]

    no_seg_change_trial_bool = decode_and_trial_info_manydays.groupby(['nwb_file_name','epoch_number','trial_number_by_epoch']).apply(
        lambda x_df: (x_df['segment_count_within_trial'] == x_df['segment_count_within_trial'].min()) & (x_df['segment_count_within_trial'] == x_df['segment_count_within_trial'].max()))
    decode_and_trial_info_manydays['zero_seg_change_trial'] = no_seg_change_trial_bool.values

    #newer filtering stuff
    # Filter out rows for trials without seg changes
    decode_and_trial_info_manydays = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['zero_seg_change_trial']==False)]

    #test that this data cleaning worked 
    only_one_seg = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_first_seg_of_trial']==True) & (decode_and_trial_info_manydays['is_last_seg_of_trial']==True)]
    assert len(np.unique(only_one_seg['trial_number_by_epoch']))==0, f'at least one trial has zero-seg-changes or some other first/last seg parsing issue where first and last overalp'

    # now how much is the first or last seg a stem rather than a leaf
    first_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_first_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]
    last_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_last_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]
    
    # handle chimi 0221 and 0306 triasl 4 and 6 epochs 8 and 6 
    if len(np.unique(first_is_stem['trial_number_by_epoch'])) != 0:
        print('WARNING: At least one trial has first seg as a stem!')
        trials_to_remove = decode_and_trial_info_manydays[(( decode_and_trial_info_manydays['is_first_seg_of_trial']==True ) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) ) )][['nwb_file_name','epoch_number','trial_number_by_epoch']].value_counts().index.values
        print(f'WARNING: Removing trials with first seg as a stem, which are {trials_to_remove}.')
        for t in trials_to_remove:
            decode_and_trial_info_manydays = decode_and_trial_info_manydays[~((decode_and_trial_info_manydays['nwb_file_name']==t[0]) & (decode_and_trial_info_manydays['epoch_number']==t[1]) & (decode_and_trial_info_manydays['trial_number_by_epoch']==t[2]))]
        print(f'Removed n={len(trials_to_remove)} trials.')
        #recalculate this number for new assertion test
        first_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_first_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]

    assert len(np.unique(first_is_stem['trial_number_by_epoch']))==0, f'at least one trial still has first seg as a stem.'

    if len(np.unique(last_is_stem['trial_number_by_epoch'])) != 0:
        print('WARNING: At least one trial has last seg as a stem!')
        trials_to_remove = decode_and_trial_info_manydays[(( decode_and_trial_info_manydays['is_last_seg_of_trial']==True ) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) ) )][['nwb_file_name','epoch_number','trial_number_by_epoch']].value_counts().index.values
        print(f'WARNING: Removing trials with last seg as a stem, which are {trials_to_remove}.')
        for t in trials_to_remove:
            decode_and_trial_info_manydays = decode_and_trial_info_manydays[~((decode_and_trial_info_manydays['nwb_file_name']==t[0]) & (decode_and_trial_info_manydays['epoch_number']==t[1]) & (decode_and_trial_info_manydays['trial_number_by_epoch']==t[2]))]
        print(f'Removed n={len(trials_to_remove)} trials.')
        #recalculate this number for new assertion test
        last_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_last_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]
    
    assert len(np.unique(last_is_stem['trial_number_by_epoch']))==0, f'at least one trial still has last seg as a stem.'

    # Now can filter       
    if require_nonlocal_by_segment:
        decode_and_trial_info_manydays = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['nonlocal_by_segment']==True)]
    if remove_hpd_timepoints:
        decode_and_trial_info_manydays = decode_and_trial_info_manydays[(decode_and_trial_info_manydays[f'spatial_coverage_{hpd_percent}_hpd']<hpd_threshold)]
    if remove_low_speed_timepoints:
        decode_and_trial_info_manydays = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['head_speed']>head_speed_threshold)]

    return decode_and_trial_info_manydays

def plot_hist2d_actual_mental_seg_by_group(decode_and_trial_info_base, nwb_file_names,
                                           is_actual_first_seg, is_actual_last_seg,
                                           hpd_percent,
                                           hpd_threshold,
                                           remove_hpd_timepoints,
                                           remove_low_speed_timepoints,
                                           head_speed_threshold,
                                           is_nosepoking=False,
                                           filter_by_content=True, nonlocal_by_seg=False,
                                           ahbeh_abs_max=None, ahbeh_abs_min=None, groupby_names=['stem_switch'],
                                           normalization_method='filtered_column_sums_remove_local_diag', cmap_max=.2):
    ''' Based on dataframe of decode and trial info, plot 2d histogram and x2 normalized versions
    Visualize actual vs mental segment decoding with a variety of normlization approaches
    Separates by stay and switch trials, can also filter by first/final segment

    Parameters
    ----------
    decode_and_trial_info_base  : pd.DataFrame
        output from get_custerless_trial_info_clean_and_filter() fxn
    nwb_file_names: list
        nwb file name strings, presumed to be one rat at a time
        just including some of these params to print in plot title
    is_actual_first_seg: bool
        only include the first segment in plots?
    is_actual_last_seg : bool
        only include the last segment in plots?
    is_nosepoking : bool
        exclude nosepoked times from plots
    filter_by_content : bool
        remove data based on first/last seg, ahbeh distance, and nonlocal_by_segment/
    nonlocal_by_seg : bool
        only plot when actual seg != mental seg
    ahbeh_abs_max : int
        only look at data < this many cm from animal
    ahbeh_abs_min : into
        only look at data > this many cm from animal
    groupby_names : list of strings that are columsn in decode_and_trial_info_base
        how to group data when iterating through plots. Expects to just be stems rn.
    normalization_method : str
        options: filtered_column_sums, occupancy_column_sums, filtered_column_sums_remove_local_diag
    cmap_max : float (0,1]
        for colorbar max on normalized plots    
    remove_hpd_timepoints : bool
        whether or not to filter by hpd
    hpd_percent : int
        50 or 95 % hpd value to threshold
    hpd_threshold : int
        cm over which hpd percent is distributed. Anything < thresh is included.
    remove_low_speed_timepoints : bool
        whether to remove stationary times or not
    head_speed_thrsehold : int
        minimum head speed threshold to include in analyses
    
    Returns
    -------
    Just shows figs, does not have saving option right now!
    '''
    
    # set up the main data to use for the histograms
    # this is just filtered for dealing with occupancy not content
    decode_and_trial_info_day_occupancy_filtered = decode_and_trial_info_base[ (decode_and_trial_info_base['is_nosepoking'] == is_nosepoking) &
                                                                                 (decode_and_trial_info_base['is_last_seg_of_trial'] == is_actual_last_seg) &
                                                                                 (decode_and_trial_info_base['is_first_seg_of_trial'] == is_actual_first_seg) ] 
    # group the data by stay/switch
    grouped = decode_and_trial_info_day_occupancy_filtered.groupby(groupby_names)

    # iterate through groups
    for name, group_df in grouped:
        # further filter by content, on top of the occupancy filters
        if filter_by_content==True:
            group_df_filtered = group_df.copy(deep=True)
            if nonlocal_by_seg==True:
                group_df_filtered = group_df_filtered[(group_df_filtered['nonlocal_by_segment']==nonlocal_by_seg)]
            if (ahbeh_abs_min is not None):
                group_df_filtered = group_df_filtered[(np.abs(group_df_filtered['nonlocal_by_segment']) > ahbeh_abs_min)]
            if (ahbeh_abs_max is not None):
                group_df_filtered = group_df_filtered[(np.abs(group_df_filtered['nonlocal_by_segment']) < ahbeh_abs_max)]
        else:
            group_df_filtered = group_df
            
        # just for the naming, organize some variables
        if is_actual_first_seg==True:
            seg = 'first'
        if is_actual_last_seg==True:
            seg = 'last'

        # plot without normalization at all - just counts
        fig1, ax1 = plt.subplots(ncols=1)
        hist1, xedges1, yedges1, mesh1 = ax1.hist2d(group_df_filtered['actual_segment_mapped'], group_df_filtered['mental_segment_mapped'], bins=[np.arange(0,10,1),np.arange(0,10,1)])
        ax1.set_title('counts')
        ax1.set_xlabel('actual segment')
        ax1.set_ylabel('mental segment')
        ax1.set_xticks(np.arange(0.5,9.5,1))
        ax1.set_yticks(np.arange(0.5,9.5,1))
        ax1.set_xticklabels(segment_renumbered_and_relabeled, rotation = 30)
        ax1.set_yticklabels(segment_renumbered_and_relabeled)
        fig1.colorbar(mesh1, label='count of 2 ms bins')
        fig1.suptitle(f'rat={nwb_file_names[0][:-4]},stem_switch={name}, segment={seg}, ndays={len(nwb_file_names)}\n'
                      f'is_nosepoking={is_nosepoking}, nonlocal_by_seg={nonlocal_by_seg}\n'
                      f'abs_ahbeh_min={ahbeh_abs_min}, abs_ahebeh_max={ahbeh_abs_max}\n'
                      f'remove_hpd_times={remove_hpd_timepoints}, hpd_percent={hpd_percent}, hpd_cm={hpd_threshold}\n'
                      f'remove_low_speed_times={remove_low_speed_timepoints}, speed_min={head_speed_threshold}',
                      x=.5,y=1.15)
        fig1.show()

        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='black') #current_cmap.colors[0])
        #current_cmap.set_extremes

        # print(hist1.astype(int))
        # plot with normalization
        fig2, ax2 = plt.subplots(ncols=1)

        if normalization_method=='occupancy_column_sums':
            hist_occupancy, xedges_occupancy, yedges_occupancy = np.histogram2d(group_df['actual_segment_mapped'], group_df['mental_segment_mapped'], bins=[np.arange(0,10,1),np.arange(0,10,1)])
            #print(hist_occupancy.astype(int))
            column_totals = hist_occupancy.sum(axis=1, keepdims=True) # sum over columns. Now divide all columns
            #print(column_totals)
            hist_prop_of_actual_time = hist1/column_totals
            # print(hist_prop_of_actual_time)
            # print(hist_prop_of_actual_time.T)
            mesh2 = ax2.pcolormesh(xedges_occupancy, yedges_occupancy, hist_prop_of_actual_time.T)
        elif normalization_method=='filtered_column_sums':
            column_totals = hist1.sum(axis=1, keepdims=True)
            hist_prop_of_actual_time = hist1/column_totals
            mesh2 = ax2.pcolormesh(xedges1,yedges1, hist_prop_of_actual_time.T)
        elif normalization_method=='filtered_column_sums_remove_local_diag':
            assert nonlocal_by_seg==False, 'nonlocal_by_seg must be False for this method to make sense.'
            column_totals = hist1.sum(axis=1, keepdims=True)
            hist_prop_of_actual_time = hist1/column_totals
            
            fig3, ax3 = plt.subplots(ncols=1)
            mesh3 = ax3.pcolormesh(xedges1, yedges1, hist_prop_of_actual_time.T)
            ax3.set_xlabel('actual segment')
            ax3.set_ylabel('mental segment')
            ax3.set_title(f'normalized\n'
                          f'normalization=filtered_column_sums')
            ax3.set_xticks(np.arange(0.5,9.5,1))
            ax3.set_yticks(np.arange(0.5,9.5,1))
            ax3.set_xticklabels(segment_renumbered_and_relabeled, rotation = 30)
            ax3.set_yticklabels(segment_renumbered_and_relabeled)
            fig3.colorbar(mesh3,label='proportion of actual seg time')
            fig3.show()
            
            #print(hist_prop_of_actual_time.round(decimals=2), 'original\n')
            diag_inds = np.diag_indices(9)
            hist_prop_of_actual_time[diag_inds] = np.nan
            #print(hist_prop_of_actual_time.round(decimals=2), 'changed diag\n')       
            max_after_remove_diag = np.nanmax(hist_prop_of_actual_time)
            #print(f'max val after removing diag is: {max_after_remove_diag}')
            if max_after_remove_diag>cmap_max:
                print(f'WARNING: max value after removing diag is greater than cmap_max: increase cmap_max arg! Instead try: {max_after_remove_diag}')
            mesh2 = ax2.pcolormesh(xedges1, yedges1, hist_prop_of_actual_time.T, vmax=cmap_max)
            

        #mesh2 = ax2.pcolormesh(xedges, yedges, hist_prop_of_actual_time.T)
        ax2.set_xlabel('actual segment')
        ax2.set_ylabel('mental segment')
        ax2.set_title(f'normalized\n'
                      f'normalization={normalization_method}')
        ax2.set_xticks(np.arange(0.5,9.5,1))
        ax2.set_yticks(np.arange(0.5,9.5,1))
        ax2.set_xticklabels(segment_renumbered_and_relabeled, rotation = 30)
        ax2.set_yticklabels(segment_renumbered_and_relabeled)
        fig2.colorbar(mesh2,label='proportion of actual seg time')

        fig2.show()

def plot_ahbeh_by_trial_over_epoch_contingencies(decode_and_trial_info_base,filter_by_stem_switch=False,apply_stat='mean',is_nosepoking=False, show_scatter=True):
    '''plot ahbeh avgd on each trial across many epochs overlaid and look at contingency change trials'''

    data_filtered = decode_and_trial_info_base[ (decode_and_trial_info_base['is_nosepoking'] == is_nosepoking) ]
    #  add filtering param to limit to first or last seg of trials
    #&
                                    #         (decode_and_trial_info_base['is_last_seg_of_trial'] == is_actual_last_seg) &
                                    #           (decode_and_trial_info_base['is_first_seg_of_trial'] == is_actual_first_seg)]
    if filter_by_stem_switch == True:
        data_filtered = data_filtered[data_filtered['stem_switch'] == filter_by_stem_switch]

    data_grouped_by_day_ep_trial = data_filtered.groupby(['nwb_file_name', 'epoch_number','trial_number_by_epoch'])

    nwbs = np.unique(decode_and_trial_info_base.nwb_file_name)

    if apply_stat == 'mean':
        day_ep_trial_stat = data_grouped_by_day_ep_trial.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).mean()).reset_index()
    elif apply_stat == 'median':
        day_ep_trial_stat = data_grouped_by_day_ep_trial.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).median()).reset_index()
    elif apply_stat == 'std':
        day_ep_trial_stat = data_grouped_by_day_ep_trial.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).std()).reset_index()
    avg_trace = day_ep_trial_stat.groupby('trial_number_by_epoch')[0].mean()
    sem_trace = day_ep_trial_stat.groupby('trial_number_by_epoch')[0].sem()
    avg_trace_plus_sem = avg_trace+sem_trace
    avg_trace_minus_sem = avg_trace-sem_trace

    day_ep_trial_stat.plot(x='trial_number_by_epoch', y=0, kind='scatter', figsize=(20,5), alpha=.5)
    avg_trace.plot(color='red')
    avg_trace_plus_sem.plot(color='pink', alpha=.8)
    avg_trace_minus_sem.plot(color='pink', alpha=.8)

    for i in [0,59,119]:
        plt.axvline(i,c='grey', zorder=0, linestyle='--')
    #plt.ylim([0,20])
    plt.xlim(0,180)
    plt.ylabel(f'{apply_stat} ahbeh per trial')
    plt.title(f'trial {apply_stat}, overlay mean+/-sem\n{nwbs}')
    #last_seg={is_actual_last_seg}, first_seg={is_actual_first_seg},\nfilter_by_switch={filter_by_stem_switch}, stem_switch={stem_switch}')
    sns.despine()

    if show_scatter ==True:
        day_ep_trial_stat.plot(x='trial_number_by_epoch', y=0, kind='scatter', figsize=(20,5), alpha=.5)
    else:
        day_ep_trial_stat.plot(x='trial_number_by_epoch', y=0, kind='scatter', figsize=(20,5), alpha=0)
    avg_trace.plot(color='red')
    avg_trace_plus_sem.plot(color='pink', alpha=.8)
    avg_trace_minus_sem.plot(color='pink', alpha=.8)

    for i in [0,59,119]:
        plt.axvline(i,c='grey', zorder=0, linestyle='--')
    plt.ylim([0,25])
    plt.xlim(0,180)
    plt.ylabel(f'{apply_stat} ahbeh per trial')
    sns.despine()
    plt.show()


def plot_prop_nonlocal_by_trial_over_epoch_contingencies(decode_and_trial_info_base,filter_by_stem_switch=False,apply_stat='mean',is_nosepoking=False, show_scatter=True):
    '''plot ahbeh avgd on each trial across many epochs overlaid and look at contingency change trials'''

    data_filtered = decode_and_trial_info_base[ (decode_and_trial_info_base['is_nosepoking'] == is_nosepoking) ]
    #  add filtering param to limit to first or last seg of trials
    #&
                                    #         (decode_and_trial_info_base['is_last_seg_of_trial'] == is_actual_last_seg) &
                                    #           (decode_and_trial_info_base['is_first_seg_of_trial'] == is_actual_first_seg)]
    if filter_by_stem_switch == True:
        data_filtered = data_filtered[data_filtered['stem_switch'] == filter_by_stem_switch]

    data_grouped_by_day_ep_trial = data_filtered.groupby(['nwb_file_name', 'epoch_number','trial_number_by_epoch'])

    nwbs = np.unique(decode_and_trial_info_base.nwb_file_name)

    if apply_stat == 'mean':
        day_ep_trial_stat = data_grouped_by_day_ep_trial.apply(lambda x_df: len(x_df[x_df['nonlocal_by_segment']])/len(x_df) ).reset_index()
    elif apply_stat == 'median':
        day_ep_trial_stat = data_grouped_by_day_ep_trial.apply(lambda x_df: len(x_df[x_df['nonlocal_by_segment']])/len(x_df) ).reset_index(name='prop_nonlocal_by_segment')
    elif apply_stat == 'std':
        day_ep_trial_stat = data_grouped_by_day_ep_trial.apply(lambda x_df: len(x_df[x_df['nonlocal_by_segment']])/len(x_df) ).reset_index(name='prop_nonlocal_by_segment')
    avg_trace = day_ep_trial_stat.groupby('trial_number_by_epoch')[0].mean()
    sem_trace = day_ep_trial_stat.groupby('trial_number_by_epoch')[0].sem()
    avg_trace_plus_sem = avg_trace+sem_trace
    avg_trace_minus_sem = avg_trace-sem_trace

    day_ep_trial_stat.plot(x='trial_number_by_epoch', y=0, kind='scatter', figsize=(20,5), alpha=.5)
    avg_trace.plot(color='red')
    avg_trace_plus_sem.plot(color='pink', alpha=.8)
    avg_trace_minus_sem.plot(color='pink', alpha=.8)

    for i in [0,59,119]:
        plt.axvline(i,c='grey', zorder=0, linestyle='--')
    #plt.ylim([0,20])
    plt.xlim(0,180)
    plt.ylabel(f'{apply_stat} prop nonlocal per trial')
    plt.title(f'trial {apply_stat}, overlay mean+/-sem,\n{nwbs}')
    #last_seg={is_actual_last_seg}, first_seg={is_actual_first_seg},\nfilter_by_switch={filter_by_stem_switch}, stem_switch={stem_switch}')
    sns.despine()

    if show_scatter == True:
        day_ep_trial_stat.plot(x='trial_number_by_epoch', y=0, kind='scatter', figsize=(20,5), alpha=.5)
    else:
        day_ep_trial_stat.plot(x='trial_number_by_epoch', y=0, kind='scatter', figsize=(20,5), alpha=.0)
    avg_trace.plot(color='red')
    avg_trace_plus_sem.plot(color='pink', alpha=.8)
    avg_trace_minus_sem.plot(color='pink', alpha=.8)

    for i in [0,59,119]:
        plt.axvline(i,c='grey', zorder=0, linestyle='--')
    plt.ylim([0,.25])
    plt.xlim(0,180)
    plt.ylabel(f'{apply_stat} prop nonlocal per trial')
    sns.despine()
    plt.show()



##########

def add_clusterless_trialinfo_rlresults(nwb_file_name, interval_list_name, behavior_model_params_name='default_hmm'):
    '''
    DEPRECATED now use "plus" version of this fxn instead, see below
    merge rl_results for one epoch with clusterless decode results for one epoch
    with trial info for one epoch, including poking info. This can be used for
    concatenating all sorts of analysis info across days, epochs, etc for an animal.    
    '''
    # Get rl results initial df for the day of data, it is stored one day at a time
    rl_results = (BehaviorModelResults.ByDay & {'nwb_file_name':nwb_file_name, 'behavior_model_params_name':behavior_model_params_name}).fetch1_dataframe()
    # Map pos interval to epoch number
    epoch = (PosValidTimesToEpoch & {'nwb_file_name':nwb_file_name, 'pos_interval_list_name':interval_list_name}).fetch1('epoch')
    # Trim to the one epoch
    rl_results = rl_results[rl_results['epoch']==epoch]
    # Add a couple columns made for convenience previously
    rl_results = add_rl_results_analysis_columns(rl_results)
    
    # Get clusterless and trial info merged
    clusterless_and_trial_info = add_trial_info_to_clusterless_results_withpokes(nwb_file_name, interval_list_name)
    clusterless_trialinfo_rlresults = clusterless_and_trial_info.reset_index().merge(rl_results, on=['nwb_file_name','trial_number_by_epoch'],
                            how='left', suffixes = (None,'_rl')).drop(columns=['leaf_rl','stem_rl','reward_rl']).set_index('time')
    assert len(clusterless_trialinfo_rlresults) == len(clusterless_and_trial_info), 'Columns were added or removed in the merging process?!'
    
    return clusterless_trialinfo_rlresults


def add_clusterless_trialinfo_rlresults_plus(nwb_file_name, interval_list_name, behavior_model_params_name):
    '''merge rl_results for one epoch with clusterless decode results for one epoch
    with trial info for one epoch, including poking info. This can be used for
    concatenating all sorts of analysis info across days, epochs, etc for an animal.    
    20240211 adding functionality for adding extra hmm info for delta q etc
    '''
    # Get rl results initial df for the day of data, it is stored one day at a time
    rl_results = (BehaviorModelResults.ByDay & {'nwb_file_name':nwb_file_name, 'behavior_model_params_name':behavior_model_params_name}).fetch1_dataframe()
    # Map pos interval to epoch number
    epoch = (PosValidTimesToEpoch & {'nwb_file_name':nwb_file_name, 'pos_interval_list_name':interval_list_name}).fetch1('epoch')
    # Trim to the one epoch
    rl_results = rl_results[rl_results['epoch']==epoch]
    # Add a couple columns made for convenience previously
    #rl_results = add_rl_results_analysis_columns(rl_results)
    if behavior_model_params_name=='default_hmm_0623':
        rl_results = create_hmm_plus(rl_results)
        rl_results = rescale_hmm_plus(rl_results)
        rl_results = add_rpe_deltaq_to_hmm_plus(rl_results)
    elif behavior_model_params_name=="beta_stable_withleaf":
        print(f"\n\nUSING {behavior_model_params_name} FOR RL VARIABLES TO ALIGN TO TRIALS AND DECODING\n\n")
        rl_results = make_full_beta_bernoulli_dv_df(rl_results)

    # Get clusterless and trial info merged
    clusterless_and_trial_info = add_trial_info_to_clusterless_results_withpokes(nwb_file_name, interval_list_name)
    clusterless_trialinfo_rlresults = clusterless_and_trial_info.reset_index().merge(rl_results, on=['nwb_file_name','trial_number_by_epoch'],
                            how='left', suffixes = (None,'_rl')).drop(columns=['leaf_rl','stem_rl','reward_rl'], errors='ignore').set_index('time')
    assert len(clusterless_trialinfo_rlresults) == len(clusterless_and_trial_info), 'Rows of time were added or removed in the merging process?!'
    
    return clusterless_trialinfo_rlresults


def get_linpos_aligned_to_clusterless_results(nwb_file_name, interval_list_name, position_info_param_name='default_decoding', classifier_param_name="default_decoding_gpu"):
    key = {'nwb_file_name':nwb_file_name,'interval_list_name':interval_list_name,'position_info_param_name':position_info_param_name, 'classifier_param_name':classifier_param_name}
    
    # get valid time sliced
    try:
        valid_time_slice = (ClusterlessResults & key).fetch1('valid_time_slice')
    except:
        print('DATAJOINTERROR: fetch1 failed, trying more specific key with default_clusterless as sorter_params_name.')
        valid_time_slice = (ClusterlessResults & key & {'sorter_params_name':'default_clusterless'}).fetch1('valid_time_slice')
    valid_time_sliced =  slice(valid_time_slice[0], valid_time_slice[1])
    
    # get linear position
    linear_position_df = (IntervalLinearizedPosition & key).fetch1_dataframe()
    linear_position_df = linear_position_df.loc[valid_time_sliced]
    
    # get results 
    clusterless_and_trial_info = add_trial_info_to_clusterless_results_withpokes(nwb_file_name, interval_list_name)
    
    # trim linpos if needed
    linear_position_df = linear_position_df[ (linear_position_df.index>=clusterless_and_trial_info.index[0]) & (linear_position_df.index<=clusterless_and_trial_info.index[-1])]
    
    # check  lengths 
    assert len(clusterless_and_trial_info) == len(linear_position_df), 'Clusterless results and trial info length does NOT match linear position length!'
    
    return linear_position_df

def add_clusterless_trialinfo_linpos_rlresults(nwb_file_name, interval_list_name, behavior_model_params_name, position_info_param_name='default_decoding'):
    '''merge rl_results for one epoch with clusterless decode results for one epoch
    with trial info for one epoch, including poking info. This can be used for
    concatenating all sorts of analysis info across days, epochs, etc for an animal. 
    
    This fxn only differs from adding clusterless trialinfo rlresults (without linpos) byt adding the one linpos column. that's all.    
    '''
    # clusterless_trialinfo_rlresults = add_clusterless_trialinfo_rlresults(nwb_file_name, interval_list_name, behavior_model_params_name)
    clusterless_trialinfo_rlresults = add_clusterless_trialinfo_rlresults_plus(nwb_file_name, interval_list_name, behavior_model_params_name)
    linear_position_df = get_linpos_aligned_to_clusterless_results(nwb_file_name, interval_list_name, position_info_param_name)
    assert len(clusterless_trialinfo_rlresults) == len(linear_position_df), "Clusterless results adn trial info length do NOT match aligned lin pos length!!"
    clusterless_trialinfo_rlresults['linear_position'] = linear_position_df['linear_position'].values
    
    return clusterless_trialinfo_rlresults

def get_clusterless_trial_rl_info_clean_and_filter(nwb_file_names, behavior_model_params_name,
                                                   position_info_param_name='default_decoding',
                                                remove_hpd_timepoints = True,
                                                hpd_percent = 50,
                                                hpd_threshold = 50,
                                                require_nonlocal_by_segment = False,
                                                remove_low_speed_timepoints = True,
                                                head_speed_threshold = 10, classifier_param_name="default_decoding_gpu"):
    '''Create large dataframe across days and add some new columns and apply some filters to only include subset of the data.
    Initially meant to be used to assess run time decodes by segment.
    Removed 0th trials and any others where first and last seg identities are for some reason equal.
    Maps segment ids to a more intuitive ordering.

    Same as get_clusterless_trial_info_clean_and_filter, but now also have rl and linpos info in it!

    Parameters
    ----------
    nwb_file_names: list
        nwb file name strings, presumed to be one rat at a time
    remove_hpd_timepoints : bool
        whether or not to filter by hpd
    hpd_percent : int
        50 or 95 % hpd value to threshold
    hpd_threshold : int
        cm over which hpd percent is distributed. Anything < thresh is included.
    require_nonlocal_by_segment : bool
        only include data where mental and actual segments disagree
    remove_low_speed_timepoints : bool
        whether to remove stationary times or not
    head_speed_thrsehold : int
        minimum head speed threshold to include in analyses
    behavior_model_params_name : str
        which rl model to use
    position_info_param_name : str
        which linpos entry to use
    
    Returns
    -------
    decode_and_trial_info_manydays : pd.DataFrame
        large df concatenated over days of data and filtered
    '''
    # Get decode info concatenated across all trials of the whole list of eps/nwbs!
    for n, nwb_file_name in enumerate(nwb_file_names):
        # Iterate through intervals
        interval_list_names = np.unique((ClusterlessAcausalResultsSummary() & {'nwb_file_name':nwb_file_name}).fetch('interval_list_name'))
        interval_list_names = sorted(interval_list_names, key=lambda x: int(x.split()[1]))
        for i,interval_list_name in enumerate(interval_list_names):
            print(f'starting with interval list name: {interval_list_name}')
            decode_and_trial_info = add_clusterless_trialinfo_linpos_rlresults(nwb_file_name,interval_list_name, behavior_model_params_name, position_info_param_name)
            if i == 0:
                decode_and_trial_info_day = decode_and_trial_info
            else:
                decode_and_trial_info_day = pd.concat([decode_and_trial_info_day, decode_and_trial_info])
            print(len(decode_and_trial_info))
        if n == 0:
            decode_and_trial_info_manydays = decode_and_trial_info_day
        else:
            decode_and_trial_info_manydays = pd.concat([decode_and_trial_info_manydays, decode_and_trial_info_day])

    # Now have all the data, so can do some adding of extra columns, etc BEFORE FILTERING

    # create segment count within trial
    seg_count = decode_and_trial_info_manydays.groupby(['nwb_file_name','epoch_number','trial_number_by_epoch']).apply(lambda x_df: (x_df['actual_segment'] != x_df['actual_segment'].shift()).cumsum()-1)
    # add seg count within trial as column
    decode_and_trial_info_manydays['segment_count_within_trial'] = seg_count.values

    # add first segment bool column
    decode_and_trial_info_manydays['is_first_seg_of_trial'] = (decode_and_trial_info_manydays['segment_count_within_trial'] == 0)

    # create bool for final segment per trial
    final_seg_bool = decode_and_trial_info_manydays.groupby(['nwb_file_name','epoch_number','trial_number_by_epoch']).apply(lambda x_df: x_df['segment_count_within_trial'] == x_df['segment_count_within_trial'].max())
    decode_and_trial_info_manydays['is_last_seg_of_trial'] = final_seg_bool.values

    # make a new df that has the mapping of seg ids to make it plot-able
    decode_and_trial_info_manydays_mapped = decode_and_trial_info_manydays[['actual_segment','mental_segment']].applymap(segment_id_to_renumbered_id.get)
    decode_and_trial_info_manydays['actual_segment_mapped'] = decode_and_trial_info_manydays_mapped['actual_segment']
    decode_and_trial_info_manydays['mental_segment_mapped'] = decode_and_trial_info_manydays_mapped['mental_segment']
    
    decode_and_trial_info_manydays = decode_and_trial_info_manydays[decode_and_trial_info_manydays['trial_number_by_epoch'] != 0]

    
    no_seg_change_trial_bool = decode_and_trial_info_manydays.groupby(['nwb_file_name','epoch_number','trial_number_by_epoch']).apply(
        lambda x_df: (x_df['segment_count_within_trial'] == x_df['segment_count_within_trial'].min()) & (x_df['segment_count_within_trial'] == x_df['segment_count_within_trial'].max()))
    decode_and_trial_info_manydays['zero_seg_change_trial'] = no_seg_change_trial_bool.values

    #newer filtering stuff
    # Filter out rows for trials without seg changes
    decode_and_trial_info_manydays = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['zero_seg_change_trial']==False)]

    #test that this data cleaning worked 
    only_one_seg = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_first_seg_of_trial']==True) & (decode_and_trial_info_manydays['is_last_seg_of_trial']==True)]
    assert len(np.unique(only_one_seg['trial_number_by_epoch']))==0, f'at least one trial has zero-seg-changes or some other first/last seg parsing issue where first and last overalp'

    # first_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_first_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]
    # last_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_last_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]
    # assert len(np.unique(first_is_stem['trial_number_by_epoch']))==0, f'at least one trial has first seg as a stem.'
    # assert len(np.unique(last_is_stem['trial_number_by_epoch']))==0, f'at least one trial has last seg as a stem.'

    first_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_first_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]
    last_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_last_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]
    
    # handle chimi 0221 and 0306 triasl 4 and 6 epochs 8 and 6 
    if len(np.unique(first_is_stem['trial_number_by_epoch'])) != 0:
        print('WARNING: At least one trial has first seg as a stem!')
        trials_to_remove = decode_and_trial_info_manydays[(( decode_and_trial_info_manydays['is_first_seg_of_trial']==True ) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) ) )][['nwb_file_name','epoch_number','trial_number_by_epoch']].value_counts().index.values
        print(f'WARNING: Removing trials with first seg as a stem, which are {trials_to_remove}.')
        for t in trials_to_remove:
            decode_and_trial_info_manydays = decode_and_trial_info_manydays[~((decode_and_trial_info_manydays['nwb_file_name']==t[0]) & (decode_and_trial_info_manydays['epoch_number']==t[1]) & (decode_and_trial_info_manydays['trial_number_by_epoch']==t[2]))]
        print(f'Removed n={len(trials_to_remove)} trials.')
        #recalculate this number for new assertion test
        first_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_first_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]

    assert len(np.unique(first_is_stem['trial_number_by_epoch']))==0, f'at least one trial still has first seg as a stem.'

    if len(np.unique(last_is_stem['trial_number_by_epoch'])) != 0:
        print('WARNING: At least one trial has last seg as a stem!')
        trials_to_remove = decode_and_trial_info_manydays[(( decode_and_trial_info_manydays['is_last_seg_of_trial']==True ) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) ) )][['nwb_file_name','epoch_number','trial_number_by_epoch']].value_counts().index.values
        print(f'WARNING: Removing trials with last seg as a stem, which are {trials_to_remove}.')
        for t in trials_to_remove:
            decode_and_trial_info_manydays = decode_and_trial_info_manydays[~((decode_and_trial_info_manydays['nwb_file_name']==t[0]) & (decode_and_trial_info_manydays['epoch_number']==t[1]) & (decode_and_trial_info_manydays['trial_number_by_epoch']==t[2]))]
        print(f'Removed n={len(trials_to_remove)} trials.')
        #recalculate this number for new assertion test
        last_is_stem = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['is_last_seg_of_trial']==True) & (decode_and_trial_info_manydays['actual_segment_mapped'].isin(stem_segs_mapped) )]
    
    assert len(np.unique(last_is_stem['trial_number_by_epoch']))==0, f'at least one trial still has last seg as a stem.'

    # Now can filter      
    if require_nonlocal_by_segment:
        decode_and_trial_info_manydays = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['nonlocal_by_segment']==True)]
    if remove_hpd_timepoints:
        decode_and_trial_info_manydays = decode_and_trial_info_manydays[(decode_and_trial_info_manydays[f'spatial_coverage_{hpd_percent}_hpd']<hpd_threshold)]
    if remove_low_speed_timepoints:
        decode_and_trial_info_manydays = decode_and_trial_info_manydays[(decode_and_trial_info_manydays['head_speed']>head_speed_threshold)]

    return decode_and_trial_info_manydays

def add_mapped_cols(big_df, min_ahbeh_dist = 15, max_ahbeh_dist=400):
    '''
    add columns to big_df that are mapped to patch and segment info
    where big_df comes from get_clusterless_trial_rl_info_clean_and_filter or similar
    '''
    patch_to_segs_mapped = {'A':[0,1,2], 'B':[3,4,5], 'C':[6,7,8]}
    segs_to_patch_mapped = {0:'A', 1:'A', 2:'A', 3:'B', 4:'B', 5:'B', 6:'C', 7:'C', 8:'C'}

    # add trials to next switch with neg sign, and abs val ahead behind distance
    big_df['trials_from_next_switch'] = big_df['trials_to_next_switch']*-1
    big_df['abs_ahead_behind_distance'] = np.abs(big_df['ahead_behind_distance'])

    # nonlocal by ahbeh range using params above
    big_df['nonlocal_by_ahbeh_range'] = (big_df['abs_ahead_behind_distance'] >= min_ahbeh_dist) & (big_df['abs_ahead_behind_distance'] <= max_ahbeh_dist)

    # add patch info
    try_patch_mapping = big_df[['actual_segment_mapped', 'mental_segment_mapped']].applymap(segs_to_patch_mapped.get)
    #print(sum(try_patch_mapping['actual_segment_mapped'] != try_patch_mapping['mental_segment_mapped']))
    big_df['actual_patch'] = try_patch_mapping['actual_segment_mapped']
    big_df['mental_patch'] = try_patch_mapping['mental_segment_mapped']
    big_df['nonlocal_by_patch'] = (big_df['mental_patch'] != big_df['actual_patch'])
    
    return big_df

def remove_100all_50all(big_df):
    '''
    remove rows where all p_rew_leafs are 100 or 50
    (run after get_clusterless_trial_rl_info_clean_and_filter)
    '''
    print(f'orig len: {len(big_df)}')
    p_rew_cols = [f"p_rew_leaf{i}" for i in [1,2,3,4,5,6]]
    big_df[~big_df[p_rew_cols].eq(big_df['p_rew_leaf1'], axis=0).all(axis=1)]
    print(f'trimmed len: {len(big_df)}')
    return big_df

