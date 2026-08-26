import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
import seaborn as sns
from scipy import stats

from plot_content import segment_renumbered_and_relabeled

def get_all_rat_big_dfs(
    out_path = '/stelmo/alison/big_df_pkls/',
    today_now = '20230113',
    with_RL = False):
    '''warning: hard coded right now! addapt for other future uses'''
    
    # reading in latest dfs based on the path and file name from how were saved
    if with_RL:
        file_descriptor = '_big_df_RL_uncertainty_'
        
    else:
        file_descriptor = '_big_df_stabledecayNoRL_'

    subject_id = 'senor'
    senor_big_df=pd.read_pickle(out_path+subject_id+file_descriptor+today_now+'.pkl')

    subject_id = 'j16'
    j16_big_df = pd.read_pickle(out_path+subject_id+file_descriptor+today_now+'.pkl')

    subject_id = 'chimi'
    chimi_big_df = pd.read_pickle(out_path+subject_id+file_descriptor+today_now+'.pkl')

    subject_id = 'peanut'
    peanut_big_df = pd.read_pickle(out_path+subject_id+file_descriptor+today_now+'.pkl')

    subject_id = 'wilbur'
    wilbur_big_df = pd.read_pickle(out_path+subject_id+file_descriptor+today_now+'.pkl')
    
    all_rat_big_dfs = {'j16':j16_big_df, 'wilbur':wilbur_big_df, 'senor':senor_big_df, 'peanut':peanut_big_df, 'chimi':chimi_big_df}
    
    return all_rat_big_dfs

def get_all_rat_stable_nwb_file_names(all_rat_big_dfs):
    '''warning: hard coded dates to include in stable task for now'''
    # careful with indices after adding to db
    all_nwb_file_names_dict = {}
    stable_nwb_file_names_dict = {}
    rats = ['j16', 'chimi', 'senor', 'peanut', 'wilbur']
    for rat in rats:
        try:
            animal_big_df = all_rat_big_dfs[rat]
        except:
            rat = rat.lower()
            animal_big_df = all_rat_big_dfs[rat]
        nwb_file_names = list(np.unique(animal_big_df['nwb_file_name']))
        
        all_nwb_file_names_dict[rat] = nwb_file_names
        
        # hard coded rn!
        if rat == 'j16':
            stable_nwb_file_names_dict[rat] = nwb_file_names[0:8]
        elif rat=='wilbur':
            stable_nwb_file_names_dict[rat] = nwb_file_names[0:7]
        elif rat=='senor':
            stable_nwb_file_names_dict[rat] = nwb_file_names[0:5]
        elif rat=='chimi':
            stable_nwb_file_names_dict[rat] = nwb_file_names[0:14]
        elif rat=='peanut':
            stable_nwb_file_names_dict[rat] = nwb_file_names[0:5]
    
    return all_nwb_file_names_dict, stable_nwb_file_names_dict

def plot_hist2d_actual_mental_seg_type_inpatch_stemleafleaf(rat, all_rat_big_dfs, stable_nwb_file_names_dict,                                                 
                                           is_actual_first_seg, is_actual_last_seg,
                                            save_fig = False,
                                           hpd_percent = 50,
                                           hpd_threshold = 50,
                                           remove_hpd_timepoints = True,
                                           remove_low_speed_timepoints = True,
                                           head_speed_threshold = 10,
                                           fig_path = None,
                                           is_nosepoking=False,
                                           filter_by_content=True, nonlocal_by_seg=False,
                                           ahbeh_abs_max=None, ahbeh_abs_min=None, groupby_names=['stem_switch'],
                                           normalization_method='filtered_column_sums_remove_local_diag', cmap_max=.12, data_name=''):
    '''
    plots stem leaf leaf version of hist2d for in patch data only
    can run this on first or final seg
    run on one rat at a time
    has switch vs stay subplots, and a couple versions of normalization
    '''
    decode_and_trial_info_base = all_rat_big_dfs[rat]
    nwb_file_names = stable_nwb_file_names_dict[rat]
    decode_and_trial_info_base = decode_and_trial_info_base[decode_and_trial_info_base.nwb_file_name.isin(nwb_file_names)]

    #add soem columns to separate out leaves and stems
    seg_mapped_to_seg_type = {0:'stem', 1:'leaf_L', 2:'leaf_R',
                             3:'stem', 4:'leaf_L', 5:'leaf_R',
                             6:'stem', 7:'leaf_L', 8:'leaf_R',}

    decode_and_trial_info_base['actual_seg_type'] = decode_and_trial_info_base['actual_segment_mapped'].map(seg_mapped_to_seg_type)
    decode_and_trial_info_base['mental_seg_type'] = decode_and_trial_info_base['mental_segment_mapped'].map(seg_mapped_to_seg_type)

    seg_mapped_to_seg_type_num = {0:0, 1:1, 2:2,
                             3:0, 4:1, 5:2,
                             6:0, 7:1, 8:2}

    decode_and_trial_info_base['actual_seg_type_num'] = decode_and_trial_info_base['actual_segment_mapped'].map(seg_mapped_to_seg_type_num)
    decode_and_trial_info_base['mental_seg_type_num'] = decode_and_trial_info_base['mental_segment_mapped'].map(seg_mapped_to_seg_type_num)

    #filter
    decode_and_trial_info_day_occupancy_filtered = decode_and_trial_info_base[ (decode_and_trial_info_base['is_nosepoking'] == is_nosepoking) &
                                                                                 (decode_and_trial_info_base['is_last_seg_of_trial'] == is_actual_last_seg) &
                                                                                 (decode_and_trial_info_base['is_first_seg_of_trial'] == is_actual_first_seg) ] 
    decode_and_trial_info_day_occupancy_filtered2 = decode_and_trial_info_day_occupancy_filtered[decode_and_trial_info_day_occupancy_filtered['actual_patch']==decode_and_trial_info_day_occupancy_filtered['mental_patch']]
    
    # group the data by stay/switch
    grouped = decode_and_trial_info_day_occupancy_filtered2.groupby(groupby_names)

    segment_type_names = ['stem', 'leaf_L', 'leaf_R']

    # iterate through groups
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12,16))
    fig.subplots_adjust(hspace=.5)
    fig.subplots_adjust(wspace=.5)
    for name, group_df in grouped:
        if name == False: #jsut work with the first group first
            col_id = 0
        elif name == True:
            col_id = 1
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
        hist1, xedges1, yedges1, mesh1 = axes[0,col_id].hist2d(group_df_filtered['actual_seg_type_num'], group_df_filtered['mental_seg_type_num'], bins=[np.arange(0,4,1),np.arange(0,4,1)])
        axes[0,col_id].set_title(f'counts\n'
                                f'stem_switch={name}', x=.5,y=1.05, )
        axes[0,col_id].set_xlabel('actual segment type')
        axes[0,col_id].set_ylabel('mental segment type')
        axes[0,col_id].set_xticks(np.arange(0.5,3.5,1))
        axes[0,col_id].set_yticks(np.arange(0.5,3.5,1))
        axes[0,col_id].set_xticklabels(segment_type_names, rotation = 30)
        axes[0,col_id].set_yticklabels(segment_type_names)
        fig.colorbar(mesh1, label='count of 2 ms bins', ax=axes[0,col_id])
        #axes[0,col_id].set_title()
        fig.suptitle(f'rat={rat},segment={seg},restrict_to_within_patch: True\n'
                      f'ndays={len(nwb_file_names)}, is_nosepoking={is_nosepoking}, nonlocal_by_seg={nonlocal_by_seg}\n'
                      f'abs_ahbeh_min={ahbeh_abs_min}, abs_ahebeh_max={ahbeh_abs_max}\n'
                      f'remove_hpd_times={remove_hpd_timepoints}, hpd_percent={hpd_percent}, hpd_cm={hpd_threshold}\n'
                      f'remove_low_speed_times={remove_low_speed_timepoints}, speed_min={head_speed_threshold}', x=.5,y=1, 
                      )
        fig.show()

        plt.set_cmap('viridis')
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='black') #current_cmap.colors[0])

        if normalization_method=='occupancy_column_sums':
            hist_occupancy, xedges_occupancy, yedges_occupancy = np.histogram2d(group_df['actual_seg_type_num'], group_df['mental_seg_type_num'], bins=[np.arange(0,4,1),np.arange(0,4,1)])
            #print(hist_occupancy.astype(int))
            column_totals = hist_occupancy.sum(axis=1, keepdims=True) # sum over columns. Now divide all columns
            #print(column_totals)
            hist_prop_of_actual_time = hist1/column_totals
            # print(hist_prop_of_actual_time)
            # print(hist_prop_of_actual_time.T)
            mesh2 = axes[1,col_id].pcolormesh(xedges_occupancy, yedges_occupancy, hist_prop_of_actual_time.T)
        elif normalization_method=='filtered_column_sums':
            column_totals = hist1.sum(axis=1, keepdims=True)
            hist_prop_of_actual_time = hist1/column_totals
            mesh2 = axes[1,col_id].pcolormesh(xedges1,yedges1, hist_prop_of_actual_time.T)
        elif normalization_method=='filtered_column_sums_remove_local_diag':
            assert nonlocal_by_seg==False, 'nonlocal_by_seg must be False for this method to make sense.'
            column_totals = hist1.sum(axis=1, keepdims=True)
            hist_prop_of_actual_time = hist1/column_totals

            #fig3, ax3 = plt.subplots(ncols=1)
            mesh3 = axes[2,col_id].pcolormesh(xedges1, yedges1, hist_prop_of_actual_time.T)
            axes[2,col_id].set_xlabel('actual segment type')
            axes[2,col_id].set_ylabel('mental segment type')
            axes[2,col_id].set_title(f'normalized\n'
                          f'norm=filtered_column_sums', y=1.05)
            axes[2,col_id].set_xticks(np.arange(0.5,3.5,1))
            axes[2,col_id].set_yticks(np.arange(0.5,3.5,1))
            axes[2,col_id].set_xticklabels(segment_type_names, rotation = 30)
            axes[2,col_id].set_yticklabels(segment_type_names)
            fig.colorbar(mesh3,label='proportion of actual segment time', ax=axes[2,col_id])
            fig.show()
            axes[2,col_id].set_rasterization_zorder(2)

            #print(hist_prop_of_actual_time.round(decimals=2), 'original\n')
            diag_inds = np.diag_indices(3)
            hist_prop_of_actual_time[diag_inds] = np.nan
            #print(hist_prop_of_actual_time.round(decimals=2), 'changed diag\n')       
            max_after_remove_diag = np.nanmax(hist_prop_of_actual_time)
            #print(f'max val after removing diag is: {max_after_remove_diag}')
            if max_after_remove_diag>cmap_max:
                print(f'WARNING: max value after removing diag is greater than cmap_max: increase cmap_max arg! Instead try: {max_after_remove_diag}')
            mesh2 = axes[1,col_id].pcolormesh(xedges1, yedges1, hist_prop_of_actual_time.T, vmax=cmap_max, vmin=0)


        #mesh2 = ax2.pcolormesh(xedges, yedges, hist_prop_of_actual_time.T)
        axes[1,col_id].set_xlabel('actual segment type')
        axes[1,col_id].set_ylabel('mental segment type')
        axes[1,col_id].set_title(f'normalized\n'
                      f'norm={normalization_method}', y=1.05)
        axes[1,col_id].set_xticks(np.arange(0.5,3.5,1))
        axes[1,col_id].set_yticks(np.arange(0.5,3.5,1))
        axes[1,col_id].set_xticklabels(segment_type_names, rotation = 30)
        axes[1,col_id].set_yticklabels(segment_type_names)
        fig.colorbar(mesh2,label='proportion of actual segment time', ax=axes[1,col_id])
 
        fig.show()
        axes[1,col_id].set_rasterization_zorder(2)

    if save_fig: 
        fig_name = f'all_days_stable_hist2d_stem_leaf_leaf_6subplots_during_firstseg{is_actual_first_seg}_lastseg{is_actual_last_seg}'
        fig.savefig(f'{fig_path}{rat}_{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)

# instead of using the mean, try to calculate some other different summary stats, and look at the results
def get_first_last_seg_ahbeh_summary_metric(big_df, summary_metric='mean', quantile=.9, restrict_to_within_patch=False):
    '''
    return grouped df with ahbeh summary metric calculated
    only for data when animal is in the first or final segment of a trial
    quantile only applies to a quantile calculation, otherwise is irrelevant
    
    big_df: df over time of extracted decode info and many other variables incl ahbeh distance
    summary_metric
    ----
    returns:
    grouped_df after applying summary metric to each first/final segmetn of data
    summary_metric to keep track of for plot labeling etc
    quantile if applicable, otherwise returns None
    '''
    
    if not restrict_to_within_patch:
        big_df_first_last_seg = big_df[np.logical_or(big_df['is_first_seg_of_trial'], big_df['is_last_seg_of_trial'])]
    elif restrict_to_within_patch:
        big_df_within_patch = big_df[big_df['actual_patch']==big_df['mental_patch']]
        big_df_first_last_seg = big_df_within_patch[np.logical_or(big_df_within_patch['is_first_seg_of_trial'], big_df_within_patch['is_last_seg_of_trial'])]
    
    group_df = big_df_first_last_seg.groupby(by=[
        'nwb_file_name','epoch_number','trial_number_by_epoch', 'is_first_seg_of_trial',
        'stem_switch','trials_from_prior_switch','trials_to_next_switch', 
        'trials_from_next_switch'])
    
    if summary_metric == 'mean':        
        group_apply_df = group_df.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).mean()).reset_index(name='ahbeh_metric')
    elif summary_metric == 'median':
        group_apply_df = group_df.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).median()).reset_index(name='ahbeh_metric')
    elif summary_metric == 'max':
        group_apply_df = group_df.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).max()).reset_index(name='ahbeh_metric')
    elif summary_metric == 'quantile':
        group_apply_df = group_df.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).quantile(q=quantile)).reset_index(name='ahbeh_metric')
    elif summary_metric == 'std':
        group_apply_df = group_df.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).std()).reset_index(name='ahbeh_metric')
    elif summary_metric == 'var':
        group_apply_df = group_df.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).var()).reset_index(name='ahbeh_metric')
    
    #grouped_df = group_apply_df.reset_index('ahbeh_metric') #ahbeh_metric instead of avg_ahbeh to be more general
    
    if summary_metric != 'quantile':
        quantile=None
    return (group_apply_df, summary_metric, quantile, restrict_to_within_patch)

def plot_all_first_last_seg_ahbeh_metric_around_switch(rats, all_rat_first_last_seg_ahbeh_summary_metric, stable_nwb_file_names_dict,
                                                       x_lim=None, y_lim=None, ci=95,
                                                       fig_path = None, save_fig = False):
    '''
    stable only
    '''
    
    # set up custom plotting properties
    if x_lim is not None:
        xlim = x_lim
    else:
        xlim = (-5.5, 5.5)
    if y_lim is not None:
        ylim=y_lim
    else:
        ylim=(0,30)
    
    # color cycles for rats in each of the two subplots
#     first_seg_colors = iter(cm.Paired(np.linspace(0, .5, len(rats)))) #first seg
#     final_seg_colors = iter(cm.Paired(np.linspace(.1, .6, len(rats)))) #final seg
#     first_seg_colors = iter(cm.Paired(np.arange(0, 10, 2))) #first seg
#     final_seg_colors = iter(cm.Paired(np.arange(1, 11, 2))) #final seg
#     first_seg_colors = iter(cm.tab20b(np.linspace(0, .8, 5))) #first seg
#     final_seg_colors = iter(cm.tab20b(np.linspace(.05, .85, 5))) #final seg
    first_seg_colors = iter(cm.tab20(np.linspace(0, .8, 5))) #first seg
    final_seg_colors = iter(cm.tab20(np.linspace(0, .8, 5))) #first seg
#     final_seg_colors = iter(cm.tab20(np.linspace(.05, .85, 5))) #final seg
     
    for i,rat in enumerate(rats):
        # for the first rat, set up the subplots, for subsequent rats, will reuse these axes
        if i == 0:
            fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(15,10))
        
        # extract variables from the dict of all-rat data
        grouped_df = all_rat_first_last_seg_ahbeh_summary_metric[rat][0]
        summary_metric = all_rat_first_last_seg_ahbeh_summary_metric[rat][1]
        if summary_metric == 'quantile':
            quantile = all_rat_first_last_seg_ahbeh_summary_metric[rat][2]
            summary_metric += str(quantile)
        restrict_to_within_patch = all_rat_first_last_seg_ahbeh_summary_metric[rat][3]
        
        # filter data to the first seg, and stable data only
        grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == True) & (grouped_df.nwb_file_name.isin(stable_nwb_file_names_dict[rat]))]
        
        # update colors to use for this rat
        first_seg_color = next(first_seg_colors)
        final_seg_color = next(final_seg_colors)
        #used to color with hue first/final seg, but now setting color cycle manually #hue=grouped_df.is_first_seg_of_trial
        
        # plot dot for each rat, disconnected over trials, with 95 percent ci
        sns.lineplot(x=grouped_df_subset.trials_from_prior_switch+.1*(i+1)-.3, y=grouped_df_subset.ahbeh_metric, linestyle='', marker='o', err_style='bars', ci=ci, alpha=1, ax=axes[0], label=rat, color=first_seg_color)
        sns.lineplot(x=grouped_df_subset.trials_from_next_switch+.1*(i+1)-.3, y=grouped_df_subset.ahbeh_metric, linestyle='', marker='o', err_style='bars',  ci=ci, alpha=1, ax=axes[0], color=first_seg_color)
        axes[0].set(xlim=xlim)
        axes[0].set(ylim=ylim)

        grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == False) & (grouped_df.nwb_file_name.isin(stable_nwb_file_names_dict[rat]))]

        sns.lineplot(x=grouped_df_subset.trials_from_prior_switch+.1*(i+1)-.25, y=grouped_df_subset.ahbeh_metric,  linestyle='', marker='o', err_style='bars',  ci=ci, alpha=1, ax=axes[1], label=rat, color=final_seg_color)
        sns.lineplot(x=grouped_df_subset.trials_from_next_switch+.1*(i+1)-.25, y=grouped_df_subset.ahbeh_metric,  linestyle='', marker='o', err_style='bars',  ci=ci,  alpha=1, ax=axes[1], color=final_seg_color)

    axes[0].set_title('First segment')
    axes[1].set_title('Final segment')

    #ahbeh metric carries through here 
    axes[1].set_ylabel(f'Absolute ahead-behind distance {summary_metric} (+/- {ci}% ci)', loc='bottom')
    axes[0].set_ylabel('')
    axes[0].set_xlabel('Trials from switch trial')
    axes[1].set_xlabel('Trials from switch trial')

    axes[0].legend(frameon=False, loc='upper right')
    axes[1].legend(frameon=False, loc='upper right')

    axes[0].axvline(x=0, linestyle='--', color='lightgrey', zorder=0, label='stem_switch')
    axes[1].axvline(x=0, linestyle='--', color='lightgrey', zorder=0, label='stem_switch')
        
    plt.suptitle(f'{rats}\nstable task, in_patch_only: {restrict_to_within_patch}\n', y=1)

    sns.despine()

    if save_fig:
        fig_name = f'all_rats_stable_task_first_final_seg_ahbeh_around_switches_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_xlim{xlim[1]}_ylim{ylim[1]}'
        plt.savefig(f'{fig_path}{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)

def plot_all_first_last_seg_ahbeh_metric_around_switch_sem(rats, all_rat_first_last_seg_ahbeh_summary_metric, stable_nwb_file_names_dict,
                                                       x_lim=None, y_lim=None, 
                                                       fig_path = None, save_fig = False):
    '''stable only'''
    
    # set up custom plotting properties
    if x_lim is not None:
        xlim = x_lim
    else:
        xlim = (-5.5, 5.5)
    if y_lim is not None:
        ylim=y_lim
    else:
        ylim=(0,30)
    
    first_seg_colors = iter(cm.tab20(np.linspace(0, .8, 5))) #first seg
    final_seg_colors = iter(cm.tab20(np.linspace(0, .8, 5))) #first seg
    
    for i,rat in enumerate(rats):
        if i == 0:
            fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(20,5))
        
        # extract variables from the dict of all-rat data
        grouped_df = all_rat_first_last_seg_ahbeh_summary_metric[rat][0]
        summary_metric = all_rat_first_last_seg_ahbeh_summary_metric[rat][1]
        if summary_metric == 'quantile':
            quantile = all_rat_first_last_seg_ahbeh_summary_metric[rat][2]
            summary_metric += str(quantile)
        restrict_to_within_patch = all_rat_first_last_seg_ahbeh_summary_metric[rat][3]
        
        #filter
        grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == True) & (grouped_df.nwb_file_name.isin(stable_nwb_file_names_dict[rat]))]

        #hue=grouped_df.is_first_seg_of_trial,
        first_seg_color = next(first_seg_colors)
        final_seg_color = next(final_seg_colors)

        sns.lineplot(x=grouped_df_subset.trials_from_prior_switch, y=grouped_df_subset.ahbeh_metric, linestyle='-', marker='', alpha=1, ax=axes[0], label=rat, color=first_seg_color)
        sns.lineplot(x=grouped_df_subset.trials_from_next_switch, y=grouped_df_subset.ahbeh_metric, linestyle='-', marker='',   alpha=1, ax=axes[0], color=first_seg_color)
        axes[0].set(xlim=xlim)
        axes[0].set(ylim=ylim)

        #filter
        grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == False) & (grouped_df.nwb_file_name.isin(stable_nwb_file_names_dict[rat]))]

        sns.lineplot(x=grouped_df_subset.trials_from_prior_switch, y=grouped_df_subset.ahbeh_metric,  linestyle='-', marker='', alpha=1, ax=axes[1], label=rat, color=final_seg_color)
        sns.lineplot(x=grouped_df_subset.trials_from_next_switch, y=grouped_df_subset.ahbeh_metric,  linestyle='-', marker='',  alpha=1, ax=axes[1], color=final_seg_color)

    axes[0].set_title('First segment')
    axes[1].set_title('Final segment')
    
    #ahbeh metric carries through here 
    axes[0].set_ylabel(f'Absolute ahead-behind distance {summary_metric} (+/- sem)', loc='bottom')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('Trials from switch trial')
    axes[1].set_xlabel('Trials from switch trial')

    axes[0].legend('',frameon=False, loc='upper right')
    axes[1].legend(frameon=False, loc='upper right')

    axes[0].axvline(x=0, linestyle='--', color='lightgrey', zorder=0, label='stem_switch')
    axes[1].axvline(x=0, linestyle='--', color='lightgrey', zorder=0, label='stem_switch')
        
    plt.suptitle(f'{rats}\nstable task, in_patch_only: {restrict_to_within_patch}\n', y=1)

    sns.despine()

    if save_fig:
        fig_name = f'all_rats_stable_task_first_final_seg_ahbeh_around_switches_sem_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_xlim{xlim[1]}_ylim{ylim[1]}'
        plt.savefig(f'{fig_path}{rat}_{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)

def plot_first_last_seg_ahbeh_metric_around_switch(rat, all_rat_first_last_seg_ahbeh_summary_metric, stable_nwb_file_names_dict,
                                                ci=95, x_lim = (-5.5,5.5), y_lim=(0,30),
                                                fig_path = None, save_fig = False):
    '''stable only'''
    grouped_df = all_rat_first_last_seg_ahbeh_summary_metric[rat][0]
    summary_metric = all_rat_first_last_seg_ahbeh_summary_metric[rat][1]
    if summary_metric == 'quantile':
        quantile = all_rat_first_last_seg_ahbeh_summary_metric[rat][2]
        summary_metric += str(quantile)
    restrict_to_within_patch = all_rat_first_last_seg_ahbeh_summary_metric[rat][3]
     
    grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == True) & (grouped_df.nwb_file_name.isin(stable_nwb_file_names_dict[rat]))]
    
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(20,5))
    sns.lineplot(x=grouped_df_subset.trials_from_prior_switch, y=grouped_df_subset.ahbeh_metric, hue=grouped_df.is_first_seg_of_trial, linestyle='', marker='o', err_style='bars', ax=axes[0])
    sns.lineplot(x=grouped_df_subset.trials_from_next_switch, y=grouped_df_subset.ahbeh_metric, hue=grouped_df.is_first_seg_of_trial, linestyle='', marker='o', err_style='bars',  ax=axes[0])
    axes[0].set(xlim=x_lim)
    axes[0].set(ylim=y_lim)

    grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == False) & (grouped_df.nwb_file_name.isin(stable_nwb_file_names_dict[rat]))]

    sns.lineplot(x=grouped_df_subset.trials_from_prior_switch, y=grouped_df_subset.ahbeh_metric, hue=grouped_df.is_first_seg_of_trial, linestyle='', marker='o', err_style='bars', ci=ci, ax=axes[1])
    sns.lineplot(x=grouped_df_subset.trials_from_next_switch, y=grouped_df_subset.ahbeh_metric, hue=grouped_df.is_first_seg_of_trial, linestyle='', marker='o', err_style='bars',  ci=ci, ax=axes[1])

    axes[0].set_title('First segment')
    axes[1].set_title('Final segment')

    axes[0].set_ylabel(f'Absolute ahead-behind distance {summary_metric} (+/- {ci}% ci)', loc='bottom')
    axes[0].set_xlabel('Trials from switch trial')
    axes[1].set_xlabel('Trials from switch trial')

    axes[0].legend([],[],frameon=False)
    axes[1].legend([],[],frameon=False)

    axes[0].axvline(x=0, linestyle='--', color='lightgrey', zorder=0)
    axes[1].axvline(x=0, linestyle='--', color='lightgrey', zorder=0)

    #nwb_file_names = np.unique(grouped_df[grouped_df.nwb_file_name.isin(stable_nwb_file_names)].nwb_file_name.values)
    plt.suptitle(f'{rat} stable task, in_patch_only: {restrict_to_within_patch}\n{np.unique(stable_nwb_file_names_dict[rat])}', y=1.3)
    sns.despine()
    
    if save_fig:
        fig_name = f'stable_task_first_final_seg_ahbeh_around_switches_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_xlim{x_lim[1]}_ylim{y_lim[1]}'
        plt.savefig(f'{fig_path}{rat}_{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)

def plot_first_last_seg_ahbeh_metric_around_switch_learning(rat, all_rat_first_last_seg_ahbeh_summary_metric, stable_nwb_file_names_dict,
                                                     ci=95, x_lim = (-5.5,5.5), y_lim=(0,30),
                                                fig_path = None, save_fig = False):
    '''stable only'''
    grouped_df = all_rat_first_last_seg_ahbeh_summary_metric[rat][0]
    summary_metric = all_rat_first_last_seg_ahbeh_summary_metric[rat][1]
    if summary_metric == 'quantile':
        quantile = all_rat_first_last_seg_ahbeh_summary_metric[rat][2]
        summary_metric += str(quantile)
    restrict_to_within_patch = all_rat_first_last_seg_ahbeh_summary_metric[rat][3]
    stable_nwb_file_names = stable_nwb_file_names_dict[rat]
    
    # stable only, over pairs of days
    #nwb_file_name_subsets = [[stable_nwb_file_names[i]] for i in range(0,len(stable_nwb_file_names)) ]
    nwb_file_name_subsets = [stable_nwb_file_names[i:i+2] for i in range(0,len(stable_nwb_file_names), 2) ]
    
    xlim = x_lim
    ylim = y_lim

    fig, axes = plt.subplots(ncols=2, nrows= len(nwb_file_name_subsets), sharex=True, sharey=True, figsize=(20,8))

    for i,nwb_file_name_subset in enumerate(nwb_file_name_subsets):
        grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == True) & (grouped_df.nwb_file_name.isin(nwb_file_name_subset))]

        #print(i, nwb_file_name_subset)

        sns.lineplot(x=grouped_df_subset.trials_from_prior_switch, y=grouped_df_subset.ahbeh_metric, hue=grouped_df.is_first_seg_of_trial, linestyle='', marker='o', err_style='bars', ax=axes[0+i,0])
        sns.lineplot(x=grouped_df_subset.trials_from_next_switch, y=grouped_df_subset.ahbeh_metric, hue=grouped_df.is_first_seg_of_trial, linestyle='', marker='o', err_style='bars',  ax=axes[0+i,0])
        axes[0+i,0].set(xlim=xlim)
        axes[0+i,0].set(ylim=ylim)

        grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == False) & (grouped_df.nwb_file_name.isin(nwb_file_name_subset))]

        sns.lineplot(x=grouped_df_subset.trials_from_prior_switch, y=grouped_df_subset.ahbeh_metric, hue=grouped_df.is_first_seg_of_trial, linestyle='', marker='o', err_style='bars', ax=axes[0+i,1])
        sns.lineplot(x=grouped_df_subset.trials_from_next_switch, y=grouped_df_subset.ahbeh_metric, hue=grouped_df.is_first_seg_of_trial, linestyle='', marker='o', err_style='bars',  ax=axes[0+i,1])

        if i==0:
            axes[0+i,0].set_title('First segment', color='orange')
            axes[0+i,1].set_title('Final segment', color='blue')

        if i==2:
            axes[0+i,0].set_ylabel(f'Absolute ahead-behind distance {summary_metric} (+/- {ci}% ci)')
        else:
            axes[0+i,0].set_ylabel('')

        axes[0+i,0].set_xlabel('Trials from switch trial')
        axes[0+i,1].set_xlabel('Trials from switch trial')

        axes[0+i,0].legend([],[],frameon=False)
        axes[0+i,1].legend([],[],frameon=False)

        axes[0+i,0].axvline(x=0, linestyle='--', color='lightgrey', zorder=0)
        axes[0+i,1].axvline(x=0, linestyle='--', color='lightgrey', zorder=0)

        #nwb_file_names = np.unique(grouped_df[grouped_df.nwb_file_name.isin(stable_nwb_file_names)].nwb_file_name.values)
        plt.suptitle(f'{rat} stable task, in_patch_only: {restrict_to_within_patch}\n{nwb_file_name_subsets}')

        sns.despine()
    if save_fig:
        fig_name = f'stable_task_first_final_seg_ahbeh_around_switches_learning_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_xlim{xlim[1]}_ylim{ylim[1]}'
        plt.savefig(f'{fig_path}{rat}_{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)

def plot_first_last_seg_ahbeh_metric_around_switch_learning_overlaid(rat, all_rat_first_last_seg_ahbeh_summary_metric, stable_nwb_file_names_dict,
                                                 ci=95, x_lim = (-5.5,5.5), y_lim=(0,30),
                                                fig_path = None, save_fig = False):
    '''stable only'''
    grouped_df = all_rat_first_last_seg_ahbeh_summary_metric[rat][0]
    summary_metric = all_rat_first_last_seg_ahbeh_summary_metric[rat][1]
    if summary_metric == 'quantile':
        quantile = all_rat_first_last_seg_ahbeh_summary_metric[rat][2]
        summary_metric += str(quantile)
    restrict_to_within_patch = all_rat_first_last_seg_ahbeh_summary_metric[rat][3]
    stable_nwb_file_names = stable_nwb_file_names_dict[rat]
    
    nwb_file_name_subsets = [stable_nwb_file_names[i:i+2] for i in range(0,len(stable_nwb_file_names), 2) ]

    xlim = x_lim
    ylim = y_lim

    fig, axes = plt.subplots(ncols=1, nrows= 2, sharex=True, sharey=True, figsize=(20,8))

    oranges = iter(cm.Oranges(np.linspace(1, .5, len(nwb_file_name_subsets))))
    blues = iter(cm.Blues(np.linspace(1, .5, len(nwb_file_name_subsets))))

    for i,nwb_file_name_subset in enumerate(nwb_file_name_subsets):
        grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == True) & (grouped_df.nwb_file_name.isin(nwb_file_name_subset))]

        #print(i, nwb_file_name_subset)
        orange_color = next(oranges)
        sns.lineplot(x=grouped_df_subset.trials_from_prior_switch+.1*i, y=grouped_df_subset.ahbeh_metric, color=orange_color, linestyle='', marker='o', err_style='bars', ax=axes[0])
        sns.lineplot(x=grouped_df_subset.trials_from_next_switch+.1*i, y=grouped_df_subset.ahbeh_metric, color=orange_color, linestyle='', marker='o', err_style='bars',  ax=axes[0])
        axes[0].set(xlim=xlim)
        axes[0].set(ylim=ylim)

        grouped_df_subset = grouped_df[(grouped_df.is_first_seg_of_trial == False) & (grouped_df.nwb_file_name.isin(nwb_file_name_subset))]
        blue_color = next(blues)
        sns.lineplot(x=grouped_df_subset.trials_from_prior_switch+.1*i, y=grouped_df_subset.ahbeh_metric, color=blue_color, linestyle='', marker='o', err_style='bars', ax=axes[1])
        sns.lineplot(x=grouped_df_subset.trials_from_next_switch+.1*i, y=grouped_df_subset.ahbeh_metric, color=blue_color, linestyle='', marker='o', err_style='bars',  ax=axes[1])

        if i==0:
            axes[0].set_title('First segment', color=orange_color)
            axes[1].set_title('Final segment', color=blue_color)

        axes[0].set_ylabel('')
        axes[1].set_ylabel('')

        axes[0].set_xlabel('Trials from switch trial')
        axes[1].set_xlabel('Trials from switch trial')

        axes[0].legend([],[],frameon=False)
        axes[1].legend([],[],frameon=False)

        axes[0].axvline(x=0, linestyle='--', color='lightgrey', zorder=0)
        axes[1].axvline(x=0, linestyle='--', color='lightgrey', zorder=0)

    #nwb_file_names = np.unique(grouped_df[grouped_df.nwb_file_name.isin(stable_nwb_file_names)].nwb_file_name.values)
    plt.suptitle(f'{rat} stable task, in_patch_only: {restrict_to_within_patch}\n{nwb_file_name_subsets}')
    fig.supylabel(f'Absolute ahead-behind distance {summary_metric} (+/- {ci}% ci)', x=.01)
    sns.despine()
    
    if save_fig:
        fig_name = f'stable_task_first_final_seg_ahbeh_around_switches_learning_overlaid_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_xlim{xlim[1]}_ylim{ylim[1]}'
        plt.savefig(f'{fig_path}{rat}_{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)


def get_all_seg_ahbeh_summary_metric(big_df, summary_metric='mean', quantile=.9, restrict_to_within_patch=False):
    '''
    return grouped df with ahbeh calculated for data in each segment
    '''
    # filter to not nosepoking
    big_df_filtered = big_df[(big_df.is_nosepoking==False)] # & (big_df.nwb_file_name.isin(stable_nwb_file_names))]
    # filter for whether to include data that is out of patch
    if restrict_to_within_patch:
        big_df_filtered = big_df_filtered[big_df_filtered['actual_patch']==big_df_filtered['mental_patch']]

    # get first last seg, group, calc ahbeh metric for first last seg only
    first_last_seg = big_df_filtered[np.logical_or(big_df_filtered.is_first_seg_of_trial==True, big_df_filtered.is_last_seg_of_trial==True)]
    
    # make groups, before applying summary stats
    first_last_seg_grouped = first_last_seg.groupby(by=['nwb_file_name','epoch_number','trial_number_by_epoch', 'is_first_seg_of_trial', 'is_last_seg_of_trial', 'stem_switch'])
    all_seg_grouped = big_df_filtered.groupby(by=['nwb_file_name','epoch_number','trial_number_by_epoch', 'segment_count_within_trial','stem_switch','is_first_seg_of_trial','is_last_seg_of_trial'])

    # apply summary stats
    if summary_metric == 'mean':        
        first_last_seg_grouped_applied = first_last_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).mean()).reset_index(name='ahbeh_metric')
        all_seg_grouped_applied = all_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).mean()).reset_index(name='ahbeh_metric')
    elif summary_metric == 'median':        
        first_last_seg_grouped_applied = first_last_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).median()).reset_index(name='ahbeh_metric')
        all_seg_grouped_applied = all_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).median()).reset_index(name='ahbeh_metric')
    elif summary_metric == 'max':        
        first_last_seg_grouped_applied = first_last_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).max()).reset_index(name='ahbeh_metric')
        all_seg_grouped_applied = all_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).max()).reset_index(name='ahbeh_metric')
    elif summary_metric == 'quantile':        
        first_last_seg_grouped_applied = first_last_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).quantile(q=quantile)).reset_index(name='ahbeh_metric')
        all_seg_grouped_applied = all_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).quantile(q=quantile)).reset_index(name='ahbeh_metric')
    elif summary_metric == 'std':        
        first_last_seg_grouped_applied = first_last_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).std()).reset_index(name='ahbeh_metric')
        all_seg_grouped_applied = all_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).std()).reset_index(name='ahbeh_metric')
    elif summary_metric == 'var':        
        first_last_seg_grouped_applied = first_last_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).var()).reset_index(name='ahbeh_metric')
        all_seg_grouped_applied = all_seg_grouped.apply(lambda x_df: np.abs(x_df['ahead_behind_distance']).var()).reset_index(name='ahbeh_metric')
    
    # merge the summary stat with the trials with 3 seg changes, before filtering down further
    all_seg = big_df_filtered.groupby(by=['nwb_file_name','epoch_number','trial_number_by_epoch']).apply(lambda x_df: x_df['segment_count_within_trial'].max()==3).reset_index(name='3_total_seg_changes')
    all_seg_grouped_with_3_total_seg_changes = all_seg.merge(all_seg_grouped_applied)
    
    # filter down to the ones where it's a switch trial with three seg changes
    all_seg_grouped_with_3_total_seg_changes_switch = all_seg_grouped_with_3_total_seg_changes[(all_seg_grouped_with_3_total_seg_changes.stem_switch == True) &
                                                                                                all_seg_grouped_with_3_total_seg_changes['3_total_seg_changes'] == True]
    # find just the stem parts of those valid switch trials
    all_seg_grouped_with_3_total_seg_changes_switch_stems = all_seg_grouped_with_3_total_seg_changes_switch[ all_seg_grouped_with_3_total_seg_changes_switch['segment_count_within_trial'].isin([1,2]) ]
    # drop off the column about total number of seg changes
    all_seg_grouped_with_3_total_seg_changes_switch_stems_trim = all_seg_grouped_with_3_total_seg_changes_switch_stems.drop(inplace = False, columns='3_total_seg_changes')
    # map the stem segmetn count to a more descriptive string for easy sns plotting
    seg_count_to_seg_name = {1:'first_stem', 2:'second_stem'}
    all_seg_grouped_with_3_total_seg_changes_switch_stems_trim['segment'] = all_seg_grouped_with_3_total_seg_changes_switch_stems_trim['segment_count_within_trial'].map(seg_count_to_seg_name)
    all_seg_grouped_with_3_total_seg_changes_switch_stems_trim.drop(inplace = True, columns='segment_count_within_trial')
    # also map the names for the first / final leaf segments
    is_first_seg_to_seg_name = {True:'first_leaf', False:'last_leaf'}
    first_last_seg_grouped_applied['segment'] = first_last_seg_grouped_applied['is_first_seg_of_trial'].map(is_first_seg_to_seg_name)
    
    # combine the stay trial and valid switch trial data
    all_seg_data = pd.concat([first_last_seg_grouped_applied,all_seg_grouped_with_3_total_seg_changes_switch_stems_trim],ignore_index=True)
    
    if summary_metric != 'quantile':
        quantile=None
    return (all_seg_data, summary_metric, quantile, restrict_to_within_patch)

# continue rewriting/editing this to adapt to the new params, referencing the below cell for details
def plot_all_all_seg_ahbeh_metric(rats, all_rat_all_seg_ahbeh_metric, stable_nwb_file_names_dict, ci=95, y_lim=None,
                                                 fig_path = None, save_fig = False):
    '''stable only'''
    custom_colors_by_rat = [sns.color_palette('Paired')[0:2],
                            sns.color_palette('Paired')[2:4],
                            sns.color_palette('Paired')[4:6],
                            sns.color_palette('Paired')[6:8],
                            sns.color_palette('Paired')[8:10],]
    
    for i,rat in enumerate(rats):
        all_seg_data = all_rat_all_seg_ahbeh_metric[rat][0]
        # make it stable only
        all_seg_data = all_seg_data[all_seg_data.nwb_file_name.isin(stable_nwb_file_names_dict[rat])]

        if y_lim is not None:
            ylim = y_lim
        else:
            ylim = (0,30)

        if i==0:
            fig, axes = plt.subplots(ncols=1, figsize=(10,5), sharex=True, sharey=False)
        
        # extract variables from the dict of all-rat data
        summary_metric = all_rat_all_seg_ahbeh_metric[rat][1]
        if summary_metric == 'quantile':
            quantile = all_rat_all_seg_ahbeh_metric[rat][2]
            summary_metric += str(quantile)
        restrict_to_within_patch = all_rat_all_seg_ahbeh_metric[rat][3]
        
        #hue='stem_switch',
        #palette=sns.color_palette('dark')
        #sns.violinplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='stem_switch', hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], cut=0, scale='area', inner='quartile', alpha=.05, dodge=.4, palette=sns.color_palette('pastel'), ax=axes)
        sns.pointplot(data=all_seg_data, x='segment', y='ahbeh_metric',  ci=ci, join=False, hue='stem_switch', dodge=.4, hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], palette=custom_colors_by_rat[i], capsize=.05, ax=axes)
    
    axes.set_ylim(ylim)
    #axes.legend(rats,frameon=False)
    axes.set_ylabel(f'Absolute ahead-behind distance {summary_metric} (+/- {ci}% ci)', loc='bottom')
    axes.set_xlabel('Trial subsection')
    axes.set_xticklabels(['First choice', 'Second choice', 'Third choice', 'Final segment'])
    #sns.violinplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='stem_switch', hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], cut=0, scale='area', inner='quartile', alpha=.05, dodge=.4, palette=sns.color_palette('pastel'), ax=axes[1])
    #sns.pointplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='stem_switch', ci=95, join=False, dodge=.4, hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], palette=sns.color_palette('dark'),capsize=.05, ax=axes[1])
    hands, labs = axes.get_legend_handles_labels()
    #print(hands, labs)
    axes.legend(handles = hands, labels=[f'{rats[0]}, False', f'{rats[0]}, True',
                f'{rats[1]}, False', f'{rats[1]}, True',
                f'{rats[2]}, False', f'{rats[2]}, True',
                f'{rats[3]}, False', f'{rats[3]}, True',
                f'{rats[4]}, False', f'{rats[4]}, True',],
                title='rat, stem_switch',frameon=False, loc='upper right', bbox_to_anchor=(1.4, 0.9))

    plt.suptitle(f'{rats}\nstable task, in_patch_only: {restrict_to_within_patch}\n', y=1)
    #plt.suptitle(f'rat: {rat}, mean +/- 95% ci\n{np.unique(all_seg_data.nwb_file_name)}', y=1.2)
    sns.despine()
    #plt.show()

    if save_fig:
        fig_name = f'all_rats_stable_task_all_segs_ahbeh_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_ylim{ylim[1]}'
        plt.savefig(f'{fig_path}{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)

def plot_all_seg_ahbeh_metric_violin(rat, all_rat_all_seg_ahbeh_metric, stable_nwb_file_names_dict, y_lim=None, ci=95,
                                                 fig_path = None, save_fig = False):
    all_seg_data = all_rat_all_seg_ahbeh_metric[rat][0]
    
    all_seg_data = all_seg_data[all_seg_data.nwb_file_name.isin(stable_nwb_file_names_dict[rat])]
    
    # extract variables from the dict of all-rat data
    summary_metric = all_rat_all_seg_ahbeh_metric[rat][1]
    if summary_metric == 'quantile':
        quantile = all_rat_all_seg_ahbeh_metric[rat][2]
        summary_metric += str(quantile)
    restrict_to_within_patch = all_rat_all_seg_ahbeh_metric[rat][3]
    
    if y_lim is not None:
        ylim = y_lim
    else:
        ylim = (0,40)

    fig, axes = plt.subplots(ncols=1, figsize=(10,5), sharex=True, sharey=False)

    sns.violinplot(data=all_seg_data, x='segment', y='ahbeh_metric', hue='stem_switch', hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], cut=0, scale='area', inner='quartile', alpha=.05, dodge=.4, palette=sns.color_palette('pastel'), ax=axes)
    sns.pointplot(data=all_seg_data, x='segment', y='ahbeh_metric', hue='stem_switch', ci=ci, join=False, dodge=.4, hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], palette=sns.color_palette('dark'),capsize=.05, ax=axes)
    axes.set_ylim(ylim)
    axes.legend([], frameon=False)
    axes.set_ylabel(f'Absolute ahead-behind distance [cm] {summary_metric}')
    axes.set_xlabel('Trial subsection')
    axes.set_xticklabels(['First choice', 'Second choice', 'Third choice', 'Final segment'])
    #sns.violinplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='stem_switch', hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], cut=0, scale='area', inner='quartile', alpha=.05, dodge=.4, palette=sns.color_palette('pastel'), ax=axes[1])
    #sns.pointplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='stem_switch', ci=95, join=False, dodge=.4, hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], palette=sns.color_palette('dark'),capsize=.05, ax=axes[1])
    axes.legend(title='stem_switch',frameon=False, loc='upper right', bbox_to_anchor=(1.4, 0.9))

    plt.suptitle(f'rat: {rat}, mean +/- {ci}% ci\n{np.unique(all_seg_data.nwb_file_name)}', y=1.2)
    sns.despine()
    
    if save_fig:
        fig_name = f'stable_task_stay_switch_violin_points_overlaid_four_segs_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_ylim{ylim[1]}'
        plt.savefig(f'{fig_path}{rat}_{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)  

def plot_all_seg_ahbeh_metric_learning(rat, all_rat_all_seg_ahbeh_metric, stable_nwb_file_names_dict, ci=95, y_lim = None, save_fig = False, fig_path=None):
    all_seg_data = all_rat_all_seg_ahbeh_metric[rat][0]
    all_seg_data = all_seg_data[all_seg_data.nwb_file_name.isin(stable_nwb_file_names_dict[rat])]
    
    # extract variables from the dict of all-rat data
    summary_metric = all_rat_all_seg_ahbeh_metric[rat][1]
    if summary_metric == 'quantile':
        quantile = all_rat_all_seg_ahbeh_metric[rat][2]
        summary_metric += str(quantile)
    restrict_to_within_patch = all_rat_all_seg_ahbeh_metric[rat][3]
    
    #learning
    #dark to light blue for all False, dark to light orange for all True, then sample from every other
    oranges = cm.Oranges(np.linspace(1, .5, len(np.unique(all_seg_data.nwb_file_name))))
    blues = cm.Blues(np.linspace(1, .5, len(np.unique(all_seg_data.nwb_file_name))))
    custom_colors = np.concatenate((oranges, blues))
    custom_colors[::2] = blues
    custom_colors[1::2] = oranges
    if y_lim is not None:
        ylim=y_lim
    fig, axes = plt.subplots(ncols=1, figsize=(8,5.5), sharex=True, sharey=False)

    #sns.violinplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='nwb_file_name', order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], cut=0, scale='area', inner='quartile', alpha=.05, dodge=.4, palette=sns.color_palette('pastel'), ax=axes[0])
    sns.pointplot(data=all_seg_data, x='segment', y='ahbeh_metric', hue=all_seg_data.sort_values(by=['nwb_file_name','stem_switch'], ascending=[True, True])[['nwb_file_name', 'stem_switch']].apply(tuple, axis=1), ci=ci, join=False, dodge=.8, order=['first_leaf', 'first_stem', 'second_stem','last_leaf'],capsize=.0, markersize=12, ax=axes, palette=custom_colors)
    axes.set_ylim(ylim)
    axes.legend(title=f'nwb_file_name, stem_switch',frameon=False, loc='upper right', bbox_to_anchor=(2.5, 1.5))
    axes.set_ylabel(f'Absolute ahead-behind\ndistance [cm] {summary_metric}')
    axes.set_xlabel('Trial subsection',y=-18)
    axes.set_xticklabels(['First\nchoice', 'Second\nchoice', 'Third\nchoice', 'Final\nsegment'])
    #sns.violinplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='nwb_file_name', order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], cut=0, scale='area', inner='quartile', alpha=.05, dodge=.4, palette=sns.color_palette('pastel'), ax=axes[1])
    #sns.pointplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue=all_seg_data[['nwb_file_name','stem_switch']].apply(tuple, axis=1), ci=95, join=False, dodge=.8, order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], palette=custom_colors,capsize=.05, ax=axes[1])
    #axes[1].legend(title='',frameon=False, loc='upper right', bbox_to_anchor=(1.5, 1))

    plt.suptitle(f'rat: {rat}, mean +/- {ci}% ci\n{np.unique(all_seg_data.nwb_file_name)}', y=1.2)
    sns.despine()
    
    if save_fig:
        fig_name = f'stable_task_stay_switch_four_segs_learning_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_ylim{y_lim[1]}'
        plt.savefig(f'{fig_path}{rat}_{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5) 

def plot_all_all_seg_ahbeh_metric_switches_only_line_format(rats, all_rat_all_seg_ahbeh_metric, stable_nwb_file_names_dict, ci=95, y_lim=None,
                                                 fig_path = None, save_fig = False, connect_lines=False, figwidth=6,figheight=4):
    '''stable only'''
#     Rainbow colors
#     custom_colors_by_rat = [sns.color_palette('Paired')[0:2],
#                             sns.color_palette('Paired')[2:4],
#                             sns.color_palette('Paired')[4:6],
#                             sns.color_palette('Paired')[6:8],
#                             sns.color_palette('Paired')[8:10],]
    custom_colors_by_rat = iter(cm.tab20b([0,.8, .85, .1, .05]))
    
    for i,rat in enumerate(rats):
        all_seg_data = all_rat_all_seg_ahbeh_metric[rat][0]
        # make it stable only
        all_seg_data = all_seg_data[all_seg_data.nwb_file_name.isin(stable_nwb_file_names_dict[rat])]
        all_seg_data = all_seg_data[all_seg_data.stem_switch == True]

        if y_lim is not None:
            ylim = y_lim
        else:
            ylim = (0,30)

        if i==0:
#             fig, axes = plt.subplots(ncols=1, figsize=(9,5.5), sharex=True, sharey=False)
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(figwidth, figheight))

        
        # extract variables from the dict of all-rat data
        summary_metric = all_rat_all_seg_ahbeh_metric[rat][1]
        if summary_metric == 'quantile':
            quantile = all_rat_all_seg_ahbeh_metric[rat][2]
            summary_metric += str(quantile)
        restrict_to_within_patch = all_rat_all_seg_ahbeh_metric[rat][3]
        
        #map back to int labels for segment
        seg_label_to_num = {'first_leaf':1, 'first_stem':2, 'second_stem':3,'last_leaf':4}
        all_seg_data.segment = all_seg_data.segment.map(seg_label_to_num)
        
        custom_color = next(custom_colors_by_rat)
        
        #hue='stem_switch',
        #palette=sns.color_palette('dark')
        #sns.violinplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='stem_switch', hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], cut=0, scale='area', inner='quartile', alpha=.05, dodge=.4, palette=sns.color_palette('pastel'), ax=axes)
        #sns.pointplot(data=all_seg_data, x='segment', y='ahbeh_metric',  ci=ci, join=False,  order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], palette=custom_colors_by_rat[i], capsize=.05, ax=axes)
        sns.lineplot(x=all_seg_data.segment+.05*(i+1)-.15, y=all_seg_data.ahbeh_metric, linestyle='', ci=ci,marker='o', err_style='bars',
                     markersize=5, label=f'Rat {rat.capitalize()[0]}', color=custom_color)
        if connect_lines:
            sns.lineplot(x=all_seg_data.segment+.05*(i+1)-.15, y=all_seg_data.ahbeh_metric, linestyle='-',alpha=.2, size=.1, ci=ci,marker='o', err_style='bars',  color=custom_color,legend=False)

        hstat, pval = stats.kruskal(all_seg_data[all_seg_data.segment == 1].ahbeh_metric,
                                    all_seg_data[all_seg_data.segment == 2].ahbeh_metric,
                                    all_seg_data[all_seg_data.segment == 3].ahbeh_metric,
                                    all_seg_data[all_seg_data.segment == 4].ahbeh_metric,)
        print(f'{rat.capitalize()}')
        print(f'\nKW test between four segs\n  hstat: {hstat}\n  pval: {pval}\n')
        if pval < .01:
            wresults = stats.ranksums(all_seg_data[all_seg_data.segment == 1].ahbeh_metric,
                                    all_seg_data[all_seg_data.segment == 2].ahbeh_metric,)
            print(f'Wrank test between 1 to 2 segs\n  stat: {wresults.statistic}\n  pval: {wresults.pvalue}\n')
            print(f'n_seg1={len(all_seg_data[all_seg_data.segment == 1].ahbeh_metric)}')
            print(f'n_seg2={len(all_seg_data[all_seg_data.segment == 2].ahbeh_metric)}')

            wresults = stats.ranksums(all_seg_data[all_seg_data.segment == 2].ahbeh_metric,
                                    all_seg_data[all_seg_data.segment == 3].ahbeh_metric,)
            print(f'Wrank test between 2 to 3 segs\n  stat: {wresults.statistic}\n  pval: {wresults.pvalue}\n')
            print(f'n_seg2={len(all_seg_data[all_seg_data.segment == 2].ahbeh_metric)}')
            print(f'n_seg3={len(all_seg_data[all_seg_data.segment == 3].ahbeh_metric)}')
            
            wresults = stats.ranksums(all_seg_data[all_seg_data.segment == 3].ahbeh_metric,
                                    all_seg_data[all_seg_data.segment == 4].ahbeh_metric,)
            print(f'Wrank test between 2 to 3 segs\n  stat: {wresults.statistic}\n  pval: {wresults.pvalue}\n')
            print(f'n_seg3={len(all_seg_data[all_seg_data.segment == 3].ahbeh_metric)}')
            print(f'n_seg4={len(all_seg_data[all_seg_data.segment == 4].ahbeh_metric)}')

            wresults = stats.ranksums(all_seg_data[all_seg_data.segment == 1].ahbeh_metric,
                                    all_seg_data[all_seg_data.segment == 4].ahbeh_metric,)
            print(f'Wrank test between 1 to 4 segs\n  stat: {wresults.statistic}\n  pval: {wresults.pvalue}\n')
            print(f'n_seg1={len(all_seg_data[all_seg_data.segment == 1].ahbeh_metric)}')
            print(f'n_seg4={len(all_seg_data[all_seg_data.segment == 4].ahbeh_metric)}')
            
    axes.set_ylim(ylim)
    #axes.legend(rats,frameon=False)
    axes.set_ylabel(f'Max. Non-local Distance [cm]', loc='center')
    axes.set_xlabel('Switch trial subsection', y=-20)
    axes.set_xlim([.5,4.5])
    axes.spines['bottom'].set_bounds(1,4)
    #axes.set_xticklabels(['First choice', 'Second choice', 'Third choice', 'Final segment'])
    axes.set_xticks([1,2,3,4], labels=[1,2,3,4]) #['First\nchoice', 'Second\nchoice', 'Third\nchoice', 'Final\nsegment'])
    #sns.violinplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='stem_switch', hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], cut=0, scale='area', inner='quartile', alpha=.05, dodge=.4, palette=sns.color_palette('pastel'), ax=axes[1])
    #sns.pointplot(data=all_seg_data, x='segment', y='avg_ahbeh', hue='stem_switch', ci=95, join=False, dodge=.4, hue_order=[False, True], order=['first_leaf', 'first_stem', 'second_stem','last_leaf'], palette=sns.color_palette('dark'),capsize=.05, ax=axes[1])
    hands, labs = axes.get_legend_handles_labels()
    axes.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 0.9))
    #print(hands, labs)
#     axes.legend(handles = hands, labels=[f'{rats[0]}, False', f'{rats[0]}, True',
#                 f'{rats[1]}, False', f'{rats[1]}, True',
#                 f'{rats[2]}, False', f'{rats[2]}, True',
#                 f'{rats[3]}, False', f'{rats[3]}, True',
#                 f'{rats[4]}, False', f'{rats[4]}, True',],
#                 title='rat, stem_switch',frameon=False, loc='upper right', bbox_to_anchor=(1.4, 0.9))

    plt.suptitle(f'{rats}\nstable task, in_patch_only: {restrict_to_within_patch}\n', y=1.01, fontsize=6)
    #plt.suptitle(f'rat: {rat}, mean +/- 95% ci\n{np.unique(all_seg_data.nwb_file_name)}', y=1.2)
    sns.despine(offset=5)
    #plt.show()

    if save_fig:
        fig_name = f'all_rats_reformat_stable_task_all_segs_switch_only_line{connect_lines}_ahbeh_metric{summary_metric}_inpatchonly{restrict_to_within_patch}_ylim{ylim[1]}_w{figwidth}_h{figheight}'
        plt.savefig(f'{fig_path}{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5)
        # save_figure(fig_path, fig_name, facecolor=None, transparent=True, pad_inches=.5, save_png=False)