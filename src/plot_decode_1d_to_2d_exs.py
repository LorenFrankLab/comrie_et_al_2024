import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns

from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

from paths import to_local # for dandi based file reconstruction


from ripple_detection.core import gaussian_smooth

# custom schema
from alison_position import PosValidTimesToEpoch
from alison_decoding import ClusterlessResults, ClusterlessAcausalResultsSummary
from alison_decoding import _align_pos_linpos_marks_to_interval
from plot_decode import trial_to_time_slice
from fig_helpers import set_figure_defaults, TWO_COLUMN

def plot_event_v3_1_mua_ahbeh_speed(event_dict,
                  subject_epoch_data, linear_position_df, position_df, acausal_results_summary,fig_path,
                  mua=None,
                  save_fig=False,
                  nonlocal_cmap = 'custom', default_cmap='PuBu', shading_named_color = 'slateblue',arrow_color = 'violet',
                  peri_nonlocal_time=.2, use_manual=True, extra_hpd =False, show_cbar_ticks=True,min_nonlocal_duration_s = 0.02,
                    between_bin_buffer_s = 0.004):
    '''Plot 1d decoding data projected to 2d
    with colors for actual and decoded nonlocal position
    color nonlocal position by time to illustrate sequence
    for a longer time snippet than just the nonlocal event, plot mua, ahead behind distance, and speed
    highlight the nonlocal time on the subplots below the position decoding plot
    also keep track of relevant rat, day, epoch, trial, and time information
    '''
    set_figure_defaults()

    nwb_file_name = event_dict['nwb_file_name']
    # interval_list_name = event_dict['interval_list_name']
    epoch = event_dict['epoch']
    trial = event_dict['trial']
    time_slice = slice(event_dict['time_slice_start'],event_dict['time_slice_stop'])
    is_first_seg_of_trial = event_dict['is_first_seg_of_trial']
    is_last_seg_of_trial = event_dict['is_last_seg_of_trial']
    e = event_dict['event_num_in_trial']
    if use_manual==True:
        event_start_t = event_dict['event_start_t_manual']
        event_stop_t = event_dict['event_stop_t_manual']
    elif use_manual==False:
        event_start_t = event_dict['event_start_t']
        event_stop_t = event_dict['event_stop_t']
    
    start_t = position_df.iloc[time_slice].index[0]
    stop_t = position_df.iloc[time_slice].index[-1]
    subject_epoch_data = subject_epoch_data.reset_index()
    subject_epoch_data_filtered = subject_epoch_data[(subject_epoch_data.time>=start_t) & (subject_epoch_data.time<=stop_t)]
    if is_first_seg_of_trial:
        subject_epoch_data_filtered_seg = subject_epoch_data_filtered[subject_epoch_data_filtered.is_first_seg_of_trial==True]
    elif is_last_seg_of_trial:
        subject_epoch_data_filtered_seg = subject_epoch_data_filtered[subject_epoch_data_filtered.is_last_seg_of_trial==True]
    else:
        raise Exception('only first OR last seg can be true, and one must be true')
    
    if event_start_t < subject_epoch_data_filtered_seg.time.iloc[0]:
        print(f'WARNING: Event start {event_start_t} is outside valid time {subject_epoch_data_filtered_seg.time.iloc[0]}\n')
    if event_stop_t > subject_epoch_data_filtered_seg.time.iloc[-1]:
        print(f'WARNING: Event stop {event_stop_t} is outside valid time {subject_epoch_data_filtered_seg.time.iloc[-1]}\n')
    
    duration = event_stop_t - event_start_t #valid_segment_durations.values[e]
#     print(duration)
#     print(event_stop_t, event_start_t)
    fig, axes = plt.subplots(nrows=4, ncols=1,
                             figsize=(TWO_COLUMN/2,(TWO_COLUMN/2)*(1.5)),
                             gridspec_kw={'height_ratios':[5,.8,.8,.8]}, )
    axes[0].set_title(f'{nwb_file_name[0:-5]}, {epoch}, {trial}, E: {e}, first seg {True if is_first_seg_of_trial else False}\nNon-local event example, duration {np.round(duration*1000,2)} ms',y=1.0,fontsize=4)
    print(f'{nwb_file_name[0:-5]}, {epoch}, {trial}, E: {e}, first seg {True if is_first_seg_of_trial else False}')
          
    # Plot track occupied positions
    axes[0].scatter(linear_position_df['projected_x_position'].iloc[::50],
                    linear_position_df['projected_y_position'].iloc[::50],
                    s=50, color='lightgrey', zorder=0, label='_')
    axes[0].set_aspect('equal')

    # plot actual position during trial run time
    axes[0].plot(linear_position_df['projected_x_position'].iloc[time_slice],
                 linear_position_df['projected_y_position'].iloc[time_slice],
                 color=arrow_color, lw=1, zorder=1, label='Trial path', linestyle='--')
    #draw arrow over last 50 linpos points to indicate motion direction for this trial
    arrow_start = (linear_position_df['projected_x_position'].iloc[time_slice].values[-50],
                   linear_position_df['projected_y_position'].iloc[time_slice].values[-50])
    arrow_end = (linear_position_df['projected_x_position'].iloc[time_slice].values[-1],
                 linear_position_df['projected_y_position'].iloc[time_slice].values[-1])
    arrow = FancyArrowPatch(arrow_start, arrow_end, arrowstyle='-|>',color=arrow_color, lw=1,mutation_scale=15,zorder=20000,)
    axes[0].add_patch(arrow)

    # make a scale bar and label it
    axes[0].axis('off')
    scale_bar_length = 25 #cm
    x_start = 200
    y_position = 58
    tick_length = 3
    axes[0].plot([x_start, x_start+scale_bar_length], [y_position, y_position], color='lightgrey', lw=1) #, zorder=5000)
    axes[0].text(x_start+scale_bar_length / 2, y_position-6, f'{scale_bar_length} cm', ha='center', va='top', color='lightgrey')
    axes[0].plot([x_start, x_start], [y_position-tick_length, y_position+tick_length*0], color='lightgrey', lw=1)
    axes[0].plot([x_start+scale_bar_length, x_start+scale_bar_length], [y_position-tick_length, y_position+tick_length*0], color='lightgrey', lw=1)

    offset=10
    axes[0].set_xlim(linear_position_df['projected_x_position'].min()-offset,linear_position_df['projected_x_position'].max()+offset)
    axes[0].set_ylim(linear_position_df['projected_y_position'].min()-offset,linear_position_df['projected_y_position'].max()+offset)

    # get event times of df
    event_filtered_data = subject_epoch_data_filtered_seg[(subject_epoch_data_filtered_seg.time>=event_start_t) &
                                                              (subject_epoch_data_filtered_seg.time<=event_stop_t)]
    # plot event actual and mental positions
    color_times = event_filtered_data.time - event_start_t

    # create custom colormap
#             colors = ['slateblue', 'dodgerblue', 'mediumseagreen', 'greenyellow', 'lightyellow']
#             custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors)
    new_cmap = plt.get_cmap('PuBu')(np.linspace(0.3,1,100))#.3,1,100))
    custom_cmap = LinearSegmentedColormap.from_list('custom_PuBu',new_cmap)

    # plot nonlocal actual and mental pos
    scatter1 = axes[0].scatter(event_filtered_data['actual_2d_x_projected_position'],
                                event_filtered_data['actual_2d_y_projected_position'],
                                s=35, marker='o', facecolors='none',
                                c='magenta', label='Rat pos.') #c=color_times, cmap='RdPu',  )
    scatter2 = axes[0].scatter(event_filtered_data['mental_2d_x_projected_position'],
                                event_filtered_data['mental_2d_y_projected_position'],
                                s=50, marker='o', facecolors='none',
                                c=color_times, cmap=custom_cmap if nonlocal_cmap=='custom' else default_cmap, 
                                label='Represented\nnon-local pos.', vmin=0)
    # legend for actual and mental pos
    legend = axes[0].legend(bbox_to_anchor=(-.03,.5), loc='lower left', frameon=False) #-.22, .15,or if up left -.22 and .55 
    text_colors = ['dimgrey','magenta', shading_named_color] #slateblue
    for label, color in zip(legend.get_texts(),text_colors):
        label.set_color(color)

    cbar_ax2 = fig.add_axes([.45, .48,.15,.02]) #x as .08 or .45 works #left bottom width height #  .45, .42,.15,.01
#     cbar2 = plt.colorbar(scatter2, cax=cbar_ax2, orientation='horizontal')
    # match cbars fuly
    norm = mcolors.Normalize(vmin=0, vmax = duration)
    scatter2bar = cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    cbar2 = plt.colorbar(scatter2bar, cax=cbar_ax2, orientation='horizontal')

    cbar2.set_label(f'ms',color=shading_named_color,alpha=1)
    cbar2.outline.set_visible(False)
    cbar2.ax.spines['left'].set_visible(False)
    cbar2.ax.tick_params(color=shading_named_color)#, alpha=.5) #slateblue
    print(duration)
    cbar2.set_ticks([0,duration])
    cbar2.set_ticklabels([0,int(np.round(1000*duration,0))])
#     cbar2.ax.spines['bottom'].set_visible(False)
    if show_cbar_ticks==False:
        cbar2.ax.xaxis.set_ticks_position('none')
    for label in cbar2.ax.get_xticklabels():
        label.set_color(shading_named_color) #slateblue

    # trial_times = acausal_results_summary.iloc[time_slice].index  - acausal_results_summary.iloc[time_slice].index.values[0]

    snippet_time_start = event_start_t - peri_nonlocal_time
    snippet_time_stop = event_stop_t + peri_nonlocal_time

    snippet_times = acausal_results_summary.loc[snippet_time_start:snippet_time_stop].index - snippet_time_start # the array of x axis times
    print(len(snippet_times))

    # or use .loc instead
    mua_snippet = mua.firing_rate[(mua.firing_rate.index >= snippet_time_start) & (mua.firing_rate.index <= snippet_time_stop) ]
    mua_tet_snippet = acausal_results_summary.multiunit_firing_rate[(acausal_results_summary.multiunit_firing_rate.index >= snippet_time_start) & 
                                                                    (acausal_results_summary.multiunit_firing_rate.index <= snippet_time_stop)]
    head_speed_snippet = acausal_results_summary.head_speed[(acausal_results_summary.head_speed.index >= snippet_time_start) & 
                                                                    (acausal_results_summary.head_speed.index <= snippet_time_stop)]
    ahbeh_snippet = acausal_results_summary.ahead_behind_distance[(acausal_results_summary.ahead_behind_distance.index >= snippet_time_start) & 
                                                                    (acausal_results_summary.ahead_behind_distance.index <= snippet_time_stop)]
    
    #for ahebeh  only want the hpd valid times
    if extra_hpd:
        acausal_results_summary_hpd = acausal_results_summary[acausal_results_summary.spatial_coverage_50_hpd <= 50]
        snippet_times_hpd = acausal_results_summary_hpd.loc[snippet_time_start:snippet_time_stop].index - snippet_time_start
        ahbeh_snippet_hpd = acausal_results_summary_hpd.ahead_behind_distance[(acausal_results_summary_hpd.ahead_behind_distance.index >= snippet_time_start) & 
                                                                    (acausal_results_summary_hpd.ahead_behind_distance.index <= snippet_time_stop)]
        axes[2].scatter(snippet_times_hpd, ahbeh_snippet_hpd, color='darkgrey',alpha=1, s=4, zorder=5000)    #'darkgrey' 
    else:
        axes[2].scatter(snippet_times, ahbeh_snippet, color='darkgrey',alpha=1, s=4)   
    axes[2].set_ylabel(f'cm',)#rotation=0)
    axes[2].axhline(0,color='magenta',alpha=.3,linewidth=1,zorder=0, linestyle='-')
    
    # ymin2,ymax2 = axes[2].get_ylim()
    
    print(len(mua_snippet))
    if mua is not None:
        axes[1].fill_between(snippet_times, mua_snippet, color='dimgrey',alpha=1)    
    else:
        axes[1].fill_between(snippet_times, mua_tet_snippet ,color='darkgrey',alpha=1)            
    axes[1].set_ylabel('MUA\n(spikes/s)')
    _,ymax = axes[1].get_ylim()
    ymax1 = int(np.ceil(ymax/100.0))*100
#     if ymax1 < 500:
#         ymax1=500
    axes[1].set_ylim([0,ymax1])
    axes[1].set_yticks([0,ymax1])
#     axes[1].set_yticklabels([0,ymax1])

    axes[3].fill_between(snippet_times, head_speed_snippet, color='darkgrey', alpha=1)
    axes[3].set_ylabel('Speed\n(cm/s)')
    axes[3].set_xlabel('Time (s)',labelpad=6)
    
    shaded_times = acausal_results_summary.loc[event_start_t:event_stop_t].index - snippet_time_start #event_filtered_data.time - snippet_time_start

    _,ymax3 = axes[3].get_ylim()
    if ymax3 <=55:
        ymax3 = 55
        axes[3].set_ylim(0,ymax3)
#     axes[1].vlines(shaded_times, ymin=0, ymax=ymax1, color=shading_named_color, alpha =.2, zorder=-100) #lw=2,) # zorder=0)
#     axes[3].vlines(shaded_times, ymin=0, ymax=ymax3, color=shading_named_color, alpha =.2, zorder=-100 ) #lw=2, ) #zorder=0)
#     axes[2].vlines(shaded_times, ymin=ymin2, ymax=ymax2, color=shading_named_color, alpha =.2, zorder=-100) #lw=2,) # zorder=0)
#     print(len(shaded_times),shaded_times[0],shaded_times[-1])

    axes[1].axvspan(shaded_times[0], shaded_times[-1], ymin=0, ymax=1, color=shading_named_color, alpha=0.5, zorder=-100, edgecolor='none',linestyle='None')
    axes[3].axvspan(shaded_times[0], shaded_times[-1], ymin=0, ymax=1, color=shading_named_color, alpha=0.5, zorder=-100, edgecolor='none',linestyle='None')
    axes[2].axvspan(shaded_times[0], shaded_times[-1], ymin=0, ymax=1, color=shading_named_color, alpha=0.5, zorder=-100, edgecolor='none',linestyle='None')


    axes[1].set_xlim(0,snippet_time_stop-snippet_time_start)
    axes[1].set_xticks([])
    axes[1].set_xticklabels([],)
    axes[1].set_xlabel('')
    
    axes[2].set_xlim(0,snippet_time_stop-snippet_time_start)
    axes[2].set_xticks([])
    axes[2].set_xticklabels([],)
    axes[2].set_xlabel('')

    axes[3].set_xlim(0,snippet_time_stop-snippet_time_start)
    axes[3].set_ylim(bottom=0)
    

    for ax in axes.flat:
        sns.despine(offset=5, ax=ax)

    axes[1].spines['bottom'].set_visible(False)
    axes[2].spines['bottom'].set_visible(False)

    plt.subplots_adjust(hspace=0.1)

    pos_ax1 = axes[1].get_position()
    pos_ax2 = axes[2].get_position()
    pos_ax3 = axes[3].get_position()
    new_left = 0.25
    new_width = .6

    axes[1].set_position([new_left, pos_ax1.y0, new_width, pos_ax1.height])
    axes[2].set_position([new_left, pos_ax2.y0, new_width, pos_ax2.height])
    axes[3].set_position([new_left, pos_ax3.y0, new_width, pos_ax3.height])

    if save_fig:
        fig_name = f'snippet_ex_decode_MUA_AHBEH_first{is_first_seg_of_trial}_{nwb_file_name[0:-5]}_{epoch}_{trial}_event{e}_mindur{min_nonlocal_duration_s}_binbuff{between_bin_buffer_s}_perievent{peri_nonlocal_time}_shade{shading_named_color}_{event_start_t}_{event_stop_t}'
        plt.savefig(f'{fig_path}{fig_name}.pdf', format='pdf', bbox_inches="tight", pad_inches=.5, dpi=300)    

    plt.show()

def get_multiunit_population_firing_rate_sum(multiunit, sampling_frequency,
                                         smoothing_sigma=0.015):
    '''Calculates the multiunit population firing rate.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    sampling_frequency : float
        Number of samples per second.
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.


    Returns
    -------
    multiunit_population_firing_rate : ndarray, shape (n_time,)

    '''
    return gaussian_smooth(multiunit.sum(axis=1) * sampling_frequency,
                           smoothing_sigma, sampling_frequency)

def _summarize_mua_sum(key, marks_xr):
    SAMPLING_FREQUENCY = key['sampling_rate']
    multiunit_spikes = (np.any(~np.isnan(marks_xr.values), axis=1)
                        ).astype(float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate_sum(
            multiunit_spikes, SAMPLING_FREQUENCY), index=marks_xr.time, 
        columns=['firing_rate'])
    return multiunit_firing_rate #.to_numpy().flatten()

# load some data from one epoch
def load_decode_ex_data(subject_id, nwb_file_name, epoch, all_rat_big_dfs_stable):
    '''Load up many useful bits of data for a subject, day, epoch
    
    Params
    ---------
    subject_id: str rat name
    nwb_file_name: str
    epoch: int
    all_rat_big_dfs_stable: dict of pd.DataFrame per subject with lots of aligned and parsed data

    Returns
    ----------
    acausal_results_summary: decoding info extracted as pd.DataFrame
    position_df: pd.DataFrame of 2d position
    linear_position_df: pd.DataFrame of linearized position
    multiunit_spike_rate_sum: updated mua array (overall, not per tet)
    time_slices: dict of slices for subsets of trial time
    subject_epoch_data: pd.DataFrame trimmed to relevant data
    '''
#     subject_id = 'j16'
#     nwb_file_name = 'j1620210707_.nwb'
    interval_list_name = f'pos {epoch-1} valid times' 

    classifier_param_name = "default_decoding_gpu"

    run = True
    well= False
    pre_sec = 0
    post_sec = 0 

    epoch = (PosValidTimesToEpoch() & {'nwb_file_name':nwb_file_name, 'pos_interval_list_name':interval_list_name}).fetch1('epoch')

    cr_key = (ClusterlessResults & {'nwb_file_name':nwb_file_name, 'interval_list_name':interval_list_name, 'classifier_param_name':classifier_param_name}).fetch1('KEY')

    # get raw decode results
    # filename = (ClusterlessResults & cr_key).fetch1('clusterless_results_path') #of form: save_results_path + f'{nwb_file_name}_{track_graph_name}_{interval_list_name}_1D.nc'
    # results = xr.open_dataset(filename)
    filename = (ClusterlessResults & cr_key).fetch1('clusterless_results_path') #of form: save_results_path + f'{nwb_file_name}_{track_graph_name}_{interval_list_name}_1D.nc'
    # data-sharing / DANDI compatibility (added for code release): fall back to the reconstructed
    # copy under DATA_DIR when the original /stelmo path isn't present
    try:
        results = xr.open_dataset(filename)
    except (FileNotFoundError, OSError):
        results = xr.open_dataset(to_local(filename))

    # alternative decode results with ahbeh and mua parsed
    acausal_results_summary = (ClusterlessAcausalResultsSummary & cr_key).fetch1_dataframe()

    # position data, aligned to results and acausal results summary
    position_df, linear_position_df,  marks_xr= _align_pos_linpos_marks_to_interval(cr_key)

    multiunit_spike_rate_sum = _summarize_mua_sum(cr_key, marks_xr)

    # get trial run time slices
    time_slices = {}
    for trial_number in range(1,180):
        time_slice = trial_to_time_slice(nwb_file_name, results, epoch, trial_number, run, well, pre_sec, post_sec)
        time_slices[trial_number] = time_slice

    # from earlier data parsing into big df
    subject_data = all_rat_big_dfs_stable[subject_id]
    subject_day_data = subject_data[subject_data['nwb_file_name']==nwb_file_name]
    subject_epoch_data = subject_day_data[subject_day_data['interval_list_name']==interval_list_name]

    return acausal_results_summary, position_df, linear_position_df, multiunit_spike_rate_sum, time_slices, subject_epoch_data