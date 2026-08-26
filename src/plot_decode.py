import os
import copy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from trajectory_analysis_tools import (
#     get_HPD_spatial_coverage,
#     get_highest_posterior_threshold,
#     maximum_a_posteriori_estimate,
# )

from spyglass.common.common_interval import interval_list_intersect

from alison_behav import TrialsInfoByEpoch


def trial_to_time_slice(
    nwb_copy_file_name,
    results,
    epoch_number,
    trial_number,
    run=True,
    well=True,
    pre_sec=0,
    post_sec=0,
):
    in_times = (
        TrialsInfoByEpoch.ByTrial()
        & {"nwb_file_name": nwb_copy_file_name, "epoch": epoch_number}
    ).fetch("poke_in_ts")
    out_times = (
        TrialsInfoByEpoch.ByTrial()
        & {"nwb_file_name": nwb_copy_file_name, "epoch": epoch_number}
    ).fetch("poke_out_ts")
    if run and well:
        trial_indices = np.where(
            np.logical_and(
                results.time.values > out_times[trial_number - 1] - pre_sec,
                results.time.values < out_times[trial_number] + post_sec,
            )
        )[0]
        time_slice = slice(trial_indices[0], trial_indices[len(trial_indices) - 1])
    elif run and not well:
        run_indices = np.where(
            np.logical_and(
                results.time.values > out_times[trial_number - 1] - pre_sec,
                results.time.values < in_times[trial_number] + post_sec,
            )
        )[0]
        time_slice = slice(run_indices[0], run_indices[len(run_indices) - 1])
    elif well and not run:
        well_indices = np.where(
            np.logical_and(
                results.time.values > in_times[trial_number] - pre_sec,
                results.time.values < out_times[trial_number] + post_sec,
            )
        )[0]
        time_slice = slice(well_indices[0], well_indices[len(well_indices) - 1])
    # print('nwb: ', nwb_copy_file_name, '\nepoch: ', epoch_number, '\ntrial: ', trial_number, '\nslice for run ', run, ' and well ', well, ' with pre sec ', pre_sec, ' post sec ', post_sec, ': ', time_slice)
    return time_slice


def plot_classifier(
    time_slice,
    results,
    environment,
    position_info,
    linear_position_df,
    ahead_behind_distance,
    multiunit_firing_rate,
    multiunit_high_synchrony_times,
    nwb_copy_file_name,
    epoch_number,
    trial_number,
    run,
    well,
    pre_sec,
    post_sec,
    actual_position_edge_id,
    mental_position_edge_id,
    theta_times,
    theta_data,
    swr_times,
    swr_data,
    kk_ripple_times,
    win_len,
    cmap="bone_r",
    figsize=(30, 20),
):

    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axes = plt.subplots(
        10,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 5, 1, 1, 1, 1, 1]},
    )

    time = results.isel(time=time_slice).time

    posterior = (
        results.isel(time=time_slice)
        .acausal_posterior.sum("state")
        .where(environment.is_track_interior_)
    )

    actual_segment = actual_position_edge_id  # [:len(actual_position_edge_id)-2]
    mental_segment = mental_position_edge_id
    nonlocal_by_segment = np.not_equal(actual_segment, mental_segment)
    ahead_or_not = np.asarray(ahead_behind_distance > 0)
    nonlocal_ahead = np.logical_and(nonlocal_by_segment, ahead_or_not)
    nonlocal_behind = np.logical_and(nonlocal_by_segment, ~ahead_or_not)
    reward_bool = (
        TrialsInfoByEpoch.ByTrial()
        & {
            "nwb_file_name": nwb_copy_file_name,
            "epoch": epoch_number,
            "trial_number_by_epoch": trial_number,
        }
    ).fetch1("reward")

    ax = 0  # speed
    axes[ax].fill_between(
        time,
        position_info.iloc[time_slice].head_speed.values.squeeze(),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[ax].set_ylim([0, max(position_info.head_speed.values.squeeze())])
    axes[ax].set_title("Speed")
    axes[ax].set_ylabel("Speed [cm / s]")
    axes[ax].set_xlabel("Time [s]")

    axes[ax].set_title(
        f"nwb: {nwb_copy_file_name[:-4]}\nepoch: {epoch_number}, trial: {trial_number}\nincludes run: {run}, and well: {well}\nplus {pre_sec} sec before, {post_sec} sec after\nis this trial rewarded: {reward_bool}",
        fontsize=14,
    )

    ax += 1  # swr
    lim = max(np.abs(swr_data))
    swr_times_sliced = swr_times[
        np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])
    ]
    swr_data_sliced = swr_data[
        np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])
    ]
    axes[ax].plot(swr_times_sliced, swr_data_sliced)
    axes[ax].set_ylim([-lim, lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title(
        "Ripple filtered LFP - NB: THIS IS NOT REFERENCED CORRECTLY RIGHT NOW"
    )

    cur_kk_ripples = interval_list_intersect(
        np.asarray(kk_ripple_times), np.asarray([(time[0], time[-1])])
    )
    for start_time, end_time in cur_kk_ripples:
        axes[ax].axvspan(start_time, end_time, color="red", alpha=0.3, zorder=100)

    ax += 1  # theta
    lim = max(np.abs(theta_data))
    theta_times_sliced = theta_times[
        np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])
    ]
    theta_data_sliced = theta_data[
        np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])
    ]
    axes[ax].plot(theta_times_sliced, theta_data_sliced)
    axes[ax].set_ylim([-lim, lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title("Theta filtered LFP")

    ax += 1  # mua and HSE
    axes[ax].fill_between(
        multiunit_firing_rate.iloc[time_slice].index.values,
        multiunit_firing_rate.iloc[time_slice].values.squeeze(),
        color="black",
    )
    axes[ax].set_ylabel("Firing Rate\n[spikes / s]")
    axes[ax].set_title("Multiunit")
    axes[ax].set_ylim((0.0, np.max(np.asarray(multiunit_firing_rate))))

    cur_multiunit_HSE = interval_list_intersect(
        np.asarray(multiunit_high_synchrony_times), np.asarray([(time[0], time[-1])])
    )

    for start_time, end_time in cur_multiunit_HSE:
        axes[ax].axvspan(start_time, end_time, color="blue", alpha=0.3, zorder=100)

    ax += 1  # decode
    (posterior.plot(x="time", y="position", ax=axes[ax], robust=True, cmap=cmap))
    axes[ax].scatter(
        time,
        linear_position_df.iloc[time_slice].linear_position.values,
        s=1,
        color="magenta",
        zorder=10,
    )
    axes[ax].scatter(
        time, maximum_a_posteriori_estimate(posterior), marker="x", color="green", s=1
    )

    ax += 1  # track segment
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}
    axes[ax].scatter(
        time,
        pd.Series(mental_segment[time_slice]),
        c=pd.Series(mental_segment[time_slice]).map(segment2color),
        s=3,
    )
    axes[ax].scatter(time, pd.Series(actual_segment[time_slice]), s=0.5, color="black")
    axes[ax].set_ylim([0, 8])
    axes[ax].set_ylabel("Track segment ID")
    axes[ax].set_title("track segment mental (color), actual (black)")

    ax += 1  # nonlocal amount smooth
    axes[ax].plot(
        time,
        pd.Series(nonlocal_by_segment).rolling(win_len).mean()[time_slice],
        color="blue",
    )
    axes[ax].plot(
        time,
        pd.Series(nonlocal_ahead).rolling(win_len).mean()[time_slice],
        color="green",
    )
    axes[ax].plot(
        time,
        pd.Series(nonlocal_behind).rolling(win_len).mean()[time_slice],
        color="grey",
    )
    # axes[ax].set_ylim([0,1])
    axes[ax].set_ylabel("Prop. nonlocal by seg.")
    axes[ax].set_title(
        "smoothed amount of nonlocal by segment (blue), nonlocal ahead (green), nonlocal behind (grey), over window = "
        + str(win_len)
    )

    ax += 1  # ahead behind
    axes[ax].plot(time, ahead_behind_distance[time_slice], color="black", linewidth=2)
    axes[ax].axhline(0, color="magenta", linestyle="--")
    axes[ax].set_title("Mental distance ahead or behind animal")
    axes[ax].set_ylabel("Distance [cm]")
    max_dist = 50  # np.max(np.abs(ahead_behind_distance)) + 5
    axes[ax].set_ylim((-max_dist, max_dist))
    axes[ax].text(time[0], max_dist - 1, "Ahead", color="grey", va="top")
    axes[ax].text(time[0], -max_dist + 1, "Behind", color="grey")

    ax += 1  # hpd
    hpd_threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
    spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    axes[ax].plot(posterior.time, spatial_coverage, color="black", linewidth=2)
    hpd_threshold = get_highest_posterior_threshold(posterior, coverage=0.50)
    spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    axes[ax].plot(posterior.time, spatial_coverage, color="grey", linewidth=2)
    axes[ax].set_ylim([0, 100])
    axes[ax].set_title("Spatial coverage by .95 & .50 highest posterior density region")
    axes[ax].set_ylabel("Distance [cm]")

    ax += 1  # state
    results.isel(time=time_slice).acausal_posterior.sum("position").plot(
        x="time", hue="state", ax=axes[ax]
    )

    # ax += 1 #smooth ahead and nonlocal, behind and nonlocal
    # axes[ax].plot(time, pd.Series(ahead_behind_distance[ahead_behind_distance>0]).rolling(win_len).mean()[time_slice], color='green')
    # axes[ax].plot(time, pd.Series(np.abs(ahead_behind_distance[ahead_behind_distance<=0])).rolling(win_len).mean()[time_slice], color='grey')
    # axes[ax].set_ylabel("")
    # axes[ax].set_title('smoothed ahead (green) and behind (grey), over window = '+str(win_len))

    sns.despine()

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(
        -1 * position_info.head_position_x,
        position_info.head_position_y,
        c=linear_position_df.track_segment_id.map(segment2color),
        s=0.5,
        alpha=0.4,
    )
    plt.axis("square")
    plt.scatter(
        -1 * position_info.head_position_x.iloc[time_slice],
        position_info.head_position_y.iloc[time_slice],
        color="black",
        s=0.25,
        zorder=10,
    )
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.xlabel("cm", fontsize=15)
    plt.ylabel("cm", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([0, 250])
    plt.xlim([-250, 0])


def plot_classifier_no_lfp(
    time_slice,
    results,
    environment,
    position_info,
    linear_position_df,
    ahead_behind_distance,
    multiunit_firing_rate,
    multiunit_high_synchrony_times,
    nwb_copy_file_name,
    epoch_number,
    trial_number,
    run,
    well,
    pre_sec,
    post_sec,
    actual_position_edge_id,
    mental_position_edge_id,
    win_len,
    cmap="bone_r",
    figsize=(30, 20),
):

    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axes = plt.subplots(
        8,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 5, 1, 1, 1, 1, 1]},
    )

    time = results.isel(time=time_slice).time

    posterior = (
        results.isel(time=time_slice)
        .acausal_posterior.sum("state")
        .where(environment.is_track_interior_)
    )

    actual_segment = actual_position_edge_id  # [:len(actual_position_edge_id)-2]
    mental_segment = mental_position_edge_id
    nonlocal_by_segment = np.not_equal(actual_segment, mental_segment)
    ahead_or_not = np.asarray(ahead_behind_distance > 0)
    nonlocal_ahead = np.logical_and(nonlocal_by_segment, ahead_or_not)
    nonlocal_behind = np.logical_and(nonlocal_by_segment, ~ahead_or_not)
    reward_bool = (
        TrialsInfoByEpoch.ByTrial()
        & {
            "nwb_file_name": nwb_copy_file_name,
            "epoch": epoch_number,
            "trial_number_by_epoch": trial_number,
        }
    ).fetch1("reward")

    ax = 0  # speed
    axes[ax].fill_between(
        time,
        position_info.iloc[time_slice].head_speed.values.squeeze(),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    #axes[ax].set_ylim([0, max(position_info.head_speed.values.squeeze())])
    axes[ax].set_ylim([0, max(position_info.head_speed[position_info.head_speed.notna()].values.squeeze())])
    axes[ax].set_title("Speed")
    axes[ax].set_ylabel("Speed [cm / s]")
    axes[ax].set_xlabel("Time [s]")

    axes[ax].set_title(
        f"nwb: {nwb_copy_file_name[:-4]}\nepoch: {epoch_number}, trial: {trial_number}\nincludes run: {run}, and well: {well}\nplus {pre_sec} sec before, {post_sec} sec after\nis this trial rewarded: {reward_bool}",
        fontsize=14,
    )

    """
    ax += 1 #swr
    lim = max(np.abs(swr_data))
    swr_times_sliced = swr_times[np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])]
    swr_data_sliced = swr_data[np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])]
    axes[ax].plot(swr_times_sliced, swr_data_sliced)
    axes[ax].set_ylim([-lim,lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title('Ripple filtered LFP - NB: THIS IS NOT REFERENCED CORRECTLY RIGHT NOW')
    
    cur_kk_ripples = interval_list_intersect(
        np.asarray(kk_ripple_times),
        np.asarray([(time[0], time[-1])]))
    for start_time, end_time in cur_kk_ripples:
        axes[ax].axvspan(start_time, end_time, color='red', alpha=0.3, zorder=100)
    
    ax += 1 #theta
    lim = max(np.abs(theta_data))
    theta_times_sliced = theta_times[np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])]
    theta_data_sliced = theta_data[np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])]
    axes[ax].plot(theta_times_sliced, theta_data_sliced)
    axes[ax].set_ylim([-lim,lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title('Theta filtered LFP')
    """

    ax += 1  # mua and HSE
    axes[ax].fill_between(
        multiunit_firing_rate.iloc[time_slice].index.values,
        multiunit_firing_rate.iloc[time_slice].values.squeeze(),
        color="black",
    )
    axes[ax].set_ylabel("Firing Rate\n[spikes / s]")
    axes[ax].set_title("Multiunit")
    axes[ax].set_ylim((0.0, np.max(np.asarray(multiunit_firing_rate))))

    cur_multiunit_HSE = interval_list_intersect(
        np.asarray(multiunit_high_synchrony_times), np.asarray([(time[0], time[-1])])
    )

    for start_time, end_time in cur_multiunit_HSE:
        axes[ax].axvspan(start_time, end_time, color="blue", alpha=0.3, zorder=100)

    ax += 1  # decode
    (posterior.plot(x="time", y="position", ax=axes[ax], robust=True, cmap=cmap))
    axes[ax].scatter(
        time,
        linear_position_df.iloc[time_slice].linear_position.values,
        s=1,
        color="magenta",
        zorder=10,
    )
    axes[ax].scatter(
        time, maximum_a_posteriori_estimate(posterior), marker="x", color="green", s=1
    )

    ax += 1  # track segment
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}
    axes[ax].scatter(
        time,
        pd.Series(mental_segment[time_slice]),
        c=pd.Series(mental_segment[time_slice]).map(segment2color),
        s=3,
    )
    axes[ax].scatter(time, pd.Series(actual_segment[time_slice]), s=0.5, color="black")
    axes[ax].set_ylim([0, 8])
    axes[ax].set_ylabel("Track segment ID")
    axes[ax].set_title("track segment mental (color), actual (black)")

    ax += 1  # nonlocal amount smooth
    axes[ax].plot(
        time,
        pd.Series(nonlocal_by_segment).rolling(win_len).mean()[time_slice],
        color="blue",
    )
    axes[ax].plot(
        time,
        pd.Series(nonlocal_ahead).rolling(win_len).mean()[time_slice],
        color="green",
    )
    axes[ax].plot(
        time,
        pd.Series(nonlocal_behind).rolling(win_len).mean()[time_slice],
        color="grey",
    )
    # axes[ax].set_ylim([0,1])
    axes[ax].set_ylabel("Prop. nonlocal by seg.")
    axes[ax].set_title(
        "smoothed amount of nonlocal by segment (blue), nonlocal ahead (green), nonlocal behind (grey), over window = "
        + str(win_len)
    )

    ax += 1  # ahead behind
    axes[ax].plot(time, ahead_behind_distance[time_slice], color="black", linewidth=2)
    axes[ax].axhline(0, color="magenta", linestyle="--")
    axes[ax].set_title("Mental distance ahead or behind animal")
    axes[ax].set_ylabel("Distance [cm]")
    max_dist = 50  # np.max(np.abs(ahead_behind_distance)) + 5
    axes[ax].set_ylim((-max_dist, max_dist))
    axes[ax].text(time[0], max_dist - 1, "Ahead", color="grey", va="top")
    axes[ax].text(time[0], -max_dist + 1, "Behind", color="grey")

    ax += 1  # hpd
    hpd_threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
    spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    axes[ax].plot(posterior.time, spatial_coverage, color="black", linewidth=2)
    hpd_threshold = get_highest_posterior_threshold(posterior, coverage=0.50)
    spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    axes[ax].plot(posterior.time, spatial_coverage, color="grey", linewidth=2)
    axes[ax].set_ylim([0, 100])
    axes[ax].set_title("Spatial coverage by .95 & .50 highest posterior density region")
    axes[ax].set_ylabel("Distance [cm]")

    ax += 1  # state
    results.isel(time=time_slice).acausal_posterior.sum("position").plot(
        x="time", hue="state", ax=axes[ax]
    )

    # ax += 1 #smooth ahead and nonlocal, behind and nonlocal
    # axes[ax].plot(time, pd.Series(ahead_behind_distance[ahead_behind_distance>0]).rolling(win_len).mean()[time_slice], color='green')
    # axes[ax].plot(time, pd.Series(np.abs(ahead_behind_distance[ahead_behind_distance<=0])).rolling(win_len).mean()[time_slice], color='grey')
    # axes[ax].set_ylabel("")
    # axes[ax].set_title('smoothed ahead (green) and behind (grey), over window = '+str(win_len))

    sns.despine()

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(
        -1 * position_info.head_position_x,
        position_info.head_position_y,
        c=linear_position_df.track_segment_id.map(segment2color),
        s=0.5,
        alpha=0.4,
    )
    plt.axis("square")
    plt.scatter(
        -1 * position_info.head_position_x.iloc[time_slice],
        position_info.head_position_y.iloc[time_slice],
        color="black",
        s=0.25,
        zorder=10,
    )
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.xlabel("cm", fontsize=15)
    plt.ylabel("cm", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([0, 250])
    plt.xlim([-250, 0])


def boxplot_ahead_behind_by_trial(
    nwb_file_name,
    results,
    epoch_number,
    run,
    well,
    ahead_behind_distance,
    pre_sec=0,
    post_sec=0,
    showfliers=True,
    show_switches=True,
    ylim=(-250, 250),
    figsize=(40, 8),
):
    ahead_behinds_dict_by_trial = get_ahead_behind_dict_by_trial(
        nwb_file_name,
        results,
        epoch_number,
        run,
        well,
        ahead_behind_distance,
        pre_sec,
        post_sec,
    )
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(ahead_behinds_dict_by_trial.values(), showfliers=showfliers)
    ax.set_xticklabels(ahead_behinds_dict_by_trial.keys())
    ax.set_ylim(ylim)
    if show_switches:
        trial_info = pd.DataFrame(
            TrialsInfoByEpoch().ByTrial()
            & {"nwb_file_name": nwb_file_name, "epoch": epoch_number}
        )
        stem_switch_trials = [
            trial_info.index[i]
            for i in range(1, len(trial_info))
            if trial_info["stem"][i] != trial_info["stem"][i - 1]
        ]
        conting_switch_trials = [
            trial_info.index[i]
            for i in range(1, len(trial_info))
            if trial_info["contingency_count"][i]
            != trial_info["contingency_count"][i - 1]
        ]
        ax.vlines(stem_switch_trials, -100, 100, color="blue", label="changed stem")
        ax.vlines(
            conting_switch_trials, -100, 100, color="magenta", label="changed conting"
        )
        ax.legend(loc="upper right", facecolor="white")
    ax.hlines(0, 1, len(ahead_behinds_dict_by_trial), color="grey", alpha=0.3)
    ax.set_xlabel("trial")
    ax.set_ylabel("ahead behind distance [cm]")
    ax.set_title(
        f"{nwb_file_name} epoch {epoch_number} trials = {len(ahead_behinds_dict_by_trial)}\nrun={run} well={well}, pre_sec={pre_sec} post_sec={post_sec}\nshowfliers={showfliers} show_switches={show_switches}"
    )
    sns.despine()


def get_ahead_behind_dict_by_trial(
    nwb_file_name,
    results,
    epoch_number,
    run,
    well,
    ahead_behind_distance,
    pre_sec=0,
    post_sec=0,
):
    ahead_behinds_dict_by_trial = {}
    # use somethign about each trial as a counter for trials
    rewards = (
        TrialsInfoByEpoch.ByTrial()
        & {"nwb_file_name": nwb_file_name, "epoch": epoch_number}
    ).fetch("reward")
    # select out the ahead behind data by trial period
    for trial_number in range(1, len(rewards)):
        time_slice = trial_to_time_slice(
            nwb_file_name,
            results,
            epoch_number,
            trial_number,
            run,
            well,
            pre_sec,
            post_sec,
        )
        trial_ahead_behind = ahead_behind_distance[time_slice]
        ahead_behinds_dict_by_trial[trial_number] = trial_ahead_behind
    return ahead_behinds_dict_by_trial


def plot_ahead_behind_seg_change(
    nwb_file_name,
    epoch,
    results,
    ahead_behind_distance,
    actual_position_edge_id,
    pre_seg_change=1,
    post_seg_change=1,
    ylim=(-235, 235),
    bin_ms=2,
):
    seg_change_idx = np.where(
        actual_position_edge_id[:-1] != actual_position_edge_id[1:]
    )[0]
    # print(f'found {len(seg_change_idx)} segment changes')
    ep_time = results.time  # results.isel(time=time_slice).time
    seg_change_times = np.array(ep_time)[seg_change_idx]

    ahead_behind_distance_seg_all = []
    # need to keep slices in indices/ints not time floats
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    for i in seg_change_idx[1:]:
        start_idx = i - int((pre_seg_change * 1000) / bin_ms)
        stop_idx = i + int((post_seg_change * 1000) / bin_ms)
        change_t = np.array(ep_time)[i]
        start_t = np.array(ep_time)[start_idx]
        stop_t = np.array(ep_time)[stop_idx]
        time_slice = slice(start_idx, stop_idx)
        ahead_behind_distance_seg = ahead_behind_distance[time_slice]
        ahead_behind_distance_seg_all.append(np.array(ahead_behind_distance_seg))
        plt.plot(
            range(-int(pre_seg_change * 1000), int(post_seg_change * 1000), bin_ms),
            ahead_behind_distance_seg,
            color="green",
            alpha=0.2,
        )
    avg_ahead_behind_trace = np.mean(ahead_behind_distance_seg_all, axis=0)
    plt.plot(
        range(-int(pre_seg_change * 1000), int(post_seg_change * 1000), bin_ms),
        avg_ahead_behind_trace,
        color="black",
        alpha=1,
        label="mean",
    )
    ax.axvline(0, color="black", zorder=0)
    ax.set_xlabel("ms since actual segment edge id change")
    ax.set_ylabel("ahead behind distance [cm]")
    ax.set_ylim(ylim)
    ax.set_title(f"{nwb_file_name} epoch {epoch} n = {len(seg_change_idx)}")
    ax.legend(loc="upper right", facecolor="white")
    sns.despine()


def plot_classifier_present(
    time_slice,
    results,
    environment,
    position_info,
    linear_position_df,
    ahead_behind_distance,
    multiunit_firing_rate,
    nwb_copy_file_name,
    epoch_number,
    trial_number,
    run,
    well,
    pre_sec,
    post_sec,
    actual_position_edge_id,
    mental_position_edge_id,
    
    cmap="bone_r",
    figsize=(30, 20),
    fontsize=22,
    multiunit_high_synchrony_times = None,
    fig_path = '/stelmo/alison/DraftFigs/decoding_exs/',
    printpdf=False,
    printpng=False,
):

    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)
    
    fig, axes = plt.subplots(
        4,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 8, 2]},
    )

    time = results.isel(time=time_slice).time

    posterior = (
        results.isel(time=time_slice)
        .acausal_posterior.sum("state")
        .where(environment.is_track_interior_)
    )

    actual_segment = actual_position_edge_id  # [:len(actual_position_edge_id)-2]
    mental_segment = mental_position_edge_id
    nonlocal_by_segment = np.not_equal(actual_segment, mental_segment)
    ahead_or_not = np.asarray(ahead_behind_distance > 0)
    nonlocal_ahead = np.logical_and(nonlocal_by_segment, ahead_or_not)
    nonlocal_behind = np.logical_and(nonlocal_by_segment, ~ahead_or_not)
    reward_bool = (
        TrialsInfoByEpoch.ByTrial()
        & {
            "nwb_file_name": nwb_copy_file_name,
            "epoch": epoch_number,
            "trial_number_by_epoch": trial_number,
        }
    ).fetch1("reward")

    ax = 0  # speed
    axes[ax].fill_between(
        time,
        position_info.iloc[time_slice].head_speed.values.squeeze(),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[ax].set_ylim([0, max(position_info.head_speed.values.squeeze())])
    axes[ax].set_title("Speed", fontsize=fontsize)
    axes[ax].set_ylabel("Speed\n[cm/s]", fontsize=fontsize)
    # axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    axes[ax].tick_params(axis="both", labelsize=fontsize)

    # axes[ax].set_title(f'nwb: {nwb_copy_file_name[:-4]}\nepoch: {epoch_number}, trial: {trial_number}\nincludes run: {run}, and well: {well}\nplus {pre_sec} sec before, {post_sec} sec after\nis this trial rewarded: {reward_bool}',
    #                        fontsize=fontsize)

    """
    ax += 1 #swr
    lim = max(np.abs(swr_data))
    swr_times_sliced = swr_times[np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])]
    swr_data_sliced = swr_data[np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])]
    axes[ax].plot(swr_times_sliced, swr_data_sliced)
    axes[ax].set_ylim([-lim,lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title('Ripple filtered LFP - NB: THIS IS NOT REFERENCED CORRECTLY RIGHT NOW')
    
    cur_kk_ripples = interval_list_intersect(
        np.asarray(kk_ripple_times),
        np.asarray([(time[0], time[-1])]))
    for start_time, end_time in cur_kk_ripples:
        axes[ax].axvspan(start_time, end_time, color='red', alpha=0.3, zorder=100)
    
    ax += 1 #theta
    lim = max(np.abs(theta_data))
    theta_times_sliced = theta_times[np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])]
    theta_data_sliced = theta_data[np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])]
    axes[ax].plot(theta_times_sliced, theta_data_sliced)
    axes[ax].set_ylim([-lim,lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title('Theta filtered LFP')
    """

    ax += 1  # mua and HSE
    axes[ax].fill_between(
        multiunit_firing_rate.iloc[time_slice].index.values,
        multiunit_firing_rate.iloc[time_slice].values.squeeze(),
        color="black",
    )
    axes[ax].set_ylabel("Firing Rate\n[spikes/s]\n", fontsize=fontsize)
    axes[ax].set_title("Multiunit", fontsize=fontsize)
    axes[ax].set_ylim(0.0, 10+np.max(np.asarray(multiunit_firing_rate.iloc[time_slice].values.squeeze())))
    axes[ax].tick_params(axis="both", labelsize=fontsize)

    #cur_multiunit_HSE = interval_list_intersect(
    #    np.asarray(multiunit_high_synchrony_times), np.asarray([(time[0], time[-1])])
    #)

    #for start_time, end_time in cur_multiunit_HSE:
    #    axes[ax].axvspan(start_time, end_time, color="blue", alpha=0.3, zorder=100)

    ax += 1  # decode
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}
    (posterior.plot(x="time", y="position", ax=axes[ax], robust=True, cmap=cmap, vmin=0, vmax=.08, rasterized=True))
    axes[ax].scatter(
        time,
        linear_position_df.iloc[time_slice].linear_position.values,
        s=1,
        color="magenta",
        zorder=10,
    )
    # axes[ax].scatter(
    #     time,
    #     maximum_a_posteriori_estimate(posterior),
    #     marker="x",
    #     c=pd.Series(mental_segment[time_slice]).map(segment2color),
    #     s=2,
    # )
    # axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    axes[ax].set_ylabel("Position [cm]", fontsize=fontsize)
    axes[ax].set_xlabel("")
    axes[ax].tick_params(axis="both", labelsize=fontsize)
    """
    ax += 1 #track segment
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}
    axes[ax].scatter(time, pd.Series(mental_segment[time_slice]), c=pd.Series(mental_segment[time_slice]).map(segment2color),s=3)
    axes[ax].scatter(time, pd.Series(actual_segment[time_slice]), s=.5, color='black')
    axes[ax].set_ylim([0,8])
    axes[ax].set_ylabel("Track segment ID")
    axes[ax].set_title('track segment mental (color), actual (black)')

    ax += 1 #nonlocal amount smooth
    axes[ax].plot(time, pd.Series(nonlocal_by_segment).rolling(win_len).mean()[time_slice], color='blue')
    axes[ax].plot(time, pd.Series(nonlocal_ahead).rolling(win_len).mean()[time_slice], color='green')
    axes[ax].plot(time, pd.Series(nonlocal_behind).rolling(win_len).mean()[time_slice], color='grey')
    #axes[ax].set_ylim([0,1])
    axes[ax].set_ylabel("Prop. nonlocal by seg.")
    axes[ax].set_title('smoothed amount of nonlocal by segment (blue), nonlocal ahead (green), nonlocal behind (grey), over window = '+str(win_len))
     """

    ax += 1  # ahead behind
    #axes[ax].plot(time, ahead_behind_distance[time_slice], color="black", linewidth=2)
    axes[ax].scatter(time, ahead_behind_distance[time_slice], color="black", s=1)
    axes[ax].axhline(0, color="magenta", linestyle="--")
    axes[ax].set_title("Distance",fontsize=fontsize) #Mental distance ahead or behind animal", fontsize=fontsize)
    axes[ax].set_ylabel("Distance\n[cm]", fontsize=fontsize)
    max_dist = 50  # np.max(np.abs(ahead_behind_distance)) + 5
    axes[ax].set_ylim((-max_dist, max_dist))
    axes[ax].text(
        time[0], max_dist - 1, "Ahead", color="grey", va="top", fontsize=fontsize
    )
    axes[ax].text(time[0], -max_dist + 1, "Behind", color="grey", fontsize=fontsize)
    axes[ax].tick_params(axis="both", labelsize=fontsize)

    # ax += 1  # hpd
    # coverage = 0.5
    # # hpd_threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
    # # spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    # # axes[ax].plot(posterior.time, spatial_coverage,
    # #              color="black", linewidth=2)
    # hpd_threshold = get_highest_posterior_threshold(posterior, coverage=coverage)
    # spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    # axes[ax].plot(posterior.time, spatial_coverage, color="black", linewidth=2)
    # axes[ax].axhline(50, color="grey", linestyle="--", alpha=0.4)
    # axes[ax].set_ylim([0, 100])
    # axes[ax].set_title(
    #     f"Spatial coverage by {coverage} highest posterior density region",
    #     fontsize=fontsize,
    # )
    # axes[ax].set_ylabel("Distance\n[cm]", fontsize=fontsize)
    axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    # axes[ax].tick_params(axis="both", labelsize=fontsize)

    # ax += 1 #state
    # results.isel(time=time_slice).acausal_posterior.sum('position').plot(x='time', hue='state', ax=axes[ax])

    # ax += 1 #smooth ahead and nonlocal, behind and nonlocal
    # axes[ax].plot(time, pd.Series(ahead_behind_distance[ahead_behind_distance>0]).rolling(win_len).mean()[time_slice], color='green')
    # axes[ax].plot(time, pd.Series(np.abs(ahead_behind_distance[ahead_behind_distance<=0])).rolling(win_len).mean()[time_slice], color='grey')
    # axes[ax].set_ylabel("")
    # axes[ax].set_title('smoothed ahead (green) and behind (grey), over window = '+str(win_len))

    # offset time
    # time_offsets = time - time[0]
    # axes[ax].set_xticks(time)
    # axes[ax].set_xticklabels([f"{offset:.2f}" for offset in time_offsets.values])

    sns.despine()
    if printpdf:
        plt.savefig(f'{fig_path}'+
            f"decode_present_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.pdf",
            #dpi=300,
        )
    if printpng:
        plt.savefig(f'{fig_path}'+
            f"decode_present_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.png",dpi=300,
        )

    fig, ax = plt.subplots(figsize=(10, 10))
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}

    plt.scatter(
        -1 * position_info.head_position_x,
        position_info.head_position_y,
        c=linear_position_df.track_segment_id.map(segment2color),
        s=0.5,
        alpha=0.4,
    )
    plt.axis("square")
    plt.scatter(
        -1 * position_info.head_position_x.iloc[time_slice],
        position_info.head_position_y.iloc[time_slice],
        color="black",
        s=1,
        zorder=10,
    )
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.xlabel("cm", fontsize=fontsize)
    plt.ylabel("cm", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim([0, 250])
    plt.xlim([-250, 0])
    if printpdf:
        plt.savefig(f'{fig_path}'+
            f"decode_present_path_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.pdf",
            #dpi=300,
        )
    if printpng:
        plt.savefig(f'{fig_path}'+
        f"decode_present_path_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.png",dpi=300,
        )
    
def plot_classifier_present_mini(
    time_slice,
    results,
    environment,
    position_info,
    linear_position_df,
    ahead_behind_distance,
    multiunit_firing_rate,
    nwb_copy_file_name,
    epoch_number,
    trial_number,
    run,
    well,
    pre_sec,
    post_sec,
    actual_position_edge_id,
    mental_position_edge_id,
    
    cmap="bone_r",
    figsize=(30, 20),
    fontsize=22,
    multiunit_high_synchrony_times = None,
    fig_path = '/nimbus/alison/acausal_decode_exs/',
    printpdf=False,
    printpng=False,
):

    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)
    
    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 8, ]},
    )

    time = results.isel(time=time_slice).time

    posterior = (
        results.isel(time=time_slice)
        .acausal_posterior.sum("state")
        .where(environment.is_track_interior_)
    )

    actual_segment = actual_position_edge_id  # [:len(actual_position_edge_id)-2]
    mental_segment = mental_position_edge_id
    nonlocal_by_segment = np.not_equal(actual_segment, mental_segment)
    ahead_or_not = np.asarray(ahead_behind_distance > 0)
    nonlocal_ahead = np.logical_and(nonlocal_by_segment, ahead_or_not)
    nonlocal_behind = np.logical_and(nonlocal_by_segment, ~ahead_or_not)
    reward_bool = (
        TrialsInfoByEpoch.ByTrial()
        & {
            "nwb_file_name": nwb_copy_file_name,
            "epoch": epoch_number,
            "trial_number_by_epoch": trial_number,
        }
    ).fetch1("reward")

    ax = 0  # speed
    axes[ax].fill_between(
        time,
        position_info.iloc[time_slice].head_speed.values.squeeze(),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[ax].set_ylim([0, max(position_info.head_speed.values.squeeze())])
    axes[ax].set_title("Speed", fontsize=fontsize)
    axes[ax].set_ylabel("Speed\n[cm/s]", fontsize=fontsize)
    # axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    axes[ax].tick_params(axis="both", labelsize=fontsize)

    # axes[ax].set_title(f'nwb: {nwb_copy_file_name[:-4]}\nepoch: {epoch_number}, trial: {trial_number}\nincludes run: {run}, and well: {well}\nplus {pre_sec} sec before, {post_sec} sec after\nis this trial rewarded: {reward_bool}',
    #                        fontsize=fontsize)

    """
    ax += 1 #swr
    lim = max(np.abs(swr_data))
    swr_times_sliced = swr_times[np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])]
    swr_data_sliced = swr_data[np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])]
    axes[ax].plot(swr_times_sliced, swr_data_sliced)
    axes[ax].set_ylim([-lim,lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title('Ripple filtered LFP - NB: THIS IS NOT REFERENCED CORRECTLY RIGHT NOW')
    
    cur_kk_ripples = interval_list_intersect(
        np.asarray(kk_ripple_times),
        np.asarray([(time[0], time[-1])]))
    for start_time, end_time in cur_kk_ripples:
        axes[ax].axvspan(start_time, end_time, color='red', alpha=0.3, zorder=100)
    
    ax += 1 #theta
    lim = max(np.abs(theta_data))
    theta_times_sliced = theta_times[np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])]
    theta_data_sliced = theta_data[np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])]
    axes[ax].plot(theta_times_sliced, theta_data_sliced)
    axes[ax].set_ylim([-lim,lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title('Theta filtered LFP')
    """

    ax += 1  # mua and HSE
    axes[ax].fill_between(
        multiunit_firing_rate.iloc[time_slice].index.values,
        multiunit_firing_rate.iloc[time_slice].values.squeeze(),
        color="black",
    )
    axes[ax].set_ylabel("Firing Rate\n[spikes/s]\n", fontsize=fontsize)
    axes[ax].set_title("Multiunit", fontsize=fontsize)
    axes[ax].set_ylim((0.0, 55)) # 10+np.max(np.asarray(multiunit_firing_rate.iloc[time_slice].values.squeeze()))))
    axes[ax].tick_params(axis="both", labelsize=fontsize)

    #cur_multiunit_HSE = interval_list_intersect(
    #    np.asarray(multiunit_high_synchrony_times), np.asarray([(time[0], time[-1])])
    #)

    #for start_time, end_time in cur_multiunit_HSE:
    #    axes[ax].axvspan(start_time, end_time, color="blue", alpha=0.3, zorder=100)

    ax += 1  # decode
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}
    (posterior.plot(x="time", y="position", ax=axes[ax], robust=True, cmap=cmap, vmin=0, vmax=.08, rasterized=True))
    axes[ax].scatter(
        time,
        linear_position_df.iloc[time_slice].linear_position.values,
        s=1,
        color="magenta",
        zorder=10,
    )
    # axes[ax].scatter(
    #     time,
    #     maximum_a_posteriori_estimate(posterior),
    #     marker="x",
    #     c=pd.Series(mental_segment[time_slice]).map(segment2color),
    #     s=2,
    # )
    # axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    axes[ax].set_ylabel("Position [cm]", fontsize=fontsize)
    axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    axes[ax].tick_params(axis="both", labelsize=fontsize)
    """
    ax += 1 #track segment
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}
    axes[ax].scatter(time, pd.Series(mental_segment[time_slice]), c=pd.Series(mental_segment[time_slice]).map(segment2color),s=3)
    axes[ax].scatter(time, pd.Series(actual_segment[time_slice]), s=.5, color='black')
    axes[ax].set_ylim([0,8])
    axes[ax].set_ylabel("Track segment ID")
    axes[ax].set_title('track segment mental (color), actual (black)')

    ax += 1 #nonlocal amount smooth
    axes[ax].plot(time, pd.Series(nonlocal_by_segment).rolling(win_len).mean()[time_slice], color='blue')
    axes[ax].plot(time, pd.Series(nonlocal_ahead).rolling(win_len).mean()[time_slice], color='green')
    axes[ax].plot(time, pd.Series(nonlocal_behind).rolling(win_len).mean()[time_slice], color='grey')
    #axes[ax].set_ylim([0,1])
    axes[ax].set_ylabel("Prop. nonlocal by seg.")
    axes[ax].set_title('smoothed amount of nonlocal by segment (blue), nonlocal ahead (green), nonlocal behind (grey), over window = '+str(win_len))
     """

    # ax += 1  # ahead behind
    # #axes[ax].plot(time, ahead_behind_distance[time_slice], color="black", linewidth=2)
    # axes[ax].scatter(time, ahead_behind_distance[time_slice], color="black", s=1)
    # axes[ax].axhline(0, color="magenta", linestyle="--")
    # axes[ax].set_title("Mental distance ahead or behind animal", fontsize=fontsize)
    # axes[ax].set_ylabel("Distance\n[cm]", fontsize=fontsize)
    # max_dist = 50  # np.max(np.abs(ahead_behind_distance)) + 5
    # axes[ax].set_ylim((-max_dist, max_dist))
    # axes[ax].text(
    #     time[0], max_dist - 1, "Ahead", color="grey", va="top", fontsize=fontsize
    # )
    # axes[ax].text(time[0], -max_dist + 1, "Behind", color="grey", fontsize=fontsize)
    # axes[ax].tick_params(axis="both", labelsize=fontsize)

    # ax += 1  # hpd
    # coverage = 0.5
    # # hpd_threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
    # # spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    # # axes[ax].plot(posterior.time, spatial_coverage,
    # #              color="black", linewidth=2)
    # hpd_threshold = get_highest_posterior_threshold(posterior, coverage=coverage)
    # spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    # axes[ax].plot(posterior.time, spatial_coverage, color="black", linewidth=2)
    # axes[ax].axhline(50, color="grey", linestyle="--", alpha=0.4)
    # axes[ax].set_ylim([0, 100])
    # axes[ax].set_title(
    #     f"Spatial coverage by {coverage} highest posterior density region",
    #     fontsize=fontsize,
    # )
    # axes[ax].set_ylabel("Distance\n[cm]", fontsize=fontsize)
  #  axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    # axes[ax].tick_params(axis="both", labelsize=fontsize)

    # ax += 1 #state
    # results.isel(time=time_slice).acausal_posterior.sum('position').plot(x='time', hue='state', ax=axes[ax])

    # ax += 1 #smooth ahead and nonlocal, behind and nonlocal
    # axes[ax].plot(time, pd.Series(ahead_behind_distance[ahead_behind_distance>0]).rolling(win_len).mean()[time_slice], color='green')
    # axes[ax].plot(time, pd.Series(np.abs(ahead_behind_distance[ahead_behind_distance<=0])).rolling(win_len).mean()[time_slice], color='grey')
    # axes[ax].set_ylabel("")
    # axes[ax].set_title('smoothed ahead (green) and behind (grey), over window = '+str(win_len))

    sns.despine()
    if printpdf:
        plt.savefig(f'{fig_path}'+
            f"decode_present_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.pdf",
            #dpi=300,
        )
    if printpng:
        plt.savefig(f'{fig_path}'+
            f"decode_present_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.png"
        )

    fig, ax = plt.subplots(figsize=(10, 10))
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}

    plt.scatter(
        -1 * position_info.head_position_x,
        position_info.head_position_y,
        c=linear_position_df.track_segment_id.map(segment2color),
        s=0.5,
        alpha=0.4,
    )
    plt.axis("square")
    plt.scatter(
        -1 * position_info.head_position_x.iloc[time_slice],
        position_info.head_position_y.iloc[time_slice],
        color="black",
        s=0.25,
        zorder=10,
    )
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.xlabel("cm", fontsize=fontsize)
    plt.ylabel("cm", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim([0, 250])
    plt.xlim([-250, 0])
    if printpdf:
        plt.savefig(f'{fig_path}'+
            f"decode_present_path_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.pdf",
            #dpi=300,
        )
    if printpng:
        plt.savefig(f'{fig_path}'+
        f"decode_present_path_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.png"
        )

def plot_classifier_present_mini2(
    time_slice,
    results,
    environment,
    position_info,
    linear_position_df,
    ahead_behind_distance,
    multiunit_firing_rate,
    nwb_copy_file_name,
    epoch_number,
    trial_number,
    run,
    well,
    pre_sec,
    post_sec,
    actual_position_edge_id,
    mental_position_edge_id,
    
    cmap="bone_r",
    figsize=(30, 20),
    fontsize=25,
    multiunit_high_synchrony_times = None,
    fig_path = '/stelmo/alison/MarAprFigs/lastminfigsTh/',
    printpdf=False,
    printpng=False,
):

    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)
    
    fig, axes = plt.subplots(
        1,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        #gridspec_kw={"height_ratios": [1, 1, 8, ]},
    )

    time = results.isel(time=time_slice).time

    posterior = (
        results.isel(time=time_slice)
        .acausal_posterior.sum("state")
        .where(environment.is_track_interior_)
    )

    actual_segment = actual_position_edge_id  # [:len(actual_position_edge_id)-2]
    mental_segment = mental_position_edge_id
    nonlocal_by_segment = np.not_equal(actual_segment, mental_segment)
    ahead_or_not = np.asarray(ahead_behind_distance > 0)
    nonlocal_ahead = np.logical_and(nonlocal_by_segment, ahead_or_not)
    nonlocal_behind = np.logical_and(nonlocal_by_segment, ~ahead_or_not)
    reward_bool = (
        TrialsInfoByEpoch.ByTrial()
        & {
            "nwb_file_name": nwb_copy_file_name,
            "epoch": epoch_number,
            "trial_number_by_epoch": trial_number,
        }
    ).fetch1("reward")

    # ax = 0  # speed
    # axes[ax].fill_between(
    #     time,
    #     position_info.iloc[time_slice].head_speed.values.squeeze(),
    #     color="lightgrey",
    #     linewidth=1,
    #     alpha=0.5,
    # )
    # axes[ax].set_ylim([0, max(position_info.head_speed.values.squeeze())])
    # axes[ax].set_title("Speed", fontsize=fontsize)
    # axes[ax].set_ylabel("Speed\n[cm/s]", fontsize=fontsize)
    # # axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    # axes[ax].tick_params(axis="both", labelsize=fontsize)

    # axes[ax].set_title(f'nwb: {nwb_copy_file_name[:-4]}\nepoch: {epoch_number}, trial: {trial_number}\nincludes run: {run}, and well: {well}\nplus {pre_sec} sec before, {post_sec} sec after\nis this trial rewarded: {reward_bool}',
    #                        fontsize=fontsize)

    """
    ax += 1 #swr
    lim = max(np.abs(swr_data))
    swr_times_sliced = swr_times[np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])]
    swr_data_sliced = swr_data[np.logical_and(swr_times >= time.data[0], swr_times <= time.data[-1])]
    axes[ax].plot(swr_times_sliced, swr_data_sliced)
    axes[ax].set_ylim([-lim,lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title('Ripple filtered LFP - NB: THIS IS NOT REFERENCED CORRECTLY RIGHT NOW')
    
    cur_kk_ripples = interval_list_intersect(
        np.asarray(kk_ripple_times),
        np.asarray([(time[0], time[-1])]))
    for start_time, end_time in cur_kk_ripples:
        axes[ax].axvspan(start_time, end_time, color='red', alpha=0.3, zorder=100)
    
    ax += 1 #theta
    lim = max(np.abs(theta_data))
    theta_times_sliced = theta_times[np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])]
    theta_data_sliced = theta_data[np.logical_and(theta_times >= time.data[0], theta_times <= time.data[-1])]
    axes[ax].plot(theta_times_sliced, theta_data_sliced)
    axes[ax].set_ylim([-lim,lim])
    axes[ax].set_ylabel("Amplitude")
    axes[ax].set_title('Theta filtered LFP')
    """

    # ax += 1  # mua and HSE
    # axes[ax].fill_between(
    #     multiunit_firing_rate.iloc[time_slice].index.values,
    #     multiunit_firing_rate.iloc[time_slice].values.squeeze(),
    #     color="black",
    # )
    # axes[ax].set_ylabel("Firing Rate\n[spikes/s]\n", fontsize=fontsize)
    # axes[ax].set_title("Multiunit", fontsize=fontsize)
    # axes[ax].set_ylim((0.0, 55)) # 10+np.max(np.asarray(multiunit_firing_rate.iloc[time_slice].values.squeeze()))))
    # axes[ax].tick_params(axis="both", labelsize=fontsize)

    #cur_multiunit_HSE = interval_list_intersect(
    #    np.asarray(multiunit_high_synchrony_times), np.asarray([(time[0], time[-1])])
    #)

    #for start_time, end_time in cur_multiunit_HSE:
    #    axes[ax].axvspan(start_time, end_time, color="blue", alpha=0.3, zorder=100)

    # ax += 1  # decode
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}
    (posterior.plot(x="time", y="position", ax=axes, robust=True, cmap=cmap, vmin=0, vmax=.08, rasterized=True))
    axes.scatter(
        time,
        linear_position_df.iloc[time_slice].linear_position.values,
        s=1,
        color="magenta",
        zorder=10,
    )
    # axes[ax].scatter(
    #     time,
    #     maximum_a_posteriori_estimate(posterior),
    #     marker="x",
    #     c=pd.Series(mental_segment[time_slice]).map(segment2color),
    #     s=2,
    # )
    # axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    axes.set_ylabel("Position [cm]", fontsize=fontsize)
    axes.set_xlabel("Time [s]", fontsize=fontsize, ha='center' , y=-100)
    axes.tick_params(axis="both", labelsize=fontsize)
    """
    ax += 1 #track segment
    segment_cmap = plt.get_cmap("tab10")
    segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}
    axes[ax].scatter(time, pd.Series(mental_segment[time_slice]), c=pd.Series(mental_segment[time_slice]).map(segment2color),s=3)
    axes[ax].scatter(time, pd.Series(actual_segment[time_slice]), s=.5, color='black')
    axes[ax].set_ylim([0,8])
    axes[ax].set_ylabel("Track segment ID")
    axes[ax].set_title('track segment mental (color), actual (black)')

    ax += 1 #nonlocal amount smooth
    axes[ax].plot(time, pd.Series(nonlocal_by_segment).rolling(win_len).mean()[time_slice], color='blue')
    axes[ax].plot(time, pd.Series(nonlocal_ahead).rolling(win_len).mean()[time_slice], color='green')
    axes[ax].plot(time, pd.Series(nonlocal_behind).rolling(win_len).mean()[time_slice], color='grey')
    #axes[ax].set_ylim([0,1])
    axes[ax].set_ylabel("Prop. nonlocal by seg.")
    axes[ax].set_title('smoothed amount of nonlocal by segment (blue), nonlocal ahead (green), nonlocal behind (grey), over window = '+str(win_len))
     """

    # ax += 1  # ahead behind
    # #axes[ax].plot(time, ahead_behind_distance[time_slice], color="black", linewidth=2)
    # axes[ax].scatter(time, ahead_behind_distance[time_slice], color="black", s=1)
    # axes[ax].axhline(0, color="magenta", linestyle="--")
    # axes[ax].set_title("Mental distance ahead or behind animal", fontsize=fontsize)
    # axes[ax].set_ylabel("Distance\n[cm]", fontsize=fontsize)
    # max_dist = 50  # np.max(np.abs(ahead_behind_distance)) + 5
    # axes[ax].set_ylim((-max_dist, max_dist))
    # axes[ax].text(
    #     time[0], max_dist - 1, "Ahead", color="grey", va="top", fontsize=fontsize
    # )
    # axes[ax].text(time[0], -max_dist + 1, "Behind", color="grey", fontsize=fontsize)
    # axes[ax].tick_params(axis="both", labelsize=fontsize)

    # ax += 1  # hpd
    # coverage = 0.5
    # # hpd_threshold = get_highest_posterior_threshold(posterior, coverage=0.95)
    # # spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    # # axes[ax].plot(posterior.time, spatial_coverage,
    # #              color="black", linewidth=2)
    # hpd_threshold = get_highest_posterior_threshold(posterior, coverage=coverage)
    # spatial_coverage = get_HPD_spatial_coverage(posterior, hpd_threshold)
    # axes[ax].plot(posterior.time, spatial_coverage, color="black", linewidth=2)
    # axes[ax].axhline(50, color="grey", linestyle="--", alpha=0.4)
    # axes[ax].set_ylim([0, 100])
    # axes[ax].set_title(
    #     f"Spatial coverage by {coverage} highest posterior density region",
    #     fontsize=fontsize,
    # )
    # axes[ax].set_ylabel("Distance\n[cm]", fontsize=fontsize)
  #  axes[ax].set_xlabel("Time [s]", fontsize=fontsize)
    # axes[ax].tick_params(axis="both", labelsize=fontsize)

    # ax += 1 #state
    # results.isel(time=time_slice).acausal_posterior.sum('position').plot(x='time', hue='state', ax=axes[ax])

    # ax += 1 #smooth ahead and nonlocal, behind and nonlocal
    # axes[ax].plot(time, pd.Series(ahead_behind_distance[ahead_behind_distance>0]).rolling(win_len).mean()[time_slice], color='green')
    # axes[ax].plot(time, pd.Series(np.abs(ahead_behind_distance[ahead_behind_distance<=0])).rolling(win_len).mean()[time_slice], color='grey')
    # axes[ax].set_ylabel("")
    # axes[ax].set_title('smoothed ahead (green) and behind (grey), over window = '+str(win_len))

    sns.despine()
    if printpdf:
        plt.savefig(f'{fig_path}'+
            f"decode_present_mini2_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.pdf",
            #dpi=300,
        )
    if printpng:
        plt.savefig(f'{fig_path}'+
            f"decode_present_mini2_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.png"
        )

    # fig, ax = plt.subplots(figsize=(10, 10))
    # segment_cmap = plt.get_cmap("tab10")
    # segment2color = {i: color for i, color in enumerate(segment_cmap.colors)}

    # plt.scatter(
    #     -1 * position_info.head_position_x,
    #     position_info.head_position_y,
    #     c=linear_position_df.track_segment_id.map(segment2color),
    #     s=0.5,
    #     alpha=0.4,
    # )
    # plt.axis("square")
    # plt.scatter(
    #     -1 * position_info.head_position_x.iloc[time_slice],
    #     position_info.head_position_y.iloc[time_slice],
    #     color="black",
    #     s=0.25,
    #     zorder=10,
    # )
    # # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # # plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # plt.xlabel("cm", fontsize=fontsize)
    # plt.ylabel("cm", fontsize=fontsize)
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.ylim([0, 250])
    # plt.xlim([-250, 0])
    # if printpdf:
    #     plt.savefig(f'{fig_path}'+
    #         f"decode_present_path_mini2_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.pdf",
    #         #dpi=300,
    #     )
    # if printpng:
    #     plt.savefig(f'{fig_path}'+
    #     f"decode_present_path_mini2_{nwb_copy_file_name}_ep{epoch_number}_trial{trial_number}_run{run}_well{well}_pre{pre_sec}_post{post_sec}.png"
    #     )