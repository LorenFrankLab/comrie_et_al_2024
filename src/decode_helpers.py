import os
import pandas as pd
import numpy as np
import xarray as xr

from spyglass.common import (
    TrackGraph,
    IntervalList,
    IntervalLinearizedPosition,
    IntervalPositionInfo,
)
# from spyglass.decoding import UnitMarksIndicator, ClusterlessClassifierParameters # moved to v0, used to compute/populate primarily
from spyglass.decoding.v0.clusterless import UnitMarksIndicator, ClusterlessClassifierParameters

from spyglass.common.common_interval import interval_list_intersect

from alison_position import PosValidTimesToEpoch

# from replay_trajectory_classification import ClusterlessClassifier # not needed for fig generation/export, so eliminate extra dependency
# from replay_trajectory_classification.environments import Environment # not needed for fig generation/export, eliminate extra dependency

def get_decode_inputs_data(
    nwb_file_name,
    interval_list_name,
    track_graph_name, 
    position_info_param_name,
    linearization_param_name, 
    preproc_params_name
):
    """retrieve the marks, linpos, and pos used for decodeing related analyses before trimming time down to valid slice"""
    
    # Load data that goes into decoder
    linear_position_df = (
        IntervalLinearizedPosition()
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "position_info_param_name": position_info_param_name,
            "track_graph_name": track_graph_name,
            "linearization_param_name": linearization_param_name,
        }
    ).fetch1_dataframe()
    # Marks, like position, shoudl be organized by pos X valid times interval -
    # but marks have actually usually only been generatied on valid no prepost trial times + pos x, so uses intersection of those
    # HARD CODING TO USE TET PREPROC PARAMS 
    marks_xr = (
        UnitMarksIndicator()
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "preproc_params_name": preproc_params_name,
        }
    ).fetch_xarray()
    # also load position data so it is easily accessible
    # and sliced for later use,  though not used directly for decoding
    position_df = (
        IntervalPositionInfo()
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "position_info_param_name": position_info_param_name,
        }
    ).fetch1_dataframe()

    return position_df, linear_position_df, marks_xr

# remake the alignment function to save marks as a df
# One run pos interval at a time
#  want the args to be a key for linpos, a key for pos, and a key for marks, rather than sharing these variables 
# If use probes rather than tetrodes would need to remove the preproc params name argument
def get_aligned_decode_inputs_slice(
    nwb_file_name,
    interval_list_valid,
    interval_list_name,
    track_graph_name,
    position_info_param_name="default_decoding",
    linearization_param_name="default",
    preproc_params_name="franklab_tetrode_hippocampus",
    ignore_invalid_times = True,
):
    """'
    Align one pos X valid times interval_list_name worth of pos, linpos, and marks data
    to only the valid time intervals to be used for decoding of all epochs within interval_list_valid interval
    """
    position_df , linear_position_df, marks_xr= get_decode_inputs_data(nwb_file_name,
        interval_list_name, track_graph_name, position_info_param_name, linearization_param_name, preproc_params_name)
    
    # Can workaround if start to have violations here
    assert np.logical_and(
        position_df.shape[0] == marks_xr.shape[0],
        position_df.shape[0] == linear_position_df.shape[0],
    )  # Check all same time len/samp rate

    # Translate from pos X valid times to ep name
    interval_list_name_by_epoch = (
        PosValidTimesToEpoch()
        & {"nwb_file_name": nwb_file_name, "pos_interval_list_name": interval_list_name}
    ).fetch1("epoch_interval_list_name")

    # Intersect valid ephys and pos intervals for the epoch to get a valid time slice
    interval = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name_by_epoch,
        }
    ).fetch1("valid_times")
    valid_ephys_times = (
        IntervalList
        & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_valid}
    ).fetch1("valid_times")
    valid_pos_times = (
        IntervalList
        & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}
    ).fetch1("valid_times")
    intersect_interval = interval_list_intersect(
        interval_list_intersect(interval, valid_ephys_times), valid_pos_times
    )

    # Warn user about discontinuous interval intersection
    if len(intersect_interval) > 1:
        print(f'\nWARNING: intersect_interval is discontinuous! It has {len(intersect_interval)} valid intersecting intervals.\n')
        for i,v in enumerate(intersect_interval):
            if i > 0:
                invalid_duration = v[0] - intersect_interval[i-1][1]
                print(f'FYI invalid duration from intersect_interval index {i} to index {i-1}: {invalid_duration} s \nthat is the same as {invalid_duration*1000} ms\nthat is the same as {invalid_duration*500} 2ms bins\n\n')
                if invalid_duration > .5:
                    print('WARNING: DURATION BETWEEN INTERVALS IS >.5 SECONDS')

    if ignore_invalid_times:
        valid_time_slice = slice(intersect_interval[0][0], intersect_interval[len(intersect_interval)-1][1])
    else:
        valid_time_slice = slice(intersect_interval[0][0], intersect_interval[0][1])

    return valid_time_slice

def align_decode_inputs(position_df, linear_position_df, marks_xr, valid_time_slice):
    # Slice pos, linpos, and marks down to common range of times to use for decoding related analyses
    linear_position_df = linear_position_df.loc[valid_time_slice]
    position_df = position_df.loc[valid_time_slice]
    marks_xr = marks_xr.sel(time=valid_time_slice)
    assert np.logical_and(
        position_df.shape[0] == marks_xr.shape[0],
        position_df.shape[0] == linear_position_df.shape[0],
    )  # Check all same time len/samp rate

    print(
        f"Created aligned position_df, linear_position_df, and marks_xr, all of which have time indices of len({position_df.shape[0]})\n"
    )

    # To save marks into nwb, store it as a df instead of the original helper function
    # that flattens electrode dimension just keep the full thing
    # marks_full_df = marks_xr.to_dataframe(name="amplitude") #removed bc no longer storing marks trimmed in analysis nwb

    return position_df, linear_position_df, marks_xr, valid_time_slice

def marks_full_df_to_xr(marks_full_df):
    """During saving marks to nwb in the decoding pipeline, it becomes a flattened df. Now
    Recreate the xarray datarray that it started as. The marks related schema have some related methods."""
    marks_xr_recreated = (
        marks_full_df.to_xarray()
        .to_array()
        .drop_vars("variable")
        .drop_vars("electrodes")
        .squeeze("variable")
    )
    return marks_xr_recreated

def customize_decode_parameters(
    nwb_file_name,
    classifier_param_name,
    track_graph_name,
    clusterless_algorithm,
    mark_std=24.0,
    position_std=6.0,
    block_size=100,
):
    """Set up all clusterless classifier parameters to prepare for decode fitting + predicting"""
    parameters = (
        ClusterlessClassifierParameters()
        & {"classifier_param_name": classifier_param_name}
    ).fetch1()
    track_graph = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).get_networkx_track_graph()
    track_graph_params = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).fetch1()
    environment = Environment(
        track_graph=track_graph,
        edge_order=track_graph_params["linear_edge_order"],
        edge_spacing=track_graph_params["linear_edge_spacing"],
    )

    # Specify further custom parameters
    parameters["classifier_params"]["environments"] = [environment]
    parameters["classifier_params"]["clusterless_algorithm"] = clusterless_algorithm
    parameters["classifier_params"]["clusterless_algorithm_params"] = {
        "mark_std": mark_std,
        "position_std": position_std,
        "block_size": block_size,
    }
    return parameters


def save_decode_results(
    nwb_file_name,
    track_graph_name,
    interval_list_name,
    results,
    environment,
    save_results=True,
    save_env=True,
    save_results_path="/stelmo/alison/decodes/",
    save_env_path="/stelmo/alison/decodes/",
    filename_appendage=""
):
    """save decoding results to .nc and accompanying environment to .pkl at custom path"""
    # Save env that goes along w/ decode b/c can't use classifier on gpu-less machines 
    if save_env:
        save_env_filename = (
            save_env_path
            + f"environment_{nwb_file_name}_trackgraph_{track_graph_name}{filename_appendage}.pkl"
        )
        environment.save_environment(filename=save_env_filename)
        print(f"Saved environment at: {save_env_filename}")
    else:
        print(f"not saving env")
    # Save decode to nc in future use nc nwb formatting extension
    if save_results:
        save_results_filename = (
            save_results_path
            + f"{nwb_file_name}_{track_graph_name}_{interval_list_name}_1D{filename_appendage}.nc"
        )
        print(f"Writing out decode results to filename: {save_results_filename}")
        results.to_netcdf(save_results_filename)
        print(f"Finished writing out decode results to filename: {save_results_filename}\n")
    else:
        print(f"not saving results")

    return save_env_filename, save_results_filename