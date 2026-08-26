import os
import datajoint as dj
from spyglass.utils import SpyglassMixin
import numpy as np
import pandas as pd
import xarray as xr
import uuid
import joblib
import logging

from paths import to_local # for dandi based file reconstruction

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')

from spyglass.common import TrackGraph, IntervalLinearizedPosition, IntervalPositionInfo, AnalysisNwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb
# from spyglass.decoding import UnitMarksIndicator # moved to v0.clusterless in new spyglass, primarily used to populate
from spyglass.decoding.v0.clusterless import UnitMarksIndicator

# from replay_trajectory_classification import ClusterlessClassifier
# from trajectory_analysis_tools import maximum_a_posteriori_estimate
# from trajectory_analysis_tools import get_HPD_spatial_coverage, get_highest_posterior_threshold
from ripple_detection import get_multiunit_population_firing_rate
# from trajectory_analysis_tools import get_trajectory_data, get_ahead_behind_distance

from alison_position import PosValidTimesToEpoch
from decode_helpers import get_aligned_decode_inputs_slice, customize_decode_parameters, save_decode_results
from alison_behav import TrialsInfoByEpoch

_DEFAULT_DECODE_SAVE_OUT_DIR = '/stelmo/alison/decodes/'
_DEFAULT_ENVIRONMENT_SAVE_OUT_DIR = '/stelmo/alison/decodes/'

schema = dj.schema('alison_decoding')

# Here include many of the UnitMarksIndicator parameters except the ones that are unique to sort groups
@schema
class ClusterlessAlignedIntervalSelection(SpyglassMixin, dj.Manual):
    definition="""
    # Identify cohorts of data that will be used for decoding
    -> IntervalLinearizedPosition
    -> TrackGraph
    -> Session
    -> IntervalList
    sort_interval_name: varchar(200)
    preproc_params_name: varchar(200) #used to fetch from unitmarks
    team_name: varchar(200)
    sorter: varchar(200)
    sorter_params_name: varchar(200)
    mark_param_name: varchar(200)
    sampling_rate: int
    interval_list_name_valid: varchar(200) #use valid no pre prost trial raw data valid time sfor a day or epoch(s)
    ---
    """

@schema
class ClusterlessAlignedInterval(SpyglassMixin, dj.Computed):
    definition="""
    # Time interval that will be used for decoding, spanning linpos + marks
    -> ClusterlessAlignedIntervalSelection
    ---
    epoch_number: int # id of epoch in TaskEpoch
    epoch_name: varchar(40) # name of epoch in IntervalList
    valid_time_slice: blob # slice from start to end of intersection of valid pos, ephys, and marks intervals
    total_unitmarks_entries_by_sg: int # count up entries in the parts table, just for user cue til new sg cohort implementation
    """
    class BySortGroup(SpyglassMixin, dj.Part):
        definition="""
        # Sort groups within each decoding cohort
        -> master
        sort_group_id: int
        curation_id: int
        artifact_removed_interval_list_name: varchar(200)
        ---
        """
    def make(self, key):
        print(f'Computing ClusterlessAlignedInterval for a set of linpos, pos, and marks for: {key}')
        master_key = key.copy()
        nwb_file_name = key['nwb_file_name']
        interval_list_name  = key['interval_list_name']
        # Find the intersecting times for pos, linpos, and marks, as well as the vaild time slice

        try:
            valid_time_slice = get_aligned_decode_inputs_slice(key['nwb_file_name'],
                                                            key['interval_list_name_valid'],
                                                            key['interval_list_name'],
                                                            key['track_graph_name'],
                                                            ignore_invalid_times=True)
            
            master_key['valid_time_slice'] = [valid_time_slice.start, valid_time_slice.stop]

            # Find epoch number and name so it is easy to navigate for user
            epoch_number, epoch_name = (PosValidTimesToEpoch() & {"nwb_file_name": key['nwb_file_name'],
                                                                "pos_interval_list_name": key['interval_list_name']}
                                                                ).fetch1("epoch", "epoch_interval_list_name")
            master_key['epoch_number'] = epoch_number
            master_key['epoch_name'] = epoch_name

            marks_selection = (UnitMarksIndicator() & {
                    "nwb_file_name": key['nwb_file_name'],
                    "interval_list_name": key['interval_list_name'],
                    "preproc_params_name": key['preproc_params_name']})
            master_key['total_unitmarks_entries_by_sg'] = len(marks_selection)

            get_keys_by_sg = marks_selection.fetch('sort_group_id','curation_id','artifact_removed_interval_list_name', as_dict=True)

            part_key = []
            for i in get_keys_by_sg:
                part_key.append(dict(key,**i)) # make list of dicts to insert into Part table

            # Insert into ClusterlessAlignedInterval and BySortGroup part table
            ClusterlessAlignedInterval.insert1(master_key, skip_duplicates = True)
            ClusterlessAlignedInterval.BySortGroup.insert(part_key, skip_duplicates = True)
            print(f'Populated ClusterlessAlignedInterval() and ClusterlessAlignedInterval.BySortGroup().')
        except AssertionError:
            print(f'\nERROR: Skipping {nwb_file_name} epoch {interval_list_name} due to intersecting intervals issue. Either shapes of marks and pos dont match OR there is no data for this epoch!\n')
            print(f'\nFailed to populate one or both of ClusterlessAlignedInterval() and ClusterlessAlignedInterval.BySortGroup() with Exception: {e}.\n')
        except Exception as e:
            print(f'\nERROR: Failed to populate one or both of ClusterlessAlignedInterval() and ClusterlessAlignedInterval.BySortGroup() with Exception: {e}.\n')

@schema
class ClusterlessAlgorithmParameters(SpyglassMixin, dj.Manual):
    definition="""
    # Extra clusterless algorithm params not currently in ClusterlessClassifierParameters
    clusterless_algorithm_param_name: varchar(200) # unique str for clusterless alg params
    ---
    clusterless_algorithm: varchar(200) # name of clusterless alg to use
    clusterless_algorithm_param_dict: BLOB # dict of params to be used in clusterless_algorithm_parameters
    """
    def insert_default(self):
        """insert default parameters"""
        # for when using gpu
        clusterless_algorithm_param_dict={}
        clusterless_algorithm_param_dict["mark_std"]=24.0
        clusterless_algorithm_param_dict["position_std"]=6.0
        clusterless_algorithm_param_dict["block_size"]=100
        clusterless_algorithm_param_name = "default_gpu"
        clusterless_algorithm = "multiunit_likelihood_integer_gpu"
        self.insert1([clusterless_algorithm_param_name,
                     clusterless_algorithm,
                     clusterless_algorithm_param_dict],
                     skip_duplicates=True)
        # for when using cpu - this can be the incorrect clusterelss algorithm name
        # it is in replay_trajectory_classification/likelihoods/__init__.py _CLUSTERLESS_ALGORITHMS
        clusterless_algorithm_param_dict={}
        self.insert1(["default_cpu",
                     "multiunit_likelihood",
                     clusterless_algorithm_param_dict],
                     skip_duplicates=True)

# this currently forces you to use the same track graph for encoding and decoding 
# has 15 pks, or 16 if include another track graph. 
# Need to include a decode path as an env variable or just hard coded in  schema
@schema
class ClusterlessSelection(SpyglassMixin, dj.Manual): 
    definition="""
    # Select parameters and interval of linpos+marks to use for decoding
    -> ClusterlessAlignedInterval
    -> ClusterlessClassifierParameters
    -> ClusterlessAlgorithmParameters
    ---
    """

@schema
class ClusterlessResults(SpyglassMixin, dj.Computed):
    definition="""
    # Paths to clusterless results and env .nc files
    -> ClusterlessSelection
    ---
    epoch_number: int # id of epoch in TaskEpoch
    epoch_name: varchar(40) # name of epoch in IntervalList
    valid_time_slice: blob # slice from start to end of intersection of valid pos, ephys, and marks intervals
    total_unitmarks_entries_by_sg: int # count up how many entries are in the parts table, just as a cue to the user until better sort group specification is implemented
    clusterless_results_path : varchar(1000)
    environment_path : varchar(1000) 
    decode_id : varchar(40) # try to use it in filenaming??? could def design better
    """
    def make(self, key):
        print(f'Going to run clusterless decoding with key: {key}')
        
        decode_id = str(uuid.uuid4())[0:8]
        
        # For use in key eventually
        epoch_number, epoch_name, valid_time_slice, total_unitmarks_entries_by_sg = (ClusterlessAlignedInterval & key).fetch1('epoch_number', 'epoch_name', 'valid_time_slice', 'total_unitmarks_entries_by_sg')
        
        # Get the data for decoding: linpos and marks
        linear_position_df = (IntervalLinearizedPosition() & {
                "nwb_file_name": key['nwb_file_name'],
                "interval_list_name": key['interval_list_name'],
                "position_info_param_name": key['position_info_param_name'],
                "track_graph_name": key['track_graph_name'],
                "linearization_param_name": key['linearization_param_name'],
                } ).fetch1_dataframe()
        # Marks, like position, shoudl be organized by pos X valid times interval -
        # but marks have actually usually only been generatied on valid no prepost trial times + pos x, so uses intersection of those
        # note that preproc params name is getting overused here to find tets vs probes - one day make a version of this schema 
        marks_xr = (UnitMarksIndicator() & {
                "nwb_file_name": key['nwb_file_name'],
                "interval_list_name": key['interval_list_name'],
                "preproc_params_name": key['preproc_params_name'],
                "sort_interval_name": key['sort_interval_name'],
                "preproc_params_name": key['preproc_params_name'],
                "team_name": key['team_name'],
                "sorter": key['sorter'],
                "sorter_params_name": key['sorter_params_name'],
                "mark_param_name": key['mark_param_name'],
                "sampling_rate": key['sampling_rate']
                } ).fetch_xarray()
        
        # Align with valid time slice
        valid_time_sliced = slice(valid_time_slice[0], valid_time_slice[1])
        linear_position_df = linear_position_df.loc[valid_time_sliced]
        marks_xr = marks_xr.sel(time=valid_time_sliced)
        assert marks_xr.shape[0] == linear_position_df.shape[0]         # Check all same time len/samp rate
        
        # Set up full params
        clusterless_algorithm, clusterless_algorithm_param_dict = (ClusterlessAlgorithmParameters() & {
            'clusterless_algorithm_param_name':key['clusterless_algorithm_param_name']
             }).fetch1('clusterless_algorithm', 'clusterless_algorithm_param_dict')
        
       
        parameters = customize_decode_parameters(key['nwb_file_name'],
                                                 key['classifier_param_name'], 
                                                 key['track_graph_name'],
                                                 clusterless_algorithm, 
                                                 mark_std = clusterless_algorithm_param_dict['mark_std'], 
                                                 position_std = clusterless_algorithm_param_dict['position_std'],
                                                 block_size = clusterless_algorithm_param_dict['block_size'])
        #pprint.pprint(parameters)
        
        nwb_file_name = key['nwb_file_name']
        interval_list_name = key['interval_list_name']

        # Try decoding fit + predict - works on Breeze
        try:
            classifier = ClusterlessClassifier(**parameters['classifier_params'])
            classifier.fit(
                position=linear_position_df.linear_position.values,
                multiunits=marks_xr.values,
                **parameters['fit_params']
            )

            results = classifier.predict(
                multiunits=marks_xr.values,
                time=linear_position_df.index,
                **parameters['predict_params']
            )
            logging.info('Done!')
            
        except Exception as e:
            print(f'\nERROR in fit+predict step while attempting to populate ClusterlessResults for {nwb_file_name} {interval_list_name} with id {decode_id}: {e}. Not saving.')
        
        else: #only if there is no exception so far, try saving, and as long as that goes ,  insert
        #try:
            # Save environment and decode results
            environment = parameters['classifier_params']['environments'][0]

            # dont  neeed to save the  env if it is already there --  omit the decode id from the environmetn file name so it just gets overwritten when it is the same
            save_env_filename, save_results_filename = save_decode_results(key['nwb_file_name'], 
                                                                           key['track_graph_name'], 
                                                                           key['interval_list_name'], 
                                                                           results, 
                                                                           environment,
                                                                           save_results_path=_DEFAULT_DECODE_SAVE_OUT_DIR, 
                                                                           save_env_path=_DEFAULT_ENVIRONMENT_SAVE_OUT_DIR, 
                                                                           filename_appendage=('_'+decode_id))
            print('Saved results and env fully, now inserting into ClusterlessResults().')
            clusterless_results_path = os.path.join(_DEFAULT_DECODE_SAVE_OUT_DIR, save_results_filename)
            environment_path = os.path.join(_DEFAULT_ENVIRONMENT_SAVE_OUT_DIR, save_env_filename)
            #clusterless_parameters_applied = parameters
            self.insert1(dict(key,
                          epoch_number = epoch_number,
                          epoch_name = epoch_name,
                          valid_time_slice = valid_time_slice,
                          total_unitmarks_entries_by_sg = total_unitmarks_entries_by_sg,
                          clusterless_results_path = clusterless_results_path,
                          environment_path = environment_path,
                          decode_id = decode_id),
                          skip_duplicates = True)
                          #clusterless_parameters_applied = clusterless_parameters_applied)
            print(f'Populated ClusterlessResults for {nwb_file_name} {interval_list_name} with id {decode_id}.')
            #except Exception as e:
                #print(f'ERROR in saving env+results step while attempting to populate ClusterlessResults for {nwb_file_name} {interval_list_name} with id {decode_id} with exception {e}.')    
        #finally:
        #    print(f'Done working with {nwb_file_name} {interval_list_name} with id {decode_id}.')

@schema
class ClusterlessAcausalResultsSummaryParameters(SpyglassMixin, dj.Manual):
    definition="""
    # 
    acausal_results_summary_param_name: varchar(200) # unique str
    ---
    acausal_results_summary_dict: BLOB # dict of params
    """
    def insert_default(self):
        """insert default parameters"""
        param_dict={}
        param_dict['compute_summary'] = True
        self.insert1({"acausal_results_summary_param_name": "default",
                     "acausal_results_summary_dict": param_dict},
                     skip_duplicates=True)

@schema
class ClusterlessAcausalResultsSummarySelection(SpyglassMixin, dj.Manual):
    definition="""
    # Match full decode primary key set with params, use decode_id downstream from here
    -> ClusterlessResults
    -> ClusterlessAcausalResultsSummaryParameters
    acausal_results_summary_param_name='default' : varchar(200)
    ---
    decode_id: varchar(40)
    """

@schema
class ClusterlessAcausalResultsSummary(SpyglassMixin, dj.Computed):
    definition="""
    # Dataframe for easy access to processed decoding results
    -> ClusterlessAcausalResultsSummarySelection
    ---
    -> AnalysisNwbfile
    decode_id : varchar(40)
    results_df_object_id : varchar(40)
    """   
    def make(self,key):
        print(f'Computing ClusterlessAcausalResultsSummary for: {key}')        
        decode_id = (ClusterlessResults & key).fetch1('decode_id')
        nwb_file_name = key['nwb_file_name']
        track_graph_name = key['track_graph_name']
        
        # Create results_df by reading in results, rebuilding environment, and doing basic computations
        # results_df = pd.DataFrame({'test':[1,2,3]}) #  for testing
        
        # Get epoch info
        interval_list_name = key['interval_list_name']
        epoch_number, epoch_interval_list_name = (ClusterlessResults & key).fetch1('epoch_number', 'epoch_name')
        
        # Get decoding related data
        environment, track_graph, results = _retrieve_clusterless_results_data(key)
        position_df, linear_position_df, marks_xr = _align_pos_linpos_marks_to_interval(key)
        
        # Calculate info about acausal results, HPD, trajectory, and mua
        posterior_acausal, max_posterior_acausal, state_continuous_acausal_sum, state_uniform_acausal_sum = _summarize_acausal_results(results, environment)
        
        spatial_coverage_95_hpd, spatial_coverage_50_hpd = _summarize_hpd(posterior_acausal)
        
        (ahead_behind_distance, mental_position_edge_id, actual_position_edge_id, actual_segment, mental_segment,
                    nonlocal_by_segment, ahead_or_not, nonlocal_ahead, nonlocal_behind, 
                    actual_2d_projected_position, mental_2d_projected_position, actual_orientation, head_speed) = _summarize_trajectory(results, track_graph, environment, linear_position_df, position_df)
        
        multiunit_firing_rate = _summarize_mua(key, marks_xr)
        
        results_df = pd.DataFrame(data = {'time': results.time,
                                    'nwb_file_name':np.full(len(ahead_behind_distance),nwb_file_name),
                                    'track_graph_name':np.full(len(ahead_behind_distance),track_graph_name), 
                                    'interval_list_name':np.full(len(ahead_behind_distance),interval_list_name),
                                    'epoch_number':np.full(len(ahead_behind_distance),epoch_number),
                                    'epoch_interval_list_name':np.full(len(ahead_behind_distance),epoch_interval_list_name),
                                    'decode_id':np.full(len(ahead_behind_distance), decode_id),

                                    'max_posterior_acausal':max_posterior_acausal.flatten(),
                                    'ahead_behind_distance':ahead_behind_distance,
                                    'spatial_coverage_50_hpd':spatial_coverage_50_hpd,
                                    'spatial_coverage_95_hpd':spatial_coverage_95_hpd,

                                    'actual_segment':actual_segment,
                                    'mental_segment':mental_segment,
                                    'nonlocal_by_segment':nonlocal_by_segment,
                                    'ahead_or_not':ahead_or_not,
                                    'nonlocal_ahead':nonlocal_ahead,
                                    'nonlocal_behind':nonlocal_behind,

                                    'multiunit_firing_rate':multiunit_firing_rate,

                                    'actual_2d_x_projected_position':actual_2d_projected_position[:,0],
                                    'actual_2d_y_projected_position':actual_2d_projected_position[:,1],
                                    'mental_2d_x_projected_position':mental_2d_projected_position[:,0],
                                    'mental_2d_y_projected_position':mental_2d_projected_position[:,1],                                                        
                                    'actual_orientation':actual_orientation,    
                                    'head_speed': head_speed,

                                    'state_uniform_acausal_sum':state_continuous_acausal_sum,
                                    'state_continuous_acausal_sum':state_continuous_acausal_sum,
                                   }) # Separate analysis to add trial info
        
        # Create nwb, add decode_id to key
        key['analysis_file_name'] = AnalysisNwbfile().create(nwb_file_name)
        key['decode_id'] = decode_id
        
        # Add dfs as nwb objects to the analysis file
        nwb_analysis_file = AnalysisNwbfile()

        key['results_df_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=results_df)

        # Add analysis file to the analysisnwb table
        nwb_analysis_file.add(
            nwb_file_name=key['nwb_file_name'],
            analysis_file_name=key['analysis_file_name'])

        # Insert into ClusterlessAcausalResultsSummary
        self.insert1(key)
        print(f'Populated ClusterlessAlignedInterval.')

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'),
                         *attrs, **kwargs)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]['results_df'].set_index('time')

def _retrieve_clusterless_results_data(key):
    # get the environment
    # env_filename = (ClusterlessResults & key).fetch1('environment_path') #uses form : save_env_path + f'environment_{nwb_file_name}_trackgraph_{track_graph_name}.pkl'
    # environment = joblib.load(env_filename)

    env_filename = (ClusterlessResults & key).fetch1('environment_path') #uses form : save_env_path + f'environment_{nwb_file_name}_trackgraph_{track_graph_name}.pkl'
    # data-sharing / DANDI compatibility (added for code release): on the original machine the file
    # is at its /stelmo path; for shared/DANDI-hosted use it won't exist, so fall back to the
    # reconstructed copy under DATA_DIR.
    try:
        environment = joblib.load(env_filename)
    except (FileNotFoundError, OSError):
        environment = joblib.load(to_local(env_filename))

    environment.fit_place_grid()

    # get the track graph
    track_graph = (TrackGraph() & {'track_graph_name': key['track_graph_name']}).get_networkx_track_graph()

    # get the decode results
    # filename = (ClusterlessResults & key).fetch1('clusterless_results_path') #of form: save_results_path + f'{nwb_file_name}_{track_graph_name}_{interval_list_name}_1D.nc'
    # results = xr.open_dataset(filename)
    filename = (ClusterlessResults & key).fetch1('clusterless_results_path') #of form: save_results_path + f'{nwb_file_name}_{track_graph_name}_{interval_list_name}_1D.nc'
    # data-sharing / DANDI compatibility (added for code release): fall back to the reconstructed
    # copy under DATA_DIR when the original /stelmo path isn't present.
    try:
        results = xr.open_dataset(filename)
    except (FileNotFoundError, OSError):
        results = xr.open_dataset(to_local(filename))

    return environment, track_graph, results

def _align_pos_linpos_marks_to_interval(key):
    # get linpos and pos and marks, trim based on valid time sliced
    valid_time_slice = (ClusterlessResults & key).fetch1('valid_time_slice')
    valid_time_sliced =  slice(valid_time_slice[0], valid_time_slice[1])

    linear_position_df = (IntervalLinearizedPosition & key).fetch1_dataframe()
    linear_position_df = linear_position_df.loc[valid_time_sliced]

    position_df = (IntervalPositionInfo & key).fetch1_dataframe()
    position_df = position_df.loc[valid_time_sliced]

    marks_xr = (UnitMarksIndicator() & key).fetch_xarray()
    marks_xr = marks_xr.sel(time=valid_time_sliced)

    return position_df, linear_position_df, marks_xr

def _summarize_acausal_results(results, environment):
    posterior_acausal = (results
         .acausal_posterior
         .sum('state')
         .where(environment.is_track_interior_))

    max_posterior_acausal = maximum_a_posteriori_estimate(posterior_acausal)
    state_continuous_acausal_sum = results.acausal_posterior.sum('position').sel(state='Continuous').to_numpy()
    state_uniform_acausal_sum = results.acausal_posterior.sum('position').sel(state='Uniform').to_numpy()

    return posterior_acausal, max_posterior_acausal, state_continuous_acausal_sum, state_uniform_acausal_sum

def _summarize_hpd(posterior_acausal):
    hpd_threshold = get_highest_posterior_threshold(posterior_acausal, coverage=0.95)
    spatial_coverage_95_hpd = get_HPD_spatial_coverage(posterior_acausal, hpd_threshold)
    hpd_threshold = get_highest_posterior_threshold(posterior_acausal, coverage=0.50)
    spatial_coverage_50_hpd = get_HPD_spatial_coverage(posterior_acausal, hpd_threshold)
    return spatial_coverage_95_hpd, spatial_coverage_50_hpd

def _summarize_trajectory(results, track_graph, environment, linear_position_df, position_df):
    trajectory_data = get_trajectory_data(
        results.sum('state').acausal_posterior,
        track_graph,
        environment,
        linear_position_df[['projected_x_position', 'projected_y_position']],
        linear_position_df.track_segment_id,
        position_df.head_orientation,
    )

    ahead_behind_distance = get_ahead_behind_distance(
        track_graph, *trajectory_data)

    mental_position_edges = trajectory_data[-1]
    mental_position_edge_id = np.asarray(
        [track_graph.edges[edge]['edge_id'] for edge in mental_position_edges])

    actual_position_edges = trajectory_data[1]
    actual_position_edge_id = np.asarray([track_graph.edges[edge]['edge_id'] for edge in actual_position_edges])

    actual_segment = actual_position_edge_id
    mental_segment = mental_position_edge_id

    nonlocal_by_segment = np.asarray(actual_segment != mental_segment)
    ahead_or_not = np.asarray(ahead_behind_distance>0)
    nonlocal_ahead = np.logical_and(nonlocal_by_segment, ahead_or_not)
    nonlocal_behind = np.logical_and(nonlocal_by_segment, ~ahead_or_not)

    # position related variables
    actual_2d_projected_position = trajectory_data[0]
    mental_2d_projected_position = trajectory_data[3]
    actual_orientation = trajectory_data[2]
    head_speed = position_df.head_speed.values

    return (ahead_behind_distance, mental_position_edge_id, actual_position_edge_id, actual_segment, mental_segment,
            nonlocal_by_segment, ahead_or_not, nonlocal_ahead, nonlocal_behind, 
            actual_2d_projected_position, mental_2d_projected_position, actual_orientation, head_speed)

def _summarize_mua(key, marks_xr):
    SAMPLING_FREQUENCY = key['sampling_rate']
    multiunit_spikes = (np.any(~np.isnan(marks_xr.values), axis=1)
                        ).astype(float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY), index=marks_xr.time,
        columns=['firing_rate'])
    return multiunit_firing_rate.to_numpy().flatten()


def add_trial_info_to_clusterless_results(nwb_file_name, interval_list_name):
    '''Melts trial info with a few extra columns added down to the sampling rate of decoding.
    Joins two dataframes into one that has decoding and trial info.

    Parameters
    ----------
    nwb_file_name: str
    interval_list_name: str
        Of the form "pos X valid times"
    
    Returns
    -------
    clusterless_trial_results: pd.DataFrame
    '''
    # Map pos interval to epoch number
    epoch = (PosValidTimesToEpoch & {'nwb_file_name':nwb_file_name, 'pos_interval_list_name':interval_list_name}).fetch1('epoch')
    
    clusterless_results = (ClusterlessAcausalResultsSummary & {'nwb_file_name':nwb_file_name, 'interval_list_name': interval_list_name}).fetch1_dataframe()
    trials_info = pd.DataFrame(TrialsInfoByEpoch.ByTrial & {'nwb_file_name':nwb_file_name, 'epoch':epoch})
    
    # Add stem id as number instead of letter
    trials_info['stem_id'] = trials_info['stem'].replace({'A':1, 'B':2, 'C':3})
    # Add if each trial is a stem switch or not
    trials_info['stem_switch'] = trials_info['stem'].shift(1, fill_value=trials_info['stem'].head(1)) != trials_info['stem']
    
    # Add how many trials after last switch (don't include trials since put on track - this new col will start with nans)
    trials_from_prior_switch_groups = trials_info['stem_switch'].cumsum()
    trials_info['trials_from_prior_switch'] = trials_info['stem_switch'].groupby(trials_from_prior_switch_groups).cumcount().where(trials_from_prior_switch_groups.gt(0)) #this last part puts nans in first three positions
    # Add how many trials until next switch (don't include trials to last trial of epoch - this new col will end with nans)
    trials_to_next_switch_groups = trials_info['stem_switch'].shift(fill_value=False).cumsum()[::-1]
    trials_info['trials_to_next_switch'] = trials_info['stem_switch'].shift(fill_value=False)[::-1].groupby(trials_to_next_switch_groups).cumcount().where(trials_to_next_switch_groups.lt(max(trials_to_next_switch_groups)))[::-1] #.where(g.gt(0)) #this last part puts nans in first three positions
    
    # Melt trial info out over 2ms time bins, and merge with clusterless results
    trial = 0
    trial_number_by_epoch_series = clusterless_results.index.to_series().between(clusterless_results.index.values[0], trials_info.poke_out_ts[trial], inclusive='left')*trial
    for trial in trials_info['trial_number_by_epoch'].values[1:]:
        x = (trial_number_by_epoch_series.index.to_series().between(trials_info.poke_out_ts[trial-1], trials_info.poke_out_ts[trial], inclusive='left')*trial)
        trial_number_by_epoch_series += x
    # Nans for last decoding time points that aren't part of any trial
    x = trial_number_by_epoch_series.index.to_series().ge(trials_info.poke_out_ts[trial]).replace(True,np.nan)
    trial_number_by_epoch_series += x
    clusterless_results['trial_number_by_epoch'] = trial_number_by_epoch_series
    # Join the trial info to clusterless based on trial number by epoch
    clusterless_trial_results = clusterless_results.merge(trials_info, on=['trial_number_by_epoch','nwb_file_name']).drop(columns='epoch')
    
    return clusterless_trial_results

def add_trial_info_to_clusterless_results_withpokes(nwb_file_name, interval_list_name):   
    # Map pos interval to epoch number
    epoch = (PosValidTimesToEpoch & {'nwb_file_name':nwb_file_name, 'pos_interval_list_name':interval_list_name}).fetch1('epoch')
    
    clusterless_results = (ClusterlessAcausalResultsSummary & {'nwb_file_name':nwb_file_name, 'interval_list_name': interval_list_name}).fetch1_dataframe()
    trials_info = pd.DataFrame(TrialsInfoByEpoch.ByTrial & {'nwb_file_name':nwb_file_name, 'epoch':epoch})
    
    # clusterless results
    trial=0
    poke_series = clusterless_results.index.to_series().between(trials_info.poke_in_ts[trial], trials_info.poke_out_ts[trial], inclusive='left')
    reward_pump_series = clusterless_results.index.to_series().between(trials_info.reward_on_ts[trial], trials_info.reward_off_ts[trial], inclusive='left')
    for trial in trials_info.index[1:]:
        poke_series += poke_series.index.to_series().between(trials_info.poke_in_ts[trial], trials_info.poke_out_ts[trial], inclusive='left')
        reward_pump_series += reward_pump_series.index.to_series().between(trials_info.reward_on_ts[trial], trials_info.reward_off_ts[trial], inclusive='left')
    clusterless_results_df_new = clusterless_results.assign(is_nosepoking = poke_series)
    clusterless_results_df_new = clusterless_results_df_new.assign(is_rewarding = reward_pump_series)
    
    # do it slighlty differently to add to existing df with pump and poke columsn in it (instead of using  fxn made previously) - also the end of this one keeps the time index

    # Add stem id as number instead of letter
    trials_info['stem_id'] = trials_info['stem'].replace({'A':1, 'B':2, 'C':3})
    # Add if each trial is a stem switch or not
    trials_info['stem_switch'] = trials_info['stem'].shift(1, fill_value=trials_info['stem'].head(1)) != trials_info['stem']

    # Add how many trials after last switch (don't include trials since put on track - this new col will start with nans)
    trials_from_prior_switch_groups = trials_info['stem_switch'].cumsum()
    trials_info['trials_from_prior_switch'] = trials_info['stem_switch'].groupby(trials_from_prior_switch_groups).cumcount().where(trials_from_prior_switch_groups.gt(0)) #this last part puts nans in first three positions
    # Add how many trials until next switch (don't include trials to last trial of epoch - this new col will end with nans)
    trials_to_next_switch_groups = trials_info['stem_switch'].shift(fill_value=False).cumsum()[::-1]
    trials_info['trials_to_next_switch'] = trials_info['stem_switch'].shift(fill_value=False)[::-1].groupby(trials_to_next_switch_groups).cumcount().where(trials_to_next_switch_groups.lt(max(trials_to_next_switch_groups)))[::-1] #.where(g.gt(0)) #this last part puts nans in first three positions

    # Melt trial info out over 2ms time bins, and merge with clusterless results
    trial = 0
    trial_number_by_epoch_series = clusterless_results_df_new.index.to_series().between(clusterless_results_df_new.index.values[0], trials_info.poke_out_ts[trial], inclusive='left')*trial
    for trial in trials_info['trial_number_by_epoch'].values[1:]:
        x = (trial_number_by_epoch_series.index.to_series().between(trials_info.poke_out_ts[trial-1], trials_info.poke_out_ts[trial], inclusive='left')*trial)
        trial_number_by_epoch_series += x
    # Nans for last decoding time points that aren't part of any trial
    x = trial_number_by_epoch_series.index.to_series().ge(trials_info.poke_out_ts[trial]).replace(True,np.nan)
    trial_number_by_epoch_series += x
    clusterless_results_df_new['trial_number_by_epoch'] = trial_number_by_epoch_series
    # Join the trial info to clusterless based on trial number by epoch
    clusterless_trial_results = clusterless_results_df_new.reset_index().merge(trials_info, on=['trial_number_by_epoch','nwb_file_name'], how='left').drop(columns='epoch').set_index('time')
    clusterless_trial_results = clusterless_trial_results[~clusterless_trial_results['trial_number_by_epoch'].isnull()]
    
    return clusterless_trial_results