import os
import datajoint as dj
from spyglass.utils import SpyglassMixin
import numpy as np
import pandas as pd
import h5py

from paths import to_local # for dandi based file reconstruction


from spyglass.common import AnalysisNwbfile, Session, Nwbfile 
from spyglass.utils.dj_helper_fn import fetch_nwb

from alison_behav import TrialsInfoByEpoch
from alison_subject import SpatialBanditSubjects

# jl = Julia(runtime=JPATH, compiled_modules=False)
# from julia import Main as Mainjl
# from julia import Pandas

# os.chdir(SPATIAL_BANDIT_TASK_PATH)
# Mainjl.include("hmm_biases_dj.jl")
# Mainjl.include("../code/hmm.jl")
# Mainjl.include("../code/util.jl")
# Mainjl.include("../code/qlearner.jl")

# #Nov2025
# Mainjl.include("../code/beta_bernoulli/beta_bernoulli.jl")
# Mainjl.include("../code/beta_bernoulli/util.jl")


schema = dj.schema('alison_rlmodel')

@schema
class TrialsInfoByRatDfParameters(SpyglassMixin, dj.Manual):
    definition="""
    trials_info_by_rat_params_name: varchar(200)
    ---
    trials_info_by_rat_params_dict: blob #dict of params
    """
    def insert_default(self):
        trials_info_by_rat_params_dict={'use_all_available_trials':True}
        self.insert1(["default",
                     trials_info_by_rat_params_dict],
                     skip_duplicates=True)
        
        trials_info_by_rat_params_dict = {'use_all_available_trials':False, 'is_stable_task':True}
        self.insert1(["stable_default",
                     trials_info_by_rat_params_dict],
                     skip_duplicates=True)
        
        trials_info_by_rat_params_dict = {'use_all_available_trials':False, 'is_stable_task':False}
        self.insert1(["decay_default",
                     trials_info_by_rat_params_dict],
                     skip_duplicates=True)


@schema
class TrialsInfoByRatDfSelection(SpyglassMixin, dj.Manual):
    definition="""
    -> SpatialBanditSubjects #rat id
    -> TrialsInfoByRatDfParameters #param name
    ---
    """

@schema
class TrialsInfoByRatDf(SpyglassMixin, dj.Computed):
    # Key input can come from querying TrialsInfoByEpoch.ByTrial part table for the rat+params specified in the Selection table
    definition="""
    -> TrialsInfoByRatDfSelection
    ---
    -> AnalysisNwbfile # based on first date only of data nwb_file_name!!
    trials_info_by_rat_df_object_id: varchar(40)
    n_days_included: int
    n_epochs_included: int
    n_trials_included: int
    """
    class ByDayNwb(SpyglassMixin, dj.Part):
        # Keeps track of which nwbfiles went into the main analysis nwb by rat
        definition="""
        -> master
        -> Nwbfile
        ---
        -> AnalysisNwbfile
        trials_info_by_rat_df_object_id: varchar(40) # by RAT not day!
        """
                
    def make(self, key):
        subject_id = key['subject_id']

        # Turn trial-wise data into df
        trials_info_by_rat_df = pd.DataFrame(TrialsInfoByEpoch.ByTrial & (Session & {'subject_id':subject_id}))
        trials_info_by_rat_df.sort_values(by=['nwb_file_name','epoch','trial_number_by_epoch'], inplace=True)

        # Get/use parameters to limit data do a subset of trials to include
        trials_info_by_rat_params_dict = (TrialsInfoByRatDfParameters & {'trials_info_by_rat_params_name': key['trials_info_by_rat_params_name']}).fetch1('trials_info_by_rat_params_dict')
        use_all_available_trials = trials_info_by_rat_params_dict['use_all_available_trials']
        if not use_all_available_trials:
            if trials_info_by_rat_params_dict['is_stable_task']:
                #Only include stable task data, that doesn't have a decay_percent value
                data_to_include = trials_info_by_rat_df[trials_info_by_rat_df['decay_percent'].isnull()]
                #handle possible trailing  experiment data appropriately
                print("Warning: not currently removing any trailing data from fun final experiments")

            elif not trials_info_by_rat_params_dict['is_stable_task']:
                #Only include decay task data that isn't NaN for decay_percent column
                data_to_include = trials_info_by_rat_df[~trials_info_by_rat_df['decay_percent'].isnull()]
                #Note possible trailing experiment data to be handled appropriately
                print("Warning: not currently removing any trailing data from fun final experiments")
                print("Warning: going to re-index df to start at 0, so the first decay trial starts at index 0 insted of ~10k")
                data_to_include = data_to_include.reset_index(drop=True)
            
        elif use_all_available_trials:
            data_to_include = trials_info_by_rat_df #don't trim the data to include at all
            
        else:
            raise Exception(f'Functionality has not yet been implemented for the given parameters: {trials_info_by_rat_params_dict}')
        
        # Calculate some summary numbers for user-friendliness
        n_trials_included = len(data_to_include)
        n_epochs_included = data_to_include.groupby(['nwb_file_name','epoch']).ngroups
        n_days_included = data_to_include.groupby(['nwb_file_name']).ngroups

        # Create nwb, add decode_id to key
        nwb_file_name = data_to_include['nwb_file_name'][data_to_include.index[0]] # use the first date's nwb_file_name to create the full rat AnalysisNwbfile
        key['analysis_file_name'] = AnalysisNwbfile().create(nwb_file_name)
        key['n_trials_included'] = n_trials_included
        key['n_days_included'] = n_days_included
        key['n_epochs_included'] = n_epochs_included

        # Add dfs as nwb objects to the analysis file
        nwb_analysis_file = AnalysisNwbfile()

        # handle an Exception  20221207. Needed for peanut and Senor.
        columns_to_try_recasting = ['decay_percent', 'p_rew_reset_leaf1', 'p_rew_reset_leaf2', 'p_rew_reset_leaf3', 'p_rew_reset_leaf4', 'p_rew_reset_leaf5', 'p_rew_reset_leaf6']
        for c in columns_to_try_recasting:
            #data_to_include[c] = -1000 # this wokred
            #data_to_include[c].fillna(value=np.nan, inplace=True) #this also worked
            data_to_include[c] = data_to_include[c].astype('float64')
        print(data_to_include.dtypes)

        key['trials_info_by_rat_df_object_id'] = nwb_analysis_file.add_nwb_object(
                analysis_file_name=key['analysis_file_name'],
                nwb_object=data_to_include)
        
        # Don't add analysis file to the analysisnwb table because these analysis nwbs aren't uniquely indexed by nwb file name, but instead by rat
        nwb_analysis_file.add(
            nwb_file_name,
            key['analysis_file_name'])

        # Insert into computed table
        TrialsInfoByRatDf.insert1(key)
        print(f'Populated TrialsInfoByRatDf for key: {key}')

        # Insert into part table
        part_key = {'subject_id': subject_id,
                    'trials_info_by_rat_params_name': key['trials_info_by_rat_params_name'],
                    'analysis_file_name': key['analysis_file_name'],
                    'trials_info_by_rat_df_object_id': key['trials_info_by_rat_df_object_id']
                    }
        nwb_file_names = np.unique(data_to_include['nwb_file_name'].values)
        for nwb in nwb_file_names:
            part_key['nwb_file_name'] = nwb
            TrialsInfoByRatDf.ByDayNwb.insert1(part_key)
            print(f'Populated TrialsInfoByRatDf.ByDayNwb for same key as in master table, but with nwb_file_name: {nwb}')


        # key input just comes from querying TrialsInfoByEpoch.ByTrial part table for a the rat specified in selection table
        # read in those blobs, turn into df from each epoch
        # pull in params and if True
        # make a df for each day based on concatenating the info by epoch - this can go into the parts table, though redundant
        # keep track of total n trials, epochs, days that will go into the key of new tables to be inserted
        # make an analysis nwb file by rat that has the rat df in it
        # insert in AnalysisNwbfile table
        # insert into TrialsInfoByRat includign the analysis nwb
    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'),
                         *attrs, **kwargs)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]['trials_info_by_rat_df']

@schema
class TrialsInfoCsvParameters(SpyglassMixin, dj.Manual):
    definition="""
    trials_info_csv_params_name: varchar(200)
    ---
    trials_info_csv_params_dict: blob
    """
    def insert_default(self):
        trials_info_csv_params_dict={}
        self.insert1(["original_csv_format",
                     trials_info_csv_params_dict],
                     skip_duplicates=True)

@schema
class TrialsInfoCsvSelection(SpyglassMixin, dj.Manual):
    definition="""
    -> TrialsInfoByRatDf
    -> TrialsInfoCsvParameters
    ---
    """

@schema
class TrialsInfoCsv(SpyglassMixin, dj.Computed):
    # While TrialsInfoByRatDf controlled rows in csv, this table effectively controls columns
    definition="""
    -> TrialsInfoCsvSelection
    ---
    trials_info_by_rat_csv_path: varchar(200)
    """
    def make(self, key):
        subject_id = key['subject_id']
        trials_info_by_rat_params_name = key['trials_info_by_rat_params_name']
        trials_info_csv_params_name = key['trials_info_csv_params_name']
        trials_info_csv_params_dict = (TrialsInfoCsvParameters & {'trials_info_csv_params_name':trials_info_csv_params_name}).fetch1('trials_info_csv_params_dict')
        trials_info_by_rat_df = (TrialsInfoByRatDf() & key).fetch1_dataframe()
        if not trials_info_csv_params_dict:
            #make original csv column formatting if no params in dict
            
            #concatenate p_rew_leafX into one contingency column
            p_rews = ['p_rew_leaf'+str(i) for i in range(1,7)]
            for p_rew_x in p_rews:
                trials_info_by_rat_df[p_rew_x] = trials_info_by_rat_df[p_rew_x].apply(str)
            trials_info_by_rat_df['contingency'] = trials_info_by_rat_df[p_rews[0]]
            for p_rew_x in p_rews[1:]:
                trials_info_by_rat_df['contingency'] += trials_info_by_rat_df[p_rew_x]
            
            #make a date only column rather than nwb file name
            trials_info_by_rat_df['date'] = trials_info_by_rat_df['nwb_file_name'].str.rstrip('_.nwb')
            trials_info_by_rat_df['date'] = trials_info_by_rat_df['date'].str.lstrip(subject_id)
            
            #make a session column that keeps track of nth run epoch within a day 
            trials_info_by_rat_df['session'] = trials_info_by_rat_df.groupby(['date'])['epoch'].transform(lambda x: pd.factorize(x)[0]+1)
            
            trials_info_by_rat_df.rename(columns={'trial_number_by_epoch':'trial'}, inplace=True)
            COLUMNS = ['leaf','stem','reward','contingency','date','session','trial']
            csv_df = trials_info_by_rat_df[COLUMNS]
            csv_df.index.name = None
            
            csvdir = '/stelmo/alison/behavior_csvs/'
            filename = f'{subject_id}_{trials_info_by_rat_params_name}_{trials_info_csv_params_name}.csv'
            behavior_csv_path = csvdir+filename
            csv_df.to_csv(behavior_csv_path)
            
            key['trials_info_by_rat_csv_path'] = behavior_csv_path
            self.insert1(key)
            print(f'Saved behavior csv to {behavior_csv_path} and populated TrialsInfoCsv for key {key}')
            
    def fetch1_csv_as_df(self):
        # behavior_csv_path = self.fetch1('trials_info_by_rat_csv_path')
        # return pd.read_csv(behavior_csv_path, index_col=0)
        behavior_csv_path = self.fetch1('trials_info_by_rat_csv_path')
        # data-sharing / DANDI compatibility (added for code release): fall back to the reconstructed
        # copy under DATA_DIR when the original /stelmo path isn't present
        try:
            return pd.read_csv(behavior_csv_path, index_col=0)
        except (FileNotFoundError, OSError):
            return pd.read_csv(to_local(behavior_csv_path), index_col=0)

@schema
class BehaviorModelParameters(SpyglassMixin, dj.Manual):
    definition="""
    behavior_model_params_name: varchar(200)
    ---
    behavior_model_params_dict: blob
    other_params_dict: blob
    paths_dict: blob
    """
    def insert_default(self):
        # Start with hmm params
        behavior_model_params_dict={
            'hmm':True,
            'q':False,
            'rewscaled':True,
            'leaf' : True,
            'stay' : True,
            'turn' : True,
            'leafturn' : False,
            'spatial' : False,
            'leafspatial' : True,
            'y2' : False,
            'retainbelief' : False,            
            }
        other_params_dict={
            "extended": True,
            "maxiter": 100,
            "compress": True,
            }
        paths_dict={
            'csv_out_path': '/stelmo/alison/behavior_model_csvs',
            'jld2_out_path': '/stelmo/alison/behavior_model_jld2s',
            }
        self.insert1(["default_hmm",
                     behavior_model_params_dict,
                     other_params_dict,
                     paths_dict],
                     skip_duplicates=True)
        
        # Change default parameters for q learner 
        behavior_model_params_dict['hmm'] = False
        behavior_model_params_dict['q']= True
        behavior_model_params_dict['y2'] = True
        
        self.insert1(["default_qlearner",
             behavior_model_params_dict,
             other_params_dict,
             paths_dict],
             skip_duplicates=True)

@schema
class BehaviorModelSelection(SpyglassMixin, dj.Manual):
    definition="""
    -> TrialsInfoCsv
    -> BehaviorModelParameters
    ---
    """

@schema
class BehaviorModel(SpyglassMixin, dj.Computed):
    definition="""
    -> BehaviorModelSelection
    ---
    behavior_model_csv_path: varchar(200)
    behavior_model_fit_path: varchar(200)
    """
    # Make csv and jld2 files here, but only turn them into nwbs in next table. 
    def make(self, key):
        #from julia import Mainjl
        # Get variables from key
        subject_id = key['subject_id']
        trials_info_by_rat_params_name = key['trials_info_by_rat_params_name']   
        trials_info_csv_params_name = key['trials_info_csv_params_name']
        behavior_model_params_name = key['behavior_model_params_name']
        
        # Get path to one rat beh csv - model input
        trials_info_by_rat_csv_path = (TrialsInfoCsv & key).fetch1("trials_info_by_rat_csv_path")
        
        # Get params for model/jl code/file writing
        behavior_model_params_dict = (BehaviorModelParameters() & {
            'behavior_model_params_name':behavior_model_params_name}).fetch1()['behavior_model_params_dict']
        other_params_dict = (BehaviorModelParameters() & {
            'behavior_model_params_name':behavior_model_params_name}).fetch1()['other_params_dict']
        paths_dict = (BehaviorModelParameters() & {
            'behavior_model_params_name':behavior_model_params_name}).fetch1()['paths_dict']
        # Create full file output names for model fit results and q val/DV results
        behavior_model_fit_path = paths_dict['jld2_out_path']
        behavior_model_csv_path = paths_dict['csv_out_path']
        model_fit_results_filename =  f'model_fit_{subject_id}_{trials_info_by_rat_params_name}_{trials_info_csv_params_name}_{behavior_model_params_name}.jld2'
        Q_vals_filename = f'q_vals_{subject_id}_{trials_info_by_rat_params_name}_{trials_info_csv_params_name}_{behavior_model_params_name}.csv' 
        behavior_model_fit_full_path = behavior_model_fit_path + '/' + model_fit_results_filename
        behavior_model_csv_full_path = behavior_model_csv_path + '/' + Q_vals_filename    
        
        # Add a couple extra params that are specific to HMM depletion and Qlearner models respectively. or move  to the param dicts in tables above.
        add_depletion_factor = not (TrialsInfoByRatDfParameters & {'trials_info_by_rat_params_name':trials_info_by_rat_params_name}).fetch1('trials_info_by_rat_params_dict')['is_stable_task']
        add_initial_Q = False # only for q learner
        # Add some more default params for teh run_hmm and run_q jl fxns. can move these to teh param dicts in tables above.
        delay_turn_bias = False #for all models
        add_leaf = True # for hmm,  for betalik, moving to param dict for that
        full = True # was true for hmm, for betalik moving to param dict for that
        quiet = False
        # some get overwritten when model is run in julia^
        
        # For either hmm or qlearner, load data for one rat, run the model, save the model fit jld2, and save teh q value/DV csv.
        if behavior_model_params_dict['hmm']:
            data = Mainjl.load_animal_dj(subject_id, trials_info_by_rat_csv_path)
            results = Mainjl.run_hmm(data,
                        maxiter=other_params_dict['maxiter'],
                        full=full,
                        extended=other_params_dict['extended'],
                        quiet=quiet,
                        add_βleaf=behavior_model_params_dict['leaf'],
                        add_stay_bias=behavior_model_params_dict['stay'],
                        add_turn_bias=behavior_model_params_dict['turn'],
                        add_spatial_bias=behavior_model_params_dict['spatial'],
                        add_leaf_turn_bias=behavior_model_params_dict['leafturn'],
                        add_leaf_spatial_bias=behavior_model_params_dict['spatial'],
                        add_γ2=behavior_model_params_dict['y2'],
                        add_depletion_factor=add_depletion_factor,
                        add_retain_belief=behavior_model_params_dict['retainbelief'],
                        delay_turn_bias=delay_turn_bias,
                        rewscaled=behavior_model_params_dict['rewscaled'],
                        add_leaf=add_leaf) 
            # params differing from hmm to pass are: 
            try:
                Mainjl.save(behavior_model_fit_full_path, model_fit_results_filename, results, compress=other_params_dict['compress'])  # Saves to jld2 rn  but now that calling jl right here coudl save directly to nwb as np arrays  
            except:
                print("WARNING: not saving behavior model fit results jld2")

            Qs = Mainjl.find_Q_vals_by_day_hmm(data, results, add_leaf=add_leaf, rewscaled=behavior_model_params_dict['rewscaled'], delay_turn_bias=delay_turn_bias) 
            
        if behavior_model_params_dict['q']:
            data = Mainjl.load_animal_dj(subject_id, trials_info_by_rat_csv_path)
            results = Mainjl.run_q(data,
                        maxiter=other_params_dict['maxiter'],
                        full=full,
                        extended=other_params_dict['extended'],
                        quiet=quiet,
                        add_βleaf=behavior_model_params_dict['leaf'],
                        add_stay_bias=behavior_model_params_dict['stay'],
                        add_turn_bias=behavior_model_params_dict['turn'],
                        add_spatial_bias=behavior_model_params_dict['spatial'],
                        add_leaf_turn_bias=behavior_model_params_dict['leafturn'],
                        add_leaf_spatial_bias=behavior_model_params_dict['spatial'],
                        add_γ2=behavior_model_params_dict['y2'],
                        add_initial_Q=add_initial_Q,
                        add_retain_belief=behavior_model_params_dict['retainbelief'],
                        delay_turn_bias=delay_turn_bias,
                        rewscaled=behavior_model_params_dict['rewscaled'],
                        add_leaf=add_leaf) 
            try:
                Mainjl.save(behavior_model_fit_full_path, model_fit_results_filename, results, compress=other_params_dict['compress'])  
            except:
                print("WARNING: not saving behavior model fit results jld2")
            Qs = Mainjl.find_Q_vals_by_day_qlearner(data, results, add_leaf=add_leaf, rewscaled=behavior_model_params_dict['rewscaled'], delay_turn_bias=delay_turn_bias)
        
        ##Nov2025
        if behavior_model_params_dict['beta_bernoulli']: #new
            depletion = not (TrialsInfoByRatDfParameters & {'trials_info_by_rat_params_name':trials_info_by_rat_params_name}).fetch1('trials_info_by_rat_params_dict')['is_stable_task'] # similar strategy to add_depletion_factor
            data = Mainjl.load_animal_dj(subject_id.lower(),
                                         trials_info_by_rat_csv_path,
                                         depletion=depletion) #also defaults to false in more recent utils.jl
            results = Mainjl.run_beta_lik(data,
                        maxiter=other_params_dict['maxiter'],
                        full=behavior_model_params_dict['full'], #newly in params dict
                        extended=other_params_dict['extended'],
                        add_βgo = behavior_model_params_dict['add_beta_go'], #new
                        add_βstay = behavior_model_params_dict['add_beta_stay'], #new
                        quiet=quiet, # hardcoded above rn
                        add_βleaf=behavior_model_params_dict['leaf'],
                        add_stay_bias=behavior_model_params_dict['stay'],
                        add_turn_bias=behavior_model_params_dict['turn'],
                        add_spatial_bias=behavior_model_params_dict['spatial'],
                        add_leaf_turn_bias=behavior_model_params_dict['leafturn'],
                        add_leaf_spatial_bias=behavior_model_params_dict['spatial'],
                        add_beta_decay=behavior_model_params_dict['add_beta_decay'], #new
                        add_a_baseline=behavior_model_params_dict['alpha'], #new
                        add_b_baseline=behavior_model_params_dict['beta'], #new
                        add_γ2=behavior_model_params_dict['y2'],
                        add_depletion_factor=add_depletion_factor, # based on is_stable_task
                        add_retain_belief=behavior_model_params_dict['retainbelief'],
                        delay_turn_bias=delay_turn_bias, # false, hardcoded above rn
                        rewscaled=behavior_model_params_dict['rewscaled'],
                        add_leaf=behavior_model_params_dict['add_leaf']) # newly in params dict
            try:
                Mainjl.save(behavior_model_fit_full_path,
                            model_fit_results_filename,
                            results,
                            compress=other_params_dict['compress'])
            except:
                print("WARNING: not saving beahvior model fit results jld2")
            Qs = Mainjl.find_Q_vals_beta_lik(data,
                                             results,
                                             add_leaf=behavior_model_params_dict['add_leaf'],
                                             delay_turn_bias=delay_turn_bias,
                                             rewscaled=behavior_model_params_dict['rewscaled'])
        #save Qs to csv
        Pandas.DataFrame(Qs).to_csv(behavior_model_csv_full_path)
        
        # Put in table
        key['behavior_model_csv_path'] = behavior_model_csv_full_path
        key['behavior_model_fit_path'] = behavior_model_fit_full_path
        self.insert1(key)
        print(f'Saved behavior model results csv to {behavior_model_csv_full_path},', 
              f'saved jld2 model fit results to {behavior_model_fit_full_path},',
              f'and populated BehviorModel for key {key}.')
        
    def fetch1_csv_as_df(self):
        # behavior_model_csv_path = self.fetch1('behavior_model_csv_path')
        # df_behavior_model_qs = pd.DataFrame(pd.read_csv(behavior_model_csv_path, index_col=0))
        behavior_model_csv_path = self.fetch1('behavior_model_csv_path')
        # data-sharing / DANDI compatibility (added for code release): fall back to the reconstructed
        # copy under DATA_DIR when the original /stelmo path isn't present
        try:
            df_behavior_model_qs = pd.DataFrame(pd.read_csv(behavior_model_csv_path, index_col=0))
        except (FileNotFoundError, OSError):
            df_behavior_model_qs = pd.DataFrame(pd.read_csv(to_local(behavior_model_csv_path), index_col=0))
        return df_behavior_model_qs
    
    def fetch1_jld2_as_hdf5(self):
        # behavior_model_fit_path = self.fetch1('behavior_model_fit_path')
        # return h5py.File(behavior_model_fit_path, "r")
        behavior_model_fit_path = self.fetch1('behavior_model_fit_path')
        # data-sharing / DANDI compatibility (added for code release): fall back to the reconstructed
        # copy under DATA_DIR when the original /stelmo path isn't present
        # note actually jld2 which file tpype export doesn't handle but not used for figs so
        # includign for consistency only here..
        try:
            return h5py.File(behavior_model_fit_path, "r")
        except (FileNotFoundError, OSError):
            return h5py.File(to_local(behavior_model_fit_path), "r")

@schema
class BehaviorModelResults(SpyglassMixin, dj.Computed):
    definition="""
    -> BehaviorModel
    ---
    -> AnalysisNwbfile
    behavior_model_results_object_id: varchar(40)
    """
    class ByDay(SpyglassMixin, dj.Part): #keep extra nwb files that are redundant of this info by day epoch, ableto parse data that way, or alighn it with other behavior data that is broken up same way
        definition="""
        -> master
        -> Nwbfile #to keep track of date information through nwb file name
        ---
        -> AnalysisNwbfile
        behavior_model_results_by_day_object_id: varchar(40)
        n_epochs_included: int
        n_trials_included: int
        """
        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'),
                         *attrs, **kwargs)
    
        def fetch1_dataframe(self):
            return self.fetch_nwb()[0]['behavior_model_results_by_day']
    
    
    def make(self, key):
        model_results_decision_variables = (BehaviorModel & key).fetch1_csv_as_df()
        
        #test dec 22
        # print(key['subject_id'])
        # print(model_results_decision_variables['date'].astype('string'))
        #Nov2025 
        print(f"beh model params name 0:4 != beta: {key['behavior_model_params_name'][0:4] != 'beta'}")
        if key['behavior_model_params_name'][0:4] != 'beta':
            # Add nwb file name column back into df
            if key['subject_id'] == 'Senor':
                model_results_decision_variables['nwb_file_name'] = model_results_decision_variables['date'].astype('string') +'_.nwb'
            elif key['subject_id'] != 'Senor':
                model_results_decision_variables['nwb_file_name'] = key['subject_id'] + model_results_decision_variables['date'].astype('string') +'_.nwb'
        else:
            model_results_decision_variables['nwb_file_name'] = key['subject_id'].lower() + model_results_decision_variables['date'].astype('string') +'_.nwb'

        # print(f'nwbfname: { model_results_decision_variables['nwb_file_name'] } ')

        # Rename trials back to trials number by epoch
        model_results_decision_variables.rename(columns={'trial':'trial_number_by_epoch'}, inplace=True)
                
        # Recreate session and make sure it's the same
        trials_info_by_rat_df = (TrialsInfoByRatDf & key).fetch1_dataframe()
        recreated_session_test = trials_info_by_rat_df.groupby(['nwb_file_name'])['epoch'].transform(lambda x: pd.factorize(x)[0]+1)
        assert recreated_session_test.equals(model_results_decision_variables['session']), f'recreated session series (len {len(trials_info_by_rat_df)}) and sesssion series (len {len(model_results_decision_variables)}) from model results are not identical'
        # Insert epoch info back into df
        model_results_decision_variables['epoch'] = trials_info_by_rat_df['epoch']
        
        # Reorganize columns so nwb ep trial come before the rest of the data
        movecolumn = model_results_decision_variables.pop('trial_number_by_epoch')
        model_results_decision_variables.insert(0,'trial_number_by_epoch', movecolumn)
        movecolumn = model_results_decision_variables.pop('epoch')
        model_results_decision_variables.insert(0,'epoch', movecolumn)
        movecolumn = model_results_decision_variables.pop('nwb_file_name')
        model_results_decision_variables.insert(0,'nwb_file_name', movecolumn)

        #Nov2025 edits
        if key['behavior_model_params_name'][0:4] != 'beta':
            #Just like in upstream tables, use earliest dates's nwb file name for analysis file 
            if key['subject_id'] != 'Senor':
                nwb_file_name = str(key['subject_id']) + str(model_results_decision_variables['date'][0]) + '_.nwb'
            elif key['subject_id'] == 'Senor':
                nwb_file_name =  str(model_results_decision_variables['date'][0]) + '_.nwb'
        else:
            nwb_file_name = str(key['subject_id']).lower() + str(model_results_decision_variables['date'][0]) + '_.nwb'
        print(f'first date would be nwb file name: {nwb_file_name}.')
        
        key['analysis_file_name'] = AnalysisNwbfile().create(nwb_file_name)
        
        nwb_analysis_file = AnalysisNwbfile()
        
        key['behavior_model_results_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=model_results_decision_variables)
        
        nwb_analysis_file.add(
            nwb_file_name,
            key['analysis_file_name'])
        
        BehaviorModelResults.insert1(key)
        print(f'Populated BehaviorModelResults for key: {key}')
        
        part_key = key.copy()
        part_key.pop('analysis_file_name',None)
        part_key.pop('behavior_model_results_object_id',None)
        dates = np.unique(model_results_decision_variables['date'].values)
        for date in dates:
            model_results_decision_variables_by_day = model_results_decision_variables[model_results_decision_variables['date'] == date]
            
            if key['behavior_model_params_name'][0:4] != 'beta':
                if key['subject_id'] != 'Senor':
                    nwb_file_name = str(key['subject_id']) + str(date) + '_.nwb'
                elif key['subject_id'] == 'Senor':
                    nwb_file_name = str(date) + '_.nwb'
            else:
                nwb_file_name = str(key['subject_id']).lower() + str(date) + '_.nwb'
            part_key['nwb_file_name'] = nwb_file_name
            
            part_key['analysis_file_name'] = AnalysisNwbfile().create(nwb_file_name)

            nwb_analysis_file = AnalysisNwbfile()
            
            part_key['behavior_model_results_by_day_object_id'] = nwb_analysis_file.add_nwb_object(
                analysis_file_name=part_key['analysis_file_name'],
                nwb_object=model_results_decision_variables_by_day)
            
            nwb_analysis_file.add(
                nwb_file_name,
                part_key['analysis_file_name'])
            
            part_key['n_epochs_included'] = model_results_decision_variables_by_day.groupby(['nwb_file_name','epoch']).ngroups
            part_key['n_trials_included'] = len(model_results_decision_variables_by_day)
            
            BehaviorModelResults.ByDay.insert1(part_key)
            print(f'Populated BehaviorModelResults.ByDay for key {part_key}')
        print(f'Populated all entries in BehaviorModelResults and part table ByDay for key {key}.')
            
    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'),
                         *attrs, **kwargs)
    
    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]['behavior_model_results']