import os
import numpy as np
import pandas as pd
from plot_rlmodel import add_rl_results_analysis_columns

def _set_nan_at_max(group):
    max_test_series = group['trials_to_next_switch_groups'].max()
    if group['stem_switch'][-1:].values[0] == False: # if final trial of epoch isn't a switch, then last group cant be countign down to next switch, so then replace with nans
        group['trials_to_next_switch_with_nans'] = group.apply(lambda row: np.nan if row['trials_to_next_switch_groups'] == max_test_series else row['trials_to_next_switch'], axis=1)
    else:
        group['trials_to_next_switch_with_nans'] = group.apply(lambda row: row['trials_to_next_switch'] if row['trials_to_next_switch_groups'] == max_test_series else row['trials_to_next_switch'], axis=1)
    return group

def create_hmm_plus(rl_results, ):
    '''
    rl_results is a df for one epoch of BehaviorModelResults info
    return the same df with many added columns for dv calculations
    this doesn't make any checks of being a good stable epoch withoiut 100all 50all etc
    '''
    n_shifts = 10
    
    data_df = rl_results
    
    # Identify stem switch trials, and handle first trial from the shift
    data_df['stem_id'] = data_df['stem'].replace({'A':1, 'B':2, 'C':3})
    data_df['stem_shifted'] = data_df.groupby(by = ['nwb_file_name','epoch'])['stem'].shift(1).bfill(limit=1)
    data_df['stem_switch'] = data_df['stem'] != data_df['stem_shifted']
    
    # Calculate trials from prior/next switch
    data_df['trials_from_prior_switch_groups'] = data_df.groupby(by = ['nwb_file_name','epoch'])['stem_switch'].cumsum()
    data_df['trials_from_prior_switch'] = data_df.groupby(by = ['nwb_file_name','epoch','trials_from_prior_switch_groups']).cumcount()
    data_df['trials_from_prior_switch'] = data_df['trials_from_prior_switch'].where(data_df['trials_from_prior_switch_groups'].gt(0), np.nan)
    
    # This is the og line, dysfunctions with one group setup
    # data_df['trials_to_next_switch_groups'] = data_df.groupby(by = ['nwb_file_name','epoch']).apply(lambda x_df: x_df['stem_switch'].shift(fill_value=False).cumsum()).reset_index(name='group').set_index('id')['group']
    
    # New version
    tmp = data_df.groupby(by = ['nwb_file_name','epoch']).apply(lambda x_df: x_df['stem_switch'].shift(fill_value=False
                ).cumsum()).reset_index().melt(id_vars=['nwb_file_name','epoch'], var_name='id', value_name='group').set_index('id')
    data_df['trials_to_next_switch_groups'] = tmp['group']
    
    data_df['trials_to_next_switch'] = data_df.groupby(by=['nwb_file_name','epoch','trials_to_next_switch_groups'])['trials_to_next_switch_groups'].cumcount(ascending=False)
    
    # Ensure that don't count trials to end of epoch as trials to next switch 
    data_df = data_df.groupby(by=['nwb_file_name','epoch']).apply(_set_nan_at_max)    
    data_df['trials_from_next_switch'] = data_df['trials_to_next_switch_with_nans']*-1
    
    # Calculate bout lengths
    data_df['try_bout_idx'] = data_df.groupby(by = ['nwb_file_name','epoch'])['stem'].transform(lambda x_df: (x_df != x_df.shift(1)).cumsum())
    data_df['bout_len'] = data_df.groupby(by = ['nwb_file_name','epoch', 'try_bout_idx'])['try_bout_idx'].transform(len)
    data_df['bout_len_new'] = data_df.groupby(by = ['nwb_file_name','epoch', 'try_bout_idx'])['trials_from_prior_switch'].transform(lambda x: np.max(x))
    
    # Calculate recent rewards and switches for any conditional analyses
    for n in range(1,n_shifts+1):
        data_df[f"reward_{n}_ago"] = data_df.groupby(["nwb_file_name","epoch"])['reward'].shift(n)
        data_df[f"stem_switch_{n}_ago"] = data_df.groupby(["nwb_file_name","epoch"])['stem_switch'].shift(n)
    
    data_df = data_df.groupby(by = ['nwb_file_name','epoch']).apply(add_rl_results_analysis_columns)
#     data_df = add_rl_results_analysis_columns(data_df)
    
    # Find value of the chosen stem on the trial
    data_df['currentQstem'] = [f"Qstem{i}" for i in data_df.stemchoice.values]
    data_df['Qstemcurrent'] = data_df['Qstem1'].values
    data_df.loc[data_df['currentQstem']=='Qstem2', 'Qstemcurrent'] = data_df.loc[data_df['currentQstem']=='Qstem2','Qstem2'].values
    data_df.loc[data_df['currentQstem']=='Qstem3', 'Qstemcurrent'] = data_df.loc[data_df['currentQstem']=='Qstem3','Qstem3'].values
    
    # Find value of the initial stem of the trial
    data_df['stemchoice_shifted'] = data_df.groupby(by=['nwb_file_name','epoch'])['stemchoice'].shift()
    data_df['currentQstem_initial'] = [f"Qstem{int(i)}" if ~np.isnan(i) else np.nan for i in data_df.stemchoice_shifted.values]
    data_df['Qstemcurrent_initial'] = data_df['Qstem1'].where(data_df['currentQstem_initial'].notna(), np.nan)
    data_df.loc[data_df['currentQstem_initial']=='Qstem2', 'Qstemcurrent_initial'] = data_df.loc[data_df['currentQstem_initial']=='Qstem2','Qstem2'].values
    data_df.loc[data_df['currentQstem_initial']=='Qstem3', 'Qstemcurrent_initial'] = data_df.loc[data_df['currentQstem_initial']=='Qstem3','Qstem3'].values
    
    # This mostly produces negatives 
    data_df['val_difference'] = (data_df['Qstem1']+data_df['Qstem2']+data_df['Qstem3']-data_df['Qstemcurrent_initial'])/2 - data_df['Qstemcurrent_initial']
    
    return data_df
    
def rescale_hmm_plus(rl_results):
    data_df = rl_results
    
    # Simple forced rescaling/shifting to get in zone of 0 to 1
    data_df['state_entropy_scaled'] = data_df['state_entropy']/4

    data_df['Q_leaf_t_shifted'] = data_df['Q_leaf_t']+.6
    data_df['Q_leaf_tminus1_shifted'] = data_df['Q_leaf_tminus1']+.6

    data_df['Qstemcurrent_shifted_scaled'] = (data_df['Qstemcurrent']+6)/5
    data_df['val_difference_shifted_scaled'] = (data_df['val_difference']+6)/5
    
    data_df['bout_len_log_scaled'] = np.log(data_df['bout_len_new']+1)/5
    data_df['trials_from_next_switch_abs_log_scaled'] = np.log(-1*data_df['trials_from_next_switch']+1)/5
    data_df['trials_from_prior_switch_log_scaled'] = np.log(data_df['trials_from_prior_switch']+1)/5
    data_df['stem_switch_float'] = data_df['stem_switch'].astype(float)
    
    # data_df['Q_leaf_t_shifted_scaled'] = data_df['Q_leaf_t_shifted']/(round(data_df['Q_leaf_t_shifted'].max(), 2)+.1)
    # data_df['Qstemcurrent_shifted_rescaled'] = data_df['Qstemcurrent_shifted_scaled']/(round(data_df['Qstemcurrent_shifted_scaled'].max(), 2)+.5)
    
    data_df['Q_leaf_t_shifted_scaled'] = data_df['Q_leaf_t_shifted']/(1.25) # max q leaf t shifted ranges 1 to 1.2 across rats all stable days
    data_df['Qstemcurrent_shifted_rescaled'] = data_df['Qstemcurrent_shifted_scaled']/(3) # max qstem current shifted scaled ranges 2.1-2.58 across rats all stable days
    
    return data_df

def add_rpe_deltaq_to_hmm_plus(rl_results):
    data_df = rl_results
    
    # This is the rpe that is experienced in the post outcome period
    data_df['rpe_rew_minus_q'] = data_df['reward']-data_df['Q_leaf_t_shifted_scaled'] 
    
    grouped = data_df.groupby(by=['nwb_file_name','epoch'])
    # This is the rpe that was just experienced based on most recent outcome
    data_df['rpe_rew_minus_q_1_ago'] = grouped['rpe_rew_minus_q'].shift(1)
    # This is the rpe that was experienced two outcomes ago (often the same port being approached currently if stay trial sequence)
    data_df['rpe_rew_minus_q_2_ago'] = grouped['rpe_rew_minus_q'].shift(2)

    ## Calculate delta q at each leaf, current leaf, prior leaf, etc
    # The delta q at leaves 1-6 due to the outcome of the current trial
    for leaf in range(1,7):
        data_df[f'next_Q{leaf}'] = grouped[f'Q{leaf}'].shift(-1)
        data_df[f'delta_Q{leaf}'] = data_df[f'next_Q{leaf}']-data_df[f'Q{leaf}']
    
    grouped = data_df.groupby(by=['nwb_file_name','epoch'])
    data_df['prior_leaf'] = grouped.leaf.shift()
    
    # Delta q of chosen leaf on this trial
    data_df['delta_Qleaf_t_name'] = [f"delta_Q{i}" for i in data_df.leaf.values.astype(int)]
    data_df['delta_Qleaf_t'] = np.nan
    for leaf in range(1,7):
        data_df.loc[
            data_df['delta_Qleaf_t_name']==f'delta_Q{leaf}',
            'delta_Qleaf_t'] = data_df.loc[
                data_df['delta_Qleaf_t_name']==f'delta_Q{leaf}',
                f'delta_Q{leaf}'].values
    data_df['delta_Qleaf_t_abs'] = data_df['delta_Qleaf_t'].abs()
    
    # Delta q 1 ago, the delta q experienced at the chosen leaf of prior trial, right before running this trial
    grouped = data_df.groupby(by=['nwb_file_name','epoch'])
    data_df['delta_Qleaf_t_1_ago'] = grouped['delta_Qleaf_t'].shift()
    
    # Calculate local delta q and remote delta q and global delta q
    data_df['delta_Qglobal_t_net'] = data_df[[f'delta_Q{leaf}' for leaf in range(1,7)]].sum(axis=1, skipna=False) # inherit nan if any are nans
    data_df['delta_Qglobal_t_abs'] = data_df[[f'delta_Q{leaf}' for leaf in range(1,7)]].abs().sum(axis=1, skipna=False)
    
    patch_leaves = {'A':[1,2], 'B':[3,4], 'C':[5,6]}
    outofpatch_leaves = {'A':[3,4,5,6], 'B':[1,2,5,6], 'C':[1,2,3,4]}
    
    # In patch delta q (sum abs delta qs at leaves in patch (total), or just sum the delta qs overall (net))
    data_df['delta_Qpatch_t_net'] = data_df.apply(lambda row: sum(row[f'delta_Q{i}'] for i in patch_leaves[row['stem']]), axis=1)
    data_df['delta_Qpatch_t_abs'] = data_df.apply(lambda row: sum(np.abs(row[f'delta_Q{i}']) for i in patch_leaves[row['stem']]), axis=1)
    data_df['delta_Qpatch_t_pos'] = data_df.apply(lambda row: sum(row[f'delta_Q{i}'] if row[f'delta_Q{i}']>0 else 0 for i in patch_leaves[row['stem']]), axis=1)
    data_df['delta_Qpatch_t_neg_abs'] = data_df.apply(lambda row: sum(np.abs(row[f'delta_Q{i}']) if row[f'delta_Q{i}']<=0 else 0 for i in patch_leaves[row['stem']]), axis=1)
    # Out of patch delta q (sum abs delta qs at leaves out of patch, or just sum the delta qs overall)
    data_df['delta_Qoutofpatch_t_net'] = data_df.apply(lambda row: sum(row[f'delta_Q{i}'] for i in outofpatch_leaves[row['stem']]), axis=1)
    data_df['delta_Qoutofpatch_t_abs'] = data_df.apply(lambda row: sum(np.abs(row[f'delta_Q{i}']) for i in outofpatch_leaves[row['stem']]), axis=1)
    data_df['delta_Qoutofpatch_t_pos'] = data_df.apply(lambda row: sum(row[f'delta_Q{i}'] if row[f'delta_Q{i}']>0 else 0 for i in outofpatch_leaves[row['stem']]), axis=1)
    data_df['delta_Qoutofpatch_t_neg_abs'] = data_df.apply(lambda row: sum(np.abs(row[f'delta_Q{i}']) if row[f'delta_Q{i}']<=0 else 0 for i in outofpatch_leaves[row['stem']]), axis=1)
    data_df['delta_Qoutofpatch_t_neg'] = data_df.apply(lambda row: sum(row[f'delta_Q{i}'] if row[f'delta_Q{i}']<=0 else 0 for i in outofpatch_leaves[row['stem']]), axis=1)
    
    # Also make these values for 1 trial ago
    grouped = data_df.groupby(by=['nwb_file_name','epoch'])
    data_df['delta_Qglobal_t_net_1_ago'] = grouped['delta_Qglobal_t_net'].shift()
    data_df['delta_Qglobal_t_abs_1_ago'] = grouped['delta_Qglobal_t_abs'].shift()
    data_df['delta_Qpatch_t_net_1_ago'] = grouped['delta_Qpatch_t_net'].shift()
    data_df['delta_Qpatch_t_abs_1_ago'] = grouped['delta_Qpatch_t_abs'].shift()
    data_df['delta_Qoutofpatch_t_net_1_ago'] = grouped['delta_Qoutofpatch_t_net'].shift()
    data_df['delta_Qoutofpatch_t_abs_1_ago'] = grouped['delta_Qoutofpatch_t_abs'].shift()
    data_df['delta_Qoutofpatch_t_pos_1_ago'] = grouped['delta_Qoutofpatch_t_pos'].shift()
    data_df['delta_Qoutofpatch_t_neg_1_ago'] = grouped['delta_Qoutofpatch_t_neg'].shift()
    data_df['delta_Qoutofpatch_t_neg_abs_1_ago'] = grouped['delta_Qoutofpatch_t_neg_abs'].shift()
    
    return data_df

def make_full_beta_bernoulli_dv_df(df):
    '''takes session of data from beh model and compiles all other dv cols into it
    orig designed to take a full animal at a time
    but combining with neural data takes an epoch at a time
    fine when groupby has only one group
    '''
    df_out = add_ak_style_cols_to_RL_output(df)
    df_out = add_periswitch_cols(df_out)
    df_out = add_dvs_beta(df_out)
    df_out = add_switch_advantage_beta(df_out)
    df_out = add_stem_q_dvs_beta(df_out)
    return df_out

def add_ak_style_cols_to_RL_output(df):
    '''
    # Assumptions:
    #   df for beta bernoulli model, single animal, has exactly the columns you listed from Model A, including:
    #   'daynum', 'daysessionnum', 'trial_number_by_epoch',
    #   'stemchoice', 'leafchoice',
    #   betadist_α1..6, betadist_β1..6, betadist_var1..6,
    #   plus all the Q*, depletion*, *_variance columns, etc.
    '''

    # Sort in true chronological order, not needed but fine
    trial_order_cols = ["daynum", "daysessionnum", "trial_number_by_epoch"]
    df = df.sort_values(trial_order_cols).reset_index(drop=True)

    session_key = "daysessionnum"   # used for session-respecting variants

    #betadist_mu1..6 = α / (α + β)  (Model B-only)
    for i in range(1, 7):
        a = df[f"betadist_α{i}"]
        b = df[f"betadist_β{i}"]
        df[f"betadist_mu{i}"] = a / (a + b)
    
    #GLOBAL (Model B-style) per-leaf deltas and ratios
    #    - match Julia Model B: deltas and ratios along the full sequence
    
    for i in range(1, 7):
        mu_col  = f"betadist_mu{i}"
        var_col = f"betadist_var{i}"

        # Δμ_t = μ_t − μ_{t−1}, row 0 = 0 (as in Model B)
        mu_delta = df[mu_col].diff()
        mu_delta.iloc[0] = 0.0
        df[f"betadist_mu{i}_delta"] = mu_delta

        # ΔVar_t = Var_t − Var_{t−1}, row 0 = 0
        var_delta = df[var_col].diff()
        var_delta.iloc[0] = 0.0
        df[f"betadist_var{i}_delta"] = var_delta

        # Var_ratio_t = Var_t / Var_{t−1}, row 0 = 0
        prev_var = df[var_col].shift(1)
        var_ratio = pd.Series(0.0, index=df.index)
        var_ratio.iloc[1:] = df[var_col].iloc[1:].values / prev_var.iloc[1:].values
        df[f"betadist_var{i}_ratio"] = var_ratio
    
    #GLOBAL (Model B-style) global change metrics
    
    mu_delta_cols  = [f"betadist_mu{i}_delta"  for i in range(1, 7)]
    var_delta_cols = [f"betadist_var{i}_delta" for i in range(1, 7)]

    df["prev_global_mu_delta"] = df[mu_delta_cols].sum(axis=1)
    df["prev_global_mu_delta_abs"] = df[mu_delta_cols].abs().sum(axis=1)

    df["prev_global_var_delta"] = df[var_delta_cols].sum(axis=1)
    df["prev_global_var_delta_abs"] = df[var_delta_cols].abs().sum(axis=1)
    
    #GLOBAL leaf-aligned features (prev/next/upcoming) as in Model B
    n = len(df)

    # leaf index: 1..6
    leaf_index_global = (df["stemchoice"] - 1) * 2 + df["leafchoice"]

    # "upcoming" leaf index on the same stem, opposite leaf
    upcoming_index_global = (df["stemchoice"] - 1) * 2 + (3 - df["leafchoice"])

    # Allocate Model B-only leaf-aligned columns
    for col in [
        "prev_leaf_mu_delta", "prev_leaf_var_delta", "prev_leaf_var_ratio",
        "next_choice_leaf_mu_delta", "next_choice_leaf_var_delta", "next_choice_leaf_var_ratio",
        "upcoming_leaf_mu_delta", "upcoming_leaf_var_delta", "upcoming_leaf_var_ratio",
    ]:
        df[col] = 0.0   # Model B initializes all rows to 0.0

    for t in range(n):
        s = int(df.loc[t, "stemchoice"])
        l = int(df.loc[t, "leafchoice"])
        prevleaf = int((s - 1) * 2 + l)

        # previous-leaf deltas/ratio at trial t
        df.loc[t, "prev_leaf_mu_delta"]  = df.loc[t, f"betadist_mu{prevleaf}_delta"]
        df.loc[t, "prev_leaf_var_delta"] = df.loc[t, f"betadist_var{prevleaf}_delta"]
        df.loc[t, "prev_leaf_var_ratio"] = df.loc[t, f"betadist_var{prevleaf}_ratio"]

        # next-choice and upcoming only if there is a next row
        if t < n - 1:
            # next chosen leaf is leaf_index_global at t+1
            nextleaf = int(leaf_index_global.iloc[t+1])

            df.loc[t, "next_choice_leaf_mu_delta"]  = df.loc[t+1, f"betadist_mu{nextleaf}_delta"]
            df.loc[t, "next_choice_leaf_var_delta"] = df.loc[t+1, f"betadist_var{nextleaf}_delta"]
            df.loc[t, "next_choice_leaf_var_ratio"] = df.loc[t+1, f"betadist_var{nextleaf}_ratio"]

            # upcoming leaf on current stem at t, evaluated at t+1
            upcomingleaf = int(upcoming_index_global.iloc[t])
            df.loc[t, "upcoming_leaf_mu_delta"]  = df.loc[t+1, f"betadist_mu{upcomingleaf}_delta"]
            df.loc[t, "upcoming_leaf_var_delta"] = df.loc[t+1, f"betadist_var{upcomingleaf}_delta"]
            df.loc[t, "upcoming_leaf_var_ratio"] = df.loc[t+1, f"betadist_var{upcomingleaf}_ratio"]
    
    #SESSION-RESPECTING variants (suffix _sess)
    #    These do NOT overwrite anything; they are additional columns.
    
    #Per-leaf deltas and ratios within-session
    for i in range(1, 7):
        mu_col  = f"betadist_mu{i}"
        var_col = f"betadist_var{i}"

        df[f"betadist_mu{i}_delta_sess"] = df.groupby(session_key)[mu_col].diff()
        df[f"betadist_var{i}_delta_sess"] = df.groupby(session_key)[var_col].diff()
        prev_var_sess = df.groupby(session_key)[var_col].shift(1)
        df[f"betadist_var{i}_ratio_sess"] = df[var_col] / prev_var_sess

    mu_delta_sess_cols  = [f"betadist_mu{i}_delta_sess"  for i in range(1, 7)]
    var_delta_sess_cols = [f"betadist_var{i}_delta_sess" for i in range(1, 7)]

    df["prev_global_mu_delta_sess"] = df[mu_delta_sess_cols].sum(axis=1, min_count=1)
    df["prev_global_mu_delta_abs_sess"] = df[mu_delta_sess_cols].abs().sum(axis=1, min_count=1)
    df["prev_global_var_delta_sess"] = df[var_delta_sess_cols].sum(axis=1, min_count=1)
    df["prev_global_var_delta_abs_sess"] = df[var_delta_sess_cols].abs().sum(axis=1, min_count=1)


    #SESSION-RESPECTING leaf-index neighbor relationships
    leaf_index_sess = (df["stemchoice"] - 1) * 2 + df["leafchoice"]
    leaf_index_next_sess = leaf_index_sess.groupby(df[session_key]).shift(-1)

    upcoming_index_sess = (df["stemchoice"] - 1) * 2 + (3 - df["leafchoice"])
    upcoming_index_next_sess = upcoming_index_sess.groupby(df[session_key]).shift(-1)

    for col in [
        "prev_leaf_mu_delta_sess", "prev_leaf_var_delta_sess", "prev_leaf_var_ratio_sess",
        "next_choice_leaf_mu_delta_sess", "next_choice_leaf_var_delta_sess",
        "next_choice_leaf_var_ratio_sess",
        "upcoming_leaf_mu_delta_sess", "upcoming_leaf_var_delta_sess",
        "upcoming_leaf_var_ratio_sess",
        "next_choice_variance_sess", "upcoming_leaf_variance_sess",
    ]:
        df[col] = np.nan

    for t in range(n):
        s = int(df.loc[t, "stemchoice"])
        l = int(df.loc[t, "leafchoice"])
        prevleaf = int((s - 1) * 2 + l)

        # previous leaf deltas/ratio within-session
        df.loc[t, "prev_leaf_mu_delta_sess"]  = df.loc[t, f"betadist_mu{prevleaf}_delta_sess"]
        df.loc[t, "prev_leaf_var_delta_sess"] = df.loc[t, f"betadist_var{prevleaf}_delta_sess"]
        df.loc[t, "prev_leaf_var_ratio_sess"] = df.loc[t, f"betadist_var{prevleaf}_ratio_sess"]

        # within-session next-choice
        leaf_next_s = leaf_index_next_sess.iloc[t]
        if not pd.isna(leaf_next_s):
            leaf_next_s = int(leaf_next_s)
            df.loc[t, "next_choice_variance_sess"] = df.loc[t+1, f"betadist_var{leaf_next_s}"]
            df.loc[t, "next_choice_leaf_mu_delta_sess"]  = df.loc[t+1, f"betadist_mu{leaf_next_s}_delta_sess"]
            df.loc[t, "next_choice_leaf_var_delta_sess"] = df.loc[t+1, f"betadist_var{leaf_next_s}_delta_sess"]
            df.loc[t, "next_choice_leaf_var_ratio_sess"] = df.loc[t+1, f"betadist_var{leaf_next_s}_ratio_sess"]

        # within-session upcoming leaf
        leaf_up_next_s = upcoming_index_next_sess.iloc[t]
        if not pd.isna(leaf_up_next_s):
            leaf_up_next_s = int(leaf_up_next_s)
            df.loc[t, "upcoming_leaf_variance_sess"]      = df.loc[t+1, f"betadist_var{leaf_up_next_s}"]
            df.loc[t, "upcoming_leaf_mu_delta_sess"]      = df.loc[t+1, f"betadist_mu{leaf_up_next_s}_delta_sess"]
            df.loc[t, "upcoming_leaf_var_delta_sess"]     = df.loc[t+1, f"betadist_var{leaf_up_next_s}_delta_sess"]
            df.loc[t, "upcoming_leaf_var_ratio_sess"]     = df.loc[t+1, f"betadist_var{leaf_up_next_s}_ratio_sess"]
    return df

def set_nan_at_max2(group):
    max_test_series = group['trials_to_next_switch_groups'].max()
    if group['stem_switch'][-1:].values[0] == False: # if final trial of epoch isn't a switch, then last group cant be countign down to next switch, so then replace with nans
        group['trials_to_next_switch_with_nans'] = group.apply(lambda row: np.nan if row['trials_to_next_switch_groups'] == max_test_series else row['trials_to_next_switch'], axis=1)
    else:
        group['trials_to_next_switch_with_nans'] = group.apply(lambda row: row['trials_to_next_switch'] if row['trials_to_next_switch_groups'] == max_test_series else row['trials_to_next_switch'], axis=1)
#         print(f'max test group: {max_test_series} for nwb_file_name {group["nwb_file_name"][0].values[0]} and epoch {group["epoch"][0].values[0]}. Because final trial is a a stay trial, last group for trials to next switch can be nans.')
#     elif group['stem_switch'][-1:].values[0] == True:
#         print(group['stem_switch'][-1:])
#         print('Final trial is a switch, so trials to next switch does NOT need to be a nan')
    return group

# add own stem switch based on groups
def add_periswitch_cols(dfs, subject_id=None):
    '''usually works on dict of dfs, now to work on one thing add a workaround at the start
    so betadfs is a dict of dfs or a df itself..this is dirty approach
    '''
    if subject_id is None:
        beta_dfs = {'x':dfs}
        subject_id = 'x'
    elif isinstance(subject_id, str):
        beta_dfs = dfs

    beta_dfs[subject_id]['SstemOption'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name', 'epoch',])['stem'].shift(1)
    beta_dfs[subject_id]['stem_switch'] = (beta_dfs[subject_id]['SstemOption']!=beta_dfs[subject_id]['stem']) # this is functionally within epoch

    # add the trial info cols
    beta_dfs[subject_id]['trials_from_prior_switch_groups'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name','epoch'])['stem_switch'].cumsum()
    beta_dfs[subject_id]['trials_from_prior_switch'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name','epoch','trials_from_prior_switch_groups']).cumcount()
    beta_dfs[subject_id]['trials_from_prior_switch'] = beta_dfs[subject_id]['trials_from_prior_switch'].where(beta_dfs[subject_id]['trials_from_prior_switch_groups'].gt(0), np.nan)


    #beta_dfs[subject_id]['trials_to_next_switch_groups'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name','epoch']).apply(lambda x_df: x_df['stem_switch'].shift(fill_value=False).cumsum()).reset_index(name='group').set_index('level_2')['group'] #set_index('id')['group']
    # new version - 
    # beta_dfs[subject_id]['trials_to_next_switch_groups'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name','epoch']).apply(lambda x_df: x_df['stem_switch'].shift(fill_value=False).cumsum()).rename('group').reset_index().set_index('level_2')['group'] #set_index('id')['group']
    # beta_dfs[subject_id]['trials_to_next_switch_groups'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name','epoch'])['stem_switch'].apply(lambda s: s.shift(fill_value=False).cumsum()).to_frame('group').reset_index().set_index('level_2')['group'] #set_index('id')['group']
    tmp = (
        beta_dfs[subject_id]
        .groupby(by=['nwb_file_name','epoch'])['stem_switch']
        .apply(lambda s: s.shift(fill_value=False).cumsum())
        .to_frame('group')
        .reset_index()
                )
    idx_col = tmp.columns[-2]   # this is the original row index column (e.g., 'level_2' or 'index' or your index name)
    beta_dfs[subject_id]['trials_to_next_switch_groups'] = tmp.set_index(idx_col)['group']
    # print(f"Preview of trials to next switch groups column: {beta_dfs[subject_id]['trials_to_next_switch_groups'].head()}")


    beta_dfs[subject_id]['trials_to_next_switch'] = beta_dfs[subject_id].groupby(by=['nwb_file_name','epoch','trials_to_next_switch_groups'])['trials_to_next_switch_groups'].cumcount(ascending=False)

    beta_dfs[subject_id] = beta_dfs[subject_id].groupby(by=['nwb_file_name','epoch']).apply(set_nan_at_max2)
    beta_dfs[subject_id]['trials_from_next_switch'] = beta_dfs[subject_id]['trials_to_next_switch_with_nans']*-1

    beta_dfs[subject_id]['try_bout_idx'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name','epoch'])['stem'].transform(lambda x_df: (x_df != x_df.shift(1)).cumsum())
    beta_dfs[subject_id]['bout_len'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name','epoch', 'try_bout_idx'])['try_bout_idx'].transform(len)
    beta_dfs[subject_id]['bout_len_new'] = beta_dfs[subject_id].groupby(by = ['nwb_file_name','epoch', 'try_bout_idx'])['trials_from_prior_switch'].transform(lambda x: np.max(x))
    return beta_dfs[subject_id]

def add_dvs_beta(df, session_key="daysessionnum"):
    """
    Timing convention:
      - Row t contains PRE-outcome belief state for trial t
      - Outcome on trial t is incorporated into betadist_* on row t+1

    Adds (all session-respecting):

      Trial t, chosen leaf:
        - chosen_leaf_var_preoutcome_sess
        - chosen_leaf_var_postoutcome_sess

      Trial t, starting-patch sibling leaf
        (sibling of leaf chosen on trial t-1):
        - startpatch_sibling_var_preoutcome_sess
        - startpatch_sibling_var_postoutcome_sess

      Trial t, global environment uncertainty (pre-outcome):
        - global_leaf_var_sum_preoutcome_sess

      Trial t, mu update magnitudes and signed updates due to trial t outcome
      (computed from mu deltas on row t+1 and aligned back onto row t):
        Absolute:
          - global_mu_update_abs_postoutcome_sess
          - chosen_leaf_mu_update_abs_postoutcome_sess
          - chosen_patch_mu_update_abs_postoutcome_sess
          - unchosen_patches_mu_update_abs_postoutcome_sess
        Signed:
          - global_mu_update_signed_postoutcome_sess
          - chosen_leaf_mu_update_signed_postoutcome_sess
          - chosen_patch_mu_update_signed_postoutcome_sess
          - unchosen_patches_mu_update_signed_postoutcome_sess

      Trial t, variance update magnitudes and signed updates due to trial t outcome
      (computed from var deltas on row t+1 and aligned back onto row t):
        Absolute:
          - global_var_update_abs_postoutcome_sess
          - chosen_leaf_var_update_abs_postoutcome_sess
          - chosen_patch_var_update_abs_postoutcome_sess
          - unchosen_patches_var_update_abs_postoutcome_sess
        Signed:
          - global_var_update_signed_postoutcome_sess
          - chosen_leaf_var_update_signed_postoutcome_sess
          - chosen_patch_var_update_signed_postoutcome_sess
          - unchosen_patches_var_update_signed_postoutcome_sess
    """

    n = len(df)

    chosen_leaf = (df["stemchoice"] - 1) * 2 + df["leafchoice"]
    start_leaf = chosen_leaf.groupby(df[session_key]).shift(1)

    startpatch_sibling = start_leaf.copy()
    m = ~startpatch_sibling.isna()
    startpatch_sibling.loc[m] = startpatch_sibling.loc[m].apply(
        lambda x: int(x) + 1 if int(x) % 2 == 1 else int(x) - 1
    )

    var_cols = [f"betadist_var{i}" for i in range(1, 7)]
    df["global_leaf_var_sum_preoutcome_sess"] = df[var_cols].sum(axis=1)

    mu_delta_cols = [f"betadist_mu{i}_delta_sess" for i in range(1, 7)]
    missing_mu_delta = [c for c in mu_delta_cols if c not in df.columns]
    if missing_mu_delta:
        raise KeyError(f"Missing required mu-delta columns: {missing_mu_delta}")

    var_delta_cols = [f"betadist_var{i}_delta_sess" for i in range(1, 7)]
    missing_var_delta = [c for c in var_delta_cols if c not in df.columns]
    if missing_var_delta:
        raise KeyError(f"Missing required var-delta columns: {missing_var_delta}")

    df["chosen_leaf_var_preoutcome_sess"] = np.nan
    df["chosen_leaf_var_postoutcome_sess"] = np.nan
    df["startpatch_sibling_var_preoutcome_sess"] = np.nan
    df["startpatch_sibling_var_postoutcome_sess"] = np.nan

    df["global_mu_update_abs_postoutcome_sess"] = np.nan
    df["chosen_leaf_mu_update_abs_postoutcome_sess"] = np.nan
    df["chosen_patch_mu_update_abs_postoutcome_sess"] = np.nan
    df["unchosen_patches_mu_update_abs_postoutcome_sess"] = np.nan
    df["global_mu_update_signed_postoutcome_sess"] = np.nan
    df["chosen_leaf_mu_update_signed_postoutcome_sess"] = np.nan
    df["chosen_patch_mu_update_signed_postoutcome_sess"] = np.nan
    df["unchosen_patches_mu_update_signed_postoutcome_sess"] = np.nan

    df["global_var_update_abs_postoutcome_sess"] = np.nan
    df["chosen_leaf_var_update_abs_postoutcome_sess"] = np.nan
    df["chosen_patch_var_update_abs_postoutcome_sess"] = np.nan
    df["unchosen_patches_var_update_abs_postoutcome_sess"] = np.nan
    df["global_var_update_signed_postoutcome_sess"] = np.nan
    df["chosen_leaf_var_update_signed_postoutcome_sess"] = np.nan
    df["chosen_patch_var_update_signed_postoutcome_sess"] = np.nan
    df["unchosen_patches_var_update_signed_postoutcome_sess"] = np.nan

    for t in range(n):
        cl = chosen_leaf.iloc[t]

        df.loc[t, "chosen_leaf_var_preoutcome_sess"] = df.loc[t, f"betadist_var{int(cl)}"]

        has_next = (t < n - 1) and (df.loc[t, session_key] == df.loc[t+1, session_key])
        if has_next:
            df.loc[t, "chosen_leaf_var_postoutcome_sess"] = df.loc[t+1, f"betadist_var{int(cl)}"]

        sl = startpatch_sibling.iloc[t]
        if not pd.isna(sl):
            df.loc[t, "startpatch_sibling_var_preoutcome_sess"] = df.loc[t, f"betadist_var{int(sl)}"]
            if has_next:
                df.loc[t, "startpatch_sibling_var_postoutcome_sess"] = df.loc[t+1, f"betadist_var{int(sl)}"]

        if has_next:
            dmu_next = df.loc[t+1, mu_delta_cols].astype(float).values
            dvar_next = df.loc[t+1, var_delta_cols].astype(float).values

            stem = int(df.loc[t, "stemchoice"])
            stem_idx = [2 * stem - 2, 2 * stem - 1]
            unchosen_idx = [i for i in range(6) if i not in stem_idx]
            chosen_leaf_idx = int(cl) - 1

            df.loc[t, "global_mu_update_abs_postoutcome_sess"] = np.abs(dmu_next).sum()
            df.loc[t, "global_mu_update_signed_postoutcome_sess"] = dmu_next.sum()

            df.loc[t, "chosen_leaf_mu_update_abs_postoutcome_sess"] = abs(dmu_next[chosen_leaf_idx])
            df.loc[t, "chosen_leaf_mu_update_signed_postoutcome_sess"] = dmu_next[chosen_leaf_idx]

            df.loc[t, "chosen_patch_mu_update_abs_postoutcome_sess"] = np.abs(dmu_next[stem_idx]).sum()
            df.loc[t, "chosen_patch_mu_update_signed_postoutcome_sess"] = dmu_next[stem_idx].sum()

            df.loc[t, "unchosen_patches_mu_update_abs_postoutcome_sess"] = np.abs(dmu_next[unchosen_idx]).sum()
            df.loc[t, "unchosen_patches_mu_update_signed_postoutcome_sess"] = dmu_next[unchosen_idx].sum()

            df.loc[t, "global_var_update_abs_postoutcome_sess"] = np.abs(dvar_next).sum()
            df.loc[t, "global_var_update_signed_postoutcome_sess"] = dvar_next.sum()

            df.loc[t, "chosen_leaf_var_update_abs_postoutcome_sess"] = abs(dvar_next[chosen_leaf_idx])
            df.loc[t, "chosen_leaf_var_update_signed_postoutcome_sess"] = dvar_next[chosen_leaf_idx]

            df.loc[t, "chosen_patch_var_update_abs_postoutcome_sess"] = np.abs(dvar_next[stem_idx]).sum()
            df.loc[t, "chosen_patch_var_update_signed_postoutcome_sess"] = dvar_next[stem_idx].sum()

            df.loc[t, "unchosen_patches_var_update_abs_postoutcome_sess"] = np.abs(dvar_next[unchosen_idx]).sum()
            df.loc[t, "unchosen_patches_var_update_signed_postoutcome_sess"] = dvar_next[unchosen_idx].sum()

    return df

def add_switch_advantage_beta(df, session_key="daysessionnum"):
    """
    Computes trial-t decision variables evaluated PRE-outcome (row t):

    stay_value(t):
      - mu of the sibling leaf of the START leaf
      - start leaf = leaf chosen on trial t-1 (within session)

    best_other_patch_value(t):
      - maximum over the two OTHER stems of
        (mean mu of the two leaves in that stem)

    switch_advantage(t):
      - best_other_patch_value - stay_value

    Adds:
      - stay_sibling_mu_preoutcome_sess
      - best_other_patch_mean_mu_preoutcome_sess
      - switch_advantage_mu_preoutcome_sess
    """

    n = len(df)

    mu_cols = [f"betadist_mu{i}" for i in range(1, 7)]
    missing = [c for c in mu_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required mu columns: {missing}")

    # chosen leaf on trial t
    chosen_leaf = (df["stemchoice"] - 1) * 2 + df["leafchoice"]

    # start leaf and start stem for trial t = values from t-1 within session
    start_leaf_index = chosen_leaf.groupby(df[session_key]).shift(1)
    start_stem_index = df["stemchoice"].groupby(df[session_key]).shift(1)

    # sibling of start leaf (1↔2, 3↔4, 5↔6)
    startpatch_sibling_leaf_index = start_leaf_index.copy()
    valid_mask = ~startpatch_sibling_leaf_index.isna()
    startpatch_sibling_leaf_index.loc[valid_mask] = startpatch_sibling_leaf_index.loc[
        valid_mask
    ].apply(lambda x: int(x) + 1 if int(x) % 2 == 1 else int(x) - 1)

    df["stay_sibling_mu_preoutcome_sess"] = np.nan
    df["best_other_patch_mean_mu_preoutcome_sess"] = np.nan
    df["switch_advantage_mu_preoutcome_sess"] = np.nan

    for t in range(n):
        start_stem_for_trial = start_stem_index.iloc[t]
        sibling_leaf_of_start_patch = startpatch_sibling_leaf_index.iloc[t]

        if pd.isna(start_stem_for_trial) or pd.isna(sibling_leaf_of_start_patch):
            continue

        start_stem_for_trial = int(start_stem_for_trial)
        sibling_leaf_of_start_patch = int(sibling_leaf_of_start_patch)

        # value of staying (other leaf in starting patch)
        stay_mu = df.loc[t, f"betadist_mu{sibling_leaf_of_start_patch}"]

        # values of the two alternative patches
        other_stems = [1, 2, 3]
        other_stems.remove(start_stem_for_trial)

        def stem_mean_mu(stem_id):
            leaf_a = 2 * stem_id - 1
            leaf_b = 2 * stem_id
            return 0.5 * (
                df.loc[t, f"betadist_mu{leaf_a}"] +
                df.loc[t, f"betadist_mu{leaf_b}"]
            )

        other_patch_mean_mus = [stem_mean_mu(stem_id) for stem_id in other_stems]
        best_other_patch_mean_mu = max(other_patch_mean_mus)

        df.loc[t, "stay_sibling_mu_preoutcome_sess"] = stay_mu
        df.loc[t, "best_other_patch_mean_mu_preoutcome_sess"] = best_other_patch_mean_mu
        df.loc[t, "switch_advantage_mu_preoutcome_sess"] = best_other_patch_mean_mu - stay_mu

    return df

def add_stem_q_dvs_beta(df, session_key="daysessionnum"):
    """
    Timing convention:
      - Row t contains PRE-outcome decision variables for trial t
      - Outcome on trial t is incorporated into Q variables on row t+1
      - Therefore, "update due to trial t outcome" is measured as a delta on row t+1
        and aligned back onto row t (within-session).

    Switch advantage variables (PRE-outcome, row t):
      These are based on the initiating (start) stem for trial t, which is stemchoice[t-1] within-session.
      For each stem-value family, we compare:
        - stay_value = value of the start stem on row t
        - best_other_value = max value among the two non-start stems on row t
        - switch_advantage = best_other_value - stay_value

      Families:
        - Qstem*: stem choice decision logits used for stem softmax; includes β scalings, γ2-weighting for stay,
                 depletion, and explicit bias additions (stay_bias, turn_bias, spatial_bias; delay_turn_bias handling).
        - Qstem*_nobias: same construction but without explicit bias additions; still includes β scalings, γ2, depletion.
        - Qstem*_pre_update_post_bias: bias-accounting variant constructed alongside choice computation; still pre-outcome
                 on row t; includes β scalings, γ2, depletion, and the choice-relevant bias structure.

    Update-magnitude variables (POST-outcome effect of trial t, aligned onto row t):
      These are based on the destination/choice on trial t (stemchoice[t]).
      For each family, we compute within-session deltas across stems, take row t+1 deltas (effect of trial t outcome),
      and aggregate:
        - global update: sum over stems
        - chosen-stem update: stemchoice[t]
        - unchosen-stems update: the other two stems

      Families used for update metrics:
        - Qstem_nobias
        - Qstem_pre_update_post_bias

      Both absolute and signed versions are produced.
    """

    required_cols = (
        [f"Qstem{i}" for i in (1, 2, 3)] +
        [f"Qstem{i}_nobias" for i in (1, 2, 3)] +
        [f"Qstem{i}_pre_update_post_bias" for i in (1, 2, 3)] +
        ["stemchoice", session_key]
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    n = len(df)

    # initiating (start) stem for trial t is stemchoice[t-1] within session
    start_stem_index = df["stemchoice"].groupby(df[session_key]).shift(1)

    # within-session deltas for stem-value families used for update metrics
    for stem_id in (1, 2, 3):
        df[f"Qstem{stem_id}_nobias_delta_sess"] = (
            df.groupby(session_key)[f"Qstem{stem_id}_nobias"].diff()
        )
        df[f"Qstem{stem_id}_pre_update_post_bias_delta_sess"] = (
            df.groupby(session_key)[f"Qstem{stem_id}_pre_update_post_bias"].diff()
        )

    # switch advantage columns for three families
    for family in ("Qstem", "Qstem_nobias", "Qstem_pre_update_post_bias"):
        df[f"start_stem_value_preoutcome_sess_{family}"] = np.nan
        df[f"best_other_stem_value_preoutcome_sess_{family}"] = np.nan
        df[f"switch_advantage_preoutcome_sess_{family}"] = np.nan

    # update-magnitude columns for two families
    for family_tag in ("Qstem_nobias", "Qstem_pre_update_post_bias"):
        df[f"global_{family_tag}_update_abs_postoutcome_sess"] = np.nan
        df[f"chosen_stem_{family_tag}_update_abs_postoutcome_sess"] = np.nan
        df[f"unchosen_stems_{family_tag}_update_abs_postoutcome_sess"] = np.nan

        df[f"global_{family_tag}_update_signed_postoutcome_sess"] = np.nan
        df[f"chosen_stem_{family_tag}_update_signed_postoutcome_sess"] = np.nan
        df[f"unchosen_stems_{family_tag}_update_signed_postoutcome_sess"] = np.nan

    for t in range(n):
        has_next = (t < n - 1) and (df.loc[t, session_key] == df.loc[t+1, session_key])

        # switch advantage uses initiating (start) stem
        start_stem_for_trial = start_stem_index.iloc[t]
        if not pd.isna(start_stem_for_trial):
            start_stem_for_trial = int(start_stem_for_trial)
            non_start_stems = [1, 2, 3]
            non_start_stems.remove(start_stem_for_trial)

            for family in ("Qstem", "Qstem_nobias", "Qstem_pre_update_post_bias"):
                if family == "Qstem":
                    stay_value = df.loc[t, f"Qstem{start_stem_for_trial}"]
                    best_other_value = max(df.loc[t, f"Qstem{non_start_stems[0]}"],
                                           df.loc[t, f"Qstem{non_start_stems[1]}"])
                elif family == "Qstem_nobias":
                    stay_value = df.loc[t, f"Qstem{start_stem_for_trial}_nobias"]
                    best_other_value = max(df.loc[t, f"Qstem{non_start_stems[0]}_nobias"],
                                           df.loc[t, f"Qstem{non_start_stems[1]}_nobias"])
                else:  # Qstem_pre_update_post_bias
                    stay_value = df.loc[t, f"Qstem{start_stem_for_trial}_pre_update_post_bias"]
                    best_other_value = max(df.loc[t, f"Qstem{non_start_stems[0]}_pre_update_post_bias"],
                                           df.loc[t, f"Qstem{non_start_stems[1]}_pre_update_post_bias"])

                df.loc[t, f"start_stem_value_preoutcome_sess_{family}"] = stay_value
                df.loc[t, f"best_other_stem_value_preoutcome_sess_{family}"] = best_other_value
                df.loc[t, f"switch_advantage_preoutcome_sess_{family}"] = best_other_value - stay_value

        # update metrics use destination/choice stem for trial t, via deltas on row t+1
        if has_next:
            chosen_stem_for_trial = int(df.loc[t, "stemchoice"])
            unchosen_stems_for_trial = [1, 2, 3]
            unchosen_stems_for_trial.remove(chosen_stem_for_trial)

            d_next_nobias = np.array([
                float(df.loc[t+1, "Qstem1_nobias_delta_sess"]),
                float(df.loc[t+1, "Qstem2_nobias_delta_sess"]),
                float(df.loc[t+1, "Qstem3_nobias_delta_sess"]),
            ])

            d_next_preupd = np.array([
                float(df.loc[t+1, "Qstem1_pre_update_post_bias_delta_sess"]),
                float(df.loc[t+1, "Qstem2_pre_update_post_bias_delta_sess"]),
                float(df.loc[t+1, "Qstem3_pre_update_post_bias_delta_sess"]),
            ])

            for family_tag, d_next in (
                ("Qstem_nobias", d_next_nobias),
                ("Qstem_pre_update_post_bias", d_next_preupd),
            ):
                chosen_idx = chosen_stem_for_trial - 1
                unchosen_idx = [s - 1 for s in unchosen_stems_for_trial]

                df.loc[t, f"global_{family_tag}_update_abs_postoutcome_sess"] = np.abs(d_next).sum()
                df.loc[t, f"global_{family_tag}_update_signed_postoutcome_sess"] = d_next.sum()

                df.loc[t, f"chosen_stem_{family_tag}_update_abs_postoutcome_sess"] = abs(d_next[chosen_idx])
                df.loc[t, f"chosen_stem_{family_tag}_update_signed_postoutcome_sess"] = d_next[chosen_idx]

                df.loc[t, f"unchosen_stems_{family_tag}_update_abs_postoutcome_sess"] = np.abs(d_next[unchosen_idx]).sum()
                df.loc[t, f"unchosen_stems_{family_tag}_update_signed_postoutcome_sess"] = d_next[unchosen_idx].sum()

    return df
