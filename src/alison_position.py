import datajoint as dj
from spyglass.utils import SpyglassMixin
import numpy as np

from spyglass.common import TrackGraph, Session, IntervalList, TaskEpoch

schema = dj.schema('alison_position')

@schema
class SpatialBanditTrackGraph(SpyglassMixin, dj.Manual):
    definition="""
    # Stores which nwb_file_name and pos interval corresponds to which TrackGraph 
    -> Session
    interval_list_name: varchar(200)
    ---
    -> TrackGraph
    """
    def verify_entry(key):
        """check if entry refers to existing entries in other tables"""
        nwb_file_name = key['nwb_file_name']
        interval_list_name = key['interval_list_name']
        track_graph_name = key['track_graph_name']
        assert interval_list_name in (IntervalList & {'nwb_file_name':key['nwb_file_name']}).fetch('interval_list_name'), f'interval {interval_list_name} is not in IntervalList for nwb {nwb_file_name}'
        assert track_graph_name in TrackGraph.fetch('track_graph_name'), f'track graph {track_graph_name} is not in TrackGraph'
        print(f'SpatialBanditTrackGraph entry has been verified: {key}')
        
@schema
class PosValidTimesToEpoch(SpyglassMixin, dj.Computed):
    definition="""
    # for ea nwb file, map pos X valid times intervals to epochs in TaskEpoch based on overlapping time intervals
    -> Session
    pos_interval_list_name: varchar(200) # of the form 'pos X valid times' #should this be NULL??
    ---
    epoch: int
    epoch_interval_list_name: varchar(200) # of the form '02_r1' or '01_s1'
    """
    # pos int name as pk rather than ep because this mapping finds epochs for pos ints, not the other way
    # plausible that if there was no pos data for an epoch that there may be epochs without any matching pos int
    # this is related to primary dependency on Session rather than on TaskEpoch
    def make(self, key):
        nwb_file_name = key['nwb_file_name']
        epsilon = 0.1   # **NB**: not currently storing epsilon param
        ep_to_pos_dict = self._match_pos_to_epoch(nwb_file_name, epsilon) # **NB**: not currently storing epsilon param
        for epoch in ep_to_pos_dict:
            pos_interval_list_name = ep_to_pos_dict[epoch][0]
            epoch_interval_list_name = (TaskEpoch & {'nwb_file_name':nwb_file_name, 'epoch':epoch}).fetch1('interval_list_name')
            self.insert1({'nwb_file_name': nwb_file_name,
                      'epoch': epoch,
                      'pos_interval_list_name': pos_interval_list_name,
                      'epoch_interval_list_name': epoch_interval_list_name
                    }, skip_duplicates=True)
        task_epochs = (TaskEpoch & {'nwb_file_name':nwb_file_name}).fetch('epoch')
        for task_epoch in task_epochs:
            if task_epoch not in ep_to_pos_dict:
                print(f'TaskEpoch for nwb {nwb_file_name} has epoch {task_epoch}, but no pos X valid times interval matched this epoch')
        print(f'Done populating PosValidTimesToEpoch for {nwb_file_name}')
    
    def _match_pos_to_epoch(self, nwb_file_name, epsilon=.1):
        """
        # Find correspondence between pos valid times names and epochs
        # Use epsilon to tolerate small differences in epoch boundaries across epoch/pos intervals
        # adapted from prior lab code
        """
        if nwb_file_name == 'peanut20201129_.nwb':
            epsilon=.22 #this catches pos 7 valid times which is currently not as tightly aligned with its epoch #20220920
        elif nwb_file_name[0:5] == 'senor':
            if nwb_file_name == 'senor20201106_.nwb':
                epsilon = 100
            else:
                epsilon=.5
        elif nwb_file_name in ['peanut20201203_.nwb','peanut20201204_.nwb','peanut20201206_.nwb']:
            epsilon=1500 #03 needs 100, 06 needs 800, 04 needs 1500
        elif nwb_file_name[0:6] == 'peanut':
            epsilon=.5 #all peanut days aren't caught with .1 epsilon
        elif nwb_file_name in ['chimi20200212_.nwb','chimi20200219_.nwb','chimi20200221_.nwb','chimi20200226_.nwb','chimi20200312_.nwb']:
            epsilon=.22 #attempt to catch 20221129
        elif nwb_file_name == 'chimi20200213_.nwb':
            epsilon=50 #20221129 attempt to catch, or hve missing pos info
        pos_interval_list_names = [interval_list_name for interval_list_name in
                                   (IntervalList & {"nwb_file_name": nwb_file_name}).fetch("interval_list_name")
                                   if np.logical_and(interval_list_name.split(" ")[0] == "pos",
                                                     " ".join(interval_list_name.split(" ")[2:]) == "valid times")]
        # Got epoch number and corresponding interval list name
        x = (TaskEpoch & {"nwb_file_name": nwb_file_name}).fetch("epoch",
                                                                 "interval_list_name")
        epochs, epoch_interval_list_names = x[0], x[1]
        epoch_pos_valid_time_dict = {epoch: [] for epoch in
                                     epochs}
        # Store correspondence between epoch number and pos x valid time interval names
        # Match pos valid time intervals to epochs
        for epoch, epoch_interval_list_name in zip(epochs, epoch_interval_list_names):  # for each epoch
            epoch_valid_times = (IntervalList & {"nwb_file_name": nwb_file_name,
                                                 "interval_list_name": epoch_interval_list_name}).fetch1(
                "valid_times")  # get epoch valid times
            epoch_time_interval = [epoch_valid_times[0][0], epoch_valid_times[-1][-1]]  # [epoch start, epoch end]
            epoch_time_interval_widened = np.asarray([epoch_time_interval[0] - epsilon,
                                                      epoch_time_interval[
                                                          1] + epsilon])
            # Widened to tolerate small differences in epoch boundaries across epoch/pos intervals
            for pos_interval_list_name in pos_interval_list_names:  # for each pos valid time interval list
                pos_valid_times = (IntervalList & {"nwb_file_name": nwb_file_name,
                                                   "interval_list_name": pos_interval_list_name}).fetch1(
                    "valid_times")  # get interval valid times
                pos_time_interval = np.asarray([pos_valid_times[0][0], pos_valid_times[-1][
                    -1]])  # [pos valid time interval start, pos valid time interval end]
                if np.logical_and(epoch_time_interval_widened[0] < pos_time_interval[0],
                                  epoch_time_interval_widened[1] > pos_time_interval[1]): 
                    # If pos valid time interval within epoch interval widened
                    epoch_pos_valid_time_dict[epoch].append(
                        pos_interval_list_name)  # match pos valid time interval to epoch
                elif np.logical_and(epoch_time_interval[0] > pos_time_interval[0],
                                  epoch_time_interval[1] < pos_time_interval[1]):
                    # If epoch interval (not widened) is within pos interval (also not widened)
                    epoch_pos_valid_time_dict[epoch].append(pos_interval_list_name) # match pos valid time to epoch    
        # Check that each pos interval was matched to only one epoch
        import itertools
        matched_pos_interval_list_names = list(itertools.chain.from_iterable(epoch_pos_valid_time_dict.values()))
        if len(np.unique(matched_pos_interval_list_names)) != len(matched_pos_interval_list_names):
            raise Exception(f"At least one pos interval list name was matched with more than one epoch in {nwb_file_name}")
        # Check that all pos intervals were matched to an epoch
        if len(np.unique(matched_pos_interval_list_names)) != len(pos_interval_list_names):
            print(f'matched_pos_interval_list_names: {matched_pos_interval_list_names}\npos_interval_list_names: {pos_interval_list_names}')
            print(f'\nEpoch pos valid time dict: \n{epoch_pos_valid_time_dict}\n')
            if nwb_file_name == 'peanut20201202_.nwb':
                    epoch_pos_valid_time_dict[17].append('pos 16 valid times')
            else:
                raise Exception(f"should have same number of matched pos intervals as pos intervals in {nwb_file_name}")
        # Does not acct for epochs without pos intervals right now
        return epoch_pos_valid_time_dict