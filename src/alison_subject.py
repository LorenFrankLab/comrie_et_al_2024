import datajoint as dj
from spyglass.utils import SpyglassMixin
from spyglass.common import Session
import numpy as np

schema = dj.schema('alison_subject')

@schema
class SpatialBanditSubjects(SpyglassMixin, dj.Manual):
    # use key from Session table to populate
    definition = """
    subject_id: varchar(40) # rat name
    ---
    """
    def insert_default(self):
        subject_ids = np.unique((Session & {'session_description LIKE "Spatial bandit%"'}).fetch('subject_id'))
        for subject_id in subject_ids:
            self.insert1({'subject_id': subject_id
                 }, skip_duplicates = True)