import numpy as np

from .HPTracker import HPTracker

from .deep_sort.tracker import Tracker
from .deep_sort import nn_matching
from .deep_sort.detection import Detection

class HPTrackerDS(HPTracker):
    def __init__(self, trackerid=""):
        super(HPTrackerDS, self).__init__(trackerid)

        print("Setting up DeepSort...")
        hungarian_th = 100.0
        metric_type = "euclidean"
        metric  = nn_matching.NearestNeighborDistanceMetric(metric_type, hungarian_th)
        max_age_track = 50
        n_init = 3
        self.tracker = Tracker(metric, max_age=max_age_track, n_init=n_init)

    def process(self, hps):
        self.tracker.predict()

        detections = []
        for hp in hps:
            features = np.ones(50, dtype=float) # TODO : collect features somewhere else
            detections.append(Detection(hp.bbox, hp.detscore, features))

        trackidxs = self.tracker.update(detections)

        print(trackidxs)

        for idx, track_idx in enumerate(trackidxs):
            hp = hps[idx]
            hp.trackid = track_idx    
            if not hp.trackid in self.tracklets:
                self.tracklets[hp.trackid] = []
            self.tracklets[hp.trackid].append(hp)

        # for track in self.tracker.tracks:
        #     if not track.is_deleted() and track.time_since_update < 1:
        #         if not track.track_id in self.tracklets:
        #             self.tracklets[track.track_id] = []

        #         bbox = track.to_tlwh()
        #         self.tracklets[track.track_id].append(track.to_hp())