from HPManager.HPManager import HPManager
from Projector import Projector
from pathlib import Path


class Pipeline():
    def __init__(self, config):
        self.config = config

        # viewpoint management
        self.hp_manager = HPManager(config)

        # world management
        self.projector = Projector()
        self.skels3d = []

    def set_viewpoints(self, metaByVpath):
        self.metaByVid = {}
        for vpath in metaByVpath:
            self.metaByVid[Path(vpath).stem] = metaByVpath[vpath]

        self.projector.set_viewpoints(self.metaByVid)

    def process(self, himages):
        self.hp_manager.process(himages)

        if False: # need reid between cameras
            # we estimate only one person per image
            hpsByPerson = {}
            if len(self.hp_manager.trackers) > 0:
                trackerkey = next(iter(self.hp_manager.trackers))
                tracker = self.hp_manager.trackers[trackerkey]
                if len(tracker.tracklets) > 0:
                    tracklet = next(iter(tracker.tracklets))
                    hps = tracker.tracklets[tracklet]
                    if len(hps) > 0:
                        # keep only one track (person) for now
                        print("project")
                        hpsForP1 = []
                        hpsForP1 = hps
                        # hpsByPerson["P1"] = hpsForP1  
                        hpsByPerson[tracklet] = hpsForP1  
                        skel3d = self.projector.process(hpsByPerson, self.hp_manager.estimator.minScoreKpt)
                        if skel3d is not None:
                            self.skels3d.append(skel3d)

    def print_stats(self):
        print("time estimation : {est_time.avg:.3f}\t".format(est_time=self.hp_manager.estim_time))
        print("time tracking : {est_time.avg:.3f}\t".format(est_time=self.hp_manager.tracking_time))

    def last_hps(self, viewpointId):
        hps = []
        if not viewpointId in self.hp_manager.trackers:
            print("no tracker for this view : ", viewpointId, " vs ", self.hp_manager.trackers.keys())
            return hps
        tracklets = self.hp_manager.trackers[viewpointId].tracklets
        for track in tracklets.values():
            hps.append(track[-1])
        return hps