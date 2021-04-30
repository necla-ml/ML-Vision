"""
Adapted from https://github.com/ZQPei/deep_sort_pytorch.
"""

import torch as th
import numpy as np
from .... import logging
from ...ops import *
from .tracking import Tracker, Track

from .deep_sort.nn_matching import NearestNeighborDistanceMetric as NNDistMetric
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker as DSort

from .deep_sort import linear_assignment
from .deep_sort import iou_matching

class DSTrackerImpl(DSort):
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, **kwargs):
        super().__init__(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.matching_cascade_feats = kwargs.get('matching_cascade_feats', True)
        self.matching_iou_confirmed = kwargs.get('matching_iou_confirmed', True)
        self.gating_kf = kwargs.get('gating_kf', 'org')
        self.gating_thrd = kwargs.get('gating_thrd', 50)
        self.gating_alpha = kwargs.get('gating_alpha', 0.2)
        self.deleted = None

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        
        '''
        print('matched:', [(self.tracks[ti].track_id, di) for ti, di in matches],
              'unmatched tracks:', [self.tracks[ti].track_id for ti in unmatched_tracks], 
              'unmatched_detections:', unmatched_detections)
        '''
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        # XXX track index changes after deleted
        matches = {self.tracks[tidx].track_id: di for tidx, di in matches}

        # XXX expose deleted confirmed tracks
        # Track deletion:
        # - tentative -> deleted for miss detection
        # - confirmed -> deteted for time_since_update > _max_age
        self.deleted = [t for t in self.tracks if t.is_deleted()]
        # assert all(t.hits >= t._n_init for t in self.deleted), f"{[(t.track_id, t.hits, t._n_init) for t in self.deleted]}"
        # Active tracks: tentaive + confirmed
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        # XXX return matches to associate with detections
        return matches

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            # logging.info(f"track ids: {targets.tolist()}")
            # logging.info(f"det idx: {detection_indices}")
            # print('feature distance cost matrix:', cost_matrix)
            if self.gating_kf == 'org':
                # KF filtering based on state distributions
                cost_matrix = linear_assignment.gate_cost_matrix(
                    self.kf, cost_matrix, tracks, dets, track_indices,
                    detection_indices, only_position=False)
                # print('gated feature cost matrix:', cost_matrix)
            elif self.gating_kf == 'iain': 
                cost_matrix = linear_assignment.gate_cost_matrix_iain(
                    self.kf, cost_matrix, 
                    tracks, dets, 
                    track_indices,
                    detection_indices,
                    gating_alpha=self.gating_alpha,
                    gating_thrd=self.gating_thrd)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        if self.matching_cascade_feats:
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)
        else:
            # FIXME matching all at once even with less updated tracks due to potential occlusion
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    gated_metric, self.metric.matching_threshold,
                    self.tracks, detections, confirmed_tracks)
        # logging.info(f'Feature matching with {len(confirmed_tracks)} confirmed tracks: {[(self.tracks[t].track_id, d) for t, d in matches_a]}')

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        if self.matching_iou_confirmed:
            # Only consider unconfirmed and those most recent unmatched tracks
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update == 1]
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]
        else:
            # FIXME Only match unconfirmed tracks with IoU in case of id switches
            iou_track_candidates = unconfirmed_tracks

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        # logging.info(f'IoU matching with {len(iou_track_candidates)} candidates including unconfirmed tracks: {[(self.tracks[t].track_id, d) for t, d in matches_b]}')

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

class DSTracker(Tracker):
    def __init__(self, max_iou_dist=0.7,    # 0.7
                 max_feat_dist=0.2,         # TODO in case of occlusion?
                 n_init=3,                  # 3
                 max_age=13,                # 30 
                 nn_budget=100, 
                 **kwargs):
        """DeepSort Tracker wrapper.
        Args:
            max_feat_dist(float): max feature vector distance to considered a match
            max_iou_dist(float): max IoU overlap
            nn_budget(int): if not None, fix samples per class to at most this number. 
                            Removes the oldest samples when the budget is reached.
            max_age(int): maximum number of missed misses before a track is deleted
            n_init(int): number of consecutive detections before the track is confirmed
        Kwargs:
            matching_cascade_feats(bool): whether to match tracks by update recency
            matching_iou_confirmed(bool): whether to match confirmed tracks with IoU
            gating_feats(bool): whether to gate features with Kalman filter
        """
        super().__init__()
        self._tracker = DSTrackerImpl(NNDistMetric("cosine", max_feat_dist, nn_budget), 
                                      max_iou_distance=max_iou_dist, 
                                      max_age=max_age, 
                                      n_init=n_init,
                                      **kwargs)

    @property
    def hits(self):
        return self._tracker.n_init

    '''
    def trace(self, tid, history=10):
        track = self.tracks[tid]
        box = torch.from_numpy(track.to_tlwh())
        return xywh2xyxy(box)
    '''

    def update(self, xyxysc, features):
        """Track one time detection.
        Args:
            xyxysc(Tensor[N, 6]): clipped boxes in xyxysc
            features(Tensor[N, D]): pooled RoI features
            frame(Tensoor[C, H, W]): frame to detect
            size(int or Tuple[H, W]): image size to clip boxes
        """
        xyxysc = xyxysc.cpu()
        features = features.cpu().numpy()
        xywh = xyxy2xywh(xyxysc[:, :4]).numpy()
        scores = xyxysc[:, 4]
        classes = xyxysc[:, 5]
        detections = [Detection(xywh[i], score.item(), features[i]) for i, score in enumerate(scores)]

        # Update tracker
        self._tracker.predict()
        matches = self._tracker.update(detections)  # { tid: di } 
        # self._tracker.tracks: current confirmed + tentative tracks
        # self._tracker.deleted: deleted confirmed and tentative tracks

        # Incrementally filter out tentative tracks
        # self.deleted: previously confirmed
        # self.tracks: currently confirmed
        current = set(self.tracks.keys())
        confirmed = {trk.track_id: trk for trk in self._tracker.tracks if trk.is_confirmed()}
        removed = current - set(confirmed.keys())
        deleted = {}
        for tid in removed:
            deleted[tid] = self.tracks[tid]
            del self.tracks[tid]
        tracks = self.tracks
        self.deleted = deleted
        '''
        print('all tracks:', [trk.track_id for trk in self._tracker.tracks])
        print('current:', sorted(current))
        print('confirmed:', sorted(confirmed.keys()))
        print('remove:', sorted(remove))
        print('matches:', sorted(matches.items()))
        '''
        for tid, trk in confirmed.items():
            if tid in matches:
                det = xyxysc[matches[tid]]
                if tid not in tracks:
                    tracks[tid] = Track(tid, cls=int(det[5].item()))
                tracks[tid].update(det)
            else:
                # Not detected => use predicted mean
                tracked = xywh2xyxy(th.from_numpy(trk.to_tlwh()).float())
                tracks[tid].update(tracked)
                # logging.info(f"track[{tid}] not detected since update for {trk.time_since_update} time(s)")
        #print("deleted:", [(trk.track_id, trk.hits, trk.time_since_update) for trk in self._tracker.deleted])
        #print("active:", [(trk.track_id, trk.hits, trk.time_since_update) for trk in self._tracker.tracks])
        #print("tracks:", list(self.tracks.keys()))
        assert deleted.keys() == set(trk.track_id for trk in self._tracker.deleted if trk.time_since_update > trk._max_age)
        assert tracks.keys() == set(trk.track_id for trk in self._tracker.tracks if trk.is_confirmed())
        return matches
