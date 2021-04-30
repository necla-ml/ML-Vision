import torch as th
from ml import logging

class Track(object):
    def __init__(self, tid, cls=None, length=5, **kwargs):
        """
        Args:
            kwargs:
                tid(int):
                length(int): max length of history
                history(List[Tensor[4+2]]): past bboxes
        """
        self.tid = tid
        self.cls = cls
        self.score = 0
        self.age = 0
        self.origin = None
        self.velocity = None
        self.length = length
        self.history = []
        self.predicted = False
        self.__dict__.update(kwargs)
    
    @property
    def last(self):
        return self.history[-1] if self.history else None

    def update(self, instance):
        """Update track statistics and history given a new observation or last tracked.
        instance(xyxysc or xyxy): up to date detection or predicted position
        """
        instance = instance.cpu()
        info = len(instance)
        score = self.score
        if info > 4:
            # detection
            self.predicted = False
            score, cls = instance[4:6]
            score, cls = score.item(), cls.item()
            xyxysc = instance[:6]
            if cls != self.cls:
                logging.warning(f"Inconsistent detection class from {self.cls} to {cls}")
                xyxysc = instance.clone()
                xyxysc[-2] = score = self.score
                xyxysc[-1] = self.cls
        elif info == 4:
            # prediction
            self.predicted = True
            # logging.warning(f"track[{self.tid}] predicted={instance.round().int().tolist()}({self.score:.2f})")
            xyxysc = th.cat((instance, th.Tensor([self.score, self.cls])))
        
        age = self.age
        self.age += 1
        self.score = (age * self.score + score) / self.age
        if age == 0:
            self.origin = (xyxysc[2:4] + xyxysc[:2]) / 2
            self.velocity = th.zeros_like(self.origin)
        else:
            last = self.last
            prev = (last[2:4] + last[:2]) / 2
            center = (xyxysc[2:4] + xyxysc[:2]) / 2
            velocity = center - prev
            # FIXME moving average could be too slow
            # self.velocity = ((age - 1) * self.velocity + velocity) / age
            self.velocity = (self.velocity + velocity) / 2
            # print(f"snapshot[{self.tid}]:", prev.tolist(), center.tolist(), velocity.tolist(), self.velocity.tolist())
        self.history.append(xyxysc)
        if len(self.history) > self.length:
            self.history.pop(0)

class Tracker(object):
    def __init__(self, *args, **kwargs):
        self.tracks = {}
        self.deleted = None

    def __contains__(self, tid):
        return tid in self.tracks

    def get(self, tid):
        return self.tracks.get(tid, None)

    def snapshot(self, fresh=True, dead=False):
        """Return last RoIs on the tracks.
        Args:
            fresh(bool): only non-predicted tracks to return
            dead(bool): including just deleted tracks
        Returns:
            snapshot(List[Tuple(tid, Tensor[N, 6])]): [(tid, xyxysc)]
        """
        tracks = [(tid, th.cat([trk.last, trk.origin, trk.velocity])) for tid, trk in self.tracks.items() if not fresh or not trk.predicted]
        if dead:
            deleted = [(tid, th.cat([trk.last, trk.origin, trk.velocity])) for tid, trk in self.deleted.items()]
            return tracks, deleted
        else:
            return tracks
    
    def update(self, xyxysc, features):
        """Update with detection bboxes and features.
        Args:
            dets(Tensor[N, 6]): object detections in xyxysc
            features(Tensor[N, C, H, W]): pooled object features
        """
        pass