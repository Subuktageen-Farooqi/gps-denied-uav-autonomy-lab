from dataclasses import dataclass


@dataclass
class TrackSummary:
    track_id: int
    age_frames: int
    last_center: tuple[float,float]


class ImageTrackRegistry:
    """Bookkeeping for image-space IDs; deliberately separate from metric Kalman state."""
    def __init__(self): self._tracks={}
    def update(self,detections):
        for d in detections:
            if d.track_id is None: continue
            old=self._tracks.get(d.track_id)
            self._tracks[d.track_id]=TrackSummary(d.track_id,1 if old is None else old.age_frames+1,d.center_px)
        return list(self._tracks.values())

