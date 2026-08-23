from dataclasses import dataclass
import math
from .types import VisionDetection, BearingObservation


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")


def pixel_to_bearing(u: float, v: float, intrinsics: CameraIntrinsics) -> tuple[float, float]:
    return (math.degrees(math.atan((u-intrinsics.cx)/intrinsics.fx)),
            math.degrees(math.atan((v-intrinsics.cy)/intrinsics.fy)))


def detection_to_bearing(detection: VisionDetection, intrinsics: CameraIntrinsics) -> BearingObservation:
    azimuth,elevation=pixel_to_bearing(*detection.center_px,intrinsics)
    return BearingObservation(azimuth,elevation,detection.confidence,detection.timestamp,detection.track_id)

