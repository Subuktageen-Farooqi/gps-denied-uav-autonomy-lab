from dataclasses import dataclass
import numpy as np


@dataclass
class MetricDetection:
    relative: np.ndarray
    confidence: float
    timestamp: float


@dataclass
class VisionDetection:
    xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    confidence: float
    class_id: int
    class_name: str
    timestamp: float
    track_id: int | None = None
    inference_ms: float | None = None


@dataclass
class BearingObservation:
    azimuth_deg: float
    elevation_deg: float
    confidence: float
    timestamp: float
    track_id: int | None = None

