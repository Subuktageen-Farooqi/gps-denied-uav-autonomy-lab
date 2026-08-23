import numpy as np
from .types import MetricDetection

Detection = MetricDetection  # Backwards-compatible public name.


class SimulatedDetector:
    def __init__(self, probability, noise_std, occlusions, rng):
        self.probability, self.noise_std = probability, noise_std
        self.occlusions, self.rng = occlusions, rng

    def detect(self, target_position, uav_position, t):
        if any(a <= t <= b for a, b in self.occlusions) or self.rng.random() > self.probability:
            return None
        relative = target_position - uav_position
        noisy = relative + self.rng.normal(0, self.noise_std, 2)
        confidence = float(np.clip(self.probability * np.exp(-np.linalg.norm(relative)/180) * self.rng.uniform(.85, 1.05), 0, 1))
        return Detection(noisy, confidence, t)

