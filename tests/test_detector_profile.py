import json
import numpy as np
import pytest
from perception.detector_profile import load_profile,ProfiledSimulatedDetector


def test_profile_validation(tmp_path):
    path=tmp_path/"profile.json"; path.write_text(json.dumps({"overall_detection_rate":1.0,"confidence_mean":.8,"dropout_lengths_frames":[1]}))
    assert load_profile(path)["confidence_mean"]==.8


def test_profiled_detector_is_deterministic(tmp_path):
    path=tmp_path/"profile.json"; path.write_text(json.dumps({"overall_detection_rate":.8,"confidence_mean":.7,"confidence_std":.05,"dropout_lengths_frames":[1,2]}))
    def sequence():
        detector=ProfiledSimulatedDetector(path,1.0,(),np.random.default_rng(4)); return [detector.detect(np.array([10.,5.]),np.zeros(2),i*.1) is not None for i in range(20)]
    assert sequence()==sequence()


def test_original_simulation_default():
    from config import SimulationConfig
    from simulation.engine import SimulationEngine
    engine=SimulationEngine(SimulationConfig(duration=.5)); assert len(engine.run())==5
