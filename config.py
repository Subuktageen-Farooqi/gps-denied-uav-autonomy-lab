from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    scenario: str = "Obstacle Pursuit"
    mode: str = "Predictive Standoff Pursuit"
    width: float = 100.0
    height: float = 70.0
    dt: float = 0.1
    duration: float = 60.0
    target_speed: float | None = None
    detection_probability: float | None = None
    sensor_noise: float | None = None
    desired_standoff: float = 8.5
    prediction_horizon: float = 2.0
    uav_max_speed: float = 7.0
    seed: int = 7
    grid_resolution: float = 1.0
    obstacle_inflation: float = 1.2
    replan_interval: float = 1.0
    randomize_obstacles: bool = False
    obstacle_seed: int = 21
    obstacle_count: int = 6
    obstacle_min_size: float = 5.0
    obstacle_max_size: float = 14.0
    obstacle_max_height: float = 18.0
    uav_altitude: float = 12.0
    target_altitude: float = 2.0
    perception_backend: str = "simulated"
    detector_profile_path: str | None = None
    vision_model_path: str | None = None
    vision_confidence: float = 0.25
    vision_imgsz: int = 640
