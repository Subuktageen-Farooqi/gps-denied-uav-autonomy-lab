from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    uav_start: tuple[float, float]
    target_start: tuple[float, float]
    target_velocity: tuple[float, float]
    detection_probability: float
    sensor_noise: float
    occlusions: tuple[tuple[float, float], ...]
    obstacles: tuple[tuple[float, float, float, float], ...] = ()
    turn_interval: float = 12.0


SCENARIOS = {
    "Basic Tracking": Scenario("Basic Tracking", (12, 12), (30, 24), (1.2, .5), .95, .7, ((18, 20),)),
    "Occlusion Stress Test": Scenario("Occlusion Stress Test", (10, 12), (32, 22), (1.4, .65), .88, 1.4, ((15, 18), (34, 38)), (), 9),
    "Obstacle Pursuit": Scenario("Obstacle Pursuit", (8, 8), (28, 18), (1.25, .75), .92, 1.0, ((22, 25), (43, 46)),
        ((25,35,10,30),(45,55,40,60),(65,72,15,38),(15,22,45,60))),
    "Fast Target": Scenario("Fast Target", (10, 10), (30, 20), (2.6, 1.1), .90, 1.1, ((25, 28),), (), 8),
}


def get_scenario(name: str) -> Scenario:
    return SCENARIOS[name]

