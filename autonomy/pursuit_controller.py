import numpy as np
from .state_machine import State


def safe_goal(uav_position, estimated_position, estimated_velocity, standoff, horizon, predictive=True):
    future = estimated_position + (estimated_velocity*horizon if predictive else 0)
    delta = future-uav_position; distance=np.linalg.norm(delta)
    direction = delta/distance if distance > 1e-6 else np.array([1.,0.])
    return future-direction*standoff, future


def velocity_command(uav_position, uav_velocity, waypoint, estimated_velocity, state, max_speed):
    if state == State.SEARCH:
        angle = .12*np.sum(uav_position); return 1.2*np.array([np.cos(angle), np.sin(angle)])
    delta=waypoint-uav_position
    command=1.3*delta + .65*estimated_velocity
    limit=max_speed*(.55 if state == State.PREDICT else 1.0)
    n=np.linalg.norm(command)
    return command if n <= limit else command*limit/n

