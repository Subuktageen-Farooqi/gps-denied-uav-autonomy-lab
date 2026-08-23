import numpy as np


def follow_command(relative_estimate, estimated_velocity, desired_distance, max_speed):
    distance=np.linalg.norm(relative_estimate)
    direction=relative_estimate/distance if distance > 1e-6 else np.zeros(2)
    command=.7*(distance-desired_distance)*direction + estimated_velocity
    n=np.linalg.norm(command)
    return command if n <= max_speed else command*max_speed/n

