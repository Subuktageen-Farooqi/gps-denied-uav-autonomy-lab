import numpy as np


class Vehicle:
    def __init__(self, position, max_speed=7.0, max_accel=5.0):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.zeros(2)
        self.max_speed, self.max_accel = max_speed, max_accel

    def step(self, command, dt, bounds):
        command = np.asarray(command, dtype=float)
        speed = np.linalg.norm(command)
        if speed > self.max_speed:
            command *= self.max_speed / speed
        dv = command - self.velocity
        n = np.linalg.norm(dv)
        if n > self.max_accel * dt:
            dv *= self.max_accel * dt / n
        self.velocity += dv
        self.position += self.velocity * dt
        self.position[:] = np.clip(self.position, [0, 0], bounds)

