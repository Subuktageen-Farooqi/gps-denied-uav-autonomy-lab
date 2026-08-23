import numpy as np


class Target:
    def __init__(self, position, velocity, rng, turn_interval=12.0):
        self.position = np.asarray(position, float)
        self.velocity = np.asarray(velocity, float)
        self.rng, self.turn_interval, self.next_turn = rng, turn_interval, turn_interval

    def step(self, t, dt, bounds):
        if t >= self.next_turn:
            angle = self.rng.normal(0, .35)
            c, s = np.cos(angle), np.sin(angle)
            self.velocity = np.array([[c,-s],[s,c]]) @ self.velocity
            self.next_turn += self.turn_interval
        self.velocity += self.rng.normal(0, .025, 2) * dt
        self.position += self.velocity * dt
        for i in range(2):
            if self.position[i] < 2 or self.position[i] > bounds[i] - 2:
                self.velocity[i] *= -1
                self.position[i] = np.clip(self.position[i], 2, bounds[i]-2)

