import numpy as np


class KalmanFilterCV:
    def __init__(self, process_var=.8, measurement_var=1.0):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0
        self.process_var, self.measurement_var = process_var, measurement_var
        self.initialized = False

    def initialize(self, position):
        self.x[:2] = position
        self.P = np.diag([4., 4., 25., 25.])
        self.initialized = True

    def predict(self, dt):
        if not self.initialized: return self.x
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
        q = self.process_var
        Q = q*np.array([[dt**4/4,0,dt**3/2,0],[0,dt**4/4,0,dt**3/2],[dt**3/2,0,dt**2,0],[0,dt**3/2,0,dt**2]])
        self.x, self.P = F @ self.x, F @ self.P @ F.T + Q
        return self.x

    def update(self, position):
        if not self.initialized:
            self.initialize(position); return self.x
        H = np.array([[1,0,0,0],[0,1,0,0]], float)
        R = np.eye(2)*self.measurement_var
        y = position-H@self.x; S=H@self.P@H.T+R
        K=self.P@H.T@np.linalg.inv(S)
        self.x += K@y
        self.P=(np.eye(4)-K@H)@self.P
        return self.x

    @property
    def position(self): return self.x[:2].copy()
    @property
    def velocity(self): return self.x[2:].copy()
    @property
    def uncertainty(self): return float(np.sqrt(np.trace(self.P[:2,:2])))

