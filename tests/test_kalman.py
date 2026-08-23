import numpy as np
from estimation.kalman_filter import KalmanFilterCV


def test_constant_velocity_converges():
    k=KalmanFilterCV(measurement_var=.1); dt=.1
    for i in range(100):
        if k.initialized: k.predict(dt)
        k.update(np.array([i*dt,2*i*dt]))
    assert np.linalg.norm(k.velocity-[1,2]) < .15


def test_prediction_only_advances_position():
    k=KalmanFilterCV(); k.initialize([0,0]); k.x[2:]=[2,1]; k.predict(.5)
    assert np.allclose(k.position,[1,.5])

