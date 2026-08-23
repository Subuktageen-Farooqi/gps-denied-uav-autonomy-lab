import numpy as np
from perception.types import MetricDetection, VisionDetection


def test_metric_and_vision_types_are_distinct():
    metric=MetricDetection(np.array([1.,2.]),.8,0.)
    vision=VisionDetection((1,2,5,8),(3,5),.9,0,"uav",0.)
    assert metric.relative.shape==(2,)
    assert not hasattr(vision,"relative")

