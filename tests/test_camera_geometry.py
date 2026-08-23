import pytest
from perception.camera_geometry import CameraIntrinsics,pixel_to_bearing


def test_center_pixel_is_zero_bearing(): assert pixel_to_bearing(320,240,CameraIntrinsics(500,500,320,240))==pytest.approx((0,0))
def test_left_right_signs():
    intrinsics=CameraIntrinsics(500,500,320,240)
    assert pixel_to_bearing(100,240,intrinsics)[0]<0
    assert pixel_to_bearing(500,240,intrinsics)[0]>0

