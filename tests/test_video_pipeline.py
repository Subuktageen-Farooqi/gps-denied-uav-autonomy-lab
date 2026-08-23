from perception.video_pipeline import detection_rows


def test_empty_detection_rows_are_empty():
    class Intrinsics: pass
    assert detection_rows([],Intrinsics(),3)==[]
