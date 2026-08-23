import numpy as np
from PIL import Image
from evaluation.perception_metrics import detection_examples


class FakeTensor:
    def __init__(self,value): self.value=np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value


class FakeBoxes:
    xyxyn=FakeTensor([[.25,.25,.75,.75]])
    cls=FakeTensor([0])
    conf=FakeTensor([.9])


class FakeResult: boxes=FakeBoxes()
class FakeModel:
    def predict(self,*args,**kwargs): return iter([FakeResult()])


def test_detection_examples_writes_true_positive(tmp_path):
    root=tmp_path/"dataset"; (root/"images"/"val").mkdir(parents=True); (root/"labels"/"val").mkdir(parents=True)
    Image.new("RGB",(100,100),"white").save(root/"images"/"val"/"frame.jpg")
    (root/"labels"/"val"/"frame.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    data=tmp_path/"dataset.yaml"; data.write_text(f"path: {root.as_posix()}\nval: images/val\nnames:\n  0: uav\n")
    output=tmp_path/"artifacts"; summary=detection_examples(FakeModel(),data,output,device="cpu")
    assert summary["true_positives"]==1
    assert summary["false_positives"]==0
    assert summary["false_negatives"]==0
    assert (output/"sample_tp.jpg").exists()
