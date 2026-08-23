from pathlib import Path
from time import perf_counter
from .types import VisionDetection


class YoloDetector:
    """Lazy Ultralytics adapter. Pixel detections never enter the metric autonomy filter."""
    def __init__(self, model_path, conf=.25, iou=.7, imgsz=640, device=None, target_classes=None):
        path=Path(model_path)
        if not path.exists(): raise FileNotFoundError(f"Model not found: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install optional vision dependencies: pip install -r requirements-vision.txt") from exc
        self.path=path; self.conf=conf; self.iou=iou; self.imgsz=imgsz; self.device=device
        self.model=YOLO(str(path)); self.target_classes={name.lower() for name in (target_classes or ["uav","drone","quadcopter"])}
        self.last_inference_ms=None

    def _convert(self, result, timestamp, latency_ms):
        output=[]; names=result.names
        if result.boxes is None: return output
        ids=result.boxes.id.cpu().tolist() if result.boxes.id is not None else [None]*len(result.boxes)
        for box,track_id in zip(result.boxes,ids):
            xyxy=tuple(float(x) for x in box.xyxy[0].cpu().tolist()); cls=int(box.cls[0]); name=str(names[cls])
            if self.target_classes and name.lower() not in self.target_classes: continue
            output.append(VisionDetection(xyxy,((xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2),float(box.conf[0]),cls,name,float(timestamp),None if track_id is None else int(track_id),latency_ms))
        return output

    def detect(self, frame, timestamp=0.0):
        start=perf_counter(); result=self.model.predict(frame,conf=self.conf,iou=self.iou,imgsz=self.imgsz,device=self.device,verbose=False)[0]
        self.last_inference_ms=(perf_counter()-start)*1000
        return self._convert(result,timestamp,self.last_inference_ms)

    def track(self, frame, timestamp=0.0, tracker="bytetrack.yaml", persist=True):
        start=perf_counter(); result=self.model.track(frame,conf=self.conf,iou=self.iou,imgsz=self.imgsz,device=self.device,tracker=tracker,persist=persist,verbose=False)[0]
        self.last_inference_ms=(perf_counter()-start)*1000
        return self._convert(result,timestamp,self.last_inference_ms)

    def metadata(self):
        suffix=self.path.suffix.lower()
        return {"model":str(self.path),"backend":{'.pt':'PyTorch','.onnx':'ONNX','.engine':'TensorRT'}.get(suffix,suffix),"imgsz":self.imgsz,"confidence":self.conf,"device":self.device}
