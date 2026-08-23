import argparse, json, time
from pathlib import Path
import numpy as np
from .device_info import collect_device_info


def main():
    p=argparse.ArgumentParser(description="Benchmark actual model latency on image/video input")
    p.add_argument("--model",required=True); p.add_argument("--source",required=True); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--conf",type=float,default=.25); p.add_argument("--warmup",type=int,default=20); p.add_argument("--iterations",type=int,default=200); p.add_argument("--output",default="artifacts/edge/jetson_orin_nx/benchmark.json"); a=p.parse_args()
    try: import cv2, psutil
    except ImportError: raise SystemExit("Install OpenCV and psutil for benchmarking")
    from perception.yolo_detector import YoloDetector
    start=time.perf_counter(); detector=YoloDetector(a.model,a.conf,imgsz=a.imgsz,device=0); load_ms=(time.perf_counter()-start)*1000
    cap=cv2.VideoCapture(a.source); frames=[]
    while len(frames)<max(a.warmup,a.iterations):
        ok,frame=cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,frame=cap.read()
            if not ok: break
        frames.append(frame)
    cap.release()
    if not frames: raise SystemExit("No frames readable from --source")
    for i in range(a.warmup): detector.detect(frames[i%len(frames)])
    latencies=[]; detections=0
    for i in range(a.iterations):
        start=time.perf_counter(); result=detector.detect(frames[i%len(frames)]); latencies.append((time.perf_counter()-start)*1000); detections+=len(result)
    arr=np.asarray(latencies); info=collect_device_info(); suffix=Path(a.model).suffix.lower(); precision="FP16" if suffix==".engine" else "runtime-dependent"
    report={"model":str(Path(a.model).resolve()),"backend":detector.metadata()["backend"],"precision":precision,"input_size":a.imgsz,"batch":1,"load_time_ms":load_ms,"warmup_iterations":a.warmup,"iterations":a.iterations,"p50_ms":float(np.percentile(arr,50)),"p95_ms":float(np.percentile(arr,95)),"p99_ms":float(np.percentile(arr,99)),"fps":float(1000/arr.mean()),"model_size_mb":Path(a.model).stat().st_size/1048576,"process_rss_mb":psutil.Process().memory_info().rss/1048576,"detection_count":detections,"device":info,"temperature_c":None,"power_w":None,"target_p95_ms":33.3,"target_met":bool(np.percentile(arr,95)<=33.3)}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()

