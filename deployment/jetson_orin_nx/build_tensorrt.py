import argparse, json, warnings
from pathlib import Path
from .device_info import collect_device_info


def main():
    p=argparse.ArgumentParser(description="Build a target-specific TensorRT engine on Jetson Orin NX")
    p.add_argument("--model",required=True); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--precision",choices=["fp16","int8"],default="fp16"); p.add_argument("--data",help="Representative calibration dataset YAML; mandatory for INT8"); p.add_argument("--output-dir",default="artifacts/edge/jetson_orin_nx"); a=p.parse_args()
    info=collect_device_info()
    if not info["is_jetson"]: warnings.warn("This is not detected as Jetson hardware. A desktop-built engine is NOT a production Orin NX engine.",RuntimeWarning)
    if a.precision=="int8" and not a.data: raise SystemExit("INT8 requires --data with representative calibration images and subsequent accuracy re-validation")
    try: from ultralytics import YOLO
    except ImportError: raise SystemExit("Use the Ultralytics/PyTorch packages compatible with the installed JetPack release")
    model=YOLO(a.model); engine=Path(model.export(format="engine",imgsz=a.imgsz,batch=1,half=a.precision=="fp16",int8=a.precision=="int8",data=a.data,device=0))
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); destination=out/("model_int8.engine" if a.precision=="int8" else "model_fp16.engine"); engine.replace(destination)
    metadata={"engine":str(destination),"precision":a.precision,"imgsz":a.imgsz,"batch":1,"calibration_data":a.data,"device":info,"accuracy_revalidation_required":a.precision=="int8"}
    (out/"build_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); print(json.dumps(metadata,indent=2))
if __name__=="__main__": main()

