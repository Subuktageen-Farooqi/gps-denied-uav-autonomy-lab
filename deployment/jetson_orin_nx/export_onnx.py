import argparse, json, hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description="Export portable ONNX on the workstation; validate against PyTorch when --sample is supplied")
    p.add_argument("--model",required=True); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--device",default="0"); p.add_argument("--sample"); p.add_argument("--simplify",action="store_true"); a=p.parse_args()
    try: from ultralytics import YOLO
    except ImportError: raise SystemExit("Install: pip install -r requirements-vision.txt")
    pt=YOLO(a.model); exported=Path(pt.export(format="onnx",imgsz=a.imgsz,batch=1,dynamic=False,simplify=a.simplify,device=a.device))
    validation={"pytorch_model":str(Path(a.model).resolve()),"onnx_model":str(exported.resolve()),"imgsz":a.imgsz,"batch":1,"sample":a.sample,"validated":False,"pytorch_detection_count":None,"onnx_detection_count":None,"confidence_max_abs_difference":None}
    if a.sample:
        p_result=pt.predict(a.sample,imgsz=a.imgsz,verbose=False)[0]; o_result=YOLO(str(exported)).predict(a.sample,imgsz=a.imgsz,verbose=False)[0]
        pc=p_result.boxes.conf.cpu().numpy(); oc=o_result.boxes.conf.cpu().numpy(); validation.update(validated=True,pytorch_detection_count=len(pc),onnx_detection_count=len(oc),confidence_max_abs_difference=float(abs(pc[:min(len(pc),len(oc))]-oc[:min(len(pc),len(oc))]).max()) if len(pc) and len(oc) else None)
    exported.with_suffix(".validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    digest=hashlib.sha256(exported.read_bytes()).hexdigest(); source_manifest=Path(a.model).parent/"model_manifest.json"
    manifest=json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.exists() else {}
    manifest.update(artifact_type="ONNX",source_weights=str(Path(a.model).resolve()),artifact_path=str(exported.resolve()),artifact_sha256=digest,imgsz=a.imgsz,batch=1,dynamic=False,validation=validation)
    exported.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(validation,indent=2))
if __name__=="__main__": main()
