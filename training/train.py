import argparse, json
from pathlib import Path
from .dataset_audit import audit_dataset
from .make_manifest import make_manifest


def main():
    p=argparse.ArgumentParser(description="Transfer-learn a custom UAV YOLO detector (RTX 3060: use --device 0)")
    p.add_argument("--data",required=True); p.add_argument("--model",default="yolo26n.pt"); p.add_argument("--epochs",type=int,default=50); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--batch",default="-1"); p.add_argument("--device",default="0"); p.add_argument("--seed",type=int,default=7); p.add_argument("--patience",type=int,default=10); p.add_argument("--project",default="artifacts/models"); p.add_argument("--name",default="uav_yolo26n_640"); a=p.parse_args()
    report=audit_dataset(a.data,Path(a.project)/a.name)
    if not report["passed"]: raise SystemExit("Dataset audit failed; inspect dataset_report.json")
    try: from ultralytics import YOLO
    except ImportError: raise SystemExit("Install: pip install -r requirements-vision.txt")
    selected=a.model
    try: model=YOLO(selected)
    except Exception as exc:
        if selected=="yolo26n.pt":
            selected="yolo11n.pt"; print(f"YOLO26n unavailable ({exc}); explicitly falling back to {selected}"); model=YOLO(selected)
        else: raise
    batch=-1 if a.batch.lower()=="auto" else (float(a.batch) if "." in a.batch else int(a.batch))
    results=model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,batch=batch,device=a.device,seed=a.seed,patience=a.patience,project=a.project,name=a.name,pretrained=True)
    best=Path(results.save_dir)/"weights"/"best.pt"; metrics={k:float(v) for k,v in getattr(results,"results_dict",{}).items() if isinstance(v,(int,float))}
    manifest=make_manifest(best,a.data,selected,a.imgsz,a.seed,metrics); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
