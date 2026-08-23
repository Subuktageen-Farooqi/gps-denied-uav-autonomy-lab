import argparse, json, shutil, platform
from pathlib import Path
from evaluation.perception_metrics import normalize_ultralytics_metrics, save_metrics, small_object_recall, detection_examples


def main():
    p=argparse.ArgumentParser(description="Evaluate custom UAV weights and save actual validation artifacts"); p.add_argument("--model",required=True); p.add_argument("--data",required=True); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--device",default="0"); p.add_argument("--output"); a=p.parse_args()
    try: from ultralytics import YOLO
    except ImportError: raise SystemExit("Install: pip install -r requirements-vision.txt")
    model=YOLO(a.model); result=model.val(data=a.data,imgsz=a.imgsz,device=a.device,plots=True,save_json=True)
    out=Path(a.output or Path(a.model).parent); out.mkdir(parents=True,exist_ok=True); metrics=normalize_ultralytics_metrics(result); metrics["small_object_recall"]=small_object_recall(model,a.data,a.imgsz,a.device); metrics["detection_summary"]=detection_examples(model,a.data,out,a.imgsz,a.device)
    save_metrics(metrics,out/"metrics.json")
    for name in ("confusion_matrix.png","confusion_matrix_normalized.png","BoxPR_curve.png","PR_curve.png","P_curve.png","R_curve.png","F1_curve.png"):
        source=Path(result.save_dir)/name
        if source.exists(): shutil.copy2(source,out/name.lower().replace("box", ""))
    manifest_path=out/"model_manifest.json"
    if not manifest_path.exists():
        manifest={"evaluation_only":True,"model_path":str(Path(a.model).resolve()),"model_classes":model.names,"dataset_yaml":str(Path(a.data).resolve()),"imgsz":a.imgsz,"device":a.device,"ultralytics_version":__import__("ultralytics").__version__,"python_version":platform.python_version()}
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
