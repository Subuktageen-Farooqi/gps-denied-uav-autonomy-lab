import argparse, csv, json, hashlib
from collections import Counter
from pathlib import Path
import numpy as np


IMAGE_EXT={".jpg",".jpeg",".png",".bmp",".webp"}


def _load_yaml(path):
    try: import yaml
    except ImportError as exc: raise RuntimeError("PyYAML is required for dataset YAML files") from exc
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def dataset_hash(root):
    digest=hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT|{".txt"}):
        digest.update(str(path.relative_to(root)).encode()); digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def audit_dataset(data_yaml,output_dir="artifacts"):
    from PIL import Image
    yaml_path=Path(data_yaml).resolve(); cfg=_load_yaml(yaml_path); root=Path(cfg.get("path",yaml_path.parent))
    if not root.is_absolute():
        cwd_candidate=(Path.cwd()/root).resolve(); yaml_candidate=(yaml_path.parent/root).resolve(); root=cwd_candidate if cwd_candidate.exists() else yaml_candidate
    report={"dataset_yaml":str(yaml_path),"dataset_root":str(root),"dataset_hash":dataset_hash(root),"splits":{},"class_distribution":{},"invalid_labels":[],"corrupted_images":[],"duplicate_filenames":[],"bbox_area_ratios":[],"image_sizes":[]}
    names=[]
    for split in ("train","val","test"):
        rel=cfg.get(split)
        if not rel: continue
        image_dir=(root/rel).resolve(); images=sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT); names.extend(p.name for p in images)
        empty=missing=boxes=0
        for image_path in images:
            try:
                with Image.open(image_path) as im: width,height=im.size; im.verify()
                report["image_sizes"].append([width,height])
            except Exception as exc: report["corrupted_images"].append({"path":str(image_path),"error":str(exc)}); continue
            label_path=Path(str(image_path).replace(str(image_dir),str(root/"labels"/split))).with_suffix(".txt")
            if not label_path.exists(): missing+=1; empty+=1; continue
            lines=[line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines: empty+=1
            for line_no,line in enumerate(lines,1):
                parts=line.split()
                try:
                    if len(parts)!=5: raise ValueError("expected class + 4 coordinates")
                    cls=int(parts[0]); x,y,w,h=map(float,parts[1:])
                    if any(not 0<=v<=1 for v in (x,y,w,h)) or w<=0 or h<=0: raise ValueError("box outside normalized [0,1] bounds")
                    report["class_distribution"][str(cls)]=report["class_distribution"].get(str(cls),0)+1
                    report["bbox_area_ratios"].append(w*h); boxes+=1
                except Exception as exc: report["invalid_labels"].append({"path":str(label_path),"line":line_no,"error":str(exc)})
        report["splits"][split]={"images":len(images),"boxes":boxes,"missing_labels":missing,"hard_negative_images":empty}
    counts=Counter(names); report["duplicate_filenames"]=[name for name,count in counts.items() if count>1]
    areas=np.asarray(report["bbox_area_ratios"],float)
    report["small_object_proportion"]=float(np.mean(areas<.01)) if len(areas) else 0.
    report["bbox_area_summary"]={"min":float(areas.min()),"median":float(np.median(areas)),"p95":float(np.percentile(areas,95)),"max":float(areas.max())} if len(areas) else None
    report["passed"]=not report["invalid_labels"] and not report["corrupted_images"]
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"dataset_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (out/"dataset_report.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["split","images","boxes","missing_labels","hard_negative_images"]); writer.writeheader()
        for split,values in report["splits"].items(): writer.writerow({"split":split,**values})
    return report


def main():
    p=argparse.ArgumentParser(description="Audit a YOLO UAV dataset before training"); p.add_argument("--data",required=True); p.add_argument("--output",default="artifacts"); args=p.parse_args()
    report=audit_dataset(args.data,args.output); print(json.dumps(report,indent=2)); raise SystemExit(0 if report["passed"] else 2)
if __name__=="__main__": main()
