import json, math
from pathlib import Path
import numpy as np


def normalize_ultralytics_metrics(result):
    values=getattr(result,"results_dict",{}) or {}; out={}
    aliases={"metrics/precision(B)":"precision","metrics/recall(B)":"recall","metrics/mAP50(B)":"map50","metrics/mAP50-95(B)":"map50_95"}
    for source,target in aliases.items():
        value=values.get(source); out[target]=float(value) if value is not None else None
    p,r=out.get("precision"),out.get("recall"); out["f1"]=2*p*r/(p+r) if p is not None and r is not None and p+r else None
    return out


def save_metrics(metrics,path): Path(path).write_text(json.dumps(metrics,indent=2),encoding="utf-8")


def _iou(box,boxes):
    if not len(boxes): return np.array([])
    left=np.maximum(box[:2],boxes[:,:2]); right=np.minimum(box[2:],boxes[:,2:]); inter=np.prod(np.maximum(0,right-left),axis=1)
    return inter/(np.prod(box[2:]-box[:2])+np.prod(boxes[:,2:]-boxes[:,:2],axis=1)-inter+1e-9)


def small_object_recall(model,data_yaml,imgsz=640,device="0"):
    """Measure GT recall at IoU .5 in dataset-relative tiny/small/medium+ buckets."""
    import yaml
    cfg=yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")); root=Path(cfg.get("path",Path(data_yaml).parent))
    if not root.is_absolute():
        cwd=(Path.cwd()/root).resolve(); root=cwd if cwd.exists() else (Path(data_yaml).resolve().parent/root).resolve()
    image_dir=root/cfg.get("val","images/val"); images=sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
    counts={name:{"matched":0,"total":0,"recall":None} for name in ("tiny","small","medium_plus")}
    results=model.predict([str(p) for p in images],imgsz=imgsz,device=device,verbose=False,stream=True)
    for image_path,result in zip(images,results):
        label=Path(str(image_path).replace(str(image_dir),str(root/"labels"/"val"))).with_suffix(".txt")
        gt=[]
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                values=line.split()
                if len(values)==5:
                    cls=int(values[0]); x,y,w,h=map(float,values[1:]); gt.append((cls,np.array([x-w/2,y-h/2,x+w/2,y+h/2]),w*h))
        pred_boxes=result.boxes.xyxyn.cpu().numpy() if result.boxes is not None else np.empty((0,4)); pred_cls=result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([],int)
        used=set()
        for cls,box,area in gt:
            bucket="tiny" if area<.0025 else "small" if area<.01 else "medium_plus"; counts[bucket]["total"]+=1
            candidates=np.where(pred_cls==cls)[0]; scores=_iou(box,pred_boxes[candidates]); order=np.argsort(scores)[::-1]
            match=next((int(candidates[j]) for j in order if scores[j]>=.5 and int(candidates[j]) not in used),None)
            if match is not None: used.add(match); counts[bucket]["matched"]+=1
    for values in counts.values(): values["recall"]=values["matched"]/values["total"] if values["total"] else None
    return counts


def detection_examples(model,data_yaml,output_dir,imgsz=640,device="0",iou_threshold=.5):
    """Create representative, evidence-backed TP/FP/FN images from labeled validation data."""
    import yaml
    from PIL import Image,ImageDraw
    yaml_path=Path(data_yaml).resolve(); cfg=yaml.safe_load(yaml_path.read_text(encoding="utf-8")); root=Path(cfg.get("path",yaml_path.parent))
    if not root.is_absolute():
        cwd=(Path.cwd()/root).resolve(); root=cwd if cwd.exists() else (yaml_path.parent/root).resolve()
    image_dir=root/cfg.get("val","images/val"); label_dir=root/"labels"/"val"
    images=sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
    if not images: raise ValueError(f"No validation images found in {image_dir}")
    names=cfg.get("names",{}); names={int(k):v for k,v in names.items()} if isinstance(names,dict) else dict(enumerate(names))
    totals={"true_positives":0,"false_positives":0,"false_negatives":0,"validation_images":len(images),"iou_threshold":iou_threshold}
    best={"tp":(-1,None),"fp":(-1,None),"fn":(-1,None)}
    results=model.predict([str(p) for p in images],imgsz=imgsz,device=device,verbose=False,stream=True)
    for image_path,result in zip(images,results):
        with Image.open(image_path) as source: width,height=source.size
        relative=image_path.relative_to(image_dir); label_path=(label_dir/relative).with_suffix(".txt"); gt=[]
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values=line.split()
                if len(values)==5:
                    cls=int(values[0]); x,y,w,h=map(float,values[1:]); gt.append((cls,np.array([x-w/2,y-h/2,x+w/2,y+h/2]),w*h))
        pred_boxes=result.boxes.xyxyn.cpu().numpy() if result.boxes is not None else np.empty((0,4)); pred_cls=result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([],int); pred_conf=result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        matched_gt=set(); matched_pred=set()
        for pred_index in np.argsort(pred_conf)[::-1]:
            candidates=[i for i,(cls,_,_) in enumerate(gt) if cls==pred_cls[pred_index] and i not in matched_gt]
            if not candidates: continue
            scores=_iou(pred_boxes[pred_index],np.asarray([gt[i][1] for i in candidates])); chosen=int(np.argmax(scores))
            if scores[chosen]>=iou_threshold: matched_pred.add(int(pred_index)); matched_gt.add(candidates[chosen])
        tp=len(matched_pred); fp=len(pred_boxes)-tp; fn=len(gt)-len(matched_gt)
        totals["true_positives"]+=tp; totals["false_positives"]+=fp; totals["false_negatives"]+=fn
        payload=(image_path,width,height,gt,pred_boxes,pred_cls,pred_conf,matched_gt,matched_pred,names)
        for key,score in (("tp",tp),("fp",fp),("fn",fn)):
            if score>best[key][0]: best[key]=(score,payload)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); paths={}
    for kind,(score,payload) in best.items():
        if score<=0 or payload is None: paths[kind]=None; continue
        image_path,width,height,gt,pred_boxes,pred_cls,pred_conf,matched_gt,matched_pred,names=payload; image=Image.open(image_path).convert("RGB"); draw=ImageDraw.Draw(image)
        if kind in {"tp","fn"}:
            for index,(cls,box,_) in enumerate(gt):
                selected=(kind=="tp" and index in matched_gt) or (kind=="fn" and index not in matched_gt)
                if selected:
                    x1,y1,x2,y2=box*np.array([width,height,width,height]); color="#16a34a" if kind=="tp" else "#f97316"; draw.rectangle((x1,y1,x2,y2),outline=color,width=4); draw.text((x1,max(0,y1-16)),f"{kind.upper()} GT {names.get(cls,cls)}",fill=color)
        if kind in {"tp","fp"}:
            for index,box in enumerate(pred_boxes):
                selected=(kind=="tp" and index in matched_pred) or (kind=="fp" and index not in matched_pred)
                if selected:
                    x1,y1,x2,y2=box*np.array([width,height,width,height]); color="#15803d" if kind=="tp" else "#dc2626"; draw.rectangle((x1,y1,x2,y2),outline=color,width=3); draw.text((x1,min(height-16,y2+2)),f"{kind.upper()} {names.get(int(pred_cls[index]),int(pred_cls[index]))} {pred_conf[index]:.2f}",fill=color)
        destination=out/f"sample_{kind}.jpg"; image.save(destination,quality=92); paths[kind]=str(destination)
    totals["precision_at_iou"]=(totals["true_positives"]/(totals["true_positives"]+totals["false_positives"])) if totals["true_positives"]+totals["false_positives"] else None
    totals["recall_at_iou"]=(totals["true_positives"]/(totals["true_positives"]+totals["false_negatives"])) if totals["true_positives"]+totals["false_negatives"] else None
    totals["example_paths"]=paths; return totals
