import json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st


ROOT=Path(__file__).resolve().parents[1]


def available_models():
    paths=[]
    for folder in (ROOT/"artifacts"/"models",ROOT/"models"):
        if folder.exists(): paths.extend(p for p in folder.rglob("*") if p.suffix.lower() in {".pt",".onnx",".engine"})
    return sorted(set(paths))


def render_perception_lab():
    st.header("Perception Lab")
    st.caption("Image-space detection, tracking, and monocular bearing. Range: N/A — bearing-only.")
    models=available_models(); supplied=st.text_input("Model path",value=str(models[0]) if models else "",placeholder="artifacts/models/<run>/weights/best.pt")
    c1,c2,c3=st.columns(3); conf=c1.slider("Confidence",.05,.95,.25,.05); iou=c2.slider("IoU threshold",.1,.9,.7,.05); imgsz=c3.select_slider("Input resolution",[320,480,640,960],value=640)
    c1,c2,c3,c4=st.columns(4); fx=c1.number_input("fx",value=640.); fy=c2.number_input("fy",value=640.); cx=c3.number_input("cx",value=320.); cy=c4.number_input("cy",value=320.)
    tracking=st.checkbox("Enable ByteTrack for video",True); upload=st.file_uploader("Image or video",type=["jpg","jpeg","png","webp","mp4","mov","avi"])
    if not supplied: st.info("Add custom `.pt`, `.onnx`, or Jetson `.engine` weights to `models/` or enter a path. The autonomy simulator remains available without weights."); return
    if upload is None: st.info("Upload an image or video to run actual inference."); return
    if st.button("Run perception",type="primary"):
        try:
            from PIL import Image,ImageDraw
            from perception.yolo_detector import YoloDetector
            from perception.camera_geometry import CameraIntrinsics,detection_to_bearing
            import torch
            detector=YoloDetector(supplied,conf,iou,imgsz,device=0 if torch.cuda.is_available() else "cpu"); intrinsics=CameraIntrinsics(fx,fy,cx,cy)
            if upload.type.startswith("image"):
                image=Image.open(upload).convert("RGB"); detections=detector.detect(np.asarray(image)); draw=ImageDraw.Draw(image); rows=[]
                for d in detections:
                    b=detection_to_bearing(d,intrinsics); draw.rectangle(d.xyxy,outline="#22c55e",width=3); draw.text((d.xyxy[0],max(0,d.xyxy[1]-15)),f"{d.class_name} {d.confidence:.2f} az {b.azimuth_deg:+.1f} el {b.elevation_deg:+.1f}",fill="#16a34a")
                    rows.append({**d.__dict__,"azimuth_deg":b.azimuth_deg,"elevation_deg":b.elevation_deg,"range":"N/A — monocular bearing-only"})
                st.image(image,width="stretch"); st.dataframe(pd.DataFrame(rows),width="stretch"); st.json(detector.metadata())
            else:
                from perception.video_pipeline import process_video
                suffix=Path(upload.name).suffix
                with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as handle: handle.write(upload.getbuffer()); source=handle.name
                try:
                    frames,table=process_video(source,detector,intrinsics,tracking,max_frames=300)
                finally:
                    Path(source).unlink(missing_ok=True)
                if frames: st.image(frames[-1][:,:,::-1],caption="Last processed frame",width="stretch")
                st.dataframe(table,width="stretch"); latency=table.get("inference_ms",pd.Series(dtype=float)).dropna()
                m1,m2,m3,m4=st.columns(4); m1.metric("Backend",detector.metadata()["backend"]); m2.metric("Range","N/A — bearing-only"); m3.metric("p50 latency",f"{latency.quantile(.5):.1f} ms" if len(latency) else "N/A"); m4.metric("p95 latency",f"{latency.quantile(.95):.1f} ms" if len(latency) else "N/A")
        except Exception as exc: st.error(str(exc))


def render_model_evaluation():
    st.header("Model Evaluation")
    model_root=ROOT/"artifacts"/"models"; runs=sorted({p.parent for p in model_root.rglob("metrics.json")}) if model_root.exists() else []
    if not runs: st.info("NOT MEASURED — run `python -m training.evaluate` with weights and labeled validation data."); return
    selected=Path(st.selectbox("Model run",runs,format_func=lambda p:str(p.relative_to(ROOT))))
    manifest_path=selected/"model_manifest.json"; manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}; metrics_path=selected/"metrics.json"; metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
    cols=st.columns(5)
    for col,key,label in zip(cols,["map50","map50_95","precision","recall","f1"],["mAP50","mAP50-95","Precision","Recall","F1"]): col.metric(label,"NOT MEASURED" if metrics.get(key) is None else f"{metrics[key]:.3f}")
    summary=metrics.get("detection_summary",{}); counts=st.columns(3); counts[0].metric("True positives",summary.get("true_positives","NOT MEASURED")); counts[1].metric("False positives",summary.get("false_positives","NOT MEASURED")); counts[2].metric("False negatives",summary.get("false_negatives","NOT MEASURED"))
    images=[p for p in selected.glob("*.png") if "confusion" in p.name.lower() or "pr_curve" in p.name.lower()]
    if images:
        for image in images: st.image(str(image),caption=image.name)
    st.subheader("Representative validation outcomes")
    examples=st.columns(3)
    for column,kind,label in zip(examples,["tp","fp","fn"],["True positive","False positive","False negative"]):
        path=selected/f"sample_{kind}.jpg"
        if path.exists(): column.image(str(path),caption=label,width="stretch")
        else: column.info(f"No {label.lower()} example occurred.")
    st.write("Small-object recall:",metrics.get("small_object_recall","NOT MEASURED"))
    with st.expander("Training and runtime metadata"): st.json(manifest)


def render_edge_deployment():
    st.header("Edge Deployment")
    st.markdown("**TARGET EDGE COMPUTER**  \nNVIDIA Jetson Orin NX 16GB  \nTensorRT FP16 · 640×640 · batch 1 · 15–25 W target profile")
    models=available_models(); suffixes={p.suffix.lower() for p in models}; benchmark=ROOT/"artifacts"/"edge"/"jetson_orin_nx"/"benchmark.json"
    rows=[("PyTorch weights","READY" if ".pt" in suffixes else "MISSING"),("ONNX","READY" if ".onnx" in suffixes else "MISSING"),("TensorRT FP16","READY" if ".engine" in suffixes else "BUILD ON JETSON"),("TensorRT INT8","OPTIONAL — calibration required"),("Jetson benchmark","AVAILABLE" if benchmark.exists() else "NOT MEASURED")]
    st.dataframe(pd.DataFrame(rows,columns=["Artifact","Status"]),hide_index=True,width="stretch")
    if benchmark.exists():
        data=json.loads(benchmark.read_text(encoding="utf-8")); st.subheader("Actual benchmark"); st.json(data)
    else: st.warning("Latency, FPS, power, and temperature: NOT MEASURED. Run the benchmark on the actual Jetson.")


def render_vision_architecture():
    st.header("Architecture")
    a,b,c=st.columns(3)
    a.code("Labeled UAV Dataset\n↓\nDataset Audit\n↓\nYOLO Fine-Tuning\n↓\nValidation → best.pt\n↓\nONNX\n↓\nJetson TensorRT")
    b.code("Camera Frame\n↓\nCustom YOLO\n↓\nBBox / Confidence\n↓\nByteTrack\n↓\nCamera Intrinsics\n↓\nBearing Observation")
    c.code("YOLO Evaluation\n↓\nReliability Profile\n↓\nMetric Sensor Simulation\n↓\nKalman + FSM\n↓\nSafe Standoff Goal\n↓\nA* + UAV Simulation")
    st.info("Pixel coordinates are never treated as metric world coordinates. Monocular YOLO supplies bearing, not range.")
