"""Bounded GPU inference, with only the final annotated preview retained."""
from dataclasses import asdict
import base64
import io
import json
from pathlib import Path
import tempfile

import numpy as np


def analyze_media(root, detector_cache, data, suffix, model_id, confidence, iou, resolution, tracking, calibration):
    from PIL import Image, ImageDraw
    from perception.camera_geometry import CameraIntrinsics, detection_to_bearing
    from perception.yolo_detector import YoloDetector
    from server.api import model_catalog
    import torch

    catalog = model_catalog(Path(root))
    if model_id not in catalog:
        raise ValueError("Model unavailable on worker")
    cache_key = (model_id, confidence, iou, resolution)
    if cache_key not in detector_cache:
        detector_cache.clear()
        detector_cache[cache_key] = YoloDetector(catalog[model_id], confidence, iou, resolution, device=0)
    detector = detector_cache[cache_key]
    intrinsics = CameraIntrinsics(*calibration)
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError("Image exceeds the 40 megapixel limit")
            image = source.convert("RGB")
        detections = detector.detect(np.asarray(image))
        draw = ImageDraw.Draw(image)
        rows = []
        for detection in detections:
            bearing = detection_to_bearing(detection, intrinsics)
            draw.rectangle(detection.xyxy, outline="#ff5a2a", width=3)
            draw.text((detection.xyxy[0], max(0, detection.xyxy[1]-16)), f"{detection.class_name} {detection.confidence:.2f}", fill="#ff5a2a")
            rows.append({**asdict(detection), "azimuth_deg": bearing.azimuth_deg, "elevation_deg": bearing.elevation_deg})
        latencies = [detector.last_inference_ms]
        frames_processed = 1
    else:
        from perception.video_pipeline import process_video
        # A new tracker is required for each independent clip.
        detector_cache.clear()
        detector = YoloDetector(catalog[model_id], confidence, iou, resolution, device=0)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            source_path = Path(handle.name)
        try:
            frames, table = process_video(source_path, detector, intrinsics, tracking, max_frames=300, retain_frames=False)
        finally:
            source_path.unlink(missing_ok=True)
        if not frames:
            raise ValueError("No readable frames found")
        image = Image.fromarray(frames[-1][:, :, ::-1])
        rows = json.loads(table.to_json(orient="records"))
        latencies = table.groupby("frame").inference_ms.first().dropna().tolist()
        frames_processed = int(table.frame.nunique())
    preview = io.BytesIO()
    image.thumbnail((1600, 1000))
    image.save(preview, format="JPEG", quality=88)
    return {
        "preview": "data:image/jpeg;base64," + base64.b64encode(preview.getvalue()).decode(),
        "rows": rows, "frames_processed": frames_processed,
        "latency_p50_ms": float(np.median(latencies)) if latencies else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "gpu": torch.cuda.get_device_name(0), "model": model_id,
        "range": None, "range_note": "Monocular bearing only; metric range is unavailable.",
        "video_frame_limit": 300,
    }
