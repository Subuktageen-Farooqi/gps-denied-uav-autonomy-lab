from pathlib import Path
import time
import pandas as pd
from .camera_geometry import detection_to_bearing


def detection_rows(detections, intrinsics, frame_index=0):
    rows=[]
    for d in detections:
        bearing=detection_to_bearing(d,intrinsics); x1,y1,x2,y2=d.xyxy
        rows.append(dict(frame=frame_index,timestamp=d.timestamp,detected=True,class_name=d.class_name,confidence=d.confidence,x1=x1,y1=y1,x2=x2,y2=y2,center_x=d.center_px[0],center_y=d.center_px[1],track_id=d.track_id,azimuth_deg=bearing.azimuth_deg,elevation_deg=bearing.elevation_deg,inference_ms=d.inference_ms))
    return rows


def annotate_frame(frame,detections,intrinsics):
    try: import cv2
    except ImportError as exc: raise RuntimeError("OpenCV is required for annotation") from exc
    image=frame.copy()
    for d in detections:
        x1,y1,x2,y2=map(int,d.xyxy); b=detection_to_bearing(d,intrinsics)
        cv2.rectangle(image,(x1,y1),(x2,y2),(40,220,90),2)
        text=f"{d.class_name} {d.confidence:.2f} ID:{d.track_id if d.track_id is not None else '-'} az:{b.azimuth_deg:+.1f} el:{b.elevation_deg:+.1f}"
        cv2.putText(image,text,(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,220,90),2)
    return image


def process_video(source,detector,intrinsics,tracking=False,max_frames=None,retain_frames=True):
    try: import cv2
    except ImportError as exc: raise RuntimeError("OpenCV is required for video inference") from exc
    cap=cv2.VideoCapture(str(source)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.; rows=[]; frames=[]; index=0
    try:
        while cap.isOpened() and (max_frames is None or index<max_frames):
            ok,frame=cap.read()
            if not ok: break
            timestamp=index/fps; detections=detector.track(frame,timestamp) if tracking else detector.detect(frame,timestamp)
            current=detection_rows(detections,intrinsics,index)
            rows.extend(current or [dict(frame=index,timestamp=timestamp,detected=False,class_name=None,confidence=0.,x1=None,y1=None,x2=None,y2=None,center_x=None,center_y=None,track_id=None,azimuth_deg=None,elevation_deg=None,inference_ms=detector.last_inference_ms)])
            annotated=annotate_frame(frame,detections,intrinsics)
            if retain_frames: frames.append(annotated)
            else: frames[:]=[annotated]
            index+=1
    finally:
        cap.release()
    return frames,pd.DataFrame(rows)
