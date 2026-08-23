import json
import argparse
from pathlib import Path
import numpy as np
from .simulated_detector import SimulatedDetector, Detection


REQUIRED={"overall_detection_rate","confidence_mean","dropout_lengths_frames"}


def load_profile(path):
    data=json.loads(Path(path).read_text(encoding="utf-8")); missing=REQUIRED-data.keys()
    if missing: raise ValueError(f"Detector profile missing: {sorted(missing)}")
    rate=float(data["overall_detection_rate"])
    if not 0<=rate<=1: raise ValueError("overall_detection_rate must be in [0,1]")
    return data


def build_profile(frame_rows,model="unknown",output_path=None):
    if "frame" in frame_rows.columns:
        frame_rows=frame_rows.groupby("frame",as_index=False).agg(detected=("detected","max"),confidence=("confidence","max"))
    detected=np.asarray(frame_rows["detected"],bool); conf=np.asarray(frame_rows.get("confidence",np.zeros(len(detected))),float)
    runs=[]; run=0
    for found in detected:
        if not found: run+=1
        elif run: runs.append(run); run=0
    if run: runs.append(run)
    profile={"model":model,"overall_detection_rate":float(detected.mean()) if len(detected) else 0.,"confidence_mean":float(conf[detected].mean()) if detected.any() else 0.,"confidence_std":float(conf[detected].std()) if detected.any() else 0.,"dropout_lengths_frames":runs,"bbox_center_jitter_px":None,"size_buckets":{}}
    if output_path: Path(output_path).write_text(json.dumps(profile,indent=2),encoding="utf-8")
    return profile


class ProfiledSimulatedDetector(SimulatedDetector):
    def __init__(self,profile_path,noise_std,occlusions,rng,dt=.1):
        self.profile=load_profile(profile_path); self.dt=dt; self.dropout_remaining=0
        super().__init__(self.profile["overall_detection_rate"],noise_std,occlusions,rng)
    def detect(self,target_position,uav_position,t):
        if self.dropout_remaining>0: self.dropout_remaining-=1; return None
        result=super().detect(target_position,uav_position,t)
        if result is None:
            lengths=self.profile.get("dropout_lengths_frames") or [1]; self.dropout_remaining=max(0,int(self.rng.choice(lengths))-1); return None
        mean=float(self.profile.get("confidence_mean",result.confidence)); std=float(self.profile.get("confidence_std",.08))
        return Detection(result.relative,float(np.clip(self.rng.normal(mean,std),0,1)),t)


def main():
    import pandas as pd
    parser=argparse.ArgumentParser(description="Build an empirical simulation reliability profile from frame-level YOLO results")
    parser.add_argument("--csv",required=True,help="CSV with one row per frame and detected/confidence columns"); parser.add_argument("--model",required=True); parser.add_argument("--output",default="artifacts/perception_profile.json"); args=parser.parse_args()
    print(json.dumps(build_profile(pd.read_csv(args.csv),args.model,args.output),indent=2))
if __name__=="__main__": main()
