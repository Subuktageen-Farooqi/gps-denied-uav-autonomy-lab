import argparse, hashlib, json, subprocess
from pathlib import Path
from .dataset_audit import dataset_hash, _load_yaml


def _version(module):
    try: return __import__(module).__version__
    except Exception: return None
def _git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def make_manifest(weights,data_yaml,base_model,imgsz=640,seed=7,metrics=None,output=None):
    weights=Path(weights).resolve(); data_yaml=Path(data_yaml).resolve(); cfg=_load_yaml(data_yaml); root=Path(cfg.get("path",data_yaml.parent))
    if not root.is_absolute():
        cwd=(Path.cwd()/root).resolve(); root=cwd if cwd.exists() else (data_yaml.parent/root).resolve()
    try:
        import torch; torch_version=torch.__version__; cuda_version=torch.version.cuda
    except Exception: torch_version=cuda_version=None
    names=cfg.get("names",{0:"uav"}); classes=list(names.values()) if isinstance(names,dict) else list(names)
    manifest={"model_family":Path(base_model).stem,"base_weights":str(base_model),"classes":classes,"imgsz":imgsz,"dataset_yaml":str(data_yaml),"dataset_hash":dataset_hash(root),"git_commit":_git_commit(),"ultralytics_version":_version("ultralytics"),"torch_version":torch_version,"cuda_version":cuda_version,"training_seed":seed,"best_metric":metrics or {},"weights_sha256":sha256(weights)}
    destination=Path(output) if output else weights.parent/"model_manifest.json"; destination.write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest


def main():
    p=argparse.ArgumentParser(description="Create reproducibility metadata for trained weights"); p.add_argument("--weights",required=True); p.add_argument("--data",required=True); p.add_argument("--base-model",required=True); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--seed",type=int,default=7); p.add_argument("--output"); a=p.parse_args(); print(json.dumps(make_manifest(a.weights,a.data,a.base_model,a.imgsz,a.seed,output=a.output),indent=2))
if __name__=="__main__": main()
