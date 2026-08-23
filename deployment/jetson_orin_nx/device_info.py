import json, os, platform, subprocess
from pathlib import Path


def _read(path):
    try: return Path(path).read_text(encoding="utf-8").strip()
    except Exception: return None
def _command(command):
    try: return subprocess.check_output(command,text=True,stderr=subprocess.DEVNULL,timeout=3).strip()
    except Exception: return None


def collect_device_info():
    try:
        import tensorrt; trt=tensorrt.__version__
    except Exception: trt=None
    try:
        import torch; cuda=torch.version.cuda
    except Exception: cuda=None
    model=_read("/proc/device-tree/model")
    return {"device_model":model or platform.node(),"is_jetson":bool(model and "jetson" in model.lower()),"architecture":platform.machine(),"platform":platform.platform(),"l4t_release":_read("/etc/nv_tegra_release"),"jetpack":None,"cuda":cuda,"tensorrt":trt,"power_mode":_command(["nvpmodel","-q"]),"tegrastats_available":bool(_command(["sh","-c","command -v tegrastats"]))}


def main(): print(json.dumps(collect_device_info(),indent=2))
if __name__=="__main__": main()

