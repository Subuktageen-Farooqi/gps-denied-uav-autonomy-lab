"""Deploy the lab API and on-demand NVIDIA L40S perception worker.

    python -m modal deploy deployment/modal_app.py

The user interface is hosted on GitHub Pages. CPU requests run independently
from the GPU worker; no GPU is allocated for opening or simulating in the UI.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("vector-uav-autonomy-lab")
artifacts = modal.Volume.from_name("vector-uav-artifacts", create_if_missing=True)
models = modal.Volume.from_name("vector-uav-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .pip_install_from_requirements(str(ROOT / "requirements-cloud.txt"))
    .pip_install_from_requirements(str(ROOT / "requirements-api.txt"))
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "LAB_REQUIRE_AUTH": "1", "PYTHONPATH": "/opt/lab"})
    .add_local_file(ROOT / "app.py", "/opt/lab/app.py")
    .add_local_file(ROOT / "config.py", "/opt/lab/config.py")
    .add_local_dir(ROOT / ".streamlit", "/opt/lab/.streamlit", ignore=["secrets.toml"])
    .add_local_dir(ROOT / "models", "/opt/bundled-models", ignore=["__pycache__", "*.pyc"])
)
for folder in ("autonomy", "estimation", "evaluation", "perception", "planning", "simulation", "training", "ui", "server", "deployment/jetson_orin_nx"):
    image = image.add_local_dir(ROOT / folder, f"/opt/lab/{folder}", ignore=["__pycache__", "*.pyc"])


def seed_models():
    import shutil
    for source in Path("/opt/bundled-models").rglob("*"):
        if source.is_file():
            target = Path("/opt/lab/models") / source.relative_to("/opt/bundled-models")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    models.commit()


@app.cls(
    image=image, gpu="L40S", cpu=2, memory=8192, timeout=600,
    min_containers=0, max_containers=1, scaledown_window=60,
    volumes={"/opt/lab/models": models, "/opt/lab/artifacts": artifacts},
)
@modal.concurrent(max_inputs=1)
class PerceptionWorker:
    @modal.enter()
    def load(self):
        seed_models()
        self.detectors = {}

    @modal.method()
    def analyze(self, data, suffix, model_id, confidence, iou, resolution, tracking, calibration):
        from server.perception import analyze_media
        return analyze_media(Path("/opt/lab"), self.detectors, data, suffix, model_id, confidence, iou, resolution, tracking, calibration)

    @modal.method()
    def check(self):
        import io
        import torch
        from PIL import Image
        from server.api import model_catalog
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        available = model_catalog(Path("/opt/lab"))
        key = next((k for k in available if k.endswith(".pt")), None)
        if not key:
            raise RuntimeError("No bundled PyTorch detector found")
        data = io.BytesIO()
        Image.new("RGB", (640, 640)).save(data, format="PNG")
        result = self.analyze.local(data.getvalue(), ".png", key, .25, .55, 640, False, [640, 640, 320, 320])
        return {"gpu": result["gpu"], "cuda": torch.version.cuda, "model": key, "inference_ms": result["latency_p50_ms"], "detections": len(result["rows"])}


@app.function(
    image=image, cpu=2, memory=4096, timeout=600,
    min_containers=0, max_containers=2, scaledown_window=60,
    volumes={"/opt/lab/models": models, "/opt/lab/artifacts": artifacts},
    secrets=[modal.Secret.from_name("vector-lab-access", required_keys=["LAB_ACCESS_CODE"])],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def api():
    from server.api import create_api
    seed_models()

    async def inference(*args):
        return await PerceptionWorker().analyze.remote.aio(*args)

    return create_api(root=Path("/opt/lab"), inference=inference, commit=artifacts.commit)


@app.local_entrypoint()
def check_gpu():
    print(PerceptionWorker().check.remote())
