"""Validated simulation API and authenticated bridge to on-demand perception."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import SimulationConfig
from simulation.engine import SimulationEngine
from simulation.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINS = [
    "https://subuktageen-farooqi.github.io", "https://subuk.github.io",
    "http://127.0.0.1:8502", "http://localhost:8502",
]
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scenario: str = "Obstacle Pursuit"
    mode: Literal["Predictive Standoff Pursuit", "GPS-Denied Follow"] = "Predictive Standoff Pursuit"
    duration: float = Field(60, ge=5, le=120)
    width: float = Field(100, ge=60, le=180)
    height: float = Field(70, ge=45, le=140)
    target_speed: float = Field(1.46, ge=.5, le=4)
    detection_probability: float = Field(.92, ge=.1, le=1)
    sensor_noise: float = Field(1, ge=.1, le=3)
    desired_standoff: float = Field(8.5, ge=5, le=20)
    prediction_horizon: float = Field(2, ge=0, le=5)
    uav_max_speed: float = Field(7, ge=2, le=12)
    seed: int = Field(7, ge=0, le=999999)
    randomize_obstacles: bool = True
    obstacle_seed: int = Field(21, ge=0, le=999999)
    obstacle_count: int = Field(6, ge=0, le=14)
    obstacle_min_size: float = Field(5, ge=3, le=25)
    obstacle_max_size: float = Field(14, ge=3, le=25)
    obstacle_max_height: float = Field(18, ge=5, le=35)
    uav_altitude: float = Field(12, ge=3, le=30)
    target_altitude: float = Field(2, ge=.5, le=20)

    @model_validator(mode="after")
    def validate_environment(self):
        if self.scenario not in SCENARIOS:
            raise ValueError("Unknown simulation scenario")
        if self.obstacle_min_size > self.obstacle_max_size:
            raise ValueError("Minimum obstacle footprint exceeds maximum")
        return self


def model_catalog(root):
    result = {}
    for folder in (root / "models", root / "artifacts" / "models"):
        if folder.exists():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".engine"}:
                    resolved = path.resolve()
                    if resolved.is_relative_to(folder.resolve()):
                        result[path.relative_to(root).as_posix()] = resolved
    return result


def execute_experiment(config):
    engine = SimulationEngine(config)
    engine.run()
    return {
        "run_id": str(uuid4()),
        "source": "computed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "metrics": engine.get_metrics(),
        "samples": json.loads(engine.get_dataframe().to_json(orient="records")),
        "obstacles": engine.visual_obstacles,
        "path": [[float(value) for value in point] for point in engine.path],
    }


def create_api(*, root=ROOT, inference=None, commit=None, access_code=None, origins=None):
    root = Path(root)
    application = FastAPI(title="UAV Simulation Lab API", docs_url=None, redoc_url=None, openapi_url=None)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        # Raw invalid values may include Infinity or non-serializable contexts.
        # Keep the error response finite and avoid reflecting submitted media.
        detail = [{key: item[key] for key in ("loc", "msg", "type")} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    application.add_middleware(
        CORSMiddleware, allow_origins=origins or DEFAULT_ORIGINS,
        allow_methods=["GET", "POST"], allow_headers=["X-Lab-Key", "Content-Type"],
        max_age=3600,
    )

    def authorize(x_lab_key: Annotated[str | None, Header()] = None):
        expected = access_code if access_code is not None else os.environ.get("LAB_ACCESS_CODE", "")
        if not expected:
            raise HTTPException(503, "Lab access has not been configured")
        if not x_lab_key or not hmac.compare_digest(x_lab_key.encode(), expected.encode()):
            raise HTTPException(401, "Enter a valid lab access code")

    protected = [Depends(authorize)]

    def save_run(result):
        folder = root / "artifacts" / "simulations"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{result['run_id']}.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        if commit:
            commit()
        return result

    @application.get("/api/health")
    def health():
        return {"status": "ok", "service": "uav-sim-lab", "configured_gpu": "L40S", "configured_gpu_vram_gb": 48, "simulation": "CPU", "perception": "on demand"}

    @application.get("/api/runtime", dependencies=protected)
    def runtime():
        import platform
        return {**health(), "python": platform.python_version(), "scenarios": list(SCENARIOS), "model_count": len(model_catalog(root))}

    @application.post("/api/simulate", dependencies=protected)
    def simulate(body: ExperimentRequest):
        return save_run(execute_experiment(SimulationConfig(**body.model_dump())))

    @application.post("/api/compare", dependencies=protected)
    def compare(body: ExperimentRequest):
        config = SimulationConfig(**body.model_dump())
        runs = [save_run(execute_experiment(replace(config, mode=mode))) for mode in ("Predictive Standoff Pursuit", "GPS-Denied Follow")]
        return {"runs": runs, "controlled": True}

    @application.get("/api/runs", dependencies=protected)
    def runs():
        folder = root / "artifacts" / "simulations"
        paths = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30] if folder.exists() else []
        result = []
        for path in paths:
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                result.append({key: run[key] for key in ("run_id", "created_at", "config", "metrics")})
            except (OSError, ValueError, KeyError):
                continue
        return result

    @application.get("/api/runs/{run_id}", dependencies=protected)
    def saved_run(run_id: UUID):
        path = root / "artifacts" / "simulations" / f"{run_id}.json"
        if not path.exists():
            raise HTTPException(404, "Experiment not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @application.get("/api/models", dependencies=protected)
    def models():
        return [{"id": key, "name": path.name, "format": path.suffix[1:].upper(), "size_bytes": path.stat().st_size} for key, path in model_catalog(root).items()]

    @application.get("/api/edge-deployment", dependencies=protected)
    def edge_deployment():
        catalog = model_catalog(root)
        artifacts = [{"format": suffix[1:].upper(), "count": sum(path.suffix.lower() == suffix for path in catalog.values())} for suffix in (".pt", ".onnx", ".engine")]
        folder = root / "artifacts" / "edge"
        path = folder / "jetson_orin_nx" / "benchmark.json"
        benchmark = None
        if path.is_file() and path.resolve().is_relative_to(folder.resolve()):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                # Reports are stored data, not trusted JSON response objects.
                # Reject non-finite metrics and malformed content without failing inventory.
                if isinstance(report, dict):
                    json.dumps(report, allow_nan=False)
                    benchmark = report
            except (OSError, ValueError):
                pass
        return {"artifacts": artifacts, "benchmark_target": "jetson-orin-nx", "benchmark": benchmark}

    @application.get("/api/evaluations", dependencies=protected)
    def evaluations():
        folder = root / "artifacts" / "models"
        results = []
        if folder.exists():
            for path in folder.rglob("metrics.json"):
                if not path.resolve().is_relative_to(folder.resolve()):
                    continue
                try:
                    metrics = json.loads(path.read_text(encoding="utf-8"))
                    results.append({"name": path.parent.relative_to(folder).as_posix(), "metrics": metrics})
                except (OSError, ValueError):
                    continue
        return results

    @application.post("/api/perception", dependencies=protected)
    async def perception(
        file: UploadFile = File(...), model_id: str = Form(...),
        confidence: float = Form(.25, ge=.05, le=.95), iou: float = Form(.55, ge=.1, le=.9),
        resolution: int = Form(640, ge=320, le=960), tracking: bool = Form(True),
        fx: float = Form(640, gt=0, le=100000), fy: float = Form(640, gt=0, le=100000),
        cx: float = Form(320, ge=0, le=20000), cy: float = Form(320, ge=0, le=20000),
    ):
        if model_id not in model_catalog(root):
            raise HTTPException(400, "Select a registered model")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".avi"}:
            raise HTTPException(400, "Unsupported image or video format")
        if resolution not in {320, 480, 640, 960}:
            raise HTTPException(400, "Unsupported input resolution")
        if not inference:
            raise HTTPException(503, "The perception worker is unavailable")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload exceeds the 200 MB limit")
        if not data:
            raise HTTPException(400, "The uploaded file is empty")
        try:
            return await inference(data, suffix, model_id, confidence, iou, resolution, tracking, [fx, fy, cx, cy])
        except (ValueError, OSError) as exc:
            raise HTTPException(400, "The media could not be processed") from exc

    return application


app = create_api()
