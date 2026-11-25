"""Contract and access tests for the Pages-to-compute boundary."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from server.api import create_api

KEY = {"X-Lab-Key": "test-lab-key"}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_api(root=tmp_path, access_code="test-lab-key"))


def test_protected_requests_require_access(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/runtime").status_code == 401
    assert client.get("/api/runtime", headers={"X-Lab-Key": "incorrect"}).status_code == 401
    runtime = client.get("/api/runtime", headers=KEY).json()
    assert runtime["configured_gpu"] == "L40S"
    assert runtime["configured_gpu_vram_gb"] == 48


@pytest.mark.parametrize("payload", [{"duration": 100000}, {"dt": 0}, {"scenario": "invalid"}, {"obstacle_min_size": 20, "obstacle_max_size": 5}, {"sensor_noise": float("inf")}])
def test_compute_inputs_are_bounded(client, payload):
    if payload.get("sensor_noise") == float("inf"):
        response = client.post("/api/simulate", headers={**KEY, "Content-Type": "application/json"}, content='{"sensor_noise":1e999}')
    else:
        response = client.post("/api/simulate", headers=KEY, json=payload)
    assert response.status_code == 422


def test_simulation_is_reproducible_and_saved(client):
    config = {"duration": 5, "seed": 34, "randomize_obstacles": False}
    first = client.post("/api/simulate", headers=KEY, json=config).json()
    second = client.post("/api/simulate", headers=KEY, json=config).json()
    assert len(first["samples"]) == 50
    assert first["samples"] == second["samples"]
    assert first["metrics"] == second["metrics"]
    assert first["run_id"] != second["run_id"]
    saved = client.get(f"/api/runs/{first['run_id']}", headers=KEY).json()
    assert saved == first
    assert len(client.get("/api/runs", headers=KEY).json()) == 2
    assert client.get("/api/runs/../../.env.modal", headers=KEY).status_code in {404, 422}


def test_comparison_changes_only_operating_mode(client):
    response = client.post("/api/compare", headers=KEY, json={"duration": 5, "seed": 17})
    assert response.status_code == 200
    first, second = response.json()["runs"]
    assert first["config"]["mode"] != second["config"]["mode"]
    first["config"].pop("mode")
    second["config"].pop("mode")
    assert first["config"] == second["config"]


def test_model_paths_are_registered_and_symlinks_confined(client):
    response = client.post("/api/perception", headers=KEY, data={"model_id": "../../.env.modal"}, files={"file": ("frame.png", b"untrusted", "image/png")})
    assert response.status_code == 400
    assert client.get("/api/evaluations", headers=KEY).json() == []


def test_edge_inventory_and_saved_benchmark(client, tmp_path):
    assert client.get("/api/edge-deployment").status_code == 401
    for relative in ("models/base.pt", "artifacts/models/custom/best.pt", "artifacts/models/custom/best.onnx", "models/detector.engine"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"inventory fixture")
    path = tmp_path / "artifacts/edge/jetson_orin_nx/benchmark.json"
    path.parent.mkdir(parents=True)
    benchmark = {"p50_ms": 12.5, "p95_ms": 16, "fps": 70, "device": {"device_model": "Jetson Orin NX"}, "power_w": None}
    path.write_text(json.dumps(benchmark), encoding="utf-8")
    result = client.get("/api/edge-deployment", headers=KEY).json()
    assert {item["format"]: item["count"] for item in result["artifacts"]} == {"PT": 2, "ONNX": 1, "ENGINE": 1}
    assert result["benchmark_target"] == "jetson-orin-nx"
    assert result["benchmark"] == benchmark


@pytest.mark.parametrize("content", [None, "invalid json", "[]", '{"p50_ms": NaN}', '{"p50_ms": 1e999}'])
def test_missing_or_invalid_edge_benchmark_keeps_inventory_available(client, tmp_path, content):
    if content is not None:
        path = tmp_path / "artifacts/edge/jetson_orin_nx/benchmark.json"
        path.parent.mkdir(parents=True)
        path.write_text(content, encoding="utf-8")
    response = client.get("/api/edge-deployment", headers=KEY)
    assert response.status_code == 200
    assert response.json()["benchmark"] is None
    assert all(item["count"] == 0 for item in response.json()["artifacts"])


def test_edge_benchmark_symlink_cannot_read_outside_artifacts(client, tmp_path):
    private = tmp_path / "private.json"
    private.write_text('{"private": "must not be returned"}', encoding="utf-8")
    path = tmp_path / "artifacts/edge/jetson_orin_nx/benchmark.json"
    path.parent.mkdir(parents=True)
    try:
        path.symlink_to(private)
    except OSError:
        pytest.skip("Symlinks unavailable on this host")
    assert client.get("/api/edge-deployment", headers=KEY).json()["benchmark"] is None


def test_cors_allows_pages_and_rejects_other_origins(client):
    headers = {"Origin": "https://subuktageen-farooqi.github.io", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-lab-key,content-type"}
    response = client.options("/api/simulate", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == headers["Origin"]
    response = client.options("/api/simulate", headers={**headers, "Origin": "https://unrelated.example"})
    assert response.status_code == 400
