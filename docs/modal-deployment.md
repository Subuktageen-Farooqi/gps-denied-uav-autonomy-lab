# GitHub Pages interface and L40S compute

The application is available at **https://subuktageen-farooqi.github.io/uav-sim-lab/**. The GitHub Pages source is maintained in [Subuktageen-Farooqi.github.io](https://github.com/Subuktageen-Farooqi/Subuktageen-Farooqi.github.io).

Only the `/uav-sim-lab/` page is published. Existing personal front-page files are never changed, and no front page is designed. GitHub's default Pages hostname follows the owner account: `subuk.github.io` requires ownership of the `subuk` GitHub account. The connected account is `Subuktageen-Farooqi`.

## Application layout

| Component | Hosting | Workload |
| --- | --- | --- |
| Interface | GitHub Pages | Static HTML/CSS/JavaScript, local fonts, Plotly, reference replay |
| API | Modal CPU containers | Authentication, validation, simulation, comparisons, saved runs, model metadata |
| Perception | Modal NVIDIA L40S, 48 GB | Image/video YOLO inference, optional ByteTrack, calibrated bearing |
| Model storage | `vector-uav-models` Modal volume | Persisted trusted detector models |
| Artifact storage | `vector-uav-artifacts` Modal volume | Saved experiments and model evaluation artifacts |
| Access | `vector-lab-access` Modal secret | `LAB_ACCESS_CODE` shared lab code |

There is no iframe, background video, or application UI hosted by Modal. The previous GPU Streamlit web endpoint has been replaced with a CPU API and an independently allocated L40S worker.

## Use the lab

The bundled, reproducible reference experiment can be replayed without cloud access. **Connect compute** accepts the access code stored in this checkout's ignored `.env.modal` file. The code is kept only in the browser tab's session storage, never in the public site source or URL, and is sent in an `X-Lab-Key` header to the lab API. Disconnect removes it from that tab.

Use **Run simulation** for a new seeded experiment, **Mode comparison** for both controllers under identical conditions, and **Run history** to reopen saved cloud experiments. JSON exports include configuration, metrics, telemetry samples, and obstacle geometry; CSV exports contain the telemetry table.

Open **Edge Deployment** in the navigation bar to choose a hardware target. The selector lists Jetson Nano, Orin, Orin NX, AGX Orin Industrial, Raspberry Pi 5, CM4/CM5, Qualcomm Dragonwing QRB5156 and QC S8550, Hailo-8 / NPU modules, Luxonis OAK, Google Coral, and AMD Xilinx Versal AI Core VC1902. Additional hardware options are selection only. Connect compute to inspect stored PyTorch, ONNX, and TensorRT artifact counts and the existing Orin NX benchmark report. The existing command-line export, build, and benchmark tools remain available in this workspace.

Perception accepts an image or video and a registered server-side model. It starts L40S compute only when media is submitted. Video runs are capped at 300 frames and retain just the final preview frame. Monocular detections provide bearing, not metric range. Latency includes initialization/warmup where applicable; it is not presented as a throughput benchmark. Model accuracy is shown only when actual labeled evaluation artifacts exist.

## Local setup

Use Python 3.11 or newer for a fresh workstation. Python 3.10 works with the pinned Modal 1.6 SDK but is deprecated by Modal.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m http.server 8502 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8502/uav-sim-lab/`. The local interface uses the configured deployed API. Browser CORS allows localhost on port 8502 and the two intended GitHub Pages origins.

To run a local CPU API for development, set `LAB_ACCESS_CODE`, run `python -m uvicorn server.api:app --host 127.0.0.1 --port 8503`, and point `site/uav-sim-lab/config.js` at `http://127.0.0.1:8503`. GPU perception requires the Modal worker; the local API intentionally reports it unavailable when no inference callback is supplied.

## Publish the interface

The GitHub connector did not expose repository creation or Pages configuration in this environment, so the publisher uses GitHub's REST API and Git with the existing Git Credential Manager login. Credentials stay in process memory and are not printed or written into the repository.

```powershell
python deployment/publish_pages.py --publish
```

The publisher checks the authenticated GitHub login, creates `<login>.github.io` if needed, and updates only `uav-sim-lab/` and `.nojekyll`. It preserves any root front page and other projects. It enables a branch-based Pages build if none exists; it preserves incompatible existing Pages settings rather than changing them automatically. No Node build step or third-party runtime CDN is required.

Read build status without publishing:

```powershell
python deployment/publish_pages.py
```

## Deploy the compute API

Authenticate with the intended Modal workspace, `subuk657`:

```powershell
python -m modal token new
```

The existing `vector-lab-access` secret must contain `LAB_ACCESS_CODE`. Never commit the CLI token configuration or `.env.modal` file.

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
python -m modal deploy deployment/modal_app.py
```

API base: `https://subuk657--vector-uav-autonomy-lab-api.us-east.modal.run`

Deployment dashboard: <https://modal.com/apps/subuk657/main/deployed/vector-uav-autonomy-lab>

The API has two CPU cores, 4 GiB RAM, zero warm minimum, and up to two containers. Perception uses one L40S container with two CPU cores and 8 GiB system RAM, one inference at a time, and a 60-second idle scale-down window. Opening the interface and running simulations do not hold a GPU allocation. The public health route reports configured compute; it does not claim that a GPU is already allocated.

The image uses Python 3.11, CUDA-enabled PyTorch 2.6, CUDA 12.4, and the existing detector weights. Only trusted registered models may be selected; arbitrary filesystem paths and uploaded model pickles are not accepted by the API. CORS origins are explicitly listed in [server/api.py](../server/api.py).

Explicitly validate L40S and a synthetic-image inference:

```powershell
python -m modal run deployment/modal_app.py::check_gpu
```

This allocates one L40S briefly and is not a detector accuracy evaluation.

## Persistent storage

New simulations are written to `artifacts/simulations/<run-id>.json` and committed to the artifact volume. Model weights are seeded only when a file does not already exist; redeploying preserves uploaded replacements. To update a model explicitly:

```powershell
python -m modal volume put vector-uav-models models/drone-yolov8n.pt /drone-yolov8n.pt
```

Restart active compute containers after external volume updates so they see the new volume snapshot. Upload labeled evaluation artifacts under `artifacts/models/<run>/metrics.json` in the artifact volume. Do not confuse simulation telemetry metrics with detector mAP/precision/recall.

## Checks

```powershell
python -m pytest -q
python tests/browser_smoke.py
python tests/browser_smoke.py --live --url https://subuktageen-farooqi.github.io/uav-sim-lab/
```

The browser smoke test uses installed Chrome in a separate headless session. The explicit `--live` option checks authentication, a simulation, a controlled comparison, saved history, and one synthetic-image L40S inference. It reads the local access code in memory and never outputs it.

## Troubleshooting

- **GitHub Pages returns 404 immediately after publishing:** check Pages build status and wait for the deployment to finish; the root front page may intentionally remain absent.
- **Cannot reach compute:** verify the API base in `config.js`, the deployed API, and the allowed CORS origin.
- **Access denied:** check `LAB_ACCESS_CODE` in the Modal secret and enter the matching local code.
- **First inference takes longer:** L40S, the model, and CUDA are starting from zero.
- **Evaluation is unmeasured:** upload actual labeled evaluation metrics; no illustrative scores are fabricated.
- **Windows deploy output fails with a Unicode error:** use the UTF-8 environment variables shown above.
- **TensorRT incompatibility:** build an engine for its target device/runtime. Desktop or cloud engines are not portable Jetson engines.
