"""GPS-denied autonomy research console. Run with streamlit run app.py."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import SimulationConfig
from simulation.engine import SimulationEngine
from simulation.scenarios import SCENARIOS
from ui.scene import pursuit_3d_figure
from ui.theme import apply_theme, section_heading
from ui.access import require_access
from ui.vision_sections import (
    render_edge_deployment, render_model_evaluation,
    render_perception_lab, render_vision_architecture,
)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Vector | UAV Autonomy Lab", page_icon="✦", layout="wide")
apply_theme()
require_access()

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">⌖</span><div>VECTOR<span>AUTONOMY RESEARCH LAB</span></div></div>', unsafe_allow_html=True)
    st.caption("WORKSPACE")
    workspace = st.radio("Workspace", ["Mission Control", "Perception Lab", "Model Evaluation", "Deployment", "Architecture"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="sidebar-note">GPS-DENIED ENVIRONMENT<br><strong>Simulation & validation</strong><p>Relative sensing. Estimated state. Safe-standoff observation.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="topline"><span>VECTOR / AUTONOMY LAB</span><span>SIMULATION ENVIRONMENT <b>●</b></span></div>', unsafe_allow_html=True)
title, status = st.columns([4, 1])
with title:
    st.title(workspace)
    st.caption("Configure. Simulate. Inspect. Validate." if workspace == "Mission Control" else "GPS-denied UAV research · perception, estimation & navigation")
with status:
    st.markdown('<div class="environment-tag">SIMULATION ONLY<span>2D kinematics · 3D visualization</span></div>', unsafe_allow_html=True)

if workspace != "Mission Control":
    if workspace == "Perception Lab":
        render_perception_lab()
    elif workspace == "Model Evaluation":
        render_model_evaluation()
    elif workspace == "Architecture":
        render_vision_architecture()
    else:
        from ui.runtime import render_runtime
        render_runtime()
        with st.expander("Jetson Orin NX deployment"):
            render_edge_deployment()
    st.stop()

with st.sidebar:
    with st.form("mission_configuration"):
        st.subheader("Mission configuration")
        scenario = st.selectbox("Scenario", list(SCENARIOS), index=2)
        mode = st.selectbox("Operating mode", ["Predictive Standoff Pursuit", "GPS-Denied Follow"])
        with st.expander("Sensor & autonomy", expanded=True):
            target_speed = st.slider("Target speed (m/s)", .5, 4.0, 1.8, .1)
            probability = st.slider("Detection probability", .1, 1.0, .88, .01)
            noise = st.slider("Sensor noise σ (m)", .1, 3.0, .8, .1)
            standoff = st.slider("Standoff distance (m)", 5.0, 20.0, 8.5, .5)
            horizon = st.slider("Prediction horizon (s)", 0., 5.0, 2.0, .25)
            max_speed = st.slider("UAV speed limit (m/s)", 2.0, 12.0, 7.0, .5)
            duration = st.select_slider("Duration (s)", [30, 60, 90, 120], value=60)
            seed = st.number_input("Simulation seed", 0, 999999, 7)
        with st.expander("Environment"):
            field_width = st.slider("Field width (m)", 60, 180, 100, 10)
            field_height = st.slider("Field depth (m)", 45, 140, 70, 5)
            random_obstacles = st.checkbox("Randomize obstacles", True)
            obstacle_seed = st.number_input("Obstacle seed", 0, 999999, 21)
            obstacle_count = st.slider("Obstacle count", 0, 14, 6)
            obstacle_size = st.slider("Obstacle footprint (m)", 3., 25., (5., 14.), 1.)
            obstacle_height = st.slider("Maximum obstacle height (m)", 5., 35., 18., 1.)
            uav_altitude = st.slider("UAV display altitude (m)", 3., 30., 12., 1.)
            target_altitude = st.slider("Target display altitude (m)", .5, 20., 2., .5)
        with st.expander("Detector profile"):
            profiles = sorted((ROOT / "artifacts").rglob("*perception_profile.json")) if (ROOT / "artifacts").exists() else []
            profile = st.selectbox("Detection source", [None, *profiles], format_func=lambda p: "Synthetic sensor" if p is None else str(p.relative_to(ROOT)))
            st.caption("Empirical profiles model detector reliability; image pixels are never used as metric positions.")
        run = st.form_submit_button("Run simulation", type="primary", width="stretch")
    st.caption("Settings are applied when you run a simulation. Seeds make each run reproducible.")

cfg = SimulationConfig(
    scenario=scenario, mode=mode, duration=float(duration),
    width=float(field_width), height=float(field_height), target_speed=target_speed,
    detection_probability=probability, sensor_noise=noise, desired_standoff=standoff,
    prediction_horizon=horizon, uav_max_speed=max_speed, seed=int(seed),
    randomize_obstacles=random_obstacles, obstacle_seed=int(obstacle_seed),
    obstacle_count=obstacle_count, obstacle_min_size=obstacle_size[0],
    obstacle_max_size=obstacle_size[1], obstacle_max_height=obstacle_height,
    uav_altitude=uav_altitude, target_altitude=target_altitude,
    perception_backend="profiled_simulated" if profile else "simulated",
    detector_profile_path=str(profile) if profile else None,
)

if run or "engine" not in st.session_state:
    try:
        with st.spinner("Executing deterministic scenario…"):
            candidate = SimulationEngine(cfg)
            candidate.run()
            candidate.get_metrics()
        if "engine" in st.session_state:
            st.session_state.previous_run = st.session_state.engine
        st.session_state.engine = candidate
        st.session_state.run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        st.session_state.pop("replay_cursor", None)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Unable to run this configuration: {exc}")
        if "engine" not in st.session_state:
            st.stop()

engine = st.session_state.engine
df = engine.get_dataframe()
metrics = engine.get_metrics()
active = engine.config
section_heading("ACTIVE EXPERIMENT", f"{active.scenario} · Seed {active.seed} · {active.duration:.0f}s · {active.mode}")
cards = st.columns(4)
cards[0].metric("ESTIMATION RMSE", f"{metrics['estimation_rmse']:.2f} m", help="Position error against simulator ground truth over the full run.")
cards[1].metric("DETECTION AVAILABILITY", f"{metrics['detection_availability']:.1f}%")
cards[2].metric("TIME IN STANDOFF BAND", f"{metrics['percent_time_inside_standoff_band']:.1f}%", help="Time within ±1.5 m of the configured standoff distance.")
cards[3].metric("UAV TRAVEL", f"{metrics['total_uav_travel_distance']:.1f} m")
st.write("")

scene, telemetry = st.columns([3.2, 1], gap="medium")
with scene:
    with st.container(border=True):
        section_heading("SPATIAL REPLAY", "Relative tracking / local reference frame")
        cursor = st.slider("Telemetry snapshot", 0, len(df)-1, len(df)-1, key="replay_cursor", format="sample %d", label_visibility="collapsed")
        with st.expander("Display layers"):
            layers = st.columns(4)
            truth = layers[0].checkbox("Ground truth", True)
            detections = layers[1].checkbox("Detections", True)
            estimate = layers[2].checkbox("Kalman estimate", True)
            prediction = layers[3].checkbox("Prediction", True)
            planned = layers[0].checkbox("Final A* path", True)
            trails = layers[1].checkbox("Trails", True)
            radius = layers[2].checkbox("Standoff radius", True)
        st.plotly_chart(pursuit_3d_figure(engine, df, truth, detections, estimate, prediction, planned, trails, radius, cursor), width="stretch", config={"displaylogo": False, "scrollZoom": True}, key="spatial_replay")
        st.caption("Drag to orbit · scroll to zoom · Play replay animates the scene. The snapshot slider controls telemetry. Altitude is visual only.")

row = df.iloc[cursor]
with telemetry:
    with st.container(border=True):
        section_heading("VEHICLE TELEMETRY", f"Snapshot T+{row.time:05.1f} s")
        st.metric("AUTONOMY STATE", row.fsm_state)
        st.metric("SENSOR CONTACT", "DETECTED" if row.detection_valid else "DROPOUT")
        a, b = st.columns(2)
        a.metric("Confidence", f"{row.confidence:.2f}")
        b.metric("Range", f"{row.range_to_target:.1f} m")
        a, b = st.columns(2)
        a.metric("Bearing", f"{row.bearing_deg:.1f}°")
        b.metric("Speed", f"{row.uav_speed:.1f} m/s")
        st.divider()
        st.caption("ESTIMATED TARGET VELOCITY")
        st.write(f"vx {row.estimated_vx:+.2f} / vy {row.estimated_vy:+.2f} m/s")
        st.caption(f"Uncertainty {row.uncertainty:.2f} · Path nodes {int(row.path_nodes)}")

section_heading("FLIGHT ANALYSIS", "Measured simulation outputs")
timeline, tracking, navigation, comparison, runlog = st.tabs(["State timeline", "Tracking quality", "Navigation", "Compare runs", "Run data & export"])


def style_chart(fig, height=280):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="#101820", plot_bgcolor="#101820", margin=dict(l=15, r=15, t=20, b=20), font=dict(color="#aebfc9"), legend=dict(orientation="h", y=1.12), xaxis=dict(title="Simulation time (s)", gridcolor="#23323b"), yaxis=dict(gridcolor="#23323b"))
    return fig


with timeline:
    states = ["SEARCH", "ACQUIRE", "TRACK", "PREDICT", "REACQUIRE"]
    fig = go.Figure(go.Scatter(x=df.time, y=[states.index(s) for s in df.fsm_state], mode="lines", line=dict(color="#a6c887", width=2, shape="hv"), name="Autonomy state"))
    lost = df.loc[~df.detection_valid]
    fig.add_trace(go.Scatter(x=lost.time, y=[-.5]*len(lost), mode="markers", marker=dict(color="#f4aa60", size=5, symbol="line-ns"), name="Sensor dropout"))
    style_chart(fig)
    fig.update_yaxes(tickvals=list(range(5)), ticktext=states)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
with tracking:
    a, b = st.columns(2)
    with a:
        st.caption("POSITION ERROR & UNCERTAINTY")
        fig = go.Figure()
        for col, label, color in [("tracking_error", "Position error (m)", "#63c9ed"), ("uncertainty", "Filter uncertainty", "#a6c887")]:
            fig.add_trace(go.Scatter(x=df.time, y=df[col], name=label, line=dict(color=color)))
        st.plotly_chart(style_chart(fig), width="stretch")
    with b:
        st.caption("DETECTION CONFIDENCE")
        fig = go.Figure(go.Scatter(x=df.time, y=df.confidence, fill="tozeroy", line=dict(color="#f4aa60"), name="Confidence"))
        st.plotly_chart(style_chart(fig), width="stretch")
with navigation:
    fig = go.Figure(go.Scatter(x=df.time, y=df.range_to_target, line=dict(color="#63c9ed"), name="Estimated range (m)"))
    fig.add_hrect(y0=active.desired_standoff-1.5, y1=active.desired_standoff+1.5, fillcolor="#a6c887", opacity=.12, line_width=0)
    fig.add_hline(y=active.desired_standoff, line_dash="dash", line_color="#a6c887", annotation_text="Desired standoff")
    st.plotly_chart(style_chart(fig), width="stretch")
    st.caption("Range uses the Kalman estimate. Shaded band: desired standoff ±1.5 m.")
with comparison:
    st.write("Compare operating modes under identical sensor, environment, and seed settings.")
    if st.button("Run controlled mode comparison"):
        alternative = "GPS-Denied Follow" if active.mode == "Predictive Standoff Pursuit" else "Predictive Standoff Pursuit"
        with st.spinner("Running comparison with identical seeds…"):
            other = SimulationEngine(replace(active, mode=alternative))
            other.run()
        st.session_state.comparison = {"config": asdict(active), "mode": alternative, "metrics": other.get_metrics()}
    controlled = st.session_state.get("comparison")
    if controlled and controlled["config"] == asdict(active):
        st.dataframe(pd.DataFrame({active.mode: metrics, controlled["mode"]: controlled["metrics"]}), width="stretch")
        st.caption("Both runs use identical parameters and random seeds; only the operating mode changes.")
    elif "previous_run" in st.session_state:
        previous = st.session_state.previous_run
        st.dataframe(pd.DataFrame({"Current run": metrics, "Previous run": previous.get_metrics()}), width="stretch")
        st.caption(f"Previous: {previous.config.scenario}, seed {previous.config.seed}, {previous.config.mode}. Settings may differ.")
    else:
        st.info("Run a controlled comparison, or change a parameter and run another experiment to compare results.")
with runlog:
    payload = {"created_at_utc": st.session_state.run_timestamp, "config": asdict(active), "metrics": metrics, "simulation_only": True}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("run_log.csv", df.to_csv(index=False))
        bundle.writestr("experiment.json", json.dumps(payload, indent=2))
    a, b, c = st.columns(3)
    a.download_button("Download experiment (.zip)", buffer.getvalue(), f"vector_{st.session_state.run_timestamp}.zip", "application/zip", width="stretch")
    b.download_button("Export telemetry (.csv)", df.to_csv(index=False), "run_log.csv", "text/csv", width="stretch")
    c.download_button("Export configuration (.json)", json.dumps(payload, indent=2), "experiment.json", "application/json", width="stretch")
    st.dataframe(df, width="stretch", height=340)
    with st.expander("Active configuration & metrics"):
        st.json(payload)

st.markdown('<div class="lab-footer">VECTOR AUTONOMY LAB<span>REPRODUCIBLE EXPERIMENTS · SIMULATED SENSING · SAFE STANDOFF</span></div>', unsafe_allow_html=True)
