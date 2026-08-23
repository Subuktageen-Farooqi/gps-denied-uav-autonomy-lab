import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config import SimulationConfig
from simulation.engine import SimulationEngine
from simulation.scenarios import SCENARIOS
from evaluation.plots import STATE_MAP
from ui.vision_sections import render_perception_lab,render_model_evaluation,render_edge_deployment,render_vision_architecture


def pursuit_3d_figure(engine, df, truth, detections, estimate, prediction, planned, trails, radius):
    """Build an animated 3D presentation of the 2D kinematic simulation."""
    fig = go.Figure()
    floor_x = [0, engine.config.width, engine.config.width, 0]
    floor_y = [0, 0, engine.config.height, engine.config.height]
    fig.add_trace(go.Mesh3d(x=floor_x, y=floor_y, z=[0]*4, i=[0, 0], j=[1, 2], k=[2, 3],
                            color="#eef3f8", opacity=.8, name="Ground", hoverinfo="skip",showlegend=False))
    obstacle_colors=["#4e79a7","#f28e2b","#59a14f","#e15759","#b07aa1","#76b7b2","#edc948","#ff9da7","#9c755f","#86bcb6","#af7aa1","#bab0ac"]
    for n, obstacle in enumerate(engine.visual_obstacles, 1):
        x0,x1,y0,y1=obstacle["bounds"]; z1=obstacle["height"]
        if obstacle["shape"] == "cylinder":
            angles=np.linspace(0,2*np.pi,20,endpoint=False); cx=(x0+x1)/2; cy=(y0+y1)/2; rx=(x1-x0)/2; ry=(y1-y0)/2
            x=np.r_[cx+rx*np.cos(angles),cx+rx*np.cos(angles),cx,cx]; y=np.r_[cy+ry*np.sin(angles),cy+ry*np.sin(angles),cy,cy]; z=np.r_[np.zeros(20),np.full(20,z1),0,z1]
            i=[]; j=[]; k=[]
            for q in range(20):
                nxt=(q+1)%20; i += [q,q,40,41]; j += [nxt,q+20,q,nxt+20]; k += [q+20,nxt+20,nxt,q+20]
        else:
            x=[x0,x1,x1,x0,x0,x1,x1,x0]; y=[y0,y0,y1,y1,y0,y0,y1,y1]; z=[0,0,0,0,z1,z1,z1,z1]
            i=[0,0,0,1,1,2,4,4,5,6,2,3]; j=[1,2,3,2,5,3,5,7,6,7,6,7]; k=[2,3,4,5,4,7,6,5,7,6,3,4]
        obstacle_color=obstacle_colors[(n-1)%len(obstacle_colors)]
        fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,color=obstacle_color,opacity=1.0,flatshading=True,
                                lighting=dict(ambient=.35,diffuse=.8,specular=.15),
                                name="Obstacles" if n == 1 else None,showlegend=n == 1,
                                hovertemplate=f"{obstacle['shape'].title()} {n}<br>height={z1:.1f} m<extra></extra>"))
    static_count=len(fig.data)
    colors=dict(uav="#42a5f5",target="#ff8c42",estimate="#3ddc97",detect="#ffd166",predict="#ef476f",path="#b388ff")
    dynamic_names=["UAV trail","Target trail","Estimate trail","Detections","UAV","Target","Estimate","Predicted","Standoff","Pursuit vector","A* path"]
    for name in dynamic_names:
        mode="lines" if name in {"UAV trail","Target trail","Estimate trail","Standoff","Pursuit vector","A* path"} else "markers"
        fig.add_trace(go.Scatter3d(x=[],y=[],z=[],mode=mode,name=name))
    trace_ids=list(range(static_count,static_count+len(dynamic_names)))
    indices=np.unique(np.r_[np.arange(0,len(df),5),len(df)-1]).astype(int)
    frames=[]
    for idx in indices:
        part=df.iloc[:idx+1]; row=df.iloc[idx]
        theta=np.linspace(0,2*np.pi,72)
        circle_x=row.estimated_target_x+engine.config.desired_standoff*np.cos(theta)
        circle_y=row.estimated_target_y+engine.config.desired_standoff*np.sin(theta)
        valid=part.dropna(subset=["detection_x","detection_y"]).iloc[::3]
        series=[
            go.Scatter3d(x=part.uav_x if trails else [],y=part.uav_y if trails else [],z=np.full(len(part),engine.config.uav_altitude),mode="lines",line=dict(color=colors["uav"],width=5),name="UAV trail"),
            go.Scatter3d(x=part.true_target_x if trails and truth else [],y=part.true_target_y if trails and truth else [],z=np.full(len(part),engine.config.target_altitude),mode="lines",line=dict(color=colors["target"],width=4),name="Target trail"),
            go.Scatter3d(x=part.estimated_target_x if trails and estimate else [],y=part.estimated_target_y if trails and estimate else [],z=np.full(len(part),engine.config.target_altitude+.4),mode="lines",line=dict(color=colors["estimate"],width=4),name="Estimate trail"),
            go.Scatter3d(x=valid.detection_x if detections else [],y=valid.detection_y if detections else [],z=np.full(len(valid),engine.config.target_altitude+.7),mode="markers",marker=dict(color=colors["detect"],size=2,opacity=.45),name="Detections"),
            go.Scatter3d(x=[row.uav_x],y=[row.uav_y],z=[engine.config.uav_altitude],mode="markers",marker=dict(color=colors["uav"],size=8,symbol="diamond"),name="UAV",hovertemplate=f"UAV<br>t={row.time:.1f}s<br>state={row.fsm_state}<extra></extra>"),
            go.Scatter3d(x=[row.true_target_x] if truth else [],y=[row.true_target_y] if truth else [],z=[engine.config.target_altitude] if truth else [],mode="markers",marker=dict(color=colors["target"],size=7),name="Target"),
            go.Scatter3d(x=[row.estimated_target_x] if estimate else [],y=[row.estimated_target_y] if estimate else [],z=[engine.config.target_altitude+.4] if estimate else [],mode="markers",marker=dict(color=colors["estimate"],size=6),name="Estimate"),
            go.Scatter3d(x=[row.predicted_target_x] if prediction else [],y=[row.predicted_target_y] if prediction else [],z=[engine.config.target_altitude+.6] if prediction else [],mode="markers",marker=dict(color=colors["predict"],size=8,symbol="x"),name="Predicted"),
            go.Scatter3d(x=circle_x if radius else [],y=circle_y if radius else [],z=np.full(len(theta),.25),mode="lines",line=dict(color=colors["estimate"],width=3,dash="dash"),name="Standoff"),
            go.Scatter3d(x=[row.uav_x,row.estimated_target_x],y=[row.uav_y,row.estimated_target_y],z=[engine.config.uav_altitude,engine.config.target_altitude+.4],mode="lines",line=dict(color="#d4d9e2",width=2,dash="dot"),name="Pursuit vector"),
            go.Scatter3d(x=[p[0] for p in engine.path] if planned else [],y=[p[1] for p in engine.path] if planned else [],z=np.full(len(engine.path),.2) if planned else [],mode="lines",line=dict(color=colors["path"],width=5,dash="dash"),name="A* path"),
        ]
        frames.append(go.Frame(data=series,traces=trace_ids,name=str(idx)))
    fig.frames=frames
    if frames:
        for trace_id, trace in zip(trace_ids, frames[-1].data): fig.data[trace_id].update(trace)
    steps=[dict(method="animate",args=[[f.name],dict(mode="immediate",frame=dict(duration=0,redraw=True),transition=dict(duration=0))],label=f"{df.iloc[int(f.name)].time:.0f}s") for f in frames[::max(1,len(frames)//12)]]
    axis_style=dict(backgroundcolor="#f7f9fc",gridcolor="#cbd5e1",zerolinecolor="#94a3b8",color="#334155",showbackground=True)
    fig.update_layout(height=700,margin=dict(l=38,r=38,t=55,b=115),paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",font=dict(color="#1f2937"),
        scene=dict(domain=dict(x=[.035,.965],y=[.20,.93]),
                   xaxis=dict(title="x (m)",range=[0,engine.config.width],**axis_style),
                   yaxis=dict(title="y (m)",range=[0,engine.config.height],**axis_style),
                   zaxis=dict(title="Altitude (m)",range=[0,max(20,engine.config.obstacle_max_height+3,engine.config.uav_altitude+3)],**axis_style),
                   bgcolor="#ffffff",aspectmode="manual",aspectratio=dict(x=1.45,y=1,z=.42),camera=dict(eye=dict(x=1.45,y=-1.65,z=1.15))),
        legend=dict(orientation="h",y=1.03,x=.5,xanchor="center",font=dict(color="#1f2937"),bgcolor="rgba(255,255,255,.85)"),
        updatemenus=[dict(type="buttons",direction="left",x=.5,xanchor="center",y=.105,yanchor="middle",pad=dict(l=10,r=10,t=8,b=8),
                          bgcolor="#ffffff",bordercolor="#94a3b8",font=dict(color="#1f2937"),buttons=[
            dict(label="▶ Play pursuit",method="animate",args=[None,dict(fromcurrent=True,frame=dict(duration=90,redraw=True),transition=dict(duration=0))]),
            dict(label="Ⅱ Pause",method="animate",args=[[None],dict(mode="immediate",frame=dict(duration=0,redraw=False))])])],
        sliders=[dict(active=len(frames)-1,x=.12,len=.76,y=.015,pad=dict(t=10,b=4),currentvalue=dict(prefix="Simulation time: ",font=dict(color="#334155")),font=dict(color="#475569"),steps=steps)])
    return fig

st.set_page_config(page_title="GPS-Denied UAV Autonomy Lab",layout="wide")
st.markdown("""
<style>
.stMetric {
    background: #ffffff !important;
    border: 1px solid #d7dce2 !important;
    padding: 14px !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
}
.stMetric [data-testid="stMetricLabel"],
.stMetric [data-testid="stMetricLabel"] p {
    color: #111111 !important;
}
.stMetric [data-testid="stMetricValue"],
.stMetric [data-testid="stMetricValue"] > div {
    color: #f4512c !important;
}
.stMetric [data-testid="stMetricDelta"] {
    color: #f4512c !important;
}
.badge {display:inline-block;padding:4px 9px;margin-right:6px;border:1px solid #3f8cff;border-radius:12px;font-size:12px}
</style>
""",unsafe_allow_html=True)
st.title("GPS-Denied UAV Autonomy Lab")
st.caption("Autonomous Target Tracking • State Estimation • Predictive Standoff Pursuit")
st.markdown('<span class="badge">SIMULATION</span><span class="badge">GPS DENIED</span><span class="badge">KALMAN TRACKING</span><span class="badge">A* PLANNING</span>',unsafe_allow_html=True)
workspace=st.segmented_control("Workspace",["Autonomy Lab","Perception Lab","Model Evaluation","Edge Deployment","Architecture"],default="Autonomy Lab",label_visibility="collapsed")
if workspace=="Perception Lab": render_perception_lab(); st.stop()
if workspace=="Model Evaluation": render_model_evaluation(); st.stop()
if workspace=="Edge Deployment": render_edge_deployment(); st.stop()
if workspace=="Architecture": render_vision_architecture(); st.stop()

with st.sidebar:
    st.header("Simulation Controls")
    scenario=st.selectbox("Scenario",list(SCENARIOS),index=2)
    mode=st.radio("Operating Mode",["GPS-Denied Follow","Predictive Standoff Pursuit"],index=1)
    base=SCENARIOS[scenario]; base_speed=float(np.linalg.norm(base.target_velocity))
    target_speed=st.slider("Target speed (m/s)",.5,4.0,base_speed,.1)
    probability=st.slider("Detection probability",.5,1.0,float(base.detection_probability),.01)
    noise=st.slider("Sensor noise σ (m)",.1,3.0,float(base.sensor_noise),.1)
    standoff=st.slider("Desired standoff (m)",5.0,20.0,8.5,.5)
    horizon=st.slider("Prediction horizon (s)",.0,5.0,2.0,.25)
    max_speed=st.slider("UAV max speed (m/s)",2.0,12.0,7.0,.5)
    seed=st.number_input("Random seed",0,9999,7)
    st.subheader("Perception Model")
    perception_choice=st.radio("Detector source",["Synthetic detector","YOLO-derived detector profile"])
    profiles=sorted(Path("artifacts").rglob("*perception_profile.json")) if Path("artifacts").exists() else []
    profile_path=st.selectbox("Detector profile",profiles,format_func=str) if perception_choice.startswith("YOLO") and profiles else None
    if perception_choice.startswith("YOLO") and not profiles: st.warning("No empirical profile found; add artifacts/perception_profile.json.")
    st.subheader("Environment")
    field_width=st.slider("Field width (m)",60,180,100,10)
    field_height=st.slider("Field depth (m)",45,140,70,5)
    random_obstacles=st.checkbox("Randomize obstacles",True)
    obstacle_seed=st.number_input("Obstacle seed",0,999999,21)
    obstacle_count=st.slider("Obstacle count",0,14,6)
    obstacle_size=st.slider("Obstacle size range (m)",3.0,25.0,(5.0,14.0),1.0)
    obstacle_height=st.slider("Maximum obstacle height (m)",5.0,35.0,18.0,1.0)
    uav_altitude=st.slider("UAV display altitude (m)",3.0,30.0,12.0,1.0)
    target_altitude=st.slider("Target display altitude (m)",0.5,20.0,2.0,.5)
    st.subheader("Visualization")
    truth=st.checkbox("Show ground truth",True); detections=st.checkbox("Show detections",True)
    estimate=st.checkbox("Show Kalman estimate",True); prediction=st.checkbox("Show target prediction",True)
    planned=st.checkbox("Show planned path",True); trails=st.checkbox("Show trails",True); radius=st.checkbox("Show standoff radius",True)
    run=st.button("RUN SIMULATION",type="primary",width="stretch")
    if st.button("RESET",width="stretch"): st.session_state.pop("engine",None); st.rerun()

if run or "engine" not in st.session_state:
    backend="profiled_simulated" if perception_choice.startswith("YOLO") and profile_path else "simulated"
    cfg=SimulationConfig(scenario=scenario,mode=mode,width=float(field_width),height=float(field_height),target_speed=target_speed,detection_probability=probability,sensor_noise=noise,desired_standoff=standoff,prediction_horizon=horizon,uav_max_speed=max_speed,seed=int(seed),randomize_obstacles=random_obstacles,obstacle_seed=int(obstacle_seed),obstacle_count=obstacle_count,obstacle_min_size=obstacle_size[0],obstacle_max_size=obstacle_size[1],obstacle_max_height=obstacle_height,uav_altitude=uav_altitude,target_altitude=target_altitude,perception_backend=backend,detector_profile_path=str(profile_path) if profile_path else None)
    with st.spinner("Running deterministic simulation…"):
        engine=SimulationEngine(cfg); engine.run(); st.session_state.engine=engine
engine=st.session_state.engine; df=engine.get_dataframe(); metrics=engine.get_metrics(); last=df.iloc[-1]
if engine.config.perception_backend=="profiled_simulated": st.info("Profile-driven metric simulation: empirical YOLO reliability influences detection availability/confidence; bbox pixels are not converted to meters.")

left,right=st.columns([2.4,1])
with left:
    st.plotly_chart(pursuit_3d_figure(engine,df,truth,detections,estimate,prediction,planned,trails,radius),width="stretch",config={"displaylogo":False})
    st.caption("Press Play pursuit to animate. Drag to orbit, scroll to zoom. Altitude is a visualization layer; the autonomy model remains 2D kinematic.")
with right:
    st.subheader("Telemetry")
    a,b=st.columns(2); a.metric("STATE",last.fsm_state); b.metric("Detection","VALID" if last.detection_valid else "LOST")
    a,b=st.columns(2); a.metric("Confidence",f"{last.confidence:.2f}"); b.metric("Range",f"{last.range_to_target:.1f} m")
    a,b=st.columns(2); a.metric("Bearing",f"{last.bearing_deg:.1f}°"); b.metric("Uncertainty",f"{last.uncertainty:.2f}")
    a,b=st.columns(2); a.metric("Estimated vx",f"{last.estimated_vx:.2f} m/s"); b.metric("Estimated vy",f"{last.estimated_vy:.2f} m/s")
    a,b=st.columns(2); a.metric("Planner","A*" if engine.obstacles else "Direct"); b.metric("Path nodes",int(last.path_nodes))

st.subheader("Autonomy and detection timeline")
fig,ax=plt.subplots(figsize=(12,2.4)); ax.step(df.time,[STATE_MAP[s] for s in df.fsm_state],where="post",label="FSM"); ax.fill_between(df.time,-.45,-.1,where=~df.detection_valid,color="#ef476f",step="post",label="Detection lost"); ax.set_yticks(list(STATE_MAP.values()),list(STATE_MAP)); ax.set_xlabel("Time (s)"); ax.legend(loc="upper right"); st.pyplot(fig,width="stretch"); plt.close(fig)

overview,tracking,navigation,comparison,runlog,architecture=st.tabs(["Overview","Tracking","Navigation","Comparison","Run Log","Architecture"])
with overview:
    cols=st.columns(4); cols[0].metric("Estimation RMSE",f"{metrics['estimation_rmse']:.2f} m"); cols[1].metric("Travel distance",f"{metrics['total_uav_travel_distance']:.1f} m"); cols[2].metric("Standoff RMSE",f"{metrics['standoff_rmse']:.2f} m"); cols[3].metric("Detection availability",f"{metrics['detection_availability']:.1f}%")
with tracking:
    st.line_chart(df.set_index("time")[["tracking_error","uncertainty","confidence"]])
with navigation:
    chart=df.set_index("time")[["range_to_target","uav_speed","path_length"]]; st.line_chart(chart); st.caption(f"Desired standoff: {engine.config.desired_standoff:.1f} m (valid band ±1.5 m)")
with comparison: st.info("Run either operating mode with identical controls and export each run log for a controlled comparison.")
with runlog:
    st.dataframe(df,width="stretch",height=420); st.download_button("Download run CSV",df.to_csv(index=False),"run_log.csv","text/csv")
with architecture:
    st.code("""Simulated Camera (relative detection)\n             ↓\nConstant-Velocity Kalman Estimator\n             ↓\nConfidence-Aware Autonomy FSM\n             ↓\nTarget Prediction → Safe Standoff Goal\n             ↓\nInflated-Grid A* Planner\n             ↓\nVelocity Controller → UAV Simulator""")
    st.caption("Ground truth is isolated to simulator internals, simulated sensor generation, visualization, and evaluation.")
