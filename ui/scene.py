"""3D replay of the 2D simulation; altitude is visual only."""
import numpy as np
import plotly.graph_objects as go


def pursuit_3d_figure(engine, df, truth, detections, estimate, prediction, planned, trails, radius, selected_index=None):
    """Build an animated 3D presentation of the 2D kinematic simulation."""
    fig = go.Figure()
    floor_x = [0, engine.config.width, engine.config.width, 0]
    floor_y = [0, 0, engine.config.height, engine.config.height]
    fig.add_trace(go.Mesh3d(x=floor_x, y=floor_y, z=[0]*4, i=[0, 0], j=[1, 2], k=[2, 3],
                            color="#101b24", opacity=.55, name="Ground", hoverinfo="skip",showlegend=False))
    obstacle_colors=["#334957", "#42535a", "#53605d", "#364e58"]
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
    colors=dict(uav="#63c9ed",target="#f4aa60",estimate="#a6c887",detect="#ffd166",predict="#ef476f",path="#b1a1e6")
    dynamic_names=["UAV trail","Target trail","Estimate trail","Detections","UAV","Target","Estimate","Predicted","Standoff","Pursuit vector","Final A* path"]
    for name in dynamic_names:
        mode="lines" if name in {"UAV trail","Target trail","Estimate trail","Standoff","Pursuit vector","Final A* path"} else "markers"
        fig.add_trace(go.Scatter3d(x=[],y=[],z=[],mode=mode,name=name))
    trace_ids=list(range(static_count,static_count+len(dynamic_names)))
    selected_index = len(df)-1 if selected_index is None else int(selected_index)
    indices=np.unique(np.r_[np.arange(0,len(df),10),selected_index,len(df)-1]).astype(int)
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
            go.Scatter3d(x=[p[0] for p in engine.path] if planned and idx == len(df)-1 else [],y=[p[1] for p in engine.path] if planned and idx == len(df)-1 else [],z=np.full(len(engine.path),.2) if planned and idx == len(df)-1 else [],mode="lines",line=dict(color=colors["path"],width=5,dash="dash"),name="Final A* path"),
        ]
        frames.append(go.Frame(data=series,traces=trace_ids,name=str(idx)))
    fig.frames=frames
    if frames:
        for trace_id, trace in zip(trace_ids, next(f for f in frames if f.name == str(selected_index)).data): fig.data[trace_id].update(trace)
    steps=[dict(method="animate",args=[[f.name],dict(mode="immediate",frame=dict(duration=0,redraw=True),transition=dict(duration=0))],label=f"{df.iloc[int(f.name)].time:.0f}s") for f in frames[::max(1,len(frames)//12)]]
    axis_style=dict(backgroundcolor="#101820",gridcolor="#24343e",zerolinecolor="#405564",color="#9db0ba",showbackground=True)
    fig.update_layout(height=570,margin=dict(l=38,r=38,t=35,b=95),paper_bgcolor="#101820",plot_bgcolor="#101820",font=dict(color="#cad6dc"),
        scene=dict(domain=dict(x=[.035,.965],y=[.20,.93]),
                   xaxis=dict(title="x (m)",range=[0,engine.config.width],**axis_style),
                   yaxis=dict(title="y (m)",range=[0,engine.config.height],**axis_style),
                   zaxis=dict(title="Altitude (m)",range=[0,max(20,engine.config.obstacle_max_height+3,engine.config.uav_altitude+3)],**axis_style),
                   bgcolor="#101820",aspectmode="manual",aspectratio=dict(x=1.45,y=1,z=.42),camera=dict(eye=dict(x=1.45,y=-1.65,z=1.15))),
        legend=dict(orientation="h",y=1.03,x=.5,xanchor="center",font=dict(color="#cad6dc"),bgcolor="rgba(16,24,32,.88)"),
        updatemenus=[dict(type="buttons",direction="left",x=.5,xanchor="center",y=.105,yanchor="middle",pad=dict(l=10,r=10,t=8,b=8),
                          bgcolor="#101820",bordercolor="#405564",font=dict(color="#cad6dc"),buttons=[
            dict(label="Play replay",method="animate",args=[None,dict(fromcurrent=True,frame=dict(duration=90,redraw=True),transition=dict(duration=0))]),
            dict(label="Pause",method="animate",args=[[None],dict(mode="immediate",frame=dict(duration=0,redraw=False))])])],
        sliders=[dict(active=max(0, next((i for i, f in enumerate(frames[::max(1,len(frames)//12)]) if f.name == str(selected_index)), 0)),x=.12,len=.76,y=.015,pad=dict(t=10,b=4),currentvalue=dict(prefix="Simulation time: ",font=dict(color="#9db0ba")),font=dict(color="#9db0ba"),steps=steps)])
    return fig
