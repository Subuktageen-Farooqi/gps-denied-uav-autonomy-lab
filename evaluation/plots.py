from pathlib import Path
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Lightweight artifact fallback for constrained runtimes.
    plt = None


STATE_MAP={"SEARCH":0,"ACQUIRE":1,"TRACK":2,"PREDICT":3,"REACQUIRE":4}


def save_plots(df, obstacles, desired, output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if plt is None:
        _save_pillow_plots(df, obstacles, desired, out)
        return
    fig,ax=plt.subplots(figsize=(10,6))
    for x0,x1,y0,y1 in obstacles: ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,color="#5f6670",alpha=.7))
    ax.plot(df.true_target_x,df.true_target_y,label="True target",color="#ff8c42")
    ax.plot(df.estimated_target_x,df.estimated_target_y,label="Kalman estimate",color="#3ddc97",alpha=.85)
    ax.plot(df.uav_x,df.uav_y,label="UAV",color="#42a5f5")
    ax.set(xlabel="x (m)",ylabel="y (m)",title="Top-down trajectories"); ax.axis("equal"); ax.legend(); fig.tight_layout(); fig.savefig(out/"trajectory.png",dpi=150); plt.close(fig)
    for name, y, ylabel in [("tracking_error.png",df.tracking_error,"Error (m)"),("range_error.png",df.range_to_target,"Range (m)")]:
        fig,ax=plt.subplots(figsize=(9,3.5)); ax.plot(df.time,y)
        if name.startswith("range"): ax.axhline(desired,color="orange",ls="--")
        ax.set(xlabel="Time (s)",ylabel=ylabel); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/name,dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.5)); ax.step(df.time,[STATE_MAP[s] for s in df.fsm_state],where="post"); ax.set_yticks(list(STATE_MAP.values()),list(STATE_MAP)); ax.set_xlabel("Time (s)"); fig.tight_layout(); fig.savefig(out/"state_timeline.png",dpi=150); plt.close(fig)


def _save_pillow_plots(df, obstacles, desired, out):
    from PIL import Image, ImageDraw
    def canvas():
        im=Image.new("RGB",(1000,560),"#10151d"); return im,ImageDraw.Draw(im)
    def mapxy(x,y): return (50+9*float(x),510-6.5*float(y))
    im,d=canvas()
    for x0,x1,y0,y1 in obstacles: d.rectangle((*mapxy(x0,y1),*mapxy(x1,y0)),fill="#626d7c")
    for xs,ys,color in [(df.true_target_x,df.true_target_y,"#ff8c42"),(df.estimated_target_x,df.estimated_target_y,"#3ddc97"),(df.uav_x,df.uav_y,"#42a5f5")]:
        points=[mapxy(x,y) for x,y in zip(xs,ys)]; d.line(points,fill=color,width=3)
    d.text((40,20),"Top-down trajectories (target / estimate / UAV)",fill="white"); im.save(out/"trajectory.png")
    def series_plot(values, title, reference=None):
        im,d=canvas(); vals=[float(v) for v in values]; lo=min(vals); hi=max(vals); span=max(hi-lo,1e-6)
        pts=[(50+900*i/max(len(vals)-1,1),500-430*(v-lo)/span) for i,v in enumerate(vals)]
        d.line(pts,fill="#42a5f5",width=3)
        if reference is not None:
            y=500-430*(reference-lo)/span; d.line((50,y,950,y),fill="#ff8c42",width=2)
        d.text((40,20),title,fill="white"); im.save(out/(title.lower().replace(" ","_")+".png"))
    series_plot(df.tracking_error.fillna(0),"tracking error")
    series_plot(df.range_to_target.fillna(0),"range error",desired)
    im,d=canvas(); vals=[STATE_MAP[s] for s in df.fsm_state]
    pts=[(50+900*i/max(len(vals)-1,1),480-90*v) for i,v in enumerate(vals)]; d.line(pts,fill="#b388ff",width=3)
    d.text((40,20),"FSM state timeline: SEARCH 0 / ACQUIRE 1 / TRACK 2 / PREDICT 3 / REACQUIRE 4",fill="white"); im.save(out/"state_timeline.png")
