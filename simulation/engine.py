import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import SimulationConfig
from simulation.scenarios import get_scenario
from simulation.vehicle import Vehicle
from simulation.target import Target
from perception.simulated_detector import SimulatedDetector
from perception.detector_profile import ProfiledSimulatedDetector
from estimation.kalman_filter import KalmanFilterCV
from autonomy.state_machine import AutonomyFSM, State
from autonomy.pursuit_controller import safe_goal, velocity_command
from autonomy.tracking_controller import follow_command
from planning.astar import AStarPlanner
from evaluation.metrics import compute_metrics


class SimulationEngine:
    def __init__(self, config: SimulationConfig): self.config=config; self.reset()
    def _build_obstacles(self):
        c=self.config
        if not c.randomize_obstacles:
            specs=[]
            for n,(x0,x1,y0,y1) in enumerate(self.scenario.obstacles):
                x0,x1=max(0,x0),min(c.width,x1); y0,y1=max(0,y0),min(c.height,y1)
                if x1<=x0 or y1<=y0: continue
                specs.append(dict(bounds=(x0,x1,y0,y1),height=min(8.0,c.obstacle_max_height),shape="box",name=f"Obstacle {n+1}"))
            return specs
        rng=np.random.default_rng(c.obstacle_seed); specs=[]; attempts=0
        protected=[np.asarray(self.scenario.uav_start,float),np.asarray(self.scenario.target_start,float)]
        while len(specs)<c.obstacle_count and attempts<c.obstacle_count*80:
            attempts+=1
            w=float(rng.uniform(c.obstacle_min_size,c.obstacle_max_size)); h=float(rng.uniform(c.obstacle_min_size,c.obstacle_max_size))
            if w>c.width-8 or h>c.height-8: continue
            x0=float(rng.uniform(3,c.width-w-3)); y0=float(rng.uniform(3,c.height-h-3)); bounds=(x0,x0+w,y0,y0+h)
            center=np.array([x0+w/2,y0+h/2]); clearance=max(w,h)/2+c.obstacle_inflation+4
            if any(np.linalg.norm(center-p)<clearance for p in protected): continue
            if any(not (bounds[1]+2<b[0] or bounds[0]-2>b[1] or bounds[3]+2<b[2] or bounds[2]-2>b[3]) for b in [s["bounds"] for s in specs]): continue
            specs.append(dict(bounds=bounds,height=float(rng.uniform(4,c.obstacle_max_height)),shape=str(rng.choice(["box","cylinder"])),name=f"Obstacle {len(specs)+1}"))
        return specs
    def reset(self):
        c=self.config; self.scenario=get_scenario(c.scenario); self.rng=np.random.default_rng(c.seed)
        s=self.scenario
        vel=np.asarray(s.target_velocity,float)
        if c.target_speed is not None: vel*=c.target_speed/max(np.linalg.norm(vel),1e-6)
        self.uav=Vehicle(s.uav_start,c.uav_max_speed); self.target=Target(s.target_start,vel,self.rng,s.turn_interval)
        if c.perception_backend == "profiled_simulated":
            if not c.detector_profile_path: raise ValueError("detector_profile_path is required for profiled_simulated")
            self.detector=ProfiledSimulatedDetector(c.detector_profile_path,c.sensor_noise or s.sensor_noise,s.occlusions,self.rng,c.dt)
        elif c.perception_backend == "simulated":
            self.detector=SimulatedDetector(c.detection_probability or s.detection_probability,c.sensor_noise or s.sensor_noise,s.occlusions,self.rng)
        else: raise ValueError(f"Unsupported simulation perception backend: {c.perception_backend}")
        self.kf=KalmanFilterCV(measurement_var=(c.sensor_noise or s.sensor_noise)**2)
        self.visual_obstacles=self._build_obstacles(); self.obstacles=[o["bounds"] for o in self.visual_obstacles]
        self.fsm=AutonomyFSM(); self.planner=AStarPlanner(c.width,c.height,c.grid_resolution,self.obstacles,c.obstacle_inflation)
        self.t=0.; self.log=[]; self.path=[]; self.last_plan=-999.; self.goal=np.asarray(s.uav_start,float); self.predicted=np.asarray(s.target_start,float)
    def step(self):
        c=self.config; previous=self.uav.position.copy()
        # Simulator truth is consumed only here and by the simulated sensor/evaluation log.
        self.target.step(self.t,c.dt,np.array([c.width,c.height]))
        detection=self.detector.detect(self.target.position,self.uav.position,self.t)
        if self.kf.initialized: self.kf.predict(c.dt)
        if detection is not None:
            measured_global=self.uav.position+detection.relative
            self.kf.update(measured_global)
        state=self.fsm.update(detection is not None,c.dt)
        if self.kf.initialized:
            predictive=c.mode=="Predictive Standoff Pursuit"
            self.goal,self.predicted=safe_goal(self.uav.position,self.kf.position,self.kf.velocity,c.desired_standoff,c.prediction_horizon,predictive)
            if c.mode=="GPS-Denied Follow" and not self.obstacles:
                command=follow_command(self.kf.position-self.uav.position,self.kf.velocity,c.desired_standoff,c.uav_max_speed)
                self.path=[]
            else:
                if self.t-self.last_plan>=c.replan_interval or not self.path:
                    self.path=self.planner.plan(self.uav.position,self.goal); self.last_plan=self.t
                while len(self.path)>1 and np.linalg.norm(self.path[0]-self.uav.position)<1.2: self.path.pop(0)
                waypoint=self.path[0] if self.path else self.goal
                command=velocity_command(self.uav.position,self.uav.velocity,waypoint,self.kf.velocity,state,c.uav_max_speed)
        else:
            command=velocity_command(self.uav.position,self.uav.velocity,self.uav.position,np.zeros(2),State.SEARCH,c.uav_max_speed)
        self.uav.step(command,c.dt,np.array([c.width,c.height])); self.t+=c.dt
        estimate=self.kf.position if self.kf.initialized else np.array([np.nan,np.nan]); error=np.linalg.norm(estimate-self.target.position) if self.kf.initialized else np.nan
        rel=self.kf.position-self.uav.position if self.kf.initialized else np.array([np.nan,np.nan])
        self.log.append(dict(time=self.t,true_target_x=self.target.position[0],true_target_y=self.target.position[1],estimated_target_x=estimate[0],estimated_target_y=estimate[1],predicted_target_x=self.predicted[0],predicted_target_y=self.predicted[1],uav_x=self.uav.position[0],uav_y=self.uav.position[1],detection_x=(self.uav.position[0]+detection.relative[0] if detection else np.nan),detection_y=(self.uav.position[1]+detection.relative[1] if detection else np.nan),detection_valid=detection is not None,confidence=(detection.confidence if detection else 0.),fsm_state=state.value,range_to_target=np.linalg.norm(rel),bearing_deg=np.degrees(np.arctan2(rel[1],rel[0])),tracking_error=error,uav_speed=np.linalg.norm(self.uav.velocity),estimated_vx=self.kf.velocity[0],estimated_vy=self.kf.velocity[1],uncertainty=self.kf.uncertainty,path_length=sum(np.linalg.norm(b-a) for a,b in zip(self.path[:-1],self.path[1:])),path_nodes=len(self.path),step_distance=np.linalg.norm(self.uav.position-previous)))
    def run(self):
        while self.t < self.config.duration-1e-9: self.step()
        return self.get_dataframe()
    def get_dataframe(self): return pd.DataFrame(self.log)
    def get_metrics(self): return compute_metrics(self.get_dataframe().dropna(subset=["tracking_error"]),self.config.desired_standoff)
    def get_frame_state(self): return self.log[-1] if self.log else {}
    def save_results(self, output_dir="results"):
        from evaluation.plots import save_plots
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); df=self.get_dataframe(); metrics=self.get_metrics()
        df.to_csv(out/"run_log.csv",index=False); (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        save_plots(df,self.obstacles,self.config.desired_standoff,out); return metrics
