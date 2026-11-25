const color = {ink:"#f4efe8", muted:"#a59b91", orange:"#ff5a2a", cyan:"#7bbec6", purple:"#b2a0d4", bg:"#141210", grid:"#2a231e"};
const finite = x => x !== null && Number.isFinite(x);

export function drawMap(canvas, run, cursor, layers) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width), height = Math.round(rect.height);
  if (canvas.width !== width*dpr || canvas.height !== height*dpr) {canvas.width=width*dpr;canvas.height=height*dpr;}
  const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
  ctx.fillStyle=color.bg;ctx.fillRect(0,0,width,height);
  const cfg=run.config, row=run.samples[cursor];
  const scale=Math.min((width-78)/cfg.width,(height-56)/cfg.height);
  const ox=(width-cfg.width*scale)/2, oy=(height-cfg.height*scale)/2;
  const x=v=>ox+v*scale,y=v=>height-oy-v*scale;
  ctx.lineWidth=1;ctx.strokeStyle=color.grid;ctx.font="9px 'IBM Plex Mono', monospace";ctx.fillStyle=color.muted;
  for(let n=0;n<=cfg.width;n+=10){ctx.beginPath();ctx.moveTo(x(n),y(0));ctx.lineTo(x(n),y(cfg.height));ctx.stroke();if(n%20===0){ctx.textAlign="center";ctx.fillText(n,x(n),y(0)+19);}}
  for(let n=0;n<=cfg.height;n+=10){ctx.beginPath();ctx.moveTo(x(0),y(n));ctx.lineTo(x(cfg.width),y(n));ctx.stroke();if(n%20===0){ctx.textAlign="right";ctx.fillText(n,x(0)-11,y(n)+3);}}
  ctx.strokeStyle="#5b483c";ctx.strokeRect(x(0),y(cfg.height),cfg.width*scale,cfg.height*scale);
  for(const obstacle of run.obstacles){const [x0,x1,y0,y1]=obstacle.bounds;ctx.fillStyle="#332920";ctx.strokeStyle="#725b48";ctx.beginPath();if(obstacle.shape==="cylinder")ctx.ellipse(x((x0+x1)/2),y((y0+y1)/2),(x1-x0)*scale/2,(y1-y0)*scale/2,0,0,Math.PI*2);else ctx.rect(x(x0),y(y1),(x1-x0)*scale,(y1-y0)*scale);ctx.fill();ctx.stroke();}
  const samples=run.samples.slice(0,cursor+1);
  function line(xkey,ykey,stroke,dash=[],limit=samples){ctx.beginPath();ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.setLineDash(dash);let begun=false;for(const p of limit){if(!finite(p[xkey])||!finite(p[ykey])){begun=false;continue;}if(begun)ctx.lineTo(x(p[xkey]),y(p[ykey]));else{ctx.moveTo(x(p[xkey]),y(p[ykey]));begun=true;}}ctx.stroke();ctx.setLineDash([]);}
  if(layers.trails){line("uav_x","uav_y",color.cyan);if(layers.truth)line("true_target_x","true_target_y","#ff5a2a80");if(layers.estimate)line("estimated_target_x","estimated_target_y","#f4efe84d",[3,4]);}
  if(layers.detections){ctx.fillStyle="#f4efe85c";for(let i=0;i<samples.length;i+=3){const p=samples[i];if(finite(p.detection_x)&&finite(p.detection_y))ctx.fillRect(x(p.detection_x)-1,y(p.detection_y)-1,2,2);}}
  if(layers.standoff&&finite(row.estimated_target_x)){ctx.strokeStyle="#f4efe866";ctx.lineWidth=1;ctx.setLineDash([5,5]);ctx.beginPath();ctx.arc(x(row.estimated_target_x),y(row.estimated_target_y),cfg.desired_standoff*scale,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);}
  if(layers.path&&cursor===run.samples.length-1&&run.path.length){ctx.strokeStyle=color.purple;ctx.setLineDash([5,4]);ctx.beginPath();run.path.forEach((p,i)=>i?ctx.lineTo(x(p[0]),y(p[1])):ctx.moveTo(x(p[0]),y(p[1])));ctx.stroke();ctx.setLineDash([]);}
  function cross(px,py,stroke,size=5){if(!finite(px)||!finite(py))return;ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x(px)-size,y(py));ctx.lineTo(x(px)+size,y(py));ctx.moveTo(x(px),y(py)-size);ctx.lineTo(x(px),y(py)+size);ctx.stroke();}
  if(layers.truth){ctx.strokeStyle=color.orange;ctx.lineWidth=2;ctx.strokeRect(x(row.true_target_x)-5,y(row.true_target_y)-5,10,10);}
  if(layers.estimate)cross(row.estimated_target_x,row.estimated_target_y,color.ink,6);
  if(layers.prediction&&finite(row.estimated_target_x))cross(row.predicted_target_x,row.predicted_target_y,color.purple,5);
  ctx.save();ctx.translate(x(row.uav_x),y(row.uav_y));ctx.rotate(-Math.PI/4);ctx.fillStyle=color.cyan;ctx.fillRect(-4,-4,8,8);ctx.strokeStyle=color.cyan;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(-10,0);ctx.lineTo(10,0);ctx.moveTo(0,-10);ctx.lineTo(0,10);ctx.stroke();ctx.restore();
  ctx.textAlign="left";ctx.font="10px 'IBM Plex Mono', monospace";ctx.fillStyle=color.cyan;ctx.fillText("UAV",x(row.uav_x)+13,y(row.uav_y)-9);
  if(layers.truth){ctx.fillStyle=color.orange;ctx.fillText("TARGET",x(row.true_target_x)+10,y(row.true_target_y)-10);}
  ctx.fillStyle=color.muted;ctx.textAlign="right";ctx.fillText("N",width-19,25);ctx.strokeStyle=color.muted;ctx.beginPath();ctx.moveTo(width-22,31);ctx.lineTo(width-22,49);ctx.moveTo(width-26,35);ctx.lineTo(width-22,30);ctx.lineTo(width-18,35);ctx.stroke();
}

function boxMesh(obstacle,index){const [a,b,c,d]=obstacle.bounds,h=obstacle.height;return{type:"mesh3d",x:[a,b,b,a,a,b,b,a],y:[c,c,d,d,c,c,d,d],z:[0,0,0,0,h,h,h,h],i:[0,0,0,1,1,2,4,4,5,6,2,3],j:[1,2,3,2,5,3,5,7,6,7,6,7],k:[2,3,4,5,4,7,6,5,7,6,3,4],color:"#5b483b",opacity:.92,flatshading:true,name:`Obstacle ${index+1}`,hoverinfo:"name",showlegend:false,lighting:{ambient:.65,diffuse:.65}};}

export function draw3D(element,run,cursor,layers){
  if(!window.Plotly||element.hidden)return;
  const cfg=run.config,row=run.samples[cursor],samples=run.samples.slice(0,cursor+1);
  const initialized=finite(row.estimated_target_x)&&finite(row.estimated_target_y);
  const traces=run.obstacles.map(boxMesh);
  function trail(x,y,z,name,lineColor,visible){traces.push({type:"scatter3d",mode:"lines",x:visible?samples.map(r=>r[x]):[],y:visible?samples.map(r=>r[y]):[],z:visible?samples.map(()=>z):[],name,line:{color:lineColor,width:4},showlegend:false,hoverinfo:"skip"});}
  function point(px,py,z,name,pointColor,visible,symbol="square"){traces.push({type:"scatter3d",mode:"markers",x:visible?[px]:[],y:visible?[py]:[],z:visible?[z]:[],name,marker:{color:pointColor,size:5,symbol},showlegend:false,hovertemplate:`${name}<extra></extra>`});}
  trail("uav_x","uav_y",cfg.uav_altitude,"UAV",color.cyan,layers.trails);
  trail("true_target_x","true_target_y",cfg.target_altitude,"Ground truth",color.orange,layers.trails&&layers.truth);
  trail("estimated_target_x","estimated_target_y",cfg.target_altitude+.4,"Estimate",color.ink,layers.trails&&layers.estimate);
  point(row.uav_x,row.uav_y,cfg.uav_altitude,"UAV",color.cyan,true,"diamond");
  point(row.true_target_x,row.true_target_y,cfg.target_altitude,"Ground truth",color.orange,layers.truth);
  point(row.estimated_target_x,row.estimated_target_y,cfg.target_altitude+.4,"Estimate",color.ink,layers.estimate,"cross");
  point(row.predicted_target_x,row.predicted_target_y,cfg.target_altitude+.8,"Prediction",color.purple,layers.prediction&&initialized,"x");
  const theta=Array.from({length:65},(_,i)=>2*Math.PI*i/64);
  traces.push({type:"scatter3d",mode:"lines",x:layers.standoff&&initialized?theta.map(t=>row.estimated_target_x+cfg.desired_standoff*Math.cos(t)):[],y:layers.standoff&&initialized?theta.map(t=>row.estimated_target_y+cfg.desired_standoff*Math.sin(t)):[],z:layers.standoff&&initialized?theta.map(()=>.2):[],line:{color:"#f4efe866",width:2,dash:"dash"},showlegend:false,hoverinfo:"skip"});
  const valid=samples.filter((r,i)=>r.detection_valid&&i%3===0);
  traces.push({type:"scatter3d",mode:"markers",x:layers.detections?valid.map(r=>r.detection_x):[],y:layers.detections?valid.map(r=>r.detection_y):[],z:valid.map(()=>cfg.target_altitude+.6),marker:{color:"#f4efe877",size:2},showlegend:false,hoverinfo:"skip"});
  if(layers.path&&cursor===run.samples.length-1)traces.push({type:"scatter3d",mode:"lines",x:run.path.map(p=>p[0]),y:run.path.map(p=>p[1]),z:run.path.map(()=>.2),line:{color:color.purple,width:4,dash:"dash"},name:"Final A* path",showlegend:false});
  const axis={backgroundcolor:color.bg,gridcolor:color.grid,zerolinecolor:"#5b483c",showbackground:true,color:color.muted};
  const layout={paper_bgcolor:color.bg,font:{family:"IBM Plex Mono, monospace",size:10,color:color.muted},margin:{l:0,r:0,t:0,b:0},scene:{xaxis:{...axis,title:"X / m",range:[0,cfg.width]},yaxis:{...axis,title:"Y / m",range:[0,cfg.height]},zaxis:{...axis,title:"Visual altitude / m",range:[0,Math.max(20,cfg.obstacle_max_height+3,cfg.uav_altitude+3)]},aspectmode:"manual",aspectratio:{x:1.4,y:1,z:.45},camera:{eye:{x:1.5,y:-1.5,z:1.3}},uirevision:run.run_id,bgcolor:color.bg},uirevision:run.run_id};
  return Plotly.react(element,traces,layout,{responsive:true,displaylogo:false,displayModeBar:false});
}

export function drawAnalysis(element,run,type){
  if(!window.Plotly)return;
  const samples=run.samples,t=samples.map(r=>r.time),traces=[],shapes=[];
  const layout={paper_bgcolor:color.bg,plot_bgcolor:color.bg,font:{family:"IBM Plex Mono, monospace",size:10,color:color.muted},margin:{l:90,r:30,t:35,b:45},height:265,xaxis:{title:{text:"SIMULATION TIME / s",font:{size:10}},gridcolor:color.grid,zeroline:false},yaxis:{gridcolor:color.grid,zeroline:false},legend:{orientation:"h",x:0,y:1.18,font:{size:10}},hovermode:"x unified"};
  if(type==="timeline"){
    const states=["SEARCH","ACQUIRE","TRACK","PREDICT","REACQUIRE"];
    traces.push({x:t,y:samples.map(r=>states.indexOf(r.fsm_state)),type:"scatter",mode:"lines",name:"Autonomy state",line:{color:color.orange,width:2,shape:"hv"}});
    const lost=samples.filter(r=>!r.detection_valid);
    traces.push({x:lost.map(r=>r.time),y:lost.map(()=>-.45),type:"scatter",mode:"markers",name:"Sensor dropout",marker:{symbol:"line-ns",color:color.muted,size:6}});
    layout.yaxis={...layout.yaxis,tickvals:[0,1,2,3,4],ticktext:states,range:[-.8,4.4]};
  }else if(type==="tracking"){
    traces.push({x:t,y:samples.map(r=>r.tracking_error),name:"Position error / m",line:{color:color.orange,width:1.5}});
    traces.push({x:t,y:samples.map(r=>r.uncertainty),name:"Filter uncertainty",line:{color:color.cyan,width:1.5},yaxis:"y2"});
    layout.yaxis2={overlaying:"y",side:"right",showgrid:false,color:color.cyan};layout.margin.r=55;
  }else{
    traces.push({x:t,y:samples.map(r=>r.range_to_target),name:"Estimated range / m",line:{color:color.cyan,width:1.5}});
    shapes.push({type:"rect",xref:"paper",x0:0,x1:1,y0:run.config.desired_standoff-1.5,y1:run.config.desired_standoff+1.5,line:{width:0},fillcolor:"#ff5a2a15"},{type:"line",xref:"paper",x0:0,x1:1,y0:run.config.desired_standoff,y1:run.config.desired_standoff,line:{color:color.orange,width:1,dash:"dash"}});
    layout.yaxis.title={text:"RANGE / m",font:{size:10}};
  }
  layout.shapes=shapes;
  return Plotly.react(element,traces,layout,{responsive:true,displaylogo:false,displayModeBar:false});
}
