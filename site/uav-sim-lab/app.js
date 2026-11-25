import {request, rememberAccess, hasAccess} from "./api.js";
import {drawMap, draw3D, drawAnalysis} from "./graphics.js?v=20261125c";

const $ = id => document.getElementById(id);
const state = {run:null,reference:null,cursor:0,playing:false,lastTick:0,dimension:"2d",analysis:"timeline",view:"mission",connected:false,pending:null,comparison:null,edge:null,layers:{truth:true,detections:true,estimate:true,prediction:true,trails:true,standoff:true,path:true}};
const fmt = (value,digits=1,unit="") => value===null||value===undefined||!Number.isFinite(Number(value)) ? "—" : `${Number(value).toFixed(digits)}${unit}`;
function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function notice(message,error=false){$("notice").textContent=message;$("notice").classList.toggle("error",error);$("notice").hidden=!message;}
function errorMessage(error){if(error.status===401){disconnect(false);openConnection();}notice(error.message||"The request could not be completed.",true);}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=el("a");a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);}
function toCSV(rows){if(!rows.length)return "";const keys=Object.keys(rows[0]);const quote=v=>`"${String(v??"").replaceAll('"','""')}"`;return [keys.map(quote).join(","),...rows.map(row=>keys.map(k=>quote(typeof row[k]==="object"?JSON.stringify(row[k]):row[k])).join(","))].join("\r\n");}
function table(headers,rows){const t=el("table"),head=el("thead"),tr=el("tr");headers.forEach(h=>tr.append(el("th",h)));head.append(tr);t.append(head);const body=el("tbody");rows.forEach(values=>{const row=el("tr");values.forEach(v=>row.append(el("td",v)));body.append(row);});t.append(body);return t;}
async function busy(button,label,action){const original=button.innerHTML;button.disabled=true;button.textContent=label;try{return await action();}catch(error){errorMessage(error);}finally{button.innerHTML=original;button.disabled=false;}}

function updateConnection(runtime){
  const gpu=runtime.configured_gpu;
  const name=gpu?(/^(NVIDIA|AMD|INTEL)\b/i.test(gpu)?gpu:`NVIDIA ${gpu}`):"GPU UNAVAILABLE";
  const memory=Number(runtime.configured_gpu_vram_gb);
  const vram=Number.isFinite(memory)&&memory>0?`${memory} GB`:"VRAM UNAVAILABLE";
  state.connected=true;
  $("compute-gpu").textContent=name;
  $("compute-vram").textContent=`${vram} · ${String(runtime.perception||"on demand").toUpperCase()}`;
  $("compute-source").hidden=false;
  $("connect-button").hidden=true;
  $("runtime-state").textContent="API CONNECTED";
  $("runtime-details").textContent=`CPU simulation available · Python ${runtime.python} · ${runtime.model_count} model(s) · ${name} perception allocated on demand.`;
}
function disconnect(showNotice=true){
  state.connected=false;state.pending=null;state.edge=null;rememberAccess("");
  $("compute-source").hidden=true;$("connect-button").hidden=false;
  $("compute-gpu").textContent="";$("compute-vram").textContent="";
  $("runtime-state").textContent="DISCONNECTED";
  $("runtime-details").textContent="Connect using your lab access code to inspect the backend.";
  $("access-code").value="";
  renderEdgeDeployment();
  if(showNotice)notice("Disconnected. The reference replay remains available.");
}
function openConnection(){if(!$("connect-dialog").open)$("connect-dialog").showModal();setTimeout(()=>$("access-code").focus(),0);}
function requireConnection(action){if(state.connected){action();return;}state.pending=action;openConnection();}
$("connect-button").addEventListener("click",openConnection);
$("header-disconnect-button").addEventListener("click",()=>{disconnect();$("connect-button").focus();});
document.querySelectorAll("[data-connect]").forEach(button=>button.addEventListener("click",openConnection));
document.querySelectorAll("[data-close]").forEach(button=>button.addEventListener("click",()=>button.closest("dialog").close()));
$("connect-dialog").addEventListener("close",()=>{if(!state.connected)state.pending=null;});
$("connect-form").addEventListener("submit",async event=>{
  event.preventDefault();const code=$("access-code").value.trim();if(!code)return;
  const button=$("connect-submit");button.disabled=true;button.textContent="CONNECTING…";$("connect-error").textContent="";
  try{const runtime=await request("/api/runtime",{key:code});rememberAccess(code);updateConnection(runtime);$("access-code").value="";$("connect-dialog").close();notice("Compute connected. New simulations use CPU; GPU perception is on demand.");loadRemoteView();const pending=state.pending;state.pending=null;if(pending)await pending();}
  catch(error){$("connect-error").textContent=error.message;}
  finally{button.disabled=false;button.textContent="CONNECT COMPUTE";}
});
$("disconnect-button").addEventListener("click",()=>disconnect());

function setView(view){if(!["mission","perception","evaluation","edge","deployment","architecture"].includes(view))view="mission";state.view=view;document.querySelectorAll(".workspace-view").forEach(section=>section.hidden=section.id!==`view-${view}`);document.querySelectorAll("[data-view]").forEach(button=>{const active=button.dataset.view===view;button.classList.toggle("active",active);if(active)button.setAttribute("aria-current","page");else button.removeAttribute("aria-current");});history.replaceState(null,"",`#${view}`);if(view!=="mission")pause();else{renderFrame();renderAnalysis();}loadRemoteView();}
document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>setView(button.dataset.view)));
window.addEventListener("hashchange",()=>setView(location.hash.slice(1)));

function fillForm(config){const form=$("mission-form");for(const input of form.elements){if(!(input.name in config))continue;if(input.type==="checkbox")input.checked=Boolean(config[input.name]);else input.value=config[input.name];}updateRangeOutputs();}
function updateRangeOutputs(){document.querySelectorAll("#mission-form input[type=range]").forEach(input=>{const output=input.closest("label").querySelector("output");output.textContent=`${Number(input.value)}${input.dataset.unit?` ${input.dataset.unit}`:""}`;});}
$("mission-form").addEventListener("input",updateRangeOutputs);
function formConfig(){const form=$("mission-form"),body={};for(const input of form.elements){if(!input.name)continue;if(input.type==="checkbox")body[input.name]=input.checked;else if(["number","range"].includes(input.type)||input.name==="duration")body[input.name]=Number(input.value);else body[input.name]=input.value;}return body;}
function metric(id,value,digits,unit){const element=$(id);element.replaceChildren(document.createTextNode(fmt(value,digits)),el("em",unit));}
function setRun(run,{fill=true,play=false}={}){
  pause();state.run=run;state.cursor=run.samples.length-1;state.comparison=null;
  $("comparison-results").replaceChildren();
  $("run-source").textContent=run.source==="bundled_reference"?"REFERENCE EXPERIMENT":"COMPUTED EXPERIMENT";
  $("run-summary").textContent=`${run.config.scenario} / Seed ${run.config.seed} / ${run.config.duration} s`;
  metric("metric-error",run.metrics.estimation_rmse,2,"m");metric("metric-availability",run.metrics.detection_availability,1,"%");metric("metric-standoff",run.metrics.percent_time_inside_standoff_band,1,"%");metric("metric-travel",run.metrics.total_uav_travel_distance,1,"m");
  $("replay-seek").max=run.samples.length-1;$("scene-loading").hidden=true;if(fill)fillForm(run.config);
  renderFrame();renderAnalysis();if(play){state.cursor=0;startPlayback();}
}
$("mission-form").addEventListener("submit",event=>{event.preventDefault();if(state.running)return;const config=formConfig();requireConnection(()=>busy($("run-button"),"RUNNING EXPERIMENT…",async()=>{state.running=true;$("run-top").disabled=true;$("run-top").textContent="RUNNING…";try{notice("Computing your seeded simulation…");const run=await request("/api/simulate",{method:"POST",body:config});setRun(run,{play:!matchMedia("(prefers-reduced-motion: reduce)").matches});notice("Experiment completed and saved. Replay and telemetry follow the selected frame.");}finally{state.running=false;$("run-top").disabled=false;$("run-top").textContent="RUN SIMULATION";}}));});
$("reference-button").addEventListener("click",()=>{if(state.reference){setRun(state.reference);notice("Loaded the bundled reference experiment. No compute was allocated.");}});

function renderFrame(){if(!state.run||state.view!=="mission")return;const row=state.run.samples[state.cursor];if(!row)return;$("replay-seek").value=state.cursor;$("scene-time").textContent=`T+${fmt(row.time,1)} s`;$("replay-clock").textContent=`${fmt(row.time,1)} / ${fmt(state.run.config.duration,1)} s`;$("telemetry-state").textContent=row.fsm_state;$("telemetry-contact").textContent=row.detection_valid?"SENSOR CONTACT / VALID":"SENSOR CONTACT / DROPOUT";$("telemetry-confidence").textContent=fmt(row.confidence*100,0,"%");$("confidence-fill").style.width=`${Math.max(0,Math.min(100,row.confidence*100))}%`;$("telemetry-range").textContent=fmt(row.range_to_target,1," m");$("telemetry-bearing").textContent=fmt(row.bearing_deg,1,"°");$("telemetry-speed").textContent=fmt(row.uav_speed,1," m/s");$("telemetry-uncertainty").textContent=fmt(row.uncertainty,2);$("telemetry-vx").textContent=fmt(row.estimated_vx,2," m/s");$("telemetry-vy").textContent=fmt(row.estimated_vy,2," m/s");$("telemetry-nodes").textContent=String(row.path_nodes);
  const changes=state.run.samples.slice(0,state.cursor+1).filter((r,i,rows)=>!i||r.fsm_state!==rows[i-1].fsm_state).slice(-5).reverse();$("event-log").replaceChildren(...changes.map(r=>{const li=el("li");li.append(el("span",`${fmt(r.time,1)} s`),el("b",r.fsm_state));return li;}));
  if(state.dimension==="2d")drawMap($("mission-canvas"),state.run,state.cursor,state.layers);else draw3D($("scene-3d"),state.run,state.cursor,state.layers);
}
function pause(){state.playing=false;$("play-button").textContent="PLAY";}
function tick(timestamp){if(!state.playing)return;const dt=state.run.config.dt||.1;const speed=Number($("replay-speed").value);if(!state.lastTick)state.lastTick=timestamp;const elapsed=timestamp-state.lastTick;if(elapsed>=dt*1000/speed){const advance=Math.max(1,Math.floor(elapsed/(dt*1000/speed)));state.cursor=Math.min(state.run.samples.length-1,state.cursor+advance);state.lastTick=timestamp;renderFrame();if(state.cursor===state.run.samples.length-1){pause();return;}}requestAnimationFrame(tick);}
function startPlayback(){if(!state.run||state.playing)return;if(state.cursor>=state.run.samples.length-1)state.cursor=0;state.playing=true;state.lastTick=0;$("play-button").textContent="PAUSE";renderFrame();requestAnimationFrame(tick);}
$("play-button").addEventListener("click",()=>state.playing?pause():startPlayback());
$("rewind-button").addEventListener("click",()=>{pause();state.cursor=0;renderFrame();});
$("replay-seek").addEventListener("input",event=>{pause();state.cursor=Number(event.target.value);renderFrame();});
document.querySelectorAll("[data-layer]").forEach(input=>input.addEventListener("change",()=>{state.layers[input.dataset.layer]=input.checked;renderFrame();}));
function dimension(mode){state.dimension=mode;$("mission-canvas").hidden=mode!=="2d";$("scene-3d").hidden=mode!=="3d";["2d","3d"].forEach(key=>{$(`mode-${key}`).classList.toggle("active",key===mode);$(`mode-${key}`).setAttribute("aria-pressed",String(key===mode));});renderFrame();}
$("mode-2d").addEventListener("click",()=>dimension("2d"));$("mode-3d").addEventListener("click",()=>dimension("3d"));
new ResizeObserver(()=>{if(state.run)renderFrame();}).observe($("mission-canvas").parentElement);
window.addEventListener("resize",()=>{if(window.Plotly){[$("analysis-chart"),$("scene-3d")].forEach(element=>{if(!element.hidden&&element.data)Plotly.Plots.resize(element);});}});
document.addEventListener("visibilitychange",()=>{if(document.hidden)pause();});
document.addEventListener("keydown",event=>{if(state.view!=="mission"||event.target.closest("input,select,textarea,button,dialog"))return;if(event.code==="Space"){event.preventDefault();state.playing?pause():startPlayback();}});

function renderAnalysis(){if(!state.run)return;const comparison=state.analysis==="comparison";$("analysis-chart").hidden=comparison;$("comparison-panel").hidden=!comparison;if(!comparison)drawAnalysis($("analysis-chart"),state.run,state.analysis);}
document.querySelectorAll("[data-analysis]").forEach(button=>button.addEventListener("click",()=>{state.analysis=button.dataset.analysis;document.querySelectorAll("[data-analysis]").forEach(item=>item.classList.toggle("active",item===button));renderAnalysis();}));
$("compare-button").addEventListener("click",()=>requireConnection(()=>busy($("compare-button"),"COMPUTING BOTH MODES…",async()=>{const config={};const allowed=Object.keys(formConfig());allowed.forEach(key=>config[key]=state.run.config[key]);const comparison=await request("/api/compare",{method:"POST",body:config});state.comparison=comparison;const metrics=[["Estimation RMSE","estimation_rmse"," m"],["Standoff RMSE","standoff_rmse"," m"],["Detection availability","detection_availability","%"],["Time in standoff band","percent_time_inside_standoff_band","%"],["UAV travel","total_uav_travel_distance"," m"],["Reacquisition time","mean_reacquisition_time"," s"]];$("comparison-results").replaceChildren(table(["MEASURED METRIC",...comparison.runs.map(run=>run.config.mode)],metrics.map(([label,key,unit])=>[label,...comparison.runs.map(run=>fmt(run.metrics[key],2,unit))])));notice("Controlled comparison completed. Only the operating mode differs.");})));
$("export-json").addEventListener("click",()=>{if(state.run)download(`uav-experiment-${state.run.run_id}.json`,JSON.stringify(state.run,null,2),"application/json");});
$("export-csv").addEventListener("click",()=>{if(state.run)download(`uav-telemetry-${state.run.run_id}.csv`,toCSV(state.run.samples),"text/csv");});
$("history-button").addEventListener("click",()=>requireConnection(()=>busy($("history-button"),"LOADING…",async()=>{const runs=await request("/api/runs");$("history-content").replaceChildren();if(!runs.length)$("history-content").append(el("p","No cloud experiments saved yet. Run a simulation to create one."));for(const run of runs){const button=el("button",undefined,"history-entry");button.append(el("strong",run.config.scenario),el("span",`${run.config.mode} / Seed ${run.config.seed} / ${new Date(run.created_at).toLocaleString()}`));button.addEventListener("click",()=>busy(button,"LOADING EXPERIMENT…",async()=>{const loaded=await request(`/api/runs/${run.run_id}`);setRun(loaded);$("history-dialog").close();}));$("history-content").append(button);}$("history-dialog").showModal();})));

async function loadModels(){const models=await request("/api/models");$("model-select").replaceChildren();if(!models.length)$("model-select").append(new Option("No model weights available",""));for(const model of models)$("model-select").append(new Option(`${model.name} / ${model.format}`,model.id));}
async function loadEvaluations(){const evaluations=await request("/api/evaluations");const container=$("evaluation-content");container.replaceChildren();if(!evaluations.length){container.classList.add("empty-state");container.append(el("h3","NOT MEASURED"),el("p","No labeled evaluation artifacts are stored. Run training.evaluate with the detector and labeled validation data to generate real accuracy metrics."));return;}container.classList.remove("empty-state");for(const evaluation of evaluations){const title=el("div",undefined,"panel-heading");title.append(el("h3",evaluation.name));container.append(title);const keys=[["mAP50","map50"],["mAP50–95","map50_95"],["Precision","precision"],["Recall","recall"],["F1","f1"]];container.append(table(keys.map(k=>k[0]),[keys.map(k=>fmt(evaluation.metrics[k[1]],3))]));const summary=el("details");summary.append(el("summary","MEASURED METADATA"));const pre=el("pre",JSON.stringify(evaluation.metrics,null,2));pre.style.cssText="white-space:pre-wrap;padding:20px;font-size:12px;overflow-wrap:anywhere";summary.append(pre);container.append(summary);}}
function renderEdgeDeployment(){
  const target=$("edge-target").value;
  const existing=target==="jetson-orin-nx";
  $("edge-target-note").textContent=existing?"Existing ONNX / TensorRT command-line workflow.":"Target listed for selection. Hardware packaging integration is pending.";
  $("edge-jetson-commands").hidden=!existing;
  const artifacts=$("edge-artifacts");
  if(state.edge){
    const names={PT:"PyTorch weights",ONNX:"Portable ONNX",ENGINE:"TensorRT engine"};
    artifacts.replaceChildren(table(["ARTIFACT","STATUS","FILES"],state.edge.artifacts.map(item=>[names[item.format]||item.format,item.count?"AVAILABLE":"MISSING",String(item.count)])));
    const note=el("p","Inventory shows stored files. An engine must be validated on its target device.","edge-artifact-files");
    artifacts.append(note);
  }else{
    const empty=el("div",undefined,"edge-empty");
    empty.append(el("p",state.connected?"Loading model artifact inventory…":"Connect compute to inspect PyTorch, ONNX, and TensorRT artifacts."));
    if(!state.connected){const button=el("button","CONNECT COMPUTE","outline");button.addEventListener("click",openConnection);empty.append(button);}
    artifacts.replaceChildren(empty);
  }
  const container=$("edge-benchmark");
  const benchmark=state.edge?.benchmark_target===target?state.edge.benchmark:null;
  if(benchmark){
    const metrics=el("div",undefined,"edge-benchmark-metrics");
    for(const [label,value] of [["P50 LATENCY",fmt(benchmark.p50_ms,1," ms")],["P95 LATENCY",fmt(benchmark.p95_ms,1," ms")],["THROUGHPUT",fmt(benchmark.fps,1," FPS")]]){const item=el("div");item.append(el("span",label),el("strong",value));metrics.append(item);}
    const details=el("details",undefined,"edge-report");details.append(el("summary","MEASURED REPORT"),el("pre",JSON.stringify(benchmark,null,2)));
    container.replaceChildren(metrics,details);
  }else{
    const empty=el("div",undefined,"edge-empty");empty.append(el("h3","NO BENCHMARK LOADED"));
    empty.append(el("p",!existing?"No device measurements are loaded for this target.":state.connected?"Run the existing benchmark on the target device and sync its report to lab compute. Latency, FPS, power, and temperature remain unmeasured here.":"Connect compute to inspect stored device measurements."));
    container.replaceChildren(empty);
  }
}
$("edge-target").addEventListener("change",renderEdgeDeployment);
async function loadEdgeDeployment(){renderEdgeDeployment();const status=await request("/api/edge-deployment");if(!state.connected)return;state.edge=status;renderEdgeDeployment();}
async function loadRemoteView(){if(!state.connected)return;try{if(state.view==="perception")await loadModels();if(state.view==="evaluation")await loadEvaluations();if(state.view==="edge")await loadEdgeDeployment();}catch(error){errorMessage(error);}}
$("media-file").addEventListener("change",event=>{$("upload-name").textContent=event.target.files[0]?.name||"JPG, PNG, WEBP, MP4, MOV, AVI · 200 MB max.";});
$("perception-form").addEventListener("submit",event=>{event.preventDefault();const file=$("media-file").files[0];if(!file)return;if(file.size>200*1024*1024){notice("Please choose a file smaller than 200 MB.",true);return;}requireConnection(()=>busy($("perception-button"),"RUNNING L40S INFERENCE…",async()=>{if(!$("model-select").value)await loadModels();const form=new FormData($("perception-form"));form.set("tracking",String($("perception-form").elements.tracking.checked));if(!form.get("model_id"))throw new Error("No detector is available. Add model weights before running perception.");notice("Starting GPU perception. The first request may take longer while compute initializes.");const result=await request("/api/perception",{method:"POST",body:form,timeout:600000});$("perception-empty").hidden=true;$("perception-preview").src=result.preview;$("perception-preview").hidden=false;const metrics=$("perception-metrics");metrics.hidden=false;metrics.replaceChildren();for(const [label,value] of [["ACTUAL GPU",result.gpu],["P50 INFERENCE",fmt(result.latency_p50_ms,1," ms")],["P95 INFERENCE",fmt(result.latency_p95_ms,1," ms")],["FRAMES PROCESSED",result.frames_processed]]){const item=el("div");item.append(el("span",label),el("strong",String(value)));metrics.append(item);}const rows=result.rows.map(row=>[row.class_name||"No detection",fmt(row.confidence,2),fmt(row.azimuth_deg,1,"°"),fmt(row.elevation_deg,1,"°"),row.track_id??"—",row.frame??0]);const holder=$("perception-table");holder.hidden=false;holder.replaceChildren(table(["CLASS","CONFIDENCE","AZIMUTH","ELEVATION","TRACK ID","FRAME"],rows));const button=el("button","EXPORT OBSERVATIONS / CSV","outline");button.style.margin="15px";button.addEventListener("click",()=>download("perception-observations.csv",toCSV(result.rows),"text/csv"));holder.append(button);notice(`Perception completed on ${result.gpu}. ${result.frames_processed} frame(s) processed. Monocular metric range is unavailable.`);}));});

async function initialize(){
  $("source-link").href=window.LAB_CONFIG.sourceUrl;
  updateRangeOutputs();
  renderEdgeDeployment();
  const requested=location.hash.slice(1);
  if(requested)setView(requested);
  try{const response=await fetch("./reference-run.json");if(!response.ok)throw new Error("Reference experiment could not be loaded.");state.reference=await response.json();setRun(state.reference);}
  catch(error){$("scene-loading").textContent="REFERENCE UNAVAILABLE";errorMessage(error);}
  if(hasAccess()){try{updateConnection(await request("/api/runtime"));loadRemoteView();}catch{disconnect(false);}}
}
initialize();
