import fs from 'node:fs';

export function writeReport(file, trip, graph, reference, results) {
  const data = { id: trip.id, nodes: graph.xy, arcs: trip.arcs, track: graph.trackXY, reference,
    results: results.map(r => ({ method: r.method, route: r.prediction.route, indices: r.prediction.indices,
      gaps: r.prediction.gaps, metrics: r.metrics, seconds: r.seconds })) };
  fs.writeFileSync(file, `<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPS route recovery — ${trip.id}</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101827;color:#e7edf7;font:15px system-ui}header{padding:24px 28px 12px}h1{margin:0 0 8px;font-size:26px}p{color:#aebdd3;margin:6px 0}main{padding:0 28px 24px} .controls{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}label{cursor:pointer}button{background:#23344d;color:white;border:1px solid #526681;border-radius:6px;padding:7px 12px;cursor:pointer}canvas{width:100%;height:60vh;min-height:360px;background:#0b1220;border:1px solid #31425c;border-radius:10px;touch-action:none}table{border-collapse:collapse;width:100%;margin-top:20px}th,td{text-align:left;padding:10px;border-bottom:1px solid #31425c}th{color:#aebdd3}small{color:#aebdd3}.key{width:10px;height:10px;display:inline-block;margin-right:6px;border-radius:50%}
</style>
<header><h1>Recover the route <span style="color:#56d8bf">/ ${trip.id}</span></h1>
<p>Noisy GPS → nearby roads → a route through the directed road network</p></header>
<main><div class="controls" id="controls"></div><canvas id="map"></canvas>
<p>Scroll to zoom · Drag to pan · Coordinates shown in local meters · No online maps required</p>
<div id="summary"></div><p>These are project-defined measures, not official hackathon scores. Segment overlap ignores traversal order and repeated visits.</p>
<p id="warnings"></p></main>
<script>
const data=${JSON.stringify(data).replaceAll('<', '\\u003c')};
const canvas=document.getElementById('map'),ctx=canvas.getContext('2d');
const colors={network:'#304159',gps:'#62aaff',reference:'#54dfaa',baseline:'#ffbc65',hmm:'#f28adc'};
const enabled={network:true,gps:true,reference:true,baseline:false,hmm:true};
let scale=1,ox=0,oy=0,drag=null;
const controls=document.getElementById('controls');
for(const name of Object.keys(enabled)){const label=document.createElement('label');label.innerHTML='<input type="checkbox" '+(enabled[name]?'checked':'')+'> <span class="key" style="background:'+colors[name]+'"></span>'+name;label.firstChild.onchange=e=>{enabled[name]=e.target.checked;draw()};controls.append(label)}
const reset=document.createElement('button');reset.textContent='Fit trip';reset.onclick=fit;controls.append(reset);
function fit(){let points=[...data.track];for(const id of data.reference){const a=data.arcs[id];points.push(data.nodes[a[0]],data.nodes[a[1]])}const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);scale=Math.min((canvas.clientWidth-60)/(maxX-minX||1),(canvas.clientHeight-60)/(maxY-minY||1));ox=canvas.clientWidth/2-(minX+maxX)/2*scale;oy=canvas.clientHeight/2+(minY+maxY)/2*scale;draw()}
function screen(p){return [p[0]*scale+ox,-p[1]*scale+oy]}
function roads(ids,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();for(const id of ids){const a=data.arcs[id],p=screen(data.nodes[a[0]]),q=screen(data.nodes[a[1]]);if(Math.max(p[0],q[0])<0||Math.min(p[0],q[0])>canvas.clientWidth||Math.max(p[1],q[1])<0||Math.min(p[1],q[1])>canvas.clientHeight)continue;ctx.moveTo(...p);ctx.lineTo(...q)}ctx.stroke()}
function draw(){const dpr=devicePixelRatio||1;const w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);if(enabled.network)roads(data.arcs.keys(),colors.network,1);if(enabled.reference)roads(data.reference,colors.reference,7);for(const r of data.results)if(enabled[r.method])roads(r.route,colors[r.method],r.method==='hmm'?3:2);if(enabled.gps){ctx.fillStyle=colors.gps;ctx.globalAlpha=.55;for(const p of data.track){const q=screen(p);ctx.beginPath();ctx.arc(...q,2,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1}}
canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,f=e.deltaY<0?1.15:1/1.15;ox=x+(ox-x)*f;oy=y+(oy-y)*f;scale*=f;draw()},{passive:false});
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;ox+=e.clientX-drag[0];oy+=e.clientY-drag[1];drag=[e.clientX,e.clientY];draw()};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
const pct=x=>(100*x).toFixed(1)+'%';
document.getElementById('summary').innerHTML='<table><tr><th>Method</th><th>Directed precision</th><th>Directed recall</th><th>Directed F1</th><th>Physical road F1</th><th>Broken steps</th><th>Match seconds</th></tr>'+data.results.map(r=>'<tr><td>'+r.method+'</td><td>'+pct(r.metrics.directed.precision)+'</td><td>'+pct(r.metrics.directed.recall)+'</td><td>'+pct(r.metrics.directed.f1)+'</td><td>'+pct(r.metrics.physicalRoad.f1)+'</td><td>'+r.metrics.disconnectedSteps+'</td><td>'+r.seconds.toFixed(2)+'</td></tr>').join('')+'</table>';
document.getElementById('warnings').textContent='Reference disconnected steps: '+data.results[0].metrics.referenceDisconnectedSteps+'. HMM route restarts: '+data.results.find(r=>r.method==='hmm').gaps.length+'. Reference data is loaded only after prediction.';
new ResizeObserver(fit).observe(canvas);
</script></html>`);
}
