// PathSnap evaluator UI (Laptop 2). Talks to the FastAPI backend only; holds no model or labels.
const $ = (id) => document.getElementById(id);
const fmtPct = (v) => (v === null || v === undefined || isNaN(v)) ? "–" : (100 * v).toFixed(1) + "%";
const params = new URLSearchParams(location.search);
let API = params.get("api") || safeGet("pathsnap_api") || `http://${location.hostname || "localhost"}:8000`;
$("api").value = API;
let trips = [];
let current = null;

function safeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

// ---------------- map ----------------
const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([20, 0], 2);
// OpenStreetMap tiles (reachable on networks that block other tile CDNs); the dark
// style is the same tiles with a CSS filter.
const OSM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const dark = L.tileLayer(OSM, { attribution: "&copy; OpenStreetMap", maxZoom: 19, className: "tiles-dark" }).addTo(map);
const light = L.tileLayer(OSM, { attribution: "&copy; OpenStreetMap", maxZoom: 19 });
const layers = {
  gps: L.layerGroup().addTo(map),
  matched: L.layerGroup(),
  clf: L.layerGroup(),
  route: L.layerGroup().addTo(map),
  ends: L.layerGroup().addTo(map),
};
L.control.layers({ "Dark": dark, "OpenStreetMap": light }, {
  "Raw GPS observations": layers.gps,
  "Matched positions (HMM)": layers.matched,
  "Classifier-only route": layers.clf,
  "Predicted route (classifier + HMM)": layers.route,
  "Start / end": layers.ends,
}, { collapsed: true }).addTo(map);
const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.innerHTML = `<i style="background:#4aa3ff;height:8px;width:8px;border-radius:50%"></i>Raw GPS<br>
    <i style="background:#ff4f8b"></i>Predicted route (classifier + HMM)<br>
    <i style="background:#ffa24a"></i>Classifier-only route<br>
    <i style="background:#3ccf91;height:8px;width:8px;border-radius:50%"></i>Matched positions`;
  return d;
};
legend.addTo(map);

// ---------------- logging ----------------
function logLine(s) {
  const t = new Date().toLocaleTimeString();
  $("log").textContent = `[${t}] ${s}\n` + $("log").textContent;
}
async function api(path, opts = {}) {
  const t0 = performance.now();
  const r = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...opts });
  const ms = (performance.now() - t0).toFixed(0);
  logLine(`${opts.method || "GET"} ${path} -> ${r.status} (${ms} ms)`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

// ---------------- connect ----------------
async function connect() {
  API = $("api").value.trim().replace(/\/$/, "");
  safeSet("pathsnap_api", API);
  $("status").className = "pill off"; $("status").textContent = "connecting…";
  try {
    const h = await api("/health");
    $("status").className = "pill on"; $("status").textContent = `connected · ${h.host}`;
    const m = await api("/model");
    $("modelinfo").innerHTML = `<b>${m.model}</b> on ${m.features.length} features<br>
      Bayesian opt.: ${m.bayes_opt.n_trials ?? "–"} trials (${m.bayes_opt.sampler ?? ""})<br>
      HMM β=${(+m.hmm.beta).toFixed(2)} m, emission w=${(+m.hmm.emission_weight).toFixed(2)}`;
    const t = await api("/trips");
    trips = t.trips;
    $("partition").textContent = `${t.partition} · ${t.count}`;
    renderTrips();
    $("runall").disabled = false;
  } catch (e) {
    $("status").className = "pill off"; $("status").textContent = "unreachable";
    logLine("ERROR " + e.message);
  }
}
$("connect").onclick = connect;
$("api").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });

function renderTrips() {
  $("trips").innerHTML = "";
  trips.forEach((t) => {
    const d = document.createElement("div");
    d.className = "trip" + (current === t.trip_id ? " active" : "");
    d.innerHTML = `<div class="id">${t.trip_id}</div>
      <div class="meta"><span>${t.n_obs.toLocaleString()} obs · ${(t.duration_s / 60).toFixed(0)} min</span>
      <span class="score">${t.score !== undefined ? fmtPct(t.score) : ""}</span></div>`;
    d.onclick = () => runTrip(t.trip_id);
    $("trips").appendChild(d);
  });
}

// ---------------- prediction ----------------
function busy(on, text) { $("busy").classList.toggle("hidden", !on); if (text) $("busytext").textContent = text; }

async function predict(tripId) {
  return api("/predict", { method: "POST", body: JSON.stringify({ trip_id: tripId, include_classifier_only: true }) });
}

async function runTrip(tripId) {
  current = tripId; renderTrips();
  busy(true, `Matching trip ${tripId}…`);
  try {
    const r = await predict(tripId);
    draw(r); showMetrics(r);
    const t = trips.find((x) => x.trip_id === tripId); if (t) t.score = r.metrics.hmm_top1_acc;
    renderTrips();
  } catch (e) { logLine("ERROR " + e.message); alert(e.message); }
  busy(false);
}

function draw(r) {
  Object.values(layers).forEach((l) => l.clearLayers());
  r.gps.forEach((p) => L.circleMarker(p, { radius: 2.2, color: "#4aa3ff", weight: 0, fillOpacity: .75 }).addTo(layers.gps));
  r.matched_points.forEach((p) => L.circleMarker(p, { radius: 1.8, color: "#3ccf91", weight: 0, fillOpacity: .9 }).addTo(layers.matched));
  (r.classifier_only_route || []).forEach((line) => L.polyline(line, { color: "#ffa24a", weight: 3, opacity: .85, dashArray: "6 5" }).addTo(layers.clf));
  r.route.forEach((line) => L.polyline(line, { color: "#ff4f8b", weight: 4.5, opacity: .95 }).addTo(layers.route));
  const s = r.gps[0], e = r.gps[r.gps.length - 1];
  L.circleMarker(s, { radius: 7, color: "#fff", fillColor: "#3ccf91", fillOpacity: 1, weight: 2 }).bindTooltip("Start").addTo(layers.ends);
  L.circleMarker(e, { radius: 7, color: "#fff", fillColor: "#ff6b6b", fillOpacity: 1, weight: 2 }).bindTooltip("End").addTo(layers.ends);
  map.fitBounds(L.latLngBounds(r.gps), { padding: [30, 30] });
}

function bar(label, v, color) {
  const w = Math.max(0, Math.min(100, 100 * (v || 0)));
  return `<div class="bar"><div class="lbl"><span>${label}</span><span>${fmtPct(v)}</span></div>
    <div class="track"><div class="fill" style="width:${w}%;background:${color}"></div></div></div>`;
}

function showMetrics(r) {
  const m = r.metrics;
  $("tripname").textContent = r.trip_id;
  $("k_acc").textContent = fmtPct(m.hmm_top1_acc);
  $("k_f1").textContent = fmtPct(m.hmm_route_f1_len);
  $("k_ord").textContent = fmtPct(m.hmm_route_ordered_f1);
  $("k_cont").textContent = fmtPct(m.hmm_continuity);
  $("compare").innerHTML =
    bar("Observation accuracy · classifier", m.clf_top1_acc, "#ffa24a") +
    bar("Observation accuracy · + HMM", m.hmm_top1_acc, "#ff4f8b") +
    bar("Route F1 · classifier", m.clf_route_f1_len, "#ffa24a") +
    bar("Route F1 · + HMM", m.hmm_route_f1_len, "#ff4f8b") +
    bar("Continuity · classifier", m.clf_continuity, "#ffa24a") +
    bar("Continuity · + HMM", m.hmm_continuity, "#ff4f8b");
  const t = r.timings_ms, total = t.total || 1;
  const parts = [["load", "#8b98a5"], ["network", "#4aa3ff"], ["features", "#3ccf91"], ["classifier", "#ffa24a"], ["hmm", "#ff4f8b"]];
  $("timing").innerHTML = `<div class="bar"><div class="lbl"><span>total</span><span>${(total / 1000).toFixed(2)} s</span></div>
    <div class="track">${parts.map(([k, c]) => `<div class="fill" title="${k}" style="width:${100 * (t[k] || 0) / total}%;background:${c}"></div>`).join("")}</div></div>
    <div class="small">${parts.map(([k, c]) => `<span style="color:${c}">■</span> ${k} ${((t[k] || 0) / 1000).toFixed(2)}s`).join(" &nbsp; ")}</div>`;
  const rows = [
    ["GPS observations", r.n_observations.toLocaleString()],
    ["Observations scored", (m.n_scored ?? 0).toLocaleString()],
    ["Reference coverage", fmtPct(m.label_coverage)],
    ["Candidate miss rate", fmtPct(m.candidate_miss_rate)],
    ["Route arcs predicted", r.n_route_arcs],
    ["Route precision / recall", `${fmtPct(m.hmm_route_precision_len)} / ${fmtPct(m.hmm_route_recall_len)}`],
    ["Viterbi segments", r.segments],
    ["Unbridged route gaps", r.route_gaps],
  ];
  $("details").innerHTML = rows.map(([a, b]) => `<tr><td class="muted">${a}</td><td>${b}</td></tr>`).join("");
}

// ---------------- batch ----------------
$("runall").onclick = async () => {
  $("runall").disabled = true;
  const body = $("batch").querySelector("tbody");
  body.innerHTML = "";
  const res = [];
  for (const [i, t] of trips.entries()) {
    busy(true, `Evaluating ${i + 1}/${trips.length}: ${t.trip_id}…`);
    try {
      const r = await predict(t.trip_id);
      res.push(r); t.score = r.metrics.hmm_top1_acc; current = t.trip_id;
      draw(r); showMetrics(r); renderTrips();
      const m = r.metrics;
      body.insertAdjacentHTML("beforeend", `<tr><td>${t.trip_id}</td><td>${r.n_observations}</td>
        <td>${fmtPct(m.clf_top1_acc)}</td><td>${fmtPct(m.hmm_top1_acc)}</td><td>${fmtPct(m.hmm_route_f1_len)}</td>
        <td>${fmtPct(m.hmm_route_ordered_f1)}</td><td>${fmtPct(m.hmm_continuity)}</td><td>${r.segments}</td>
        <td>${r.prediction_seconds.toFixed(2)}</td></tr>`);
    } catch (e) { logLine("ERROR " + e.message); }
  }
  const mean = (f) => res.reduce((s, r) => s + (f(r) || 0), 0) / Math.max(1, res.length);
  const tf = document.createElement("tfoot");
  tf.innerHTML = `<tr><td>Mean (${res.length})</td><td>${Math.round(mean((r) => r.n_observations))}</td>
    <td>${fmtPct(mean((r) => r.metrics.clf_top1_acc))}</td><td>${fmtPct(mean((r) => r.metrics.hmm_top1_acc))}</td>
    <td>${fmtPct(mean((r) => r.metrics.hmm_route_f1_len))}</td><td>${fmtPct(mean((r) => r.metrics.hmm_route_ordered_f1))}</td>
    <td>${fmtPct(mean((r) => r.metrics.hmm_continuity))}</td><td>–</td><td>${mean((r) => r.prediction_seconds).toFixed(2)}</td></tr>`;
  $("batch").querySelector("tfoot")?.remove();
  $("batch").appendChild(tf);
  busy(false); $("runall").disabled = false;
};

document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $("tab-batch").classList.toggle("hidden", b.dataset.tab !== "batch");
  $("tab-log").classList.toggle("hidden", b.dataset.tab !== "log");
});

connect();
