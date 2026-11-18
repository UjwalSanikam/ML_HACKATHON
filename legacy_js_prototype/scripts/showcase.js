import fs from 'node:fs';
import path from 'node:path';
import { RoadGraph } from '../src/graph.js';
import { evaluate, hmmMatch, nearestRoad, sampleIndices } from '../src/match.js';
import { writeReport } from '../src/report.js';

// Fictional parallel streets separated by 24 m, with intersections every 60 m.
const nodes = [], arcs = [], reference = [];
const rowCount = 4, columns = 8, degreesPerMeter = 180 / (Math.PI * 6371000);
function connect(from, to) {
  arcs.push([from, to], [to, from]);
}
for (let row = 0; row < rowCount; row++) {
  for (let column = 0; column < columns; column++) {
    const id = nodes.length;
    nodes.push([column * 60 * degreesPerMeter, row * 24 * degreesPerMeter]);
    if (column) connect(id - 1, id);
    if (row) connect(id - columns, id);
  }
}
const stops = [8, 9, 10, 11, 19, 20, 21, 22, 30, 31];
const track = [];
for (let step = 1; step < stops.length; step++) {
  const from = stops[step - 1], to = stops[step];
  reference.push(arcs.findIndex(([u, v]) => u === from && v === to));
  for (let sample = 0; sample < 12; sample++) {
    const k = track.length, fraction = sample / 12;
    const eastNoise = 3 * Math.sin(k * 0.73);
    const northNoise = 3 * Math.cos(k * 0.58) + (k >= 14 && k <= 26 ? 15 : 0);
    track.push([nodes[from][0] + fraction * (nodes[to][0] - nodes[from][0]) + eastNoise * degreesPerMeter,
      nodes[from][1] + fraction * (nodes[to][1] - nodes[from][1]) + northNoise * degreesPerMeter, k]);
  }
}
track.push([...nodes[stops.at(-1)], track.length]);
const trip = { id: 'Synthetic parallel streets', nodes, arcs, track };
const graph = new RoadGraph(trip), indices = sampleIndices(track.length, 5);
const results = [];
for (const [method, match] of [['baseline', nearestRoad], ['hmm', hmmMatch]]) {
  const start = performance.now(), prediction = match(graph, indices);
  const seconds = (performance.now() - start) / 1000;
  results.push({ method, prediction, seconds, metrics: evaluate(graph, prediction, reference) });
}
const output = process.argv[2] || 'outputs/showcase.html';
fs.mkdirSync(path.dirname(output), { recursive: true });
writeReport(output, trip, graph, reference, results);
console.log('Open ' + path.resolve(output));
