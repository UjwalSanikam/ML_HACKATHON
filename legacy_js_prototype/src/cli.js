import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { DEFAULT_DATA, tripIds, loadTrip, loadReference, writeJson, partition } from './data.js';
import { RoadGraph } from './graph.js';
import { sampleIndices, nearestRoad, hmmMatch, evaluate } from './match.js';
import { writeReport } from './report.js';

const [command = 'help', ...args] = process.argv.slice(2), options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!args[i].startsWith('--') || args[i + 1] === undefined) throw new Error('Options require --name value');
  options[args[i].slice(2)] = args[i + 1];
}
const root = options.data || DEFAULT_DATA, out = options.out || 'outputs';
const stride = Number(options.stride || 5);
if (!Number.isInteger(stride) || stride < 1) throw new Error('--stride must be a positive integer');
const parameters = { sigma: Number(options.sigma || 20), beta: Number(options.beta || 30), candidates: Number(options.candidates || 8), jitter: Number(options.jitter ?? 15) };
if (!Number.isFinite(parameters.jitter) || parameters.jitter < 0) throw new Error('--jitter must be a nonnegative distance in meters');
if (!(parameters.sigma > 0 && parameters.beta > 0 && Number.isInteger(parameters.candidates) && parameters.candidates > 0)) {
  throw new Error('sigma and beta must be positive; candidates must be a positive integer');
}

function runTrip(split, id, withReference) {
  const start = performance.now(), trip = loadTrip(root, split, id), graph = new RoadGraph(trip);
  const graphSeconds = (performance.now() - start) / 1000;
  const indices = sampleIndices(trip.track.length, stride), results = [];
  for (const [method, match] of [['baseline', () => nearestRoad(graph, indices)], ['hmm', () => hmmMatch(graph, indices, parameters)]]) {
    const begin = performance.now(), prediction = match();
    results.push({ method, prediction, seconds: (performance.now() - begin) / 1000 });
  }
  // Evaluation happens after both matchers have completed; no reference is passed to either.
  const reference = withReference ? loadReference(root, split, id) : null;
  for (const r of results) {
    if (reference) r.metrics = evaluate(graph, r.prediction, reference);
    writeJson(path.join(out, split, id, r.method + '.json'), {
      tripId: id, method: r.method, stride, parameters: r.method === 'hmm' ? parameters : {},
      sampledTrackIndices: indices, matchedArcIds: r.prediction.matched.map(c => c.id),
      routeArcIds: r.prediction.route, gaps: r.prediction.gaps,
      matchSeconds: r.seconds, graphSeconds, metrics: r.metrics,
    });
  }
  if (reference) writeReport(path.join(out, split, id, 'report.html'), trip, graph, reference, results);
  const summary = { id, split, graphSeconds, totalSeconds: (performance.now() - start) / 1000,
    gpsPoints: trip.track.length, sampledPoints: indices.length,
    results: results.map(({ method, seconds, metrics, prediction }) => ({ method, seconds, metrics, restartCount: prediction.gaps.length })) };
  console.log(`${id}: ` + summary.results.map(r => `${r.method} ${r.seconds.toFixed(2)}s` +
    (r.metrics ? ` directed F1=${r.metrics.directed.f1.toFixed(3)}, physical F1=${r.metrics.physicalRoad.f1.toFixed(3)}, breaks=${r.metrics.disconnectedSteps}` : '')).join(' | '));
  return summary;
}

if (command === 'audit') {
  const splits = {};
  for (const split of ['train', 'student_test']) {
    const trips = [];
    for (const id of tripIds(root, split)) {
      const trip = loadTrip(root, split, id);
      // Student-test references remain unopened until organizers explain their role.
      const reference = split === 'train' ? loadReference(root, split, id) : null;
      const invalidReferenceIds = reference?.filter(i => !Number.isInteger(i) || !trip.arcs[i]).length || 0;
      const breaks = reference?.slice(1).flatMap((arc, i) =>
        trip.arcs[reference[i]]?.[1] !== trip.arcs[arc]?.[0] ? [i + 1] : []) || [];
      trips.push({ id, nodes: trip.nodes.length, arcs: trip.arcs.length, gps: trip.track.length,
        referenceArcs: reference?.length ?? null, invalidReferenceIds, referenceBreakPositions: breaks,
        nonIncreasingTimes: trip.track.slice(1).filter((p, i) => p[2] <= trip.track[i][2]).length });
    }
    splits[split] = { tripCount: trips.length, totals: Object.fromEntries(['nodes', 'arcs', 'gps', 'invalidReferenceIds', 'nonIncreasingTimes']
      .map(key => [key, trips.reduce((sum, t) => sum + t[key], 0)])),
      referenceDisconnectedSteps: trips.reduce((sum, t) => sum + t.referenceBreakPositions.length, 0), trips };
  }
  writeJson(path.join(out, 'audit.json'), { coordinateOrder: ['longitude', 'latitude'],
    routeInterpretation: 'Zero-based directed arc row IDs; nodes are zero-based node row IDs',
    timestampUnit: 'Unconfirmed; matching uses observation order and geometric displacement', splits });
  writeJson(path.join(out, 'split.json'), partition(tripIds(root, 'train')));
  console.log(JSON.stringify(Object.fromEntries(Object.entries(splits).map(([k, v]) => [k, { ...v, trips: undefined }])), null, 2));
} else if (command === 'run') {
  const split = options.split || 'train';
  if (split !== 'train') throw new Error('Use predict for student_test; run is for labeled training data');
  runTrip(split, options.id || '00000046', true);
} else if (command === 'evaluate') {
  const subsets = partition(tripIds(root, 'train'));
  const subset = options.subset || 'validation';
  if (!subsets[subset]) throw new Error('--subset must be development or validation');
  const limit = Number(options.limit || subsets[subset].length);
  if (!Number.isInteger(limit) || limit < 1) throw new Error('--limit must be a positive integer');
  const trips = subsets[subset].slice(0, limit).map(id => runTrip('train', id, true));
  const means = ['baseline', 'hmm'].map(method => {
    const rows = trips.map(t => t.results.find(r => r.method === method));
    const mean = fn => rows.reduce((sum, r) => sum + fn(r), 0) / rows.length;
    return { method, macroDirectedF1: mean(r => r.metrics.directed.f1), macroPhysicalF1: mean(r => r.metrics.physicalRoad.f1),
      macroOrderedF1: mean(r => r.metrics.orderedF1), meanMatchSeconds: mean(r => r.seconds), totalDisconnectedSteps: rows.reduce((sum, r) => sum + r.metrics.disconnectedSteps, 0) };
  });
  writeJson(path.join(out, 'evaluation-' + subset + '.json'), { subset, stride, parameters, tripCount: trips.length, means, trips });
  console.log(JSON.stringify(means, null, 2));
} else if (command === 'predict') {
  const ids = options.id ? [options.id] : tripIds(root, 'student_test');
  for (const id of ids) runTrip('student_test', id, false);
} else {
  console.log('Commands: audit | run --id 00000046 | evaluate --subset validation | predict\nOptions: --data "Student folder" --out outputs --stride 5 --sigma 20 --beta 30 --candidates 8 --jitter 15\nEvaluation: --limit N. Predictions never load student_test .route files.');
}
