import { loadTrip, loadReference, partition, tripIds, writeJson } from '../src/data.js';
import { RoadGraph } from '../src/graph.js';
import { evaluate, hmmMatch, sampleIndices } from '../src/match.js';

// All candidates see the same trips. Only the published and selected versions
// run on validation. Ranking is by mean ordered F1 on development.
const root = process.argv[2] || 'Student folder';
const splits = partition(tripIds(root, 'train'));
const published = { stride: 5, candidates: 4, jitter: 0 };
const variants = [published,
  { stride: 5, candidates: 4, jitter: 15 },
  { stride: 5, candidates: 8, jitter: 15 },
  { stride: 3, candidates: 8, jitter: 15 },
];

function compare(ids, configs) {
  const summaries = configs.map(config => ({ config, orderedF1: 0, directedF1: 0, breaks: 0, seconds: 0 }));
  for (const id of ids) {
    const trip = loadTrip(root, 'train', id);
    const graph = new RoadGraph(trip);
    const reference = loadReference(root, 'train', id);
    for (const row of summaries) {
      const indices = sampleIndices(trip.track.length, row.config.stride);
      const start = performance.now();
      const prediction = hmmMatch(graph, indices, row.config);
      row.seconds += (performance.now() - start) / 1000;
      const metrics = evaluate(graph, prediction, reference);
      row.orderedF1 += metrics.orderedF1 / ids.length;
      row.directedF1 += metrics.directed.f1 / ids.length;
      row.breaks += metrics.disconnectedSteps;
    }
  }
  return summaries;
}
const development = compare(splits.development, variants);
const selected = development.reduce((a, b) => a.orderedF1 >= b.orderedF1 ? a : b).config;
console.table(development.map(row => ({ ...row.config, ...row, config: undefined })));
const validation = compare(splits.validation, [published, selected]);
console.table(validation.map(row => ({ ...row.config, ...row, config: undefined })));
writeJson('outputs/benchmark.json', { splits, development, selected, validation });
