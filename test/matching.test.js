import test from 'node:test';
import assert from 'node:assert/strict';
import { RoadGraph, projectSegment } from '../src/graph.js';
import { hmmMatch, sampleIndices, evaluate } from '../src/match.js';

test('segment projection clamps to endpoints and handles zero length', () => {
  assert.equal(projectSegment([12, 3], [0, 0], [10, 0]).fraction, 1);
  assert.equal(projectSegment([1, 0], [0, 0], [0, 0]).distance, 1);
});

test('HMM fills unobserved road arcs and respects direction', () => {
  const trip = { nodes: [[0, 0], [.001, 0], [.002, 0], [.003, 0]],
    arcs: [[0, 1], [1, 2], [2, 3]], track: [[.0001, .00001, 1], [.0029, -.00001, 2]] };
  const graph = new RoadGraph(trip);
  const result = hmmMatch(graph, [0, 1], { candidates: 1 });
  assert.deepEqual(result.route, [0, 1, 2]);
  assert.equal(evaluate(graph, result, [0, 1, 2]).directed.f1, 1);
  assert.equal(result.gaps.length, 0);
  assert.equal(graph.shortestPaths(3, [0], 1000).distances.has(0), false);
});

test('disconnected roads create an explicit restart', () => {
  const graph = new RoadGraph({ nodes: [[0, 0], [.001, 0], [.01, 0], [.011, 0]],
    arcs: [[0, 1], [2, 3]], track: [[.0001, 0, 1], [.0109, 0, 2]] });
  const result = hmmMatch(graph, [0, 1], { candidates: 1 });
  assert.equal(result.gaps.length, 1);
  assert.equal(evaluate(graph, result, [0, 1]).disconnectedSteps, 1);
});

test('sampling retains both endpoints', () => {
  assert.deepEqual(sampleIndices(12, 5), [0, 5, 10, 11]);
  assert.deepEqual(sampleIndices(1, 5), [0]);
});
