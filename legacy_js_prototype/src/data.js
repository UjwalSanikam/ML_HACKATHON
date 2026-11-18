import fs from 'node:fs';
import path from 'node:path';

export const DEFAULT_DATA = 'Student folder';
const rows = file => fs.readFileSync(file, 'utf8').trim().split(/\r?\n/)
  .filter(Boolean).map(line => line.trim().split(/\s+/).map(Number));

export function tripIds(root, split) {
  return fs.readdirSync(path.join(root, split), { withFileTypes: true })
    .filter(e => e.isDirectory()).map(e => e.name).sort();
}

// References are deliberately loaded separately from prediction inputs.
export function loadTrip(root, split, id) {
  const prefix = path.join(root, split, id, id);
  const nodes = rows(prefix + '.nodes');
  const arcs = rows(prefix + '.arcs');
  const track = rows(prefix + '.track');
  if (!nodes.length || !arcs.length || !track.length) throw new Error(`Empty trip: ${id}`);
  if (nodes.some(r => r.length !== 2 || r.some(x => !Number.isFinite(x))) ||
      arcs.some(r => r.length !== 2 || r.some(x => !Number.isInteger(x) || x < 0 || x >= nodes.length)) ||
      track.some(r => r.length !== 3 || r.some(x => !Number.isFinite(x)))) {
    throw new Error(`Invalid graph or track in ${id}`);
  }
  return { id, nodes, arcs, track };
}

export function loadReference(root, split, id) {
  return rows(path.join(root, split, id, id + '.route')).map(r => r[0]);
}

export function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n');
}

export function partition(ids) {
  // Fixed trip-level holdout. Never split individual points from the same trip.
  return { development: ids.filter((_, i) => i % 5 !== 0), validation: ids.filter((_, i) => i % 5 === 0) };
}
