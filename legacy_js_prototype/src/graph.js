export function projection(lon, lat, origin) {
  const rad = Math.PI / 180, radius = 6371000;
  return [(lon - origin[0]) * rad * radius * Math.cos(origin[1] * rad), (lat - origin[1]) * rad * radius];
}

export function projectSegment(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const fraction = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
  const point = [a[0] + fraction * dx, a[1] + fraction * dy];
  return { fraction, point, distance: Math.hypot(p[0] - point[0], p[1] - point[1]) };
}

export class MinHeap {
  items = [];
  push(item) {
    let i = this.items.length; this.items.push(item);
    while (i > 0) { const parent = (i - 1) >> 1; if (this.items[parent][0] <= item[0]) break;
      this.items[i] = this.items[parent]; i = parent; }
    this.items[i] = item;
  }
  pop() {
    const first = this.items[0], last = this.items.pop();
    if (this.items.length) {
      let i = 0;
      while (i * 2 + 1 < this.items.length) {
        let child = i * 2 + 1;
        if (child + 1 < this.items.length && this.items[child + 1][0] < this.items[child][0]) child++;
        if (last[0] <= this.items[child][0]) break;
        this.items[i] = this.items[child]; i = child;
      }
      this.items[i] = last;
    }
    return first;
  }
}

export class RoadGraph {
  constructor(trip, cellSize = 100) {
    this.origin = trip.nodes[0]; this.cellSize = cellSize; this.arcs = trip.arcs;
    this.xy = trip.nodes.map(p => projection(...p, this.origin));
    this.trackXY = trip.track.map(p => projection(p[0], p[1], this.origin));
    this.lengths = trip.arcs.map(([u, v]) => Math.hypot(this.xy[u][0] - this.xy[v][0], this.xy[u][1] - this.xy[v][1]));
    this.adjacency = Array.from({ length: trip.nodes.length }, () => []);
    this.grid = new Map();
    trip.arcs.forEach(([u, v], id) => {
      this.adjacency[u].push(id);
      const a = this.xy[u], b = this.xy[v];
      const x0 = Math.floor(Math.min(a[0], b[0]) / cellSize), x1 = Math.floor(Math.max(a[0], b[0]) / cellSize);
      const y0 = Math.floor(Math.min(a[1], b[1]) / cellSize), y1 = Math.floor(Math.max(a[1], b[1]) / cellSize);
      for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) {
        const key = `${x},${y}`;
        if (!this.grid.has(key)) this.grid.set(key, []);
        this.grid.get(key).push(id);
      }
    });
  }

  candidates(point, count = 6, initialRadius = 50) {
    for (let radius = initialRadius; radius <= 6400; radius *= 2) {
      const seen = new Set(), found = [];
      const s = this.cellSize;
      for (let x = Math.floor((point[0] - radius) / s); x <= Math.floor((point[0] + radius) / s); x++)
        for (let y = Math.floor((point[1] - radius) / s); y <= Math.floor((point[1] + radius) / s); y++)
          for (const id of this.grid.get(`${x},${y}`) || []) {
            if (seen.has(id)) continue; seen.add(id);
            const [u, v] = this.arcs[id], hit = projectSegment(point, this.xy[u], this.xy[v]);
            if (hit.distance <= radius) found.push({ id, ...hit });
          }
      found.sort((a, b) => a.distance - b.distance || a.id - b.id);
      // Retain both directions of a road segment: heading is resolved by the HMM.
      const physical = new Set(), selected = [];
      for (const hit of found) {
        const key = this.physicalKey(hit.id);
        if (!physical.has(key) && physical.size >= count) continue;
        physical.add(key); selected.push(hit);
      }
      if (physical.size >= count || (found.length && radius === 6400)) return selected;
    }
    throw new Error('No nearby road segments within 6.4 km');
  }

  physicalKey(id) {
    const [u, v] = this.arcs[id]; return u < v ? `${u}:${v}` : `${v}:${u}`;
  }

  shortestPaths(source, targets, maxDistance) {
    const remaining = new Set(targets), distances = new Map([[source, 0]]), parents = new Map();
    const heap = new MinHeap(); heap.push([0, source]);
    while (heap.items.length && remaining.size) {
      const [distance, u] = heap.pop();
      if (distance !== distances.get(u)) continue;
      if (distance > maxDistance) break;
      remaining.delete(u);
      for (const id of this.adjacency[u]) {
        const v = this.arcs[id][1], next = distance + this.lengths[id];
        if (next <= maxDistance && next < (distances.get(v) ?? Infinity)) {
          distances.set(v, next); parents.set(v, id); heap.push([next, v]);
        }
      }
    }
    return { distances, parents, source };
  }

  pathTo(search, target) {
    if (!search.distances.has(target)) return null;
    const route = [];
    while (target !== search.source) {
      const id = search.parents.get(target);
      if (id === undefined) return null;
      route.push(id); target = this.arcs[id][0];
    }
    return route.reverse();
  }
}
