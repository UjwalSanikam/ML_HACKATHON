const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
export function sampleIndices(length, stride = 5) {
  const indices = [];
  for (let i = 0; i < length; i += stride) indices.push(i);
  if (indices.at(-1) !== length - 1) indices.push(length - 1);
  return indices;
}

export function nearestRoad(graph, indices) {
  const matched = indices.map(i => graph.candidates(graph.trackXY[i], 1)[0]);
  return { route: matched.map(c => c.id).filter((id, i, all) => i === 0 || id !== all[i - 1]),
    matched, indices, gaps: [] };
}

export function hmmMatch(graph, indices, { sigma = 20, beta = 30, candidates = 4 } = {}) {
  const layers = indices.map(i => graph.candidates(graph.trackXY[i], candidates));
  const emission = c => -0.5 * (c.distance / sigma) ** 2;
  let scores = layers[0].map(emission);
  const history = [];
  for (let t = 1; t < layers.length; t++) {
    const previous = layers[t - 1], current = layers[t];
    const gpsDistance = distance(graph.trackXY[indices[t - 1]], graph.trackXY[indices[t]]);
    const maxDistance = Math.max(500, gpsDistance * 4 + 200);
    const searches = new Map(), targets = current.map(c => graph.arcs[c.id][0]);
    for (const c of previous) {
      const source = graph.arcs[c.id][1];
      if (!searches.has(source)) searches.set(source, graph.shortestPaths(source, targets, maxDistance));
    }
    const nextScores = [], back = [];
    current.forEach(c => {
      let best = -Infinity, choice;
      previous.forEach((p, j) => {
        const search = searches.get(graph.arcs[p.id][1]);
        let length = Infinity, connector = null;
        if (p.id === c.id && c.fraction >= p.fraction) {
          length = (c.fraction - p.fraction) * graph.lengths[c.id]; connector = [];
        }
        const between = search.distances.get(graph.arcs[c.id][0]);
        if (between !== undefined) {
          const via = (1 - p.fraction) * graph.lengths[p.id] + between + c.fraction * graph.lengths[c.id];
          if (via < length) { length = via; connector = undefined; }
        }
        if (!Number.isFinite(length)) return;
        const score = scores[j] + emission(c) - Math.abs(length - gpsDistance) / beta;
        if (score > best) {
          best = score;
          choice = { previous: j, connector: connector ?? graph.pathTo(search, graph.arcs[c.id][0]), gap: false };
        }
      });
      nextScores.push(best); back.push(choice);
    });
    // Restart only if the entire candidate lattice has no feasible transition.
    // Explicitly report the break rather than invent a connecting road.
    if (nextScores.every(s => !Number.isFinite(s))) {
      const bestPrevious = scores.indexOf(Math.max(...scores));
      current.forEach((c, j) => {
        nextScores[j] = scores[bestPrevious] + emission(c) - 100;
        back[j] = { previous: bestPrevious, connector: [], gap: true };
      });
    }
    // Normalize log scores to avoid accumulating very large values.
    const maxScore = Math.max(...nextScores);
    scores = nextScores.map(s => s - maxScore); history.push(back);
  }
  let selected = scores.indexOf(Math.max(...scores));
  const choices = [selected], links = [];
  for (let t = history.length - 1; t >= 0; t--) {
    const link = history[t][selected];
    links.push(link); selected = link.previous; choices.push(selected);
  }
  choices.reverse(); links.reverse();
  const matched = choices.map((j, t) => layers[t][j]), route = [matched[0].id], gaps = [];
  const append = id => { if (route.at(-1) !== id) route.push(id); };
  links.forEach((link, t) => {
    if (link.gap) gaps.push({ afterSample: t, fromArc: matched[t].id, toArc: matched[t + 1].id });
    for (const id of link.connector) append(id);
    append(matched[t + 1].id);
  });
  return { route, matched, indices, gaps };
}

export function evaluate(graph, prediction, reference) {
  if (reference.some(id => !Number.isInteger(id) || !graph.arcs[id])) throw new Error('Invalid reference arc ID');
  function overlap(key) {
    const predicted = new Set(prediction.route.map(key)), truth = new Set(reference.map(key));
    const intersection = [...predicted].filter(id => truth.has(id)).length;
    const precision = intersection / (predicted.size || 1), recall = intersection / (truth.size || 1);
    return { precision, recall, f1: 2 * precision * recall / (precision + recall || 1) };
  }
  function broken(route) {
    return route.slice(1).filter((id, i) => graph.arcs[route[i]][1] !== graph.arcs[id][0]).length;
  }
  return {
    directed: overlap(id => id), physicalRoad: overlap(id => graph.physicalKey(id)),
    disconnectedSteps: broken(prediction.route), referenceDisconnectedSteps: broken(reference),
    routeArcs: prediction.route.length, referenceArcs: reference.length,
    restartCount: prediction.gaps.length,
    meanGpsToRoadMeters: prediction.matched.reduce((sum, c) => sum + c.distance, 0) / prediction.matched.length,
  };
}
