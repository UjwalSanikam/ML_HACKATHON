# GPS Route Recovery

Recover a vehicle's road route from noisy GPS observations using the supplied directed road graph. This hackathon starter includes a nearest-road baseline, a Hidden Markov Model (HMM) matcher, trip-level evaluation, and offline interactive reports.

## Quick start

Requires **Node.js 20 or newer** (`node --version`). There are no third-party runtime dependencies and no API keys.

1. Clone the team's repository and open a terminal in its root:

   ```sh
   git clone https://github.com/UjwalSanikam/ML_HACKATHON.git
   cd ML_HACKATHON
   ```

2. Obtain the organizer's `Student folder.zip` separately and extract it so the structure matches the example below. The dataset and video are excluded from Git.
3. Run:

```sh
npm test
npm run audit
npm run demo
```

Open `outputs/train/00000046/report.html` in your browser. The report supports pan, zoom, and toggling the GPS observations, reference, baseline, and HMM layers. It contains its own data and needs no internet connection. The report JavaScript was syntax-checked; the development browser blocked local-file preview, so visual inspection remains pending.

```text
project/
  Student folder/
    train/
      00000001/
        00000001.nodes
        00000001.arcs
        00000001.track
        00000001.route
      ...
    student_test/
      ...
  src/
  test/
  docs/
  package.json
```

No `npm install` step is needed. If the data is elsewhere, append `-- --data "path/to/Student folder"` to an npm command.

## Commands

```sh
# Audit all inputs and training references; save a fixed split manifest.
npm run audit

# Inspect a labeled training trip.
node src/cli.js run --id 00000001

# Evaluate all 16 validation trips with fixed defaults.
npm run evaluate

# Use development trips for tuning; keep the validation split fixed.
npm run evaluate -- --subset development

# Small development run.
npm run evaluate -- --subset development --limit 3

# Try every GPS point instead of every fifth point.
node src/cli.js run --id 00000046 --stride 1

# Tune on development trips only.
npm run evaluate -- --subset development --sigma 20 --beta 30 --candidates 4

# Generate baseline and HMM predictions for the 10 student-test trips.
npm run predict
```

Predictions are saved to `outputs/<split>/<trip-id>/{baseline,hmm}.json`. Labeled runs also create `report.html`. Aggregate evaluation goes to `outputs/evaluation-validation.json` or `outputs/evaluation-development.json`. Outputs are ignored by Git and can be reproduced locally.

## File formats verified from the supplied data

All files are whitespace-delimited and have no headers. IDs are **local to each trip**.

| Extension | Each row means |
| --- | --- |
| `.nodes` | Longitude, latitude. Its zero-based row number is the node ID. |
| `.arcs` | Source node ID, destination node ID. Its zero-based row number is the directed arc ID. |
| `.track` | Longitude, latitude, timestamp. Timestamp units are unconfirmed. |
| `.route` | A zero-based directed arc ID in reference traversal order. |

The training data contains 80 trips and 208,417 GPS observations. Student-test contains 10 trips and 19,160 observations. All checked graph endpoints and training route IDs are valid. There are 13 disconnected consecutive-arc steps across training reference routes. See `outputs/audit.json` for their positions.

Student-test includes `.route` files, but their intended role is unclear. **The reusable prediction and audit commands never open student-test route files.** Ask the organizers what those files are intended for before using them. Training references are loaded only after prediction for evaluation and plotting.

## How the methods work

Both methods use the same sampled GPS observations: every fifth row by default, always including the first and last. `--stride 1` uses every observation; stride counts rows, not seconds. Coordinates are projected to an approximate local plane in meters using the first road node as origin.

- **Baseline:** project each observation onto its nearest road segment. Distance ties between opposite directions use arc ID order; this baseline has no heading or connectivity logic. Remove consecutive duplicate arc IDs. It does not fill gaps between matched observations.
- **HMM:** consider the nearest four distinct physical segments, retaining their directed alternatives. Score candidates by GPS-to-segment distance (Gaussian noise scale `sigma=20` meters). Score transitions by the difference between road travel distance and observed straight-line displacement (`beta=30` meters). Directed shortest paths provide travel distances and connecting arcs. Viterbi decoding selects a sequence, and connecting paths reconstruct the route.

Candidate searches expand from 50 meters to at most 6.4 kilometers. Shortest-path searches are bounded to `max(500, 4 × GPS displacement + 200)` meters. If an entire step has no feasible candidate transition, the matcher explicitly restarts and records a gap. Timestamp values are not used to enforce speed limits because their unit is unconfirmed.

Background: [Newson and Krumm, Hidden Markov Map Matching Through Noise and Sparseness](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/).

## Evaluation and current results

The organizers have not supplied a scoring metric. Our measures are project-defined:

- **Directed arc precision/recall/F1:** overlap between the sets of predicted and reference arc IDs.
- **Physical-road F1:** the same overlap after treating a node pair and its reverse as one road segment.
- **Disconnected steps:** predicted consecutive arc pairs whose destination/source nodes do not connect.
- **Runtime:** matching time per trip, with graph-building time recorded separately.

Precision is intersection size divided by predicted set size; recall divides by reference set size. F1 is their harmonic mean. Aggregate F1 is the unweighted mean across trips. Set overlap does **not** assess traversal order, repeat visits, or complete-route correctness. Physical-road overlap also ignores direction. GPS-to-road distance is a diagnostic, not an accuracy score.

Trips are sorted by ID; every fifth trip, starting at index zero, is assigned to validation (16 trips). The other 64 are for development. This split is deterministic but is not guaranteed to separate overlapping geographic areas.

First validation run, fixed defaults, no parameter search:

| Method | Mean directed F1 | Mean physical-road F1 | Total disconnected steps |
| --- | ---: | ---: | ---: |
| Nearest-road baseline | 50.12% | 67.30% | 2,791 |
| HMM | 94.65% | 95.92% | 65 |

These are local validation results, not official scores. The HMM's 65 disconnected steps occur in six trips; it still needs improvement before claiming all predictions are continuous routes. Some restarts occur on a repeated arc and do not create a disconnected pair in the deduplicated route. Runtime depends on the machine and excludes report rendering.

## Team workflow

Create feature branches and use pull requests. Keep the source, tests, and documentation in Git. Each teammate keeps a local copy of the dataset. Run `npm test` before opening a PR and include development-set results when changing matching logic. Keep parameter tuning on the development subset, and label any later validation-driven changes honestly.

Suggested independent work areas:

| Work area | First task |
| --- | --- |
| Data and evaluation | Investigate reference gaps and add an order-aware route measure. |
| Matching | Reduce restarts using adaptive candidate radii and graph-aware recovery. |
| Visualization and presentation | Inspect reports and build a clear successful/difficult-trip comparison. |

See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) for the remaining work. Do not upload the organizer's data publicly until its sharing terms are known.

## Code map

| File | Responsibility |
| --- | --- |
| `src/data.js` | Data loading, separate reference loading, deterministic split |
| `src/graph.js` | Local projection, spatial segment index, directed shortest paths |
| `src/match.js` | Baseline, HMM decoding, evaluation |
| `src/report.js` | Self-contained interactive HTML report |
| `src/cli.js` | Audit, demo, evaluation, and prediction commands |
| `test/matching.test.js` | Geometry, missing-road reconstruction, direction, and gap tests |
