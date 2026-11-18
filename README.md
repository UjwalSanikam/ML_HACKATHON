# PathSnap: map-matching and route reconstruction

PathSnap takes noisy GPS trajectories and matches them to a directed road network. It works in two stages:

1. A **candidate classifier** scores every nearby directed road arc for every GPS fix.
2. An **HMM with Viterbi decoding** combines those scores with road-network connectivity to rebuild one continuous route.

The system runs on two laptops:

* **Laptop 1** runs a FastAPI backend with the trained model, the decoder and the road networks.
* **Laptop 2** runs a browser UI that only talks to the API.

<!-- RESULTS-START -->
## Final inference accuracy (deployed system, 10 unseen evaluator-test trips)

Measured through the live two-laptop deployment: the FastAPI backend on Laptop 1 and the UI on Laptop 2, means over trips. Raw scores are in `results/evaluator_test_results.csv` and `.json`.

| Pipeline | Observation accuracy | Route F1 (length-weighted) | Ordered route F1 | Continuity | Time per trip |
|---|---|---|---|---|---|
| LightGBM classifier only | 93.9% | – | – | – | – |
| **LightGBM + HMM/Viterbi (final)** | **95.8%** | **0.988** | **0.940** | **100%** | **0.69 s** |

* **Generalisation.** Accuracy on the unseen evaluator trips (95.83%) matches student-test (95.76%), so the model does not overfit the development data.
* **Range.** Per-trip accuracy runs from 86.5% (trip 00000030) to 100% (trip 00000049).

## Development results (10 student-test trips, means over trips)

| Pipeline | Observation top-1 | Route F1 (length-weighted) | Ordered route F1 | Continuity |
|---|---|---|---|---|
| Nearest road (no learning) | 55.9% | 0.604 | – | 98.6% |
| HMM with distance-only emissions (no classifier) | 91.0% | 0.982 | 0.934 | 99.99% |
| LightGBM classifier only | 92.0% | 0.838 | 0.478 | 98.3% |
| **LightGBM + HMM/Viterbi, tuned (final)** | **95.8%** | **0.991** | **0.969** | **99.99%** |

* **Model comparison.** 11 classifiers from 7 algorithm families plus a heuristic baseline, scored by 5-fold GroupKFold CV over the 80 training trips: `results/model_comparison.csv` and `results/figures/models_*.png`.
* **Model selection.** LightGBM had the best CV macro F1 (0.9451, tied with XGBoost at 0.9451) and the best student-test top-1 (91.4%).
* **Bayesian optimisation.** 50 Optuna TPE trials raised CV macro F1 to 0.9471. On student-test the tuned classifier alone was not better (91.0% vs 91.4% top-1). After HMM decoding the tuned model is marginally ahead (95.76% vs 95.68%). Details: `results/bayes_opt_tuned_vs_untuned.csv` and `results/figures/bayes_opt_lightgbm*.png`.
* **HMM decoder tuning.** 40 TPE trials on 32 training trips using out-of-fold emissions (student-test not used). The objective rose from 0.940 to 0.965.
* **Speed.** About 0.1–2.5 s per trip end to end on a laptop.

All numbers are reproducible with `./run_all.sh`.
<!-- RESULTS-END -->

---

## Repository layout

| Folder / file | What it contains |
|---|---|
| `pathsnap/` | Core library: IO and cleaning (`io.py`), local projection (`geo.py`), road network with KD-tree and bounded Dijkstra (`network.py`), label alignment (`labels.py`), candidate generation and features (`features.py`), model zoo (`models.py`), HMM/Viterbi (`hmm.py`), metrics (`metrics.py`), end-to-end inference (`pipeline.py`), Folium helpers (`viz.py`) |
| `scripts/01_eda.py` | Dataset exploration: inventory, missing values, gaps, network connectivity, plots, Folium maps |
| `scripts/02_build_dataset.py` | Candidate generation, feature extraction and label derivation for train and student-test |
| `scripts/03_train_compare.py` | Trains and compares 11 models plus a heuristic baseline (5-fold GroupKFold CV, then student-test) |
| `scripts/04_bayes_opt.py` | Bayesian hyperparameter optimisation (Optuna TPE) of the selected model; exports the final model |
| `scripts/05_hmm_eval.py` | HMM decoder tuning, classifier-only vs HMM comparison, Folium prediction maps |
| `scripts/06_figures.py` | All comparison plots, confusion matrices and real training histories |
| `scripts/07_report_assets.py` | Static map exports and auto-generated markdown tables for the report |
| `scripts/08_collect_eval.py` | Saves per-trip scores from a running API (`--api http://<laptop1-ip>:8000`) to `results/<partition>_results.csv` / `.json` |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator interface (Laptop 2): `index.html`, `app.js`, `styles.css`, `serve.py` |
| `artifacts/` | `final_model.joblib`, `final_model_config.json`, `hmm_config.json`, `models/` (every compared model), `fold_models/` |
| `results/` | CSV results, experiment log, Optuna SQLite studies, `figures/`, `maps/` (interactive Folium HTML) |
| `tests/` | Unit tests, including a check that prediction never reads `.route` files |
| `legacy_js_prototype/` | The team's first Node.js HMM prototype, kept for history |

## Installation

Use Python 3.12 for the pinned dependencies. On macOS, LightGBM and XGBoost also need OpenMP: `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_api.py
```

Extract the organiser's `Student folder` into the repository root or place it next to the repository. For another location set `PATHSNAP_DATA` (PowerShell example: `$env:PATHSNAP_DATA = 'D:\data\Student folder'`). The smoke check uses the shipped model on the 10 student-test trips and saves its API log and aggregate scores under `outputs/smoke/`.

## Reproducing every result

```bash
./run_all.sh
```

The script runs the tests and then scripts 01 to 07 in order, taking about 1.5 to 2 hours on an 8-core laptop, mostly model CV and Bayesian optimisation. Every number in the report is written to `results/` by these scripts. Seeds are fixed at 42.

## Deployment: two-laptop evaluation

For Windows PowerShell, run the following from the repo root on Laptop 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On Laptop 2, replace the example IP with Laptop 1's LAN address:

```powershell
.\.venv\Scripts\python.exe ui/serve.py --port 8080 --api http://192.168.1.20:8000
```

Confirm `GET /trips` and the UI both show `evaluator_test` with ten trips before starting the final demo. The UI loads Leaflet and map tiles from the internet. See [the submission evidence guide](docs/SUBMISSION_EVIDENCE.md) for the report sections and screenshot checklist.

### Laptop 1: API

1. Copy the reserved evaluation trips to `<PATHSNAP_DATA>/evaluator_test/<trip_id>/...`. The API serves that partition. If it does not exist, the API falls back to `student_test` and logs a warning.
2. Start the API:

   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

3. Find the laptop's IP with `ipconfig getifaddr en0` on macOS or `ipconfig` on Windows. Allow incoming connections on port 8000 if the firewall asks.

Every request is logged in this terminal, which is the log the evaluators ask to see. Each log line shows the client IP, endpoint, status, latency, and the trip's accuracy, route F1 and continuity.

### Laptop 2: UI

```bash
python ui/serve.py --port 8080 --api http://<laptop1-ip>:8000
```

This opens `http://localhost:8080`. The UI:

* lists the trips returned by `GET /trips`;
* sends the selected trip ID to `POST /predict`;
* draws the raw GPS, the predicted route, the classifier-only route and the matched positions on a Leaflet map with layer controls;
* shows the scores computed on the server.

**Evaluate all trips** runs every trip and builds a summary table. The *API request log* tab shows every call.

### API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check, host name, served partition |
| GET | `/model` | Model type, features, tuned hyperparameters, HMM parameters |
| GET | `/trips` | Trip IDs with observation count and duration. Reads GPS only. |
| POST | `/predict` | `{"trip_id": "..."}`. Runs the full pipeline and returns the route geometry, matched points, timings and server-side metrics. |

Quick check from any terminal:

```bash
curl -s http://<laptop1-ip>:8000/predict -H 'Content-Type: application/json' -d '{"trip_id":"00000056"}' | python -m json.tool | head -40
```

**Ground-truth isolation.** `Predictor.predict` never reads a `.route` file. `tests/test_core.py::test_prediction_never_reads_route` fails if it tries. The reference route is loaded only afterwards, inside `pipeline.evaluate` on the API laptop. Only aggregate scores leave the server: no reference geometry and no per-point labels.

## Method summary

* **Preprocessing.** Invalid coordinates and duplicate or out-of-order timestamps are dropped (none occur in the supplied data). Each trip is projected to a local equirectangular plane in metres, centred on the trip; distortion is under 0.1% at city scale. The road network is cropped to the track's bounding box plus 1.5 km.
* **Labels.** `.route` files are ordered arc lists without timestamps. Each GPS fix is assigned a reference arc by a monotone dynamic-programming alignment (`labels.py`). Fixes more than 50 m from their aligned arc are excluded from fitting and scoring, because some reference routes cover only part of the GPS trace.
* **Candidates.** Up to 12 directed arcs within 60 m of each fix, found with a densified-segment KD-tree. The radius doubles up to 250 m when nothing is found.
* **Features.** 27 features per (fix, candidate):
  * distance, plus its rank and ratio against the other candidates;
  * heading agreement over short and long windows (this resolves direction on two-way streets);
  * speed, turn rate and time gap;
  * position along the arc;
  * node degrees and two-way flag;
  * how well the arc explains the neighbouring fixes (±2, 5, 10);
  * signed motion along the arc direction.
* **Classifier.** Binary "is this candidate the reference arc?". Probabilities are normalised over each fix's candidates.
* **HMM.**
  * Hidden states are the candidate arcs.
  * Emission is the classifier probability raised to a weight.
  * Transition is `exp(-|route distance - GPS distance| / β)`, using bounded Dijkstra on the directed graph, with a U-turn penalty, a speed limit and a backward-motion tolerance.
  * Viterbi decoding restarts at large time gaps or when every transition is impossible.
  * Consecutive matched arcs are joined by shortest paths into one route.

## Metrics

* **Observation-level top-1 accuracy:** the best candidate equals the aligned reference arc. Fixes whose reference arc is not among the candidates count as errors.
* **Macro F1 / precision / recall:** candidate-level binary decisions (top-1 = positive), macro-averaged over {other, reference}.
* **Log loss:** binary cross-entropy of the candidate probabilities.
* **Candidate miss rate:** share of scored fixes whose reference arc is not among the candidates.
* **Route continuity:** share of consecutive fixes whose predicted arcs are equal or reachable by a plausible-length directed path.
* **Route F1 (length weighted)** and **ordered route F1** (LCS based) between the predicted and reference arc sequences.

## AI assistance disclosure

Parts of this codebase were written with an AI coding assistant. Each team member must understand, run and be able to explain the code and results, and must disclose AI use as the event's policy requires. The report must be written by the team from its own analysis of these results.
