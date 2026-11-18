"""Central configuration: paths, constants and default parameters.

Every script imports from here so preprocessing is applied consistently
across training, evaluation and the API.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw data location. Override with the PATHSNAP_DATA environment variable.
# Expected layout:  <DATA_ROOT>/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
_local_data = ROOT / "Student folder"
_default_data = _local_data if _local_data.is_dir() else ROOT.parent / "Student folder"
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", _default_data))

# Partition folder names inside DATA_ROOT.
TRAIN = "train"
STUDENT_TEST = "student_test"
# The evaluator partition is placed on the API laptop only on evaluation day.
EVALUATOR_TEST = os.environ.get("PATHSNAP_EVAL_PARTITION", "evaluator_test")

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MAPS = RESULTS / "maps"
CACHE = ROOT / ".cache"

for _p in (ARTIFACTS, RESULTS, FIGURES, MAPS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
EARTH_RADIUS_M = 6_371_008.8

# ---- Candidate generation -------------------------------------------------
CAND_RADIUS_M = 60.0        # search radius around each GPS fix
CAND_RADIUS_MAX_M = 250.0   # radius is widened up to this when nothing is found
CAND_K = 12                 # max candidate (directed) arcs kept per observation
DENSIFY_STEP_M = 15.0       # sample spacing used for the segment KD-tree
BBOX_MARGIN_M = 1500.0      # road network is cropped to the track bbox + margin

# ---- Label alignment -------------------------------------------------------
ALIGN_SKIP_PENALTY_M = 2.0  # cost per skipped route arc in the monotone alignment
ALIGN_COST_CLIP_M = 300.0

# ---- Training ----------------------------------------------------------------
TRAIN_OBS_STRIDE = 3        # use every 3rd GPS fix for training rows (reduces autocorrelation)
N_FOLDS = 5

# ---- HMM defaults (overwritten by artifacts/hmm_config.json after tuning) --------
HMM_DEFAULTS = dict(
    beta=8.0,               # transition scale (m) on |route distance - GPS distance|
    emission_weight=1.0,    # exponent applied to the classifier probability
    sigma_dist=12.0,        # optional Gaussian distance term (m); 0 disables it
    dist_weight=0.0,        # weight of the Gaussian distance term
    back_tol_m=12.0,        # tolerated backward motion on the same arc (GPS noise)
    max_speed_mps=60.0,     # physically implausible speed -> transition forbidden
    gap_break_s=120.0,      # time gap that forces a new Viterbi segment
    route_factor=3.0,       # shortest-path search bound = factor * gps distance + slack
    route_slack_m=150.0,
)
