"""Save the scores of every trip served by a running PathSnap API.

    python scripts/08_collect_eval.py --api http://<laptop1-ip>:8000

Calls GET /trips and POST /predict for each trip (the same calls the UI makes) and writes
results/<partition>_results.csv (one row per trip plus a MEAN row) and
results/<partition>_results.json (raw metrics, timings and model info).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

RESULTS = Path(__file__).resolve().parent.parent / "results"

COLUMNS = {
    "clf_top1_acc": "clf_top1",
    "hmm_top1_acc": "hmm_top1",
    "hmm_route_f1_len": "route_f1_len",
    "hmm_route_ordered_f1": "route_ordered_f1",
    "hmm_continuity": "continuity",
    "hmm_route_precision_len": "route_precision_len",
    "hmm_route_recall_len": "route_recall_len",
    "candidate_miss_rate": "candidate_miss_rate",
    "label_coverage": "label_coverage",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, help="API base URL, e.g. http://10.30.201.105:8000")
    args = ap.parse_args()
    api = args.api.rstrip("/")

    with requests.Session() as s:
        health = s.get(f"{api}/health", timeout=10).json()
        model = s.get(f"{api}/model", timeout=10).json()
        listing = s.get(f"{api}/trips", timeout=30).json()
        partition = listing["partition"]
        rows, raw = [], []
        for t in listing["trips"]:
            tid = t["trip_id"]
            r = s.post(f"{api}/predict", json={"trip_id": tid}, timeout=300)
            r.raise_for_status()
            out = r.json()
            m = out["metrics"]
            row = {"trip_id": tid, "n_obs": out["n_observations"], "segments": out["segments"],
                   "prediction_s": out["prediction_seconds"]}
            row.update({new: m.get(old) for old, new in COLUMNS.items()})
            rows.append(row)
            raw.append({"trip_id": tid, "metrics": m, "timings_ms": out["timings_ms"],
                        "n_route_arcs": out["n_route_arcs"], "segments": out["segments"]})
            print(f"{tid}: obs={row['n_obs']} HMM top-1={row['hmm_top1']} route F1={row['route_f1_len']}")

    df = pd.DataFrame(rows)
    mean = df.drop(columns="trip_id").mean(numeric_only=True)
    df = pd.concat([df, pd.DataFrame([{"trip_id": "MEAN", **mean.to_dict()}])], ignore_index=True)
    csv = RESULTS / f"{partition}_results.csv"
    df.to_csv(csv, index=False, float_format="%.4f")
    (RESULTS / f"{partition}_results.json").write_text(json.dumps(
        {"collected_at": datetime.now().isoformat(timespec="seconds"), "api": api, "host": health.get("host"),
         "partition": partition, "model": model, "trips": raw}, indent=2))
    print(f"\n{partition}: mean HMM top-1 {mean['hmm_top1']:.4f}, route F1 {mean['route_f1_len']:.4f}, "
          f"ordered F1 {mean['route_ordered_f1']:.4f}, continuity {mean['continuity']:.4f}")
    print("written", csv)


if __name__ == "__main__":
    sys.exit(main())
