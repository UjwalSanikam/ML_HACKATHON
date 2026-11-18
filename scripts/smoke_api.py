"""Check the saved model and live HTTP interface on all student-test trips."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import requests


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PATHSNAP_EVAL_PARTITION": "student_test"}
    with (output / "api.log").open("w", encoding="utf-8") as log, requests.Session() as client:
        api = subprocess.Popen(
            [sys.executable, "-u", "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(120):
                if api.poll() is not None:
                    raise RuntimeError(f"API exited during startup. See {log.name}")
                try:
                    health = client.get(f"{url}/health", timeout=1)
                    health.raise_for_status()
                    break
                except requests.ConnectionError:
                    time.sleep(0.25)
            else:
                raise RuntimeError(f"API did not start. See {log.name}")
            listing = client.get(f"{url}/trips", timeout=30)
            listing.raise_for_status()
            trips = listing.json()
            if trips["partition"] != "student_test" or len(trips["trips"]) != 10:
                raise RuntimeError("Expected all 10 student-test trips. Check PATHSNAP_DATA.")
            rows = []
            for trip in trips["trips"]:
                trip_id = trip["trip_id"]
                response = client.post(f"{url}/predict", json={"trip_id": trip_id}, timeout=120)
                response.raise_for_status()
                result = response.json()
                if (result["trip_id"] != trip_id or not result["route"]
                        or not result["classifier_only_route"]
                        or result["metrics"].get("hmm_top1_acc") is None
                        or {"reference", "reference_route", "labels"} & result.keys()):
                    raise RuntimeError(f"Unexpected prediction response for {trip_id}")
                row = {key: result[key] for key in ("trip_id", "prediction_seconds", "route_gaps", "metrics")}
                rows.append(row)
                print(f"HTTP {response.status_code} | {trip_id} | "
                      f"top-1 {row['metrics']['hmm_top1_acc']:.2%} | {row['prediction_seconds']:.3f}s", flush=True)
            (output / "student_test.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        finally:
            api.terminate()
            api.wait(timeout=10)


if __name__ == "__main__":
    main()
