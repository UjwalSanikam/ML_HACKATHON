# Data-quality record (from `scripts/01_eda.py`)

Partitions analysed: `train` (80 trips) and `student_test` (10 trips). The evaluator partition was never available to us and was not used. Raw numbers are in `eda_inventory.csv`, `eda_summary.json` and `eda_missing_values.csv`.

## What the data is

* **Many cities.** Each trip has its own road-network extract, and node/arc IDs are local to the trip. Trip centroids are spread across Europe, Asia and the Americas (`figures/eda_trip_locations.png`). A single global projection is therefore unsuitable, so each trip gets its own local tangent-plane projection.
* **Network size.** Networks range from 299 to 155,003 nodes (649 to 307,392 arcs). They are much larger than the area a trip covers, so we crop to the track bounding box plus 1.5 km before indexing.
* **Track extent.** Trips span up to about 75 km. An equirectangular projection centred on the trip has under 0.1% scale error over that range, which is negligible next to 4–15 m of GPS noise.
* **Coordinate order.** Longitude comes first in both `.track` and `.nodes`. All latitudes are within ±90°, so no file has the columns swapped.

## Missing / invalid values

* **No missing values.** No NaN values, duplicate timestamps or non-monotonic timestamps in either partition.
* **Timestamps.** Not all tracks start at t=1: 16 training trips start later, one of them at t=693. Only time differences are used.
* **Sampling rate.** 96.5% of training steps are exactly 1 s apart.
* **Large gaps.** 22 training trips have at least one gap over 60 s. The largest gaps are:

  | Trip | Largest gap |
  |---|---|
  | `00000090` | 153,064 s (about 42 h) |
  | `00000078` | 52,810 s |
  | `00000069` | 2,658 s |
  | `00000021` | 2,161 s |
  | `00000011` | 909 s |
  | `00000042` | 670 s, with 17 gaps over 60 s |

  The HMM breaks the Viterbi chain at large gaps instead of forcing a connection.

## Road network

* **Connectivity.** No isolated nodes. 79 of 80 training networks have more than one weakly connected component. The largest strongly connected component holds between 64% and 99.96% of nodes, so some candidate arcs cannot be reached from others. This is why transitions can be infeasible, and why the decoder needs explicit restarts.
* **Odd arcs.** 24 self-loop arcs and 106 zero-length arcs across all trips. They are harmless: zero-length arcs are never closer than a real segment.
* **Two-way streets.** About 92% of arcs have a reverse arc, so distance alone cannot tell the two directions apart. This is the main reason the nearest-road baseline is only about 57% accurate. Heading features and connectivity are needed.

## Reference routes and their limitations

* **Valid IDs, mostly continuous.** All route arc IDs are valid. 13 of 31,256 consecutive route steps in train are not head-to-tail connected, spread over 10 trips.
* **Repeated arcs.** 10 trips traverse some arcs more than once (loops). Our monotone alignment handles this.
* **No per-point labels.** Routes have no timestamps. Per-observation labels come from a monotone DP alignment, and are an estimate rather than independent ground truth.
* **Partial coverage.** Some reference routes cover only part of the GPS track:

  | Trip | Route length | GPS path length | Fixes >50 m from aligned arc |
  |---|---|---|---|
  | `train/00000054` | 31 km | 169 km | 89% |
  | `train/00000033` | 6.4 km | 47.8 km | 59% |

  Overall, 2.5% of training fixes per trip on average (11% of all training fixes, weighted by size) are more than 50 m from their aligned arc. We treat these labels as unreliable and exclude them from fitting and scoring.

## Candidate generation quality

* **Coverage.** With a 60 m radius and up to 12 directed arcs, the candidate miss rate on reliable labels is 0.74% in train and 0.10% in student-test.
* **No-candidate fixes.** Only 2 fixes in all 90 trips needed the radius expanded beyond 250 m without finding a candidate.
