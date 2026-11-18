# Data folder (not committed)

By default the code reads the organiser data from `../Student folder` (next to this repository).
Set `PATHSNAP_DATA` to point anywhere else. Expected layout:

```
<PATHSNAP_DATA>/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
  student_test/<trip_id>/...
  evaluator_test/<trip_id>/...     # copied onto the API laptop on evaluation day only
```
