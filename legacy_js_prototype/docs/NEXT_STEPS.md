# Next steps

The first end-to-end prototype is implemented and evaluated. Its predictions are an intermediate project artifact; organizers have not specified a submission schema.

1. Confirm organizer expectations: deliverables, deadline, test `.route` purpose, external code/data rules, and whether a continuous route is mandatory.
2. Visually inspect `outputs/train/00000046/report.html`, then inspect several development trips. The automated browser could not preview local files. Check the layer toggles, zoom, and pan.
3. Investigate restarts on development trips before changing parameters. Log candidates, graph reachability, and local GPS noise around each gap. Study whether increasing candidates or retaining multiple hypotheses across gaps improves continuity.
4. Compare ordered F1 with a route-length-weighted measure to understand how small arcs affect the score.
5. Compare stride 1 and stride 5 on development trips. Report accuracy, gaps, total processing time, and memory usage.
6. Run npm run benchmark to reproduce the comparison and inspect its per-configuration results.
7. Choose a successful trip and a difficult development trip for the presentation. Explain why independent nearest-road matches fail and how connectivity helps. Show remaining failure cases honestly.
8. Export the organizer's requested submission format once it is known. Preserve explicit gap information rather than presenting a disconnected arc list as one continuous route.

## Interpretation cautions

- Arc IDs are row IDs local to each trip, not global road IDs.
- Treat arcs as directed. A reverse edge is available only when listed.
- Distance calculations use an approximate local projection; long geographical extents may need a more suitable coordinate system.
- Both methods are evaluated on the same sampled observations, but only HMM reconstructs intermediate connecting roads.
- Candidate count and routing-distance limits can exclude the correct path. Restarts represent uncertainty, not solved connectivity.
- Both references and predicted routes can have gaps. Reference gaps do not explain all prediction gaps.
- Set F1 can remain high despite an incorrect traversal order or repeated detour; it must not be called whole-route accuracy.
