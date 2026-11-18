# PathSnap submission evidence

The public repository contains the pipeline, trained artifacts, experiment records and deployment commands. The team uses these sources to write a single technical report PDF in the organizers’ required order. Keep the partition and metric definition visible in every caption and table.

| Required report section | Evidence in this repository |
| --- | --- |
| 2.2 Dataset and EDA | [Data quality notes](../results/eda_data_quality.md), [missing-value counts](../results/eda_missing_values.csv), [trip inventory](../results/eda_inventory.csv), [sample Folium map](../results/maps/eda_train_00000002.html), and the trajectory and network plots under `results/figures/`. Use `pathsnap/features.py` for feature definitions. |
| 3 Model experiments | [Comparison tables](../results/report_tables.md) and [model fact sheets](../results/report_model_fact_sheets.md). Give each trained model its own subsection. Use the training, CV, and test metrics to explain any apparent overfitting or underfitting. Document the nearest-road baseline separately. |
| 4 Bayesian optimization | [Tuned versus untuned results](../results/bayes_opt_tuned_vs_untuned.csv), [trial history](../results/optuna_lightgbm_trials.csv), [selected configuration](../artifacts/final_model_config.json), and the search ranges in `scripts/04_bayes_opt.py`. Explain the search objective and why the final end-to-end results support the choice. |
| 5 HMM and Viterbi | [Pipeline comparison](../results/hmm_comparison.csv), [per-trip results](../results/hmm_per_trip.csv), and [decoder settings](../artifacts/hmm_config.json). Explain how candidate scores become emissions, how graph distance constrains transitions, and when decoding restarts. Use the per-trip results to discuss an improvement and a remaining failure. |
| 6 Folium maps | [Interactive student-test map](../results/maps/pred_student_test_00000056.html) and [map generation code](../pathsnap/viz.py). Capture a representative Folium view with the applicable route layers and legend visible. |
| 7 API, UI, and deployment | Run the two-laptop setup in the README. Capture the UI showing a successful trip and Laptop 1’s terminal showing the matching request, HTTP 200 and scoring output. Include both screenshots in the PDF. |

For a rehearsal, run `scripts/smoke_api.py` with the project virtual environment. It writes real HTTP logs to `outputs/smoke/api.log` and aggregate scores to `outputs/smoke/student_test.json`. Label those results as student-test evidence. At the final evaluation, use the organizer-supplied partition and check that the API reports `evaluator_test` with ten trips. Keep references on Laptop 1 and include the team’s AI assistance disclosure in the report.
