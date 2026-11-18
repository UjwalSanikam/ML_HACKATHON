# Report tables (auto-generated from results/*.csv)

## Overall model comparison (CV = 5-fold GroupKFold on 80 train trips; test = 10 student-test trips)

| model | family | config | cv_macro_f1 | cv_macro_f1_std | cv_top1 | test_top1 | test_macro_f1 | test_precision | test_recall | test_log_loss | route_continuity | train_time_s | infer_ms_per_1k_obs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | Boosting | 400 trees, lr=0.05, 63 leaves (untuned) | 0.9451 | 0.0088 | 0.9014 | 0.9138 | 0.9512 | 0.9511 | 0.9514 | 0.0574 | 0.9772 | 17.6948 | 90.1796 |
| XGBoost | Boosting | 400 trees, lr=0.05, depth=7, hist | 0.9451 | 0.0091 | 0.9014 | 0.9117 | 0.9500 | 0.9499 | 0.9502 | 0.0585 | 0.9764 | 3.0523 | 8.4435 |
| LogisticRegression | Linear / probabilistic | StandardScaler + L2, C=1.0 | 0.9432 | 0.0082 | 0.8981 | 0.8890 | 0.9372 | 0.9370 | 0.9373 | 0.0775 | 0.9770 | 2.2520 | 1.9781 |
| HistGradientBoosting | Boosting | lr=0.1, max_iter=300, max_leaf_nodes=31 | 0.9424 | 0.0082 | 0.8967 | 0.9026 | 0.9449 | 0.9447 | 0.9450 | 0.0644 | 0.9743 | 1.8791 | 4.7467 |
| RandomForest | Ensemble (bagging) | 300 trees, max_depth=20, min_leaf=10, sqrt features | 0.9424 | 0.0077 | 0.8966 | 0.8979 | 0.9422 | 0.9421 | 0.9423 | 0.0626 | 0.9751 | 97.9466 | 25.5993 |
| MLP | Neural network | 128-64 ReLU, Adam lr=1e-3, early stopping, 50% obs | 0.9422 | 0.0087 | 0.8964 | 0.9042 | 0.9458 | 0.9457 | 0.9460 | 0.0666 | 0.9763 | 6.2285 | 1.6980 |
| LinearSVM_RBF_Nystroem | Kernel-based | Nystroem RBF(400 comps) + LinearSVC + sigmoid calib., 25% obs | 0.9420 | 0.0073 | 0.8961 | 0.8826 | 0.9335 | 0.9334 | 0.9336 | 0.0775 | 0.9760 | 8.6864 | 31.8772 |
| ExtraTrees | Ensemble (bagging) | 300 trees, max_depth=22, min_leaf=10 | 0.9419 | 0.0078 | 0.8957 | 0.8890 | 0.9372 | 0.9370 | 0.9373 | 0.0669 | 0.9748 | 14.5859 | 25.6922 |
| DecisionTree | Tree-based | max_depth=12, min_samples_leaf=50 | 0.9393 | 0.0084 | 0.8913 | 0.8888 | 0.9370 | 0.9369 | 0.9372 | 0.0811 | 0.9705 | 7.7832 | 0.3754 |
| KNN | Instance-based | k=40, distance-weighted, standardised, 25% obs | 0.9369 | 0.0092 | 0.8871 | 0.8793 | 0.9316 | 0.9315 | 0.9318 | 0.0940 | 0.9714 | 0.0279 | 554.9700 |
| GaussianNB | Linear / probabilistic | QuantileTransformer(normal) + GaussianNB | 0.9216 | 0.0115 | 0.8605 | 0.8453 | 0.9124 | 0.9122 | 0.9125 | 0.5701 | 0.9578 | 0.8896 | 15.2948 |
| NearestRoad (baseline) | Heuristic | argmin perpendicular distance | 0.7534 | 0.0395 | 0.5688 | 0.5362 | 0.7368 | 0.7367 | 0.7369 | N/A | 0.9169 | 0.0000 | 0.0015 |


## Overfitting check (top-1 accuracy)

| model | train_top1 | cv_top1 | test_top1 |
|---|---|---|---|
| LightGBM | 0.9579 | 0.9014 | 0.9138 |
| XGBoost | 0.9560 | 0.9014 | 0.9117 |
| LogisticRegression | 0.9002 | 0.8981 | 0.8890 |
| HistGradientBoosting | 0.9272 | 0.8967 | 0.9026 |
| RandomForest | 0.9459 | 0.8966 | 0.8979 |
| MLP | 0.9236 | 0.8964 | 0.9042 |
| LinearSVM_RBF_Nystroem | 0.9001 | 0.8961 | 0.8826 |
| ExtraTrees | 0.9247 | 0.8957 | 0.8890 |
| DecisionTree | 0.9239 | 0.8913 | 0.8888 |
| KNN | 0.9238 | 0.8871 | 0.8793 |
| GaussianNB | 0.8629 | 0.8605 | 0.8453 |


## Bayesian optimisation: untuned vs tuned

| model | cv_macro_f1 | cv_macro_f1_std | cv_top1 | cv_log_loss | test_top1 | test_macro_f1 | test_log_loss | test_precision | test_recall | train_time_s | infer_ms_per_1k_obs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm_untuned | 0.9451 | 0.0088 | 0.9014 | 0.0661 | 0.9138 | 0.9512 | 0.0574 | 0.9511 | 0.9514 | 4.5487 | 28.8236 |
| lightgbm_tuned | 0.9471 | 0.0085 | 0.9048 | 0.0691 | 0.9099 | 0.9491 | 0.0615 | 0.9489 | 0.9492 | 13.2945 | 98.9733 |


Best hyperparameters:

```json
{
  "n_estimators": 950,
  "learning_rate": 0.09898176156647576,
  "num_leaves": 201,
  "min_child_samples": 25,
  "subsample": 0.6192726807936171,
  "colsample_bytree": 0.5991896677732862,
  "reg_lambda": 0.0018881262880778734,
  "reg_alpha": 2.3568581603700394,
  "min_split_gain": 0.5320415651853762
}
```


Trials: 50 (49 complete, 1 pruned); total trial time 45.0 min; best CV macro F1 0.9471


## HMM / Viterbi: classifier-only vs decoded (student-test, mean of 10 trips)

| setting | top1_acc | continuity | route_f1_len | route_ordered_f1 | route_precision_len | route_recall_len | decode_seconds | top1_acc_obs_weighted |
|---|---|---|---|---|---|---|---|---|
| Classifier only (final model) | 0.9197 | 0.9832 | 0.8380 | 0.4780 | 0.7311 | 0.9975 | 0.0000 | 0.9100 |
| HMM, distance-only emissions | 0.9104 | 0.9999 | 0.9824 | 0.9337 | 0.9697 | 0.9957 | 0.1530 | 0.8970 |
| Classifier + HMM (default params) | 0.9376 | 0.9999 | 0.9849 | 0.9492 | 0.9725 | 0.9979 | 0.1015 | 0.9218 |
| Classifier + HMM (tuned params) | 0.9576 | 0.9999 | 0.9914 | 0.9686 | 0.9853 | 0.9976 | 0.0989 | 0.9530 |


## HMM per trip

| setting | trip_id | n_obs | top1_acc | continuity | route_f1_len | route_ordered_f1 | route_precision_len | route_recall_len | segments | decode_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| Classifier only (final model) | 0 | 2503 | 0.9584 | 0.9856 | 0.9580 | 0.4420 | 0.9193 | 1.0000 | N/A | 0.0000 |
| Classifier only (final model) | 7 | 1354 | 0.9867 | 0.9985 | 0.9763 | 0.8521 | 0.9601 | 0.9930 | N/A | 0.0000 |
| Classifier only (final model) | 9 | 1049 | 0.8827 | 0.9943 | 0.8216 | 0.6965 | 0.6975 | 0.9994 | N/A | 0.0000 |
| Classifier only (final model) | 40 | 2995 | 0.9526 | 0.9916 | 0.8674 | 0.4154 | 0.7662 | 0.9994 | N/A | 0.0000 |
| Classifier only (final model) | 52 | 262 | 0.9733 | 0.9847 | 0.8033 | 0.4052 | 0.6727 | 0.9969 | N/A | 0.0000 |
| Classifier only (final model) | 56 | 3958 | 0.9262 | 0.9517 | 0.8625 | 0.4756 | 0.7638 | 0.9903 | N/A | 0.0000 |
| Classifier only (final model) | 65 | 768 | 0.8646 | 0.9739 | 0.7425 | 0.4125 | 0.5905 | 1.0000 | N/A | 0.0000 |
| Classifier only (final model) | 76 | 1486 | 0.9367 | 0.9832 | 0.7439 | 0.4895 | 0.5935 | 0.9962 | N/A | 0.0000 |
| Classifier only (final model) | 77 | 3911 | 0.8049 | 0.9987 | 0.8598 | 0.3609 | 0.7540 | 1.0000 | N/A | 0.0000 |
| Classifier only (final model) | 91 | 874 | 0.9108 | 0.9702 | 0.7446 | 0.2307 | 0.5932 | 1.0000 | N/A | 0.0000 |
| HMM, distance-only emissions | 0 | 2503 | 0.9740 | 0.9992 | 0.9941 | 0.9077 | 0.9901 | 0.9982 | 3.0000 | 0.0901 |
| HMM, distance-only emissions | 7 | 1354 | 0.9601 | 1.0000 | 0.9911 | 0.9837 | 0.9892 | 0.9930 | 6.0000 | 0.0338 |
| HMM, distance-only emissions | 9 | 1049 | 0.8570 | 1.0000 | 0.9607 | 0.9124 | 0.9361 | 0.9866 | 2.0000 | 0.1578 |
| HMM, distance-only emissions | 40 | 2995 | 0.9649 | 1.0000 | 0.9983 | 0.9778 | 0.9966 | 1.0000 | 2.0000 | 0.0983 |
| HMM, distance-only emissions | 52 | 262 | 0.9809 | 1.0000 | 0.9984 | 0.9894 | 1.0000 | 0.9969 | 1.0000 | 0.0145 |
| HMM, distance-only emissions | 56 | 3958 | 0.9468 | 0.9997 | 0.9808 | 0.9839 | 0.9630 | 0.9992 | 3.0000 | 0.2954 |
| HMM, distance-only emissions | 65 | 768 | 0.8841 | 1.0000 | 0.9863 | 0.9603 | 0.9729 | 1.0000 | 2.0000 | 0.0735 |
| HMM, distance-only emissions | 76 | 1486 | 0.9240 | 1.0000 | 0.9603 | 0.8897 | 0.9316 | 0.9908 | 1.0000 | 0.3663 |
| HMM, distance-only emissions | 77 | 3911 | 0.7223 | 1.0000 | 0.9738 | 0.7779 | 0.9489 | 1.0000 | 4.0000 | 0.3062 |
| HMM, distance-only emissions | 91 | 874 | 0.8902 | 1.0000 | 0.9801 | 0.9542 | 0.9682 | 0.9923 | 1.0000 | 0.0938 |
| Classifier + HMM (default params) | 0 | 2503 | 0.9776 | 0.9992 | 0.9959 | 0.9368 | 0.9927 | 0.9991 | 3.0000 | 0.0770 |
| Classifier + HMM (default params) | 7 | 1354 | 0.9616 | 1.0000 | 0.9911 | 0.9837 | 0.9892 | 0.9930 | 6.0000 | 0.0337 |
| Classifier + HMM (default params) | 9 | 1049 | 0.9218 | 1.0000 | 0.9667 | 0.9529 | 0.9413 | 0.9936 | 2.0000 | 0.0895 |
| Classifier + HMM (default params) | 40 | 2995 | 0.9773 | 1.0000 | 0.9985 | 0.9841 | 0.9971 | 1.0000 | 2.0000 | 0.0983 |
| Classifier + HMM (default params) | 52 | 262 | 1.0000 | 1.0000 | 0.9984 | 0.9894 | 1.0000 | 0.9969 | 1.0000 | 0.0129 |
| Classifier + HMM (default params) | 56 | 3958 | 0.9575 | 0.9997 | 0.9795 | 0.9823 | 0.9614 | 0.9983 | 3.0000 | 0.2210 |
| Classifier + HMM (default params) | 65 | 768 | 0.8828 | 1.0000 | 0.9773 | 0.9603 | 0.9556 | 1.0000 | 2.0000 | 0.0496 |
| Classifier + HMM (default params) | 76 | 1486 | 0.9798 | 1.0000 | 0.9824 | 0.9524 | 0.9668 | 0.9984 | 1.0000 | 0.1169 |
| Classifier + HMM (default params) | 77 | 3911 | 0.7678 | 1.0000 | 0.9699 | 0.7757 | 0.9416 | 1.0000 | 4.0000 | 0.2583 |
| Classifier + HMM (default params) | 91 | 874 | 0.9497 | 1.0000 | 0.9894 | 0.9748 | 0.9791 | 1.0000 | 1.0000 | 0.0574 |
| Classifier + HMM (tuned params) | 0 | 2503 | 0.9816 | 0.9992 | 0.9974 | 0.9396 | 0.9949 | 1.0000 | 3.0000 | 0.0771 |
| Classifier + HMM (tuned params) | 7 | 1354 | 0.9993 | 1.0000 | 0.9965 | 0.9918 | 1.0000 | 0.9930 | 1.0000 | 0.0351 |
| Classifier + HMM (tuned params) | 9 | 1049 | 0.9228 | 1.0000 | 0.9752 | 0.9714 | 0.9579 | 0.9932 | 2.0000 | 0.0811 |
| Classifier + HMM (tuned params) | 40 | 2995 | 0.9863 | 1.0000 | 0.9994 | 0.9904 | 0.9989 | 1.0000 | 1.0000 | 0.0969 |
| Classifier + HMM (tuned params) | 52 | 262 | 1.0000 | 1.0000 | 0.9984 | 0.9894 | 1.0000 | 0.9969 | 1.0000 | 0.0127 |
| Classifier + HMM (tuned params) | 56 | 3958 | 0.9687 | 0.9997 | 0.9812 | 0.9849 | 0.9641 | 0.9990 | 3.0000 | 0.2132 |
| Classifier + HMM (tuned params) | 65 | 768 | 0.8997 | 1.0000 | 0.9942 | 0.9797 | 0.9884 | 1.0000 | 2.0000 | 0.0487 |
| Classifier + HMM (tuned params) | 76 | 1486 | 0.9825 | 1.0000 | 0.9849 | 0.9635 | 0.9717 | 0.9984 | 1.0000 | 0.1129 |
| Classifier + HMM (tuned params) | 77 | 3911 | 0.8811 | 1.0000 | 0.9930 | 0.8904 | 0.9860 | 1.0000 | 1.0000 | 0.2562 |
| Classifier + HMM (tuned params) | 91 | 874 | 0.9542 | 1.0000 | 0.9933 | 0.9851 | 0.9911 | 0.9956 | 1.0000 | 0.0551 |


## All models through the tuned HMM decoder

| model | clf_top1 | hmm_top1 | clf_continuity | hmm_continuity | clf_route_f1_len | hmm_route_f1_len | hmm_route_ordered_f1 |
|---|---|---|---|---|---|---|---|
| NearestRoad (baseline) | 0.5587 | 0.9319 | 0.9863 | 0.9999 | 0.6038 | 0.9871 | 0.9385 |
| LogisticRegression | 0.9007 | 0.9408 | 0.9945 | 0.9999 | 0.8290 | 0.9904 | 0.9629 |
| GaussianNB | 0.8547 | 0.9323 | 0.9893 | 0.9999 | 0.7253 | 0.9840 | 0.9320 |
| KNN | 0.8895 | 0.9481 | 0.9926 | 0.9999 | 0.8000 | 0.9885 | 0.9585 |
| DecisionTree | 0.9036 | 0.9502 | 0.9919 | 0.9999 | 0.8106 | 0.9912 | 0.9675 |
| RandomForest | 0.9099 | 0.9538 | 0.9946 | 0.9999 | 0.8121 | 0.9906 | 0.9680 |
| ExtraTrees | 0.9005 | 0.9520 | 0.9946 | 0.9999 | 0.8125 | 0.9906 | 0.9664 |
| HistGradientBoosting | 0.9119 | 0.9560 | 0.9946 | 0.9999 | 0.8088 | 0.9909 | 0.9683 |
| LightGBM | 0.9212 | 0.9568 | 0.9951 | 0.9999 | 0.8340 | 0.9909 | 0.9668 |
| XGBoost | 0.9199 | 0.9566 | 0.9951 | 0.9999 | 0.8302 | 0.9911 | 0.9698 |
| LinearSVM_RBF_Nystroem | 0.8935 | 0.9388 | 0.9950 | 0.9999 | 0.8110 | 0.9895 | 0.9628 |
| MLP | 0.9092 | 0.9516 | 0.9944 | 0.9999 | 0.8327 | 0.9905 | 0.9672 |


Tuned HMM parameters:

```json
{
  "beta": 1.3686277303338878,
  "emission_weight": 1.64250552368797,
  "sigma_dist": 12.0,
  "dist_weight": 0.07690069558101933,
  "back_tol_m": 39.35750934774539,
  "max_speed_mps": 60.0,
  "gap_break_s": 39.155628717131,
  "route_factor": 3.0,
  "route_slack_m": 150.0,
  "uturn_penalty": 9.2852780382415
}
```


Decoder tuning: 40 trials, objective mean_trip 0.5*(HMM top-1 acc + length-weighted route F1), OOF emissions, default 0.9396 -> best 0.9649


## Missing values / data-quality counts (EDA)

| partition | nan_values | duplicate_timestamps | non_monotonic_timestamps | n_gaps_gt5s | n_gaps_gt60s | no_candidate_obs | route_disconnected_steps | isolated_nodes |
|---|---|---|---|---|---|---|---|---|
| student_test | 0 | 0 | 0 | 44 | 1 | 0 | 1 | 0 |
| train | 0 | 0 | 0 | 1216 | 69 | 2 | 13 | 0 |
