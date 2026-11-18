# Per-model fact sheets (auto-generated; all numbers from results/*.csv)

## LightGBM  (Boosting)

- Configuration: 400 trees, lr=0.05, 63 leaves (untuned)
- Objective / loss: binary log loss (leaf-wise gradient boosting)
- CV macro F1 per fold: 0.9500, 0.9328, 0.9559, 0.9369, 0.9500 (mean 0.9451 ± 0.0088)
- CV top-1 per fold: 0.9089, 0.8828, 0.9216, 0.8807, 0.9131
- Student-test: top-1 0.9138, macro F1 0.9512, precision 0.9511, recall 0.9514, log loss 0.0574370062257265
- Over/under-fitting evidence: train top-1 0.9579 vs CV 0.9014 vs test 0.9138 (gap train-CV = +0.0565)
- Classifier-only route continuity: 0.9774
- Training time 4.8 s; inference 30.6 ms / 1000 obs
- Through the tuned HMM: top-1 0.9212 -> 0.9568; route F1 0.8340 -> 0.9909

## XGBoost  (Boosting)

- Configuration: 400 trees, lr=0.05, depth=7, hist
- Objective / loss: binary log loss (depth-wise gradient boosting, 2nd-order)
- CV macro F1 per fold: 0.9496, 0.9312, 0.9554, 0.9378, 0.9514 (mean 0.9451 ± 0.0091)
- CV top-1 per fold: 0.9081, 0.8800, 0.9208, 0.8824, 0.9155
- Student-test: top-1 0.9117, macro F1 0.9500, precision 0.9499, recall 0.9502, log loss 0.0584689205239129
- Over/under-fitting evidence: train top-1 0.9560 vs CV 0.9014 vs test 0.9117 (gap train-CV = +0.0547)
- Classifier-only route continuity: 0.9767
- Training time 3.5 s; inference 8.8 ms / 1000 obs
- Through the tuned HMM: top-1 0.9199 -> 0.9566; route F1 0.8302 -> 0.9911

## LogisticRegression  (Linear / probabilistic)

- Configuration: StandardScaler + L2, C=1.0
- Objective / loss: binary cross-entropy + L2 penalty (lbfgs)
- CV macro F1 per fold: 0.9481, 0.9315, 0.9496, 0.9352, 0.9516 (mean 0.9432 ± 0.0082)
- CV top-1 per fold: 0.9056, 0.8805, 0.9107, 0.8778, 0.9158
- Student-test: top-1 0.8890, macro F1 0.9372, precision 0.9370, recall 0.9373, log loss 0.0775016775370552
- Over/under-fitting evidence: train top-1 0.9002 vs CV 0.8981 vs test 0.8890 (gap train-CV = +0.0021)
- Classifier-only route continuity: 0.9772
- Training time 0.8 s; inference 0.8 ms / 1000 obs
- Through the tuned HMM: top-1 0.9007 -> 0.9408; route F1 0.8290 -> 0.9904

## HistGradientBoosting  (Boosting)

- Configuration: lr=0.1, max_iter=300, max_leaf_nodes=31
- Objective / loss: binary log loss (gradient boosting on histograms)
- CV macro F1 per fold: 0.9447, 0.9306, 0.9530, 0.9357, 0.9481 (mean 0.9424 ± 0.0082)
- CV top-1 per fold: 0.8997, 0.8789, 0.9166, 0.8786, 0.9099
- Student-test: top-1 0.9026, macro F1 0.9449, precision 0.9447, recall 0.9450, log loss 0.0644254772341246
- Over/under-fitting evidence: train top-1 0.9272 vs CV 0.8967 vs test 0.9026 (gap train-CV = +0.0305)
- Classifier-only route continuity: 0.9743
- Training time 1.9 s; inference 4.7 ms / 1000 obs
- Through the tuned HMM: top-1 0.9119 -> 0.9560; route F1 0.8088 -> 0.9909

## RandomForest  (Ensemble (bagging))

- Configuration: 300 trees, max_depth=20, min_leaf=10, sqrt features
- Objective / loss: Gini impurity per tree; bagging + feature subsampling
- CV macro F1 per fold: 0.9463, 0.9324, 0.9509, 0.9339, 0.9484 (mean 0.9424 ± 0.0077)
- CV top-1 per fold: 0.9025, 0.8820, 0.9128, 0.8754, 0.9103
- Student-test: top-1 0.8979, macro F1 0.9422, precision 0.9421, recall 0.9423, log loss 0.0626491347712176
- Over/under-fitting evidence: train top-1 0.9459 vs CV 0.8966 vs test 0.8979 (gap train-CV = +0.0493)
- Classifier-only route continuity: 0.9751
- Training time 97.9 s; inference 25.6 ms / 1000 obs
- Through the tuned HMM: top-1 0.9099 -> 0.9538; route F1 0.8121 -> 0.9906

## MLP  (Neural network)

- Configuration: 128-64 ReLU, Adam lr=1e-3, early stopping, 50% obs
- Objective / loss: binary cross-entropy, Adam, L2 alpha=1e-4, early stopping on 10% validation
- CV macro F1 per fold: 0.9473, 0.9321, 0.9520, 0.9315, 0.9483 (mean 0.9422 ± 0.0087)
- CV top-1 per fold: 0.9042, 0.8815, 0.9148, 0.8712, 0.9102
- Student-test: top-1 0.9042, macro F1 0.9458, precision 0.9457, recall 0.9460, log loss 0.0666065500493564
- Over/under-fitting evidence: train top-1 0.9236 vs CV 0.8964 vs test 0.9042 (gap train-CV = +0.0272)
- Classifier-only route continuity: 0.9763
- Training time 6.2 s; inference 1.7 ms / 1000 obs
- Through the tuned HMM: top-1 0.9092 -> 0.9516; route F1 0.8327 -> 0.9905

## LinearSVM_RBF_Nystroem  (Kernel-based)

- Configuration: Nystroem RBF(400 comps) + LinearSVC + sigmoid calib., 25% obs
- Objective / loss: squared hinge loss on Nystroem RBF features; sigmoid (Platt) calibration for probabilities
- CV macro F1 per fold: 0.9459, 0.9331, 0.9497, 0.9334, 0.9481 (mean 0.9420 ± 0.0073)
- CV top-1 per fold: 0.9018, 0.8833, 0.9108, 0.8746, 0.9097
- Student-test: top-1 0.8826, macro F1 0.9335, precision 0.9334, recall 0.9336, log loss 0.0775135964604062
- Over/under-fitting evidence: train top-1 0.9001 vs CV 0.8961 vs test 0.8826 (gap train-CV = +0.0040)
- Classifier-only route continuity: 0.9760
- Training time 8.7 s; inference 31.9 ms / 1000 obs
- Through the tuned HMM: top-1 0.8935 -> 0.9388; route F1 0.8110 -> 0.9895

## ExtraTrees  (Ensemble (bagging))

- Configuration: 300 trees, max_depth=22, min_leaf=10
- Objective / loss: Gini impurity with randomised thresholds; averaging
- CV macro F1 per fold: 0.9450, 0.9310, 0.9491, 0.9342, 0.9500 (mean 0.9419 ± 0.0078)
- CV top-1 per fold: 0.9002, 0.8796, 0.9098, 0.8761, 0.9131
- Student-test: top-1 0.8890, macro F1 0.9372, precision 0.9370, recall 0.9373, log loss 0.0669084976115441
- Over/under-fitting evidence: train top-1 0.9247 vs CV 0.8957 vs test 0.8890 (gap train-CV = +0.0290)
- Classifier-only route continuity: 0.9748
- Training time 14.6 s; inference 25.7 ms / 1000 obs
- Through the tuned HMM: top-1 0.9005 -> 0.9520; route F1 0.8125 -> 0.9906

## DecisionTree  (Tree-based)

- Configuration: max_depth=12, min_samples_leaf=50
- Objective / loss: Gini impurity reduction (greedy splits)
- CV macro F1 per fold: 0.9434, 0.9289, 0.9468, 0.9295, 0.9479 (mean 0.9393 ± 0.0084)
- CV top-1 per fold: 0.8973, 0.8760, 0.9057, 0.8676, 0.9095
- Student-test: top-1 0.8888, macro F1 0.9370, precision 0.9369, recall 0.9372, log loss 0.0810728119332467
- Over/under-fitting evidence: train top-1 0.9239 vs CV 0.8913 vs test 0.8888 (gap train-CV = +0.0326)
- Classifier-only route continuity: 0.9705
- Training time 7.8 s; inference 0.4 ms / 1000 obs
- Through the tuned HMM: top-1 0.9036 -> 0.9502; route F1 0.8106 -> 0.9912

## KNN  (Instance-based)

- Configuration: k=40, distance-weighted, standardised, 25% obs
- Objective / loss: none (lazy learner); probability = distance-weighted vote of 40 neighbours
- CV macro F1 per fold: 0.9378, 0.9268, 0.9489, 0.9263, 0.9449 (mean 0.9369 ± 0.0092)
- CV top-1 per fold: 0.8878, 0.8723, 0.9093, 0.8620, 0.9044
- Student-test: top-1 0.8793, macro F1 0.9316, precision 0.9315, recall 0.9318, log loss 0.0940415025942131
- Over/under-fitting evidence: train top-1 0.9238 vs CV 0.8871 vs test 0.8793 (gap train-CV = +0.0367)
- Classifier-only route continuity: 0.9714
- Training time 0.0 s; inference 555.0 ms / 1000 obs
- Through the tuned HMM: top-1 0.8895 -> 0.9481; route F1 0.8000 -> 0.9885

## GaussianNB  (Linear / probabilistic)

- Configuration: QuantileTransformer(normal) + GaussianNB
- Objective / loss: maximum likelihood of class-conditional Gaussians (after quantile-normal transform)
- CV macro F1 per fold: 0.9271, 0.9108, 0.9353, 0.9052, 0.9296 (mean 0.9216 ± 0.0115)
- CV top-1 per fold: 0.8691, 0.8446, 0.8856, 0.8249, 0.8782
- Student-test: top-1 0.8453, macro F1 0.9124, precision 0.9122, recall 0.9125, log loss 0.5700748397695714
- Over/under-fitting evidence: train top-1 0.8629 vs CV 0.8605 vs test 0.8453 (gap train-CV = +0.0024)
- Classifier-only route continuity: 0.9583
- Training time 0.9 s; inference 15.4 ms / 1000 obs
- Through the tuned HMM: top-1 0.8547 -> 0.9323; route F1 0.7253 -> 0.9840

## NearestRoad (baseline)  (Heuristic)

- Configuration: argmin perpendicular distance
- Objective / loss: none (heuristic)
- Student-test: top-1 0.5362, macro F1 0.7368, precision 0.7367, recall 0.7369, log loss N/A
- Classifier-only route continuity: 0.9169
- Training time 0.0 s; inference 0.0 ms / 1000 obs
- Through the tuned HMM: top-1 0.5587 -> 0.9319; route F1 0.6038 -> 0.9871
