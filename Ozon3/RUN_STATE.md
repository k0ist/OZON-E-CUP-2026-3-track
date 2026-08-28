# Run state — 2026-08-28

## Completed stages

- Production leakage-safe folds are fixed and were not rebuilt in this run.
- Clean pre-tuning LightGBM walk-forward OOF is complete and remains the primary reference.
- The bounded five-candidate LightGBM tuning run is complete; no candidates were added after viewing the held-out fold.
- CatBoost behavioral-feature walk-forward OOF is complete on exactly the same validation rows as LightGBM.
- CatBoost now uses lazy fold loading plus atomic, validated per-fold OOF/model checkpoints.
- Clean LightGBM versus CatBoost keyed alignment, fixed log blends, and expanding temporal meta-CV are complete.
- Fold-safe TabularResNet temporal OOF is complete on the same five validation folds and keyed rows.
- NN per-fold CSV, normalizer, three seed models, hashes, and metadata are crash-safe and resume-validated.
- Exact three-model alignment, fixed log blends, all-OOF diagnostic weights, and expanding temporal meta-CV are complete.
- Standalone NN was selected, memory-safely refit on all labeled snapshots, and used to create exactly one validated primary submission. Nothing was uploaded automatically.
- A later OOF-only tail-risk study supported a deterministic raw cap of `5,000` and created one separate clipped NN candidate without changing the original.
- The current ensemble/decision-only pass revalidated all canonical OOF artifacts and repeated the honest temporal model-selection gate. No model was trained and no ensemble submission was created.

## Clean LightGBM reference

- Legacy pooled OOF RMSLE: `1.7213300223011123`.
- Corrected clean pooled OOF RMSLE: `1.720624454980874`.
- OOF rows: `1,222,351`.
- Improvement versus legacy: `-0.0007055673202382984` RMSLE.

| Fold | Cutoff | RMSLE |
|---|---:|---:|
| fold_1 | 2025-09-16 | 1.725698710730942 |
| fold_2 | 2025-10-16 | 1.7145501849461056 |
| fold_3 | 2025-11-15 | 1.7332472524861082 |
| fold_4 | 2025-12-15 | 1.7414421836233143 |
| fold_5 | 2026-01-14 | 1.6879835749558092 |

Mean fold RMSLE is `1.7205843813484556`; fold standard deviation is `0.01855142414959783`.

## Bounded LightGBM tuning

Development evidence uses folds 1–4. Fold 5 was evaluated once, only for the candidate selected on development folds. These development scores are tuning evidence, not a new fully unbiased pooled OOF estimate.

| Candidate | fold_1 | fold_2 | fold_3 | fold_4 | Mean | Std | Selection score |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.725698710730942 | 1.7145501849461056 | 1.7332472524861082 | 1.7414421836233143 | 1.7287345829466174 | 0.009902789607957305 | 1.7303602419412478 |
| compact_31 | 1.7253482435850374 | 1.7146464411847546 | 1.7332286817160796 | 1.741867022024108 | 1.7287725971274948 | 0.010032386706761494 | 1.7304305570430016 |
| depth8_95 | 1.7255156970823933 | 1.7144165032517744 | 1.733233832138784 | 1.7413566861276684 | 1.728630679650155 | 0.0099358716374184 | 1.7302605671377724 |
| regularized_63 | 1.7256733362528633 | 1.7145608968864876 | 1.7330938513698826 | 1.7413388000438164 | 1.7286667211382625 | 0.009850326095749375 | 1.730285357693115 |
| wide_127 | 1.7259797211965318 | 1.7147382083393041 | 1.7329243956275766 | 1.7413723308036575 | 1.7287536639917676 | 0.009756453124276783 | 1.7303602426447897 |

Development winner: `depth8_95` with overrides:

```json
{
  "learning_rate": 0.025,
  "num_leaves": 95,
  "max_depth": 8,
  "min_data_in_leaf": 120,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.8,
  "lambda_l2": 2.0
}
```

- Held-out fold_5, `depth8_95`: `1.6879108646058907`.
- Held-out fold_5, baseline: `1.6879835749558092`.
- Exact tuned-minus-baseline delta: `-0.00007271034991851444`.
- This gain is extremely small. The clean baseline remains the primary LightGBM reference; no tuned pooled OOF was produced or presented as independent evidence.

## CatBoost temporal OOF

Configuration: CPU, `iterations=1500`, `learning_rate=0.04`, `depth=6`, RMSE on `log1p(target)`, seed 42, early stopping 100. Protocol for validation fold `i` is strictly `train = folds[:i]`.

| Fold | Cutoff | Rows | RMSLE | Best iteration | Source |
|---|---:|---:|---:|---:|---|
| fold_1 | 2025-09-16 | 236,668 | 1.7257053778093092 | 713 | recovered from exact validated CPU model |
| fold_2 | 2025-10-16 | 240,700 | 1.7143666492101044 | 812 | recovered from exact validated CPU model |
| fold_3 | 2025-11-15 | 244,983 | 1.7328411624056208 | 1349 | trained and checkpointed |
| fold_4 | 2025-12-15 | 250,000 | 1.741233531583124 | 1487 | trained and checkpointed |
| fold_5 | 2026-01-14 | 250,000 | 1.6880277080948431 | 364 | trained and checkpointed |

- Pooled RMSLE: `1.7204734227399536`.
- Mean fold RMSLE: `1.7204348858206004`.
- Fold standard deviation: `0.018446279374290043`.
- Latest fold RMSLE: `1.6880277080948431`.
- OOF rows: `1,222,351`.
- Robust final iteration would be median `[713, 812, 1349, 1487, 364] = 812`.
- Final all-fold refit was deliberately deferred because the current concat-all-snapshots implementation recreates the known RAM/OOM risk. Valid OOF evidence is preserved first.

## Clean LightGBM versus CatBoost diversity

Alignment key: `user_id + fold + cutoff_date`; exact aligned coverage is `1,222,351 / 1,222,351` rows.

- LightGBM RMSLE: `1.720624454980874`.
- CatBoost RMSLE: `1.7204734227399536`.
- Correlation of `log1p(pred)`: `0.9982875359212462`.
- Correlation of log residuals: `0.9985082257808938`.

Fixed log-space blend diagnostics:

| LGBM weight | CatBoost weight | RMSLE |
|---:|---:|---:|
| 0.25 | 0.75 | 1.7200310699439987 |
| 0.50 | 0.50 | 1.7199087753228957 |
| 0.75 | 0.25 | 1.7201066071422861 |

The best fixed blend checked is 50/50 at `1.7199087753228957`. The all-OOF optimized 47.05/52.95 score (`1.7199065470417387`) is only an in-sample diagnostic and is not treated as independent validation.

Expanding temporal meta-CV is valid on folds 2–5: each fold's weights are fit only on earlier OOF folds. It covers `985,683` rows and excludes the earliest `236,668` rows for which no earlier OOF exists.

- Expanding temporal blend RMSLE: `1.7186997256343872`.
- LightGBM standalone on the same coverage: `1.7194038683526547`.
- CatBoost standalone on the same coverage: `1.719214830634487`.

## Temporal neural-network OOF

Configuration was fixed before the run and was not changed from fold results: 436 common numeric features, TabularResNet with hidden width 256, two residual blocks, dropout 0.15, batch size 4096, seeds 42/1337/2026, AdamW, MSE on `log1p(target)`, at most 80 epochs, and patience 8. Each scaler was fit only on `folds[:i]`; validation was always `fold_i`. Dataset fingerprint is `08e2865810`.

All five checkpoints were independently resume-validated against the dataset fingerprint, exact feature/dtype schema, NN configuration, seeds, fold/cutoff/target window, scaler statistics and file hash, every seed-model payload/hash, validation IDs/order/targets, OOF CSV hash, and RMSLE. No fold was retrained during the validation pass.

| Fold | Cutoff | Rows | RMSLE | Best epochs: seed 42 / 1337 / 2026 |
|---|---:|---:|---:|---:|
| fold_1 | 2025-09-16 | 236,668 | 1.724955298207952 | 9 / 7 / 5 |
| fold_2 | 2025-10-16 | 240,700 | 1.7137532260845607 | 17 / 11 / 8 |
| fold_3 | 2025-11-15 | 244,983 | 1.7318989197293053 | 11 / 5 / 4 |
| fold_4 | 2025-12-15 | 250,000 | 1.7376863217488374 | 15 / 6 / 11 |
| fold_5 | 2026-01-14 | 250,000 | 1.6765474646565144 | 17 / 3 / 10 |

- Pooled NN OOF RMSLE: `1.7169843885804699`.
- Mean fold RMSLE: `1.7169682460854339`.
- Fold standard deviation: `0.02172617132028848`.
- Latest fold RMSLE: `1.6765474646565144`.
- OOF rows: `1,222,351`.
- Composite key `user_id + fold + cutoff_date` is unique; coverage exactly matches LightGBM and CatBoost; target/prediction values are finite and nonnegative; `pred_log == log1p(pred)`.
- Final NN refit is complete on all `1,455,328` labeled snapshots with the pre-committed seed epochs `42:15`, `1337:6`, and `2026:8`. CV artifacts remain unchanged.

## Three-model diversity and temporal ensemble

Exact alignment coverage is `1,222,351` rows for clean LightGBM, CatBoost, and NN.

Standalone pooled RMSLE:

| Model | Pooled RMSLE | Latest fold RMSLE |
|---|---:|---:|
| LightGBM | 1.7206244549808738 | 1.6879835749558092 |
| CatBoost | 1.720473422739954 | 1.6880277080948431 |
| NN | 1.7169843885804694 | 1.6765474646565144 |

`log1p(pred)` correlation matrix:

| | LightGBM | CatBoost | NN |
|---|---:|---:|---:|
| LightGBM | 1.0 | 0.9982875359213115 | 0.9967946621088727 |
| CatBoost | 0.9982875359213115 | 1.0 | 0.9968043437226867 |
| NN | 0.9967946621088727 | 0.9968043437226867 | 1.0 |

Log-residual correlation matrix:

| | LightGBM | CatBoost | NN |
|---|---:|---:|---:|
| LightGBM | 1.0 | 0.998508225781013 | 0.9972156941500718 |
| CatBoost | 0.998508225781013 | 1.0 | 0.9972152726481046 |
| NN | 0.9972156941500718 | 0.9972152726481046 | 1.0 |

Fixed log-space blends:

| Blend | Pooled RMSLE | Latest fold RMSLE |
|---|---:|---:|
| 50% LightGBM + 50% CatBoost | 1.7199087753228954 | 1.6873834894819582 |
| 50% LightGBM + 50% NN | 1.7175231682977614 | 1.6805050507426431 |
| 50% CatBoost + 50% NN | 1.7174465539815158 | 1.6804353128445138 |
| Equal thirds | 1.717936943460552 | 1.6823051293239821 |

The best nonnegative all-OOF log-space diagnostic has weights `LightGBM=0.057024817684265754`, `CatBoost=0.11724652891985994`, `NN=0.8257286533958743`, pooled RMSLE `1.7168455881894775`, and latest-fold RMSLE `1.677477908487001`. This is an in-sample weight-fit diagnostic, not independent validation evidence.

Expanding temporal three-model meta-CV fits weights only on earlier OOF folds and evaluates the next fold. It covers folds 2–5, or `985,683` rows:

| Validation fold | Historical-fit weights: LGBM / CatBoost / NN | RMSLE |
|---|---:|---:|
| fold_2 | 0.2811874679223661 / 0.22054120954998518 / 0.4982713225276488 | 1.7129112747620978 |
| fold_3 | 0.2514181844558538 / 0.24527504912099773 / 0.5033067664231484 | 1.7313493985468078 |
| fold_4 | 0.2132666130514582 / 0.2599022281496493 / 0.5268311587988925 | 1.7379310695479468 |
| fold_5 | 0.16208098440605392 / 0.21874912334415617 / 0.61916989224979 | 1.6791121119422545 |

- Three-model expanding temporal meta-CV RMSLE: `1.715420936741266`.
- Previous LightGBM/CatBoost temporal meta-CV: `1.7186997256343872` on the same `985,683` rows.
- Exact three-model minus two-model delta: `-0.003278788893121165`.
- Standalone NN on the same evaluable rows: `1.7150650133546534`, which is `0.000355923386612611` better than the three-model meta-CV blend.
- NN residual correlation is lower than the LightGBM/CatBoost residual correlation by about `0.001293`, but all correlations remain very high.
- NN clearly adds value and must be retained. However, the honest expanding evidence does not show that LightGBM/CatBoost improve over standalone NN; equal blends also degrade NN. Do not force all three models into the final solution solely because they are available.

## Final standalone NN refit and primary submission

Standalone NN is the final model. It was selected because its pooled OOF RMSLE is `1.7169843885804699`, its latest-fold RMSLE is `1.6765474646565144`, and on the honest common folds 2–5 its `1.7150650133546534` beats the expanding three-model meta-CV result `1.715420936741266`. The all-OOF optimized blend remains an in-sample diagnostic and was not deployed.

- Exact contract: 436 fixed common numeric features; TabularResNet `hidden_dim=256`, two residual blocks, dropout `0.15`; batch size `4096`; AdamW learning rate `0.001`, weight decay `0.0001`; MSE on `log1p(target)`; seeds `42`, `1337`, `2026`.
- Final rows: `1,455,328` across all six labeled snapshots.
- Fixed final epochs: seed 42 = `15`, seed 1337 = `6`, seed 2026 = `8` (`explicit_fixed_post_cv`).
- Runtime: CUDA, PyTorch `2.5.1+cu124`, NVIDIA GeForce RTX 2070.
- Memory strategy: folds were loaded sequentially into a `2,538,092,032`-byte float32 memmap on `I:`; it was scaled in place and deleted only after all three model checkpoints were atomically persisted. No pandas fold concatenation was used.
- Dataset fingerprint: `08e2865810`; feature schema SHA-256: `a077602129d78f98f041a9f41df39437eaf5817049f7f06181e545b70200fd7d`.
- Feature contract: `data/models/nn/features.json`, SHA-256 `60c4446918cf9839ac95387f74e00b6afa06b892462f053dcc725b0a9753176c`.
- Final normalizer: `data/models/nn/normalizer.npz`, file SHA-256 `6a3e770b8bd76afaa1668174f69c1dafc96e243f92e38fd8736dc2ddb01da1d6`, statistics SHA-256 `2bdab99200592b9e00778b4e1cb34b21e739d948729380b1de16675f7c3ae738`.
- Seed 42 model: `data/models/nn/model_seed_42.pt`, SHA-256 `29ab1eec1d2077650898de13ae4ef0cc9fcaa3860f92a9b3c8c05ea201525b13`.
- Seed 1337 model: `data/models/nn/model_seed_1337.pt`, SHA-256 `c650df73297d492e37180df0a0d9b6039c5ea33333becc31104eafd0d7af19af`.
- Seed 2026 model: `data/models/nn/model_seed_2026.pt`, SHA-256 `3b1bd25b66cddc38bc71d2931b459fc6499fbfe8dfac0c9e7a0fb29f75cfdb13`.
- Canonical manifest: `data/models/nn/manifest.json`, SHA-256 `6a24788ac87adbe75cad79ff50b1388f0ffb405373683121b3636fadfa0ca0d0`.

Final test predictions use exactly `expm1(max(mean(seed_pred_log), 0))`:

| Statistic | Test prediction | NN OOF prediction |
|---|---:|---:|
| min | 0.0 | 0.0 |
| median | 6.0195012102054 | 7.693299144914247 |
| mean | 108.02050972439486 | 1141.17476641686 |
| p90 | 84.2252891509418 | 109.78922493307056 |
| p95 | 150.57852678400724 | 192.34289152623825 |
| p99 | 425.04297200149085 | 509.2712444739565 |
| p99.9 | 1210.2538588918833 | 1449.051101505356 |
| max | 18304202.83339376 | 1340572377.8971505 |

Extreme-value diagnosis: robust test tails are below OOF (`p99.9` ratio `0.8352044021322667`), and no test prediction exceeds the pre-existing OOF maximum. The test maximum belongs to `user_id=168367`; its most extreme recent-GMV features are roughly 115–251 training standard deviations from the mean. Seed log-predictions are `13.779256820678711`, `21.976913452148438`, and `14.41175365447998`, so all seeds extrapolate upward and seed 1337 amplifies the effect. This user's historical 30-day GMV increased to `53,746.949219` in fold 5, but the resulting `18,304,202.83339376` prediction is still a raw-space pathology. It contributes `73.21681133357504` to the reported test mean; excluding only that maximum gives mean `34.80383760617024`. Existing NN OOF already contains a larger maximum (`1,340,572,377.8971505`), 26 rows above `10,000`, and one above `1,000,000`; excluding its single maximum gives OOF mean `44.458411262948104`. No arbitrary upper clipping or calibration was introduced after model selection; the warning is preserved because the required prediction contract has no pre-committed OOF-derived upper cap.

Primary submission: `submissions/submission_nn_primary.csv`, SHA-256 `0c34448922d4e57e975afa24e70ebb7368de9cbb6673778e65a4b14a60977fc2`. Independent validation passed: exactly `250,000` rows; columns exactly `user_id,predict`; exact official sample ID universe and order; `250,000` unique IDs; no NaN, infinity, negative predictions, or duplicate IDs. It remains unchanged. A later OOF-only tail-risk pass created the separate `submission_nn_primary_clipped.csv` candidate with a raw cap of `5,000`; neither file was uploaded automatically.

Inference report: `reports/nn_final_inference.json`. It records the distribution, OOF comparison, top 20 raw predictions, per-seed log disagreement, model/normalizer paths and hashes, and submission validation.

## Artifacts

- Clean LightGBM OOF: `data/oof/runs/lgbm_baseline/oof_lgbm.csv`.
- Clean LightGBM manifest: `data/models/runs/lgbm_baseline/lgbm/manifest.json`.
- Tuning table: `reports/runs/lgbm_tuned/lgbm_bounded_tuning.csv`.
- Tuning report: `reports/runs/lgbm_tuned/lgbm_bounded_tuning.json`.
- Selected tuning overrides: `data/runs/lgbm_tuned/best_params.json`.
- CatBoost combined OOF: `data/oof/oof_catboost.csv`.
- CatBoost checkpoints: `data/oof/oof_catboost_fold_1.csv` through `oof_catboost_fold_5.csv`, each with a matching `.meta.json` sidecar.
- CatBoost fold models: `data/models/catboost/model_fold_1.cbm` through `model_fold_5.cbm`.
- CatBoost manifest: `data/models/catboost/manifest.json`.
- CatBoost fold metrics: `reports/catboost_fold_metrics.csv`.
- LGBM/CatBoost ensemble diagnostics: `reports/lgbm_catboost_clean_ensemble.json`.
- Hash-bound ensemble manifest: `data/models/runs/lgbm_cb_clean/ensemble/manifest.json`.
- NN combined OOF: `data/oof/oof_nn.csv`.
- NN checkpoints: `data/oof/oof_nn_fold_1.csv` through `oof_nn_fold_5.csv`, each with a matching `.meta.json` sidecar.
- NN per-fold normalizers and seed models: `data/models/nn/normalizer_fold_*.npz` and `data/models/nn/model_fold_*_seed_*.pt`.
- NN manifest: `data/models/nn/manifest.json`.
- NN fold metrics: `reports/nn_fold_metrics.csv`.
- Final NN feature contract and normalizer: `data/models/nn/features.json` and `data/models/nn/normalizer.npz`.
- Final NN seed models: `data/models/nn/model_seed_42.pt`, `model_seed_1337.pt`, and `model_seed_2026.pt`.
- Final NN inference diagnostics: `reports/nn_final_inference.json`.
- Original primary NN submission: `submissions/submission_nn_primary.csv`.
- OOF-validated clipped NN candidate: `submissions/submission_nn_primary_clipped.csv`.
- Three-model diagnostics: `reports/lgbm_catboost_nn_clean_ensemble.json`.
- Hash-bound three-model ensemble manifest: `data/models/runs/lgbm_cb_nn_clean/ensemble/manifest.json`.

## Failures and limitations

- CatBoost GPU passed a small probe but failed with `bad allocation` on production fold_3; its partial GPU artifacts were not used.
- The original eager CPU runner failed with `bad allocation` because it retained all future folds in RAM. Lazy loading fixed the blocker and is preserved.
- The desktop app crashed during the first lazy CPU fold_3 fit. Fold_1/2 CPU models were validated and converted into exact checkpoints without retraining; folds 3–5 then completed.
- Fold_4 reached the configured 1500-round cap (best iteration 1487), so its optimum may lie slightly beyond the cap. No parameters or limits were changed after observing it.
- CatBoost final refit is not complete, and no CatBoost submission was generated.
- PyTorch was absent from the project environment; CUDA PyTorch 2.5.1+cu124 was installed in an isolated dependency directory on drive `I:` because drive `C:` lacked space. The RTX 2070 CUDA preflight and the complete run were stable.
- Final NN refit completed successfully. The only remaining model warning is the documented raw-space extrapolation tail; no post-selection clipping was introduced.

## Ensemble / decision-only checkpoint — 2026-08-27

External public scores are recorded only as diagnostics and were not used for any weight fit or model-selection decision: `submission_nn_primary.csv = 1.6657056695`; separate v4 branch benchmark `submission_v4_stable_logblend.csv = 1.6558771329789561`.

Canonical OOF validation passed for clean baseline LightGBM, CatBoost, and NN. Every file has `1,222,351` rows, a unique `user_id + fold + cutoff_date` key, finite nonnegative target/prediction values, and `pred_log == log1p(pred)` within `1.78e-15`. Fold rows are `236,668 / 240,700 / 244,983 / 250,000 / 250,000` at cutoffs `2025-09-16 / 2025-10-16 / 2025-11-15 / 2025-12-15 / 2026-01-14`. Alignment loses zero rows. LGBM and CatBoost targets are bit-identical; NN CSV serialization differs on 9,506 rows by at most `2.84e-14`, with zero differences above `1e-12`.

| Model | OOF SHA-256 | Pooled RMSLE | Latest fold |
|---|---|---:|---:|
| Clean LGBM | `9f1436c6c16b57140091a4ab264785cfe1c099f4744bb9a7d14978eddf56b256` | 1.720624454980874 | 1.6879835749558092 |
| CatBoost | `c67cc6e1d0042b5d175ff477a10184d04c0ccafbe4d35c129e790721c3b0c7bd` | 1.7204734227399536 | 1.6880277080948431 |
| NN | `c1db265a9e5eb3ed0e7165698a0e5102b99a3a0ed4096539c1475d048c50265d` | 1.7169843885804699 | 1.6765474646565144 |

`log1p(pred)` correlation:

| | LGBM | CatBoost | NN |
|---|---:|---:|---:|
| LGBM | 1.0 | 0.9982875359213115 | 0.9967946621088727 |
| CatBoost | 0.9982875359213115 | 1.0 | 0.9968043437226867 |
| NN | 0.9967946621088727 | 0.9968043437226867 | 1.0 |

Log-residual correlation, where residual is `pred_log - log1p(target)`:

| | LGBM | CatBoost | NN |
|---|---:|---:|---:|
| LGBM | 1.0 | 0.998508225781013 | 0.9972156941500718 |
| CatBoost | 0.998508225781013 | 1.0 | 0.9972152726481046 |
| NN | 0.9972156941500718 | 0.9972152726481046 | 1.0 |

Required fixed log-space blends:

| LGBM / CatBoost / NN | Pooled RMSLE |
|---:|---:|
| 75 / 25 / 0 | 1.720106607142286 |
| 50 / 50 / 0 | 1.7199087753228954 |
| 25 / 75 / 0 | 1.720031069943999 |
| 90 / 0 / 10 | 1.7197996975567391 |
| 75 / 0 / 25 | 1.7187540965930497 |
| 50 / 0 / 50 | 1.7175231682977614 |
| 60 / 25 / 15 | 1.7190033517234162 |
| 50 / 30 / 20 | 1.7186470806495846 |
| 40 / 30 / 30 | 1.7181033582856076 |

Every predeclared fixed blend is worse than standalone NN `1.7169843885804699`.

All-OOF nonnegative simplex weights are `LGBM=0.057024817684265754`, `CatBoost=0.11724652891985994`, `NN=0.8257286533958743`, with RMSLE `1.7168455881894775`. This is an in-sample deployment-fit diagnostic only and is not validation evidence.

Expanding temporal three-model meta-CV:

| Validation fold | Earlier-fold-fit LGBM / CatBoost / NN | RMSLE |
|---|---:|---:|
| fold_2 | 0.2811874679 / 0.2205412095 / 0.4982713225 | 1.7129112747620978 |
| fold_3 | 0.2514181845 / 0.2452750491 / 0.5033067664 | 1.7313493985468078 |
| fold_4 | 0.2132666131 / 0.2599022281 / 0.5268311588 | 1.7379310695479468 |
| fold_5 | 0.1620809844 / 0.2187491233 / 0.6191698922 | 1.6791121119422545 |

The meta gate evaluates folds 2–5, exactly `985,683` rows:

| Candidate | RMSLE | Delta versus standalone NN |
|---|---:|---:|
| Standalone NN | **1.7150650133546534** | 0 |
| CatBoost + NN expanding meta | 1.715326039878 | +0.000261027 |
| LGBM + NN expanding meta | 1.715371143953 | +0.000306131 |
| LGBM + CatBoost + NN expanding meta | 1.715420936741266 | +0.000355923386612611 |
| LGBM + CatBoost expanding meta | 1.7186997256343872 | +0.003634712 |

Chosen model set is standalone NN with deployment weights `NN=1.0`, `LGBM=0.0`, `CatBoost=0.0`. NN is included because it is the strongest standalone model on pooled OOF, latest fold, and identical expanding-meta rows. Both tree models are excluded because every honest NN-containing temporal ensemble is worse than NN alone. Their learned weights are also temporally nonstationary: the three-model NN weight rises from `0.4983` to `0.6192`, while LGBM falls from `0.2812` to `0.1621`; the all-OOF NN weight then jumps to `0.8257`.

Deployment artifact gate also fails independently: clean LGBM has only CV models and declares `final_model=null`; CatBoost declares `final_refit_completed=false` and `final_model=null`. Neither has a canonical test-prediction artifact. NN final models and its primary test predictions remain fully hash-valid. Separate v2/v4 files are not substitutes for the clean OOF-bound tree models.

No `submission_ensemble_primary.csv` was created. The refreshed hash-bound report and manifest are `reports/lgbm_catboost_nn_clean_ensemble.json` and `data/models/runs/lgbm_cb_nn_clean/ensemble/manifest.json`, both SHA-256 `5a45ba7405928855f1700d438d7feb07a4c479b7b62bb2ea19b7591b1df99027`.

## Final status

The ensemble gate fails. Do not create or upload a tree/NN ensemble: standalone NN is better by `0.000261027` than the best honest NN-containing temporal pair and by `0.000355923` than the three-model temporal ensemble on identical rows. The existing OOF-validated clipped NN candidate remains a separate standalone post-processing decision; this ensemble-only run did not modify it or use leaderboard results to select it.

## Exact public-v4 same-fold diagnostic — 2026-08-27

The exact deployed `submission_v4_stable_logblend.csv` recipe is the equal log-space mean of `deep95`, `depth8`, and a hurdle component using **global** isotonic calibration of `P(target > 0)`. The old reported `~1.716745` used segment isotonic and is not attributed to the public-v4 recipe. Saved final v4 models and calibrator reproduce the public-v4 CSV to maximum absolute difference `2.27e-13`.

For the strict same-fold projection, the calibrator remains global (not segment-specific), but is fitted only on earlier raw OOF rows: fold 1 uses cutoff `2025-08-17` (`232,977` fit rows); folds 2–5 use all prior cutoffs, reaching `1,205,328` fit rows for fold 5. The isotonic configuration is `out_of_bounds="clip", y_min=0, y_max=1`, mapping raw hurdle `p_buy` to `1[target>0]`. This removes the future-fold leakage in the legacy leave-one-fold-out calibration.

- Newly trained: auxiliary hurdle components at `2025-08-17`; all four missing v4 components for fold 1 at `2025-09-16`.
- Reused/reconstructed without retraining: raw direct/hurdle OOF for folds 2–5 from `data/oof_v4/oof_*.parquet`.
- Canonical output: `data/oof/runs/v4_stable_same_folds/oof_v4_stable.csv`, SHA-256 `b121b143b285cebd2c382b34ce4524a455219e64296bacdc4680751b6b542b60`.
- Report: `reports/v4_stable_nn_same_folds.json`, SHA-256 `001857fc29fa002242d1ed1861250fe38bba024e79dc22b7dfe31d3d7ece2f3b`.
- Run manifest: `data/oof/runs/v4_stable_same_folds/manifest.json`, SHA-256 `1c456c7ae42494a571a92d2ac59c8ab4414fb3b67077b0988255f6ef759cd5f6`.

The OOF has exactly `1,222,351` rows and columns `user_id,fold,cutoff_date,target,pred,pred_log`; composite keys are unique; all values are finite and nonnegative; all NN keys align with zero row loss; canonical targets match within `2.84e-14` CSV round-off; `pred_log == log1p(pred)` within `1.78e-15`.

| Fold | Exact v4 RMSLE | Canonical NN RMSLE |
|---|---:|---:|
| fold_1 | 1.723823540698881 | 1.724955298207952 |
| fold_2 | 1.7136752887600502 | 1.7137532260845607 |
| fold_3 | 1.7315384103623477 | 1.7318989197293053 |
| fold_4 | 1.7392447096263832 | 1.7376863217488372 |
| fold_5 | 1.683333298734295 | 1.6765474646565144 |
| **Pooled** | **1.718356214142841** | **1.7169843885804699** |

The latest-fold v4 RMSLE is `1.683333298734295`. Pooled log-prediction correlation v4/NN is `0.9962443870816545`; pooled log-residual correlation is `0.996690737350526`.

| Fold | Log-prediction correlation | Log-residual correlation |
|---|---:|---:|
| fold_1 | 0.9964107241605753 | 0.9967670929770914 |
| fold_2 | 0.996871858657849 | 0.9971961942447709 |
| fold_3 | 0.9968319037927605 | 0.9972727357296187 |
| fold_4 | 0.9970017716675555 | 0.997320298654565 |
| fold_5 | 0.9956789491208757 | 0.996021101487755 |

Fixed v4/NN log-space blends:

| v4 / NN | fold_1 | fold_2 | fold_3 | fold_4 | fold_5 | Pooled |
|---:|---:|---:|---:|---:|---:|---:|
| 95 / 5 | 1.7236125354817358 | 1.7134461156604717 | 1.7313197735363 | 1.738866977970894 | 1.682629447764342 | 1.7180068660450012 |
| 90 / 10 | 1.7234296793877566 | 1.713241450369738 | 1.7311260257750312 | 1.738520736214381 | 1.6819637651364185 | 1.7176870111558655 |
| 80 / 20 | 1.7231484490351305 | 1.712905677302423 | 1.7308132298061885 | 1.737922795970353 | 1.6807470837114185 | 1.717135845409217 |
| 70 / 30 | 1.7229799047929726 | 1.7126680272359982 | 1.7306000764560237 | 1.7374510190164816 | 1.6796835866154578 | 1.7167028305942431 |
| 60 / 40 | **1.7229240797316803** | 1.7125285410204782 | 1.7304866025450896 | 1.7371055081460647 | 1.678773564973748 | 1.7163880561332139 |
| 50 / 50 | 1.7229809848076518 | **1.712487242642909** | **1.7304728276825065** | **1.7368863387117102** | **1.6780172684845445** | **1.716191587087125** |

Folds 2–5 overlap the historical periods used to select the v4 recipe, so their scores and the pooled ranking are **model-selection-exposed diagnostics**, not fully independent validation. Fold 1 was not part of the old v4 selection and is reported separately: v4 beats NN by `0.001131757509071`, while the best predeclared fold-1 blend is 60/40 at `1.7229240797316803`, an improvement of `0.0008994609672007` versus standalone v4.

Diagnostic conclusion: v4+NN is promising. Among the predeclared pooled blends, 50/50 is best at `1.716191587087125`, improving standalone v4 by `0.002164627055716`. This is not yet a deployment decision because four of five folds are selection-exposed. No v4+NN submission was created or uploaded.

## Final v4/NN production assembly — 2026-08-27

The deployment weight was fixed before the final tail diagnostic at `exact-public-v4=0.60`, `NN=0.40`. It was selected from the primary selection-independent fold 1 evidence: standalone v4 `1.723823540698881`, fixed 60/40 `1.7229240797316803`, and fixed 50/50 `1.7229809848076518`. No weight was searched or changed during production assembly.

The only permitted NN-tail check compared the fixed 60/40 formula with raw NN versus the pre-established raw cap `5,000` applied to the NN component before `log1p`. No other cap or blend weight was evaluated:

| Scope | Raw NN component | NN capped at 5,000 | Delta capped - raw | OOF rows capped |
|---|---:|---:|---:|---:|
| fold_1 | 1.722924079731680 | 1.722921368387861 | -0.000002711343820 | 10 |
| fold_2 | 1.712528541020478 | 1.712524697160617 | -0.000003843859861 | 17 |
| fold_3 | 1.730486602545090 | 1.730489303098919 | +0.000002700553829 | 30 |
| fold_4 | 1.737105508146065 | 1.737107009214933 | +0.000001501068869 | 20 |
| fold_5 | 1.678773564973748 | 1.678770418082312 | -0.000003146891436 | 23 |
| **Pooled** | **1.716388056133214** | **1.716387000849700** | **-0.000001055283514** | **100 / 1,222,351** |

The capped version improves pooled OOF and folds 1, 2, and 5. The fold 3/4 regressions are only `1.5e-6` to `2.7e-6`, so they are not a meaningful temporal inconsistency. Version B was selected, while its blend-level gain is explicitly negligible.

Final formula:

`nn_safe = min(nn_pred, 5000)`

`final_log = 0.60 * log1p(exact_public_v4_pred) + 0.40 * log1p(nn_safe)`

`predict = expm1(final_log)`

Exact source bindings:

- Official sample: `uploads/sample_submit.csv`, SHA-256 `06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1`.
- Exact public-v4 predictions: `submissions/submission_v4_stable_logblend.csv`, SHA-256 `fb17e2fe9bf948cd2bca907aae9a854d5ccef05520db7a3ab0742d9f993b6ab9`.
- Final raw NN predictions: `submissions/submission_nn_primary.csv`, SHA-256 `0c34448922d4e57e975afa24e70ebb7368de9cbb6673778e65a4b14a60977fc2`.
- Existing clipped-NN reference, left unchanged: `submissions/submission_nn_primary_clipped.csv`, SHA-256 `8e9d07fbb16e75708d276a415ffcd237b2cd075b748e4d2fff9417aaad45bbb8`.

The cap was applied directly to the raw NN values. The clipped reference matches `min(raw NN, 5000)` within CSV precision (`4.55e-13` maximum absolute difference). Exactly `16 / 250,000` test NN predictions were capped.

Final submission: `submissions/submission_v4_nn_60_40_primary.csv`, SHA-256 `8d0fbdea2692b8deb5f457e6d61da77e585347e5b07dbdc299c7b2d3f90e5f67`.

| Statistic | Final 60/40 | Exact public v4 | Raw NN | Safe NN |
|---|---:|---:|---:|---:|
| min | 0.005851273431426877 | 0.0 | 0.0 | 0.0 |
| median | 6.157686385014459 | 6.267406576043051 | 6.0195012102054 | 6.0195012102054 |
| mean | 34.63712639630449 | 35.10681197132502 | 108.02050972439486 | 34.4098982362134 |
| p90 | 87.21206678367525 | 89.58376733538014 | 84.2252891509418 | 84.2252891509418 |
| p95 | 153.1778704405594 | 156.29852272370118 | 150.57852678400724 | 150.57852678400724 |
| p99 | 428.48388575154655 | 432.10764880216766 | 425.04297200149085 | 425.04297200149085 |
| p99.9 | 1135.9982210613684 | 1098.9867653216938 | 1210.2538588918833 | 1210.2538588918833 |
| max | 2831.9308900260557 | 1956.1554580064744 | 18304202.83339376 | 5000.0 |

Top 20 final predictions:

| Rank | user_id | Final | Exact v4 | NN raw | NN safe |
|---:|---:|---:|---:|---:|---:|
| 1 | 599863 | 2831.930890 | 1938.495294 | 5288.792662 | 5000.0 |
| 2 | 535648 | 2821.298100 | 1926.378024 | 14697.769560 | 5000.0 |
| 3 | 874034 | 2806.662348 | 1909.748644 | 8965.890143 | 5000.0 |
| 4 | 855200 | 2786.187293 | 1886.581385 | 9304.051072 | 5000.0 |
| 5 | 853521 | 2780.939201 | 1956.155458 | 4713.450724 | 4713.450724 |
| 6 | 282596 | 2766.896231 | 1864.857350 | 5068.266948 | 5000.0 |
| 7 | 168367 | 2745.629514 | 1841.025202 | 18304202.833394 | 5000.0 |
| 8 | 474806 | 2741.129351 | 1835.997897 | 6491.223577 | 5000.0 |
| 9 | 485305 | 2737.749679 | 1884.321056 | 4794.157455 | 4794.157455 |
| 10 | 596480 | 2710.160836 | 1930.678794 | 4507.020035 | 4507.020035 |
| 11 | 705221 | 2702.515525 | 1793.087182 | 5465.880457 | 5000.0 |
| 12 | 466880 | 2593.223047 | 1673.843707 | 5382.368322 | 5000.0 |
| 13 | 245048 | 2568.067021 | 1870.084480 | 4132.314872 | 4132.314872 |
| 14 | 891391 | 2566.894950 | 1918.655648 | 3971.904602 | 3971.904602 |
| 15 | 117556 | 2540.662502 | 1876.524074 | 4002.284294 | 4002.284294 |
| 16 | 1857 | 2438.875046 | 1869.218535 | 3634.636275 | 3634.636275 |
| 17 | 300258 | 2428.729719 | 1949.776489 | 3376.408915 | 3376.408915 |
| 18 | 613768 | 2414.011164 | 1866.235766 | 3551.213990 | 3551.213990 |
| 19 | 860574 | 2307.853588 | 1902.049249 | 3084.431742 | 3084.431742 |
| 20 | 673716 | 2294.820663 | 1842.835940 | 3188.785554 | 3188.785554 |

Independent final validation passed: exactly `250,000` rows and `250,001` physical CSV lines including the header; columns exactly `user_id,predict`; exact official sample user order and universe; `250,000` unique IDs; zero duplicate IDs, missing IDs, NaN, infinity, negative predictions, or schema errors. The independently recomputed formula matches the final CSV within `4.55e-13`. At this production-assembly checkpoint it was the only v4+NN candidate in `submissions/`; the later predeclared 95/5 candidate is documented below. Nothing was uploaded automatically.

Production assembly report: `reports/v4_nn_60_40_primary.json`, SHA-256 `1b2048c14e97c2c9e4786a429dbb5e38a8aa109784eaeb1bd0b798f71eaffa18`.

## Conservative fixed 95/5 candidate — 2026-08-28

New public evidence, recorded diagnostically:

- Exact-public-v4 stable public RMSLE: `1.6558771329789561`.
- Fixed 60% v4 + 40% NN public RMSLE: `1.6575178285026597`.
- Interpretation: the 40% NN share did not transfer to the public target period; it was worse than standalone v4 by `0.0016406955237036`.
- The next candidate was fixed in advance by the user at `95% v4 + 5% NN`. No other weight was searched or produced in this run, and public evidence was not used for any within-run optimization.

The NN tail rule remains unchanged: `nn_safe = min(nn_pred, 5000)` before log blending. Final formula:

`final_log = 0.95 * log1p(exact_public_v4_pred) + 0.05 * log1p(nn_safe)`

`predict = expm1(final_log)`

The only same-fold consistency check used the fixed 95/5 formula with cap 5,000:

| Scope | RMSLE | NN OOF rows capped |
|---|---:|---:|
| fold_1 | 1.7236126183471936 | 10 |
| fold_2 | 1.7134465316746406 | 17 |
| fold_3 | 1.7313210722669468 | 30 |
| fold_4 | 1.7388686644486133 | 20 |
| fold_5 | 1.6826349211253273 | 23 |
| **Pooled** | **1.7180086716475782** | **100 / 1,222,351** |

Exact source bindings were unchanged from the validated 60/40 production run:

- Official sample: `uploads/sample_submit.csv`, SHA-256 `06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1`.
- Exact-public-v4: `submissions/submission_v4_stable_logblend.csv`, SHA-256 `fb17e2fe9bf948cd2bca907aae9a854d5ccef05520db7a3ab0742d9f993b6ab9`.
- Final raw NN: `submissions/submission_nn_primary.csv`, SHA-256 `0c34448922d4e57e975afa24e70ebb7368de9cbb6673778e65a4b14a60977fc2`.
- Existing clipped-NN reference, unchanged: `submissions/submission_nn_primary_clipped.csv`, SHA-256 `8e9d07fbb16e75708d276a415ffcd237b2cd075b748e4d2fff9417aaad45bbb8`.

The cap affected exactly `16 / 250,000` test NN values. Final candidate distribution:

| Statistic | Prediction |
|---|---:|
| min | 0.0009779715389175126 |
| median | 6.255262894934484 |
| mean | 35.03636675850828 |
| p90 | 89.31476828375612 |
| p95 | 155.83568701502892 |
| p99 | 432.4321655810656 |
| p99.9 | 1101.3886163722018 |
| max | 2044.10500780647 |

Final candidate: `submissions/submission_v4_nn_95_05_primary.csv`, SHA-256 `0ec0d8ee6c3e3d728bc4fb33af47bf4816c3065e80892264de6889b5207c400a`.

Independent validation passed: exactly `250,000` rows (`250,001` physical lines including header); columns exactly `user_id,predict`; exact official user universe and order; zero missing/extra/duplicate IDs; zero NaN, infinity, or negative predictions. Independent formula reproduction matches the stored CSV within `2.27e-13`. The existing earlier 60/40 file was not changed; no 90/10, 97.5/2.5, or other new candidate was created. Nothing was uploaded automatically.

Subsequent public leaderboard evidence:

- Fixed 95% v4 + 5% NN public RMSLE: `1.6559149404776174`.
- It is worse than exact v4 stable by `0.0000378074986613` RMSLE.
- Exact v4 stable therefore remains the best known public result at `1.6558771329789561`.
- Both tested NN shares failed to improve exact v4 on the public target period.
  Stop public blend-weight tuning; treat the local/public disagreement as
  evidence of temporal/domain shift for the next predeclared research cycle.
