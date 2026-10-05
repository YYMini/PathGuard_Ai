# Reproducibility

## Environment and prerequisites

검증 환경은 Windows, Python **3.11.9**, CPU inference다. 실제 버전은 [stage7_environment.lock.txt](../configs/stage7_environment.lock.txt)에 고정했다. 기존 requirements.txt는 수정하지 않았다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configs/stage7_environment.lock.txt
python --version
```

Binary model, scaler, 원본 GeoLife, 가공 CSV와 full prediction은 Git에 포함되지 않는다. **Fresh clone만으로 최종 평가와 demo가 실행되지는 않는다.** 원본 보존자의 frozen artifact bundle과 GeoLife 입력이 필요하다. 공개된 docs/artifacts/stage7의 summary와 manifest는 수치·hash 검토용이며 모델 복원이나 원본 예측 검증을 대체하지 않는다. 서로 다른 sklearn/PyTorch/joblib 버전의 model load 호환성을 보장하지 않는다.

실제 사용 stack은 Python/Pandas/NumPy/scikit-learn/PyTorch/Folium/Matplotlib/Numba다. 폴더를 임의로 바꾸거나 frozen file의 line ending을 바꾸면 hash 검증이 실패한다. .gitattributes는 새 파일 LF와 기존 frozen CRLF 파일을 명시해 Git checkout 뒤 byte identity를 보존한다.

## Dataset structure

[Microsoft GeoLife download](https://www.microsoft.com/en-sg/download/details.aspx?id=52367)와 [공식 user guide](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)를 따른다. 실제 원본 경로는 protocol/cohort CLI 기본값을 확인한다.

```text
data/raw/geolife/Data/
  000/Trajectory/*.plt
  ...
  039/Trajectory/*.plt
data/processed/stage5/geolife_u000_019_first5_seed42_dedup/
data/processed/stage7/final_unseen_u020_039_first5/
  final_normal.csv
  final_synthetic.csv
  final_evaluation.csv
  input_manifest.csv
  excluded_duplicates.csv
  quality_summary.csv
  synthetic_anomaly_manifest.csv
  final_manifest.json
```

Final selection은 users020–039 각각 lexical filename first5, missing user report/fail이다. 기존 quality 기준과 fingerprint exclusion 뒤 eligible source당4개 synthetic sample을 생성한다. 사용자나 trajectory replacement는 없다. User021의 zero eligible 결과를 보존한다.

## Frozen artifacts

모든 model/config/scaler path, SHA256, training users, 86,067 train rows, threshold,8 feature list는 [final_model_manifest.json](artifacts/stage7/final_model_manifest.json)에 있다. **이 파일에 기재된 경로 그대로** bundle을 복원한다. 대표 경로:

```text
models/stage5/geolife_u000_019_first5_seed42_dedup/
outputs/metrics/stage6/alert_aggregation/locked_alert_policies.json
outputs/metrics/stage7_protected_snapshot.json
outputs/metrics/stage7/final/
  final_model_manifest.json
  prior_user_audit.json
  final_protocol_manifest.json
  final_cohort_integrity.json
  final_results_manifest.json
  predictions_*.csv
  final_*.csv
```

Model family는 Rule1개, IF/AE 각각 seed7/21/42/100/2026이다. Existing artifact를 사용했으며 reconstruction은 하지 않았다. Demo는 선택한 frozen model/scaler/config, generator source hashes, Stage6 alert lock, protocol과 seal receipt가 필요하며 Final CSV는 입력으로 사용하지 않는다. 전체 verifier에는 Stage1–6 protected578파일, complete final data/results bundle이 추가로 필요하다.

## Historical first execution: already completed

다음은 실제 완료한 순서의 기록이다. **현재 완료된 output 위에서 다시 실행하는 명령이 아니다.** 최초 평가 전에 protocol/config와 evaluator 세 파일을 확정하고 full515tests를 통과시킨 후 freeze commit을 원격에 게시했다.

```powershell
python -m src.stage7_protocol --phase freeze
python -m unittest discover -s tests -v
git add configs/stage7_final_protocol.json src/stage7_protocol.py src/prepare_final_cohort.py src/evaluate_final_validation.py src/verify_final_validation.py tests/test_stage7_final_validation.py
git commit -m "stage7: freeze final validation protocol"
# publish freeze commit before reading Final cohort
python -m src.stage7_protocol --phase seal
python -m src.prepare_final_cohort
python -m src.evaluate_final_validation
```

Protocol precommit SHA **3b267f7c9d6593facc4135c4992d5f2c1144d519**; protocol SHA **7e8ee719f8b0529fc753366867fa71966fe9aa97cda56f2db7e8798da446a3c4**. Original CLI authentication failed; GitHub connector published the identical Git tree and local HEAD was aligned before sealing. Local original freeze commit82c6e17 and published3b267f7 have identical tree473751e48df202136cf5e014093fe6a2fc9ad2eb. The published SHA is the authoritative receipt.

UTC chronology: seal18:26:35 → cohort18:29:10 → inference18:30:30 → results18:34:54 on2026-10-05. Evaluation is first successful actual11-model run, about263.88seconds. Unit tests use temporary small fixtures, not the real Final cohort. Evaluation attempt and result manifest refuse overwrite; cohort preparation refuses an existing data directory.

After squash, the freeze commit is retained in PR history. A fresh checkout needs its commit object for git-show evidence:

```powershell
git fetch origin 3b267f7c9d6593facc4135c4992d5f2c1144d519
```

## Verify existing results: recommended now

Restore the exact saved bundle before running these commands. They verify hashes and recompute from **saved** prediction files; they do not run model scoring or train.

```powershell
python -m src.stage7_protocol --phase verify
python -m src.evaluate_final_validation --verify-existing
python -m src.verify_final_validation
python -m unittest discover -s tests -v
```

Independent verifier checks confusion counts, ROC/AP, all220user entries, user macro/user SD, anomaly-type counts, seed ddof1 mean/std/min/max,1,848 event-policy and1,848 normal-policy scalar replay, protocol/model/threshold/hash/lineage/protected bytes. NA for no rows, missed delay, single deterministic seed SD is declared in final_metric_na_report.csv.

An identical deterministic verification rerun is allowed only in a separate workspace with archived protocol/seal/model/protected dependencies. It must preserve the original run, record that it is verification, and retain identical evaluator/config. Its timestamps/provenance manifest hashes may differ even when metric CSVs agree. Seeing these final results cannot make a rerun a new untouched holdout. A bug affecting metrics would require explicit invalidation, original-result preservation, a fix commit and separately named corrected run; no such corrected run was needed.

## Figures and public summaries

Seven figures were produced from frozen CSV summaries by src/create_final_figures.py, without recomputing metrics. Source CSV hashes, PNG hashes and public-copy hashes are in [figure manifest](artifacts/stage7/final_figure_manifest.json).

```powershell
# Historical figure step; existing outputs intentionally reject overwrite.
python -m src.create_final_figures
```

Public summary CSV/manifest copies preserve exact bytes; full predictions and raw GPS stay out of Git. Reading a figure or summary does not invoke the evaluator.

## Inference demo

Use a **new** output directory on each run. Input timestamp must be parseable and strictly increasing within each trajectory, coordinates finite/in range, required columns present. Altitude/user_id/trajectory_id are optional. Missing/invalid CSV is rejected; Stage5 quality-invalid feature rows remain unknown, rather than forced-normal.

```powershell
python -m src.run_pathguard_demo --input examples/demo_trajectory.csv --detector ae --seed 42 --output-dir outputs/demo/public_ae42 --map
python -m src.run_pathguard_demo --input examples/demo_trajectory.csv --detector if --seed 42 --output-dir outputs/demo/public_if42
python -m src.run_pathguard_demo --input examples/demo_trajectory.csv --detector rule --output-dir outputs/demo/public_rule
```

Seed42 is the prior Stage4/5 canonical seed, not a best-Final-seed selection. Example is200 analytically constructed points (sine/cosine around an artificial origin), not observed human GPS or Final020–039. The formula is in examples/README.md. Three public-example CLI smokes produce200-row CSVs; exact notification counts are recorded in their manifests. This example is not a labelled accuracy benchmark.

demo_predictions.csv contains timestamp/coordinates/trajectory/features/anomaly_score/nullable point_prediction/notification/prediction_status. demo_manifest.json records model/scaler/threshold/input/output hashes and policy. Optional Folium demo_route.html reuses the existing map; multi-trajectory input map shows only the first trajectory. Blue normal/red predicted anomaly/amber notification/gray unknown. Full CSV retains all trajectories.

## Verification status and boundaries

Full543tests PASS,85new over458baseline. Frozen578files unchanged, independent verifier passed. First final result manifest SHA **6b3b22b5a7b65bc286663577d6c2ac0264eb32c0bfdd2caf9332a5459c20cfa7**. Metric-producing files are src/stage7_protocol.py, src/prepare_final_cohort.py, src/evaluate_final_validation.py plus protocol and existing frozen sources. They were not edited after successful results. Demo/docs/figure generation do not retune models or metrics.

Reproducible scientific inference and offline replay are the validated scope. This is not real anomaly ground truth, production online throughput, operational incident precision or safety validation.
