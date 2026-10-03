# Stage 5.2: Single-seed Unseen-user Evaluation

Stage 5.2에서는 Stage 4와 동일한 Autoencoder를 유지한 채 unseen-user 일반화 성능을 평가했습니다. Test F1은 0.2098, ROC-AUC는 0.7225로 Stage 4 단일 사용자 평가보다 낮아졌습니다. 이상 유형별 direction_change Recall은 0.8026으로 높았으나 long_stop은 0.0727로 낮았습니다. 사용자별 데이터량 불균형 때문에 전체 row-level 지표와 user macro 지표를 함께 분석했습니다.

Stage 4와 동일한 row-level Autoencoder를 처음 보는 사용자에게 적용하는 실험입니다. 성능 개선이나 모델 간 우열 판정이 목적이 아닙니다. Stage 4는 사용자 000의 다른 경로를 평가했고, Stage 5는 사용자 단위 holdout을 평가합니다. 데이터 규모·경로·이상 비율도 다르므로 변화량을 일반화 조건의 차이와 함께 해석합니다.

## 실행과 고정 조건

```powershell
.\.venv\Scripts\python.exe -m src.train_multiuser_autoencoder
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

CLI는 데이터 경로와 출력 root만 받습니다. 실제 실험은 seed 42 한 번이며, epochs=100, batch_size=64, learning_rate=0.001, patience=15, threshold_percentile=95를 사용합니다. 기존 출력이 있으면 덮어쓰지 않고 실패합니다. 다른 seed·threshold percentile도 거부합니다. 테스트의 임시 fixture에서는 실행 시간을 줄이기 위해 2 epoch를 사용합니다.

`src.autoencoder.Autoencoder`, Stage 4의 `TrainingOptions`, seed 설정, scaler, `train_model`, reconstruction error 함수를 재사용합니다. Encoder 8→16→8→4, Decoder 4→8→16→8, hidden ReLU, 마지막 Linear 출력, MSELoss, Adam입니다. Stage 4 구현을 수정하지 않았습니다.

Feature 순서는 `time_diff_sec`, `distance_m`, `speed_mps`, `acceleration_mps2`, `direction_change_deg`, `stop_duration_sec`, `bearing_sin`, `bearing_cos`입니다. `bearing_deg`에서 sin/cos만 변환하며 좌표·timestamp·ID·label·quality metadata를 모델에 입력하지 않습니다.

## 데이터와 선택 경계

`data/processed/stage5/geolife_u000_019_first5_seed42_dedup/`을 읽기 전용으로 사용합니다. 실행 전 Stage 5.1 checksum·split·dedup·lineage를 재검증하고, 실행 후 입력 checksum을 다시 확인합니다. 구조 검증은 전체 CSV를 읽지만 Test 점수·성능은 checkpoint와 threshold 확정 이후에만 계산합니다.

- Train: 원본 정상·품질 유효 86,067행. 사용자 12명·51경로. 합성 및 exact duplicate 제외.
- Validation: 원본 정상 19,448행 + 품질 유효 합성 이상 1,113행. 사용자 4명·12경로.
- Test: 원본 정상 46,299행 + 품질 유효 합성 이상 2,112행. 사용자 4명·19경로.

Stage 5.1의 `validation_evaluation.csv`, `test_evaluation.csv`를 직접 사용합니다. 이미 제외된 합성 정상 복사 구간을 다시 포함하지 않습니다. Stage 4의 위치 기반 품질 연결을 호출하지 않고 명시적 `(source_trajectory_id, source_point_index)`와 저장된 `source_quality_valid`를 사용합니다. 입력을 조용히 필터링하거나 split을 변경하지 않으며, 부적절한 행이 있으면 실패합니다.

`fit_and_select()`는 Train과 Validation만 인자로 받습니다. StandardScaler는 Train 정상 86,067행에만 fit하고, Validation 정상만 transform해 trainer에 전달합니다. checkpoint는 Validation normal MSE 최소 epoch입니다. best checkpoint를 불러온 뒤 Validation 원본 정상 error의 95th percentile을 threshold로 저장합니다. 예측은 `error > threshold`입니다. Validation 합성 이상은 학습·early stopping·threshold에 들어가지 않습니다.

모델·scaler·선택 config를 저장한 뒤 Test CSV를 inference용으로 읽어 transform·평가합니다. Test 결과에 따라 선택 값이나 설정을 변경하지 않습니다.

## 평가와 NA

전체 및 Test 사용자별 Accuracy, Precision, Recall, F1, FPR, ROC-AUC, Average Precision(AP), TN/FP/FN/TP를 저장합니다. AP는 `sklearn.metrics.average_precision_score`이고 Stage 5에서 PR-AUC라는 명칭을 사용하지 않습니다.

한 클래스만 존재하면 ROC-AUC/AP는 NA이고 이유는 `requires_normal_and_anomaly_classes`입니다. 실제 이상 행이 없으면 Recall/F1은 NA, 원본 정상 행이 없으면 FPR은 NA, 예측 이상 행이 없으면 Precision은 NA입니다. JSON에서는 null과 `metric_na_reasons`, CSV에서는 NA와 `na_reasons`를 저장합니다. 실제 이상 행이 있지만 하나도 탐지하지 못한 경우 Recall/F1=0은 정의된 값입니다.

사용자 macro는 사용자별 정의된 metric의 동일 가중 평균입니다. 제외된 NA 사용자 수와 유효 사용자 수를 metric마다 기록합니다. 긴 경로가 많은 사용자의 비중을 확인하려면 전체 row-level 지표와 macro 지표 및 사용자별 row 수를 함께 봅니다.

이상 유형별로 이상 행 수·탐지 수·Recall·error mean/median/p95/p99를 저장합니다. 각 유형의 ROC-AUC/AP는 해당 유형의 이상 행과 동일 Test split의 **원본 정상 행 전체**만 비교합니다. 다른 유형의 이상을 정상으로 취급하지 않습니다.

## 산출물

모델 경로: `models/stage5/geolife_u000_019_first5_seed42_dedup/seed_42/`

- `model.pt`, `scaler.joblib`, `training_config.json`, `feature_columns.json`

지표 경로: `outputs/metrics/stage5/geolife_u000_019_first5_seed42_dedup/seed_42/`

- `training_history.csv`, `validation_predictions.csv`, `test_predictions.csv`
- `validation_metrics.json`, `test_metrics.json`
- `per_user_metrics.csv`, `user_macro_metrics.json`, `per_anomaly_type_metrics.csv`
- `stage4_stage5_comparison.csv`, `stage4_stage5_comparison.json`
- `dataset_validation.json`, `run_report.json`

prediction CSV는 source point index, 사용자, 경로, sample, split, timestamp, quality와 입력 feature를 보존합니다. 저장 후 lineage가 원본 평가 CSV와 일치하는지 다시 확인합니다. Stage 4 비교는 로컬 최종 지표의 정확한 값을 읽고 출처와 checksum을 기록합니다. 로컬 지표가 없으면 사용자가 제공한 반올림된 값을 사용합니다.

그래프 경로: `outputs/figures/stage5/geolife_u000_019_first5_seed42_dedup/seed_42/`

- `training_loss.png`
- `reconstruction_error_distribution.png` (log10(1+error))
- `reconstruction_error_distribution_zoom.png` (99th percentile zoom)
- `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`
- `anomaly_score_by_type.png` (Stage 4와 동일한 boxplot)
- `per_user_f1.png`, `per_user_fpr.png`

시각화의 log/zoom은 표시 범위에만 적용합니다. 지표·threshold·예측 error를 변형하지 않습니다. GUI 의존성 없이 PNG를 저장하도록 Matplotlib Agg backend를 사용합니다.

## 검증

신규 Stage 5.2 테스트 25개를 추가했습니다. Stage 1~5-A의 기존 120개를 유지하며 전체 145개가 통과했습니다. 실제 데이터 학습 전에 전체 테스트를 실행했습니다.

Test 값을 변경하는 파이프라인 테스트는 Test inference 전 선택 config가 저장되어 있음을 확인하고 threshold·best epoch가 동일함을 검증합니다. Validation anomaly 값을 변경하는 테스트는 threshold·checkpoint·전체 학습 history가 동일함을 확인합니다. scaler의 Train 평균·fit row 수, normal-only early stopping, 사용자 confusion 합계, 유형별 row/AUC/AP, lineage CSV round-trip, 모델/scaler 복원, Stage 4 보존, NA 처리, 9개 PNG 및 덮어쓰기 거부도 검증합니다.

## 실제 실행 결과

| 항목 | 결과 |
| --- | --- |
| 실행일 | 2026-10-02 (Asia/Seoul) |
| Device | CPU / PyTorch threads 6 |
| 실제 epoch | 100 |
| Scaler fit 행 | 86,067 |
| Best epoch | 98 |
| Best Validation normal loss | 0.044703345746 |
| Threshold (95th percentile) | 0.144246757030 |
| Threshold 결정 정상 행 | 19,448 |
| 실행 시간 | 146.78초 |

실제 Train은 86,067행, Validation 평가는 20,561행(정상 19,448 + 이상 1,113), Test 평가는 48,411행(정상 46,299 + 이상 2,112)입니다. Early stopping과 threshold에 사용한 Validation은 정상 19,448행뿐입니다.

| 지표 | Validation | Test 전체 | Test 사용자 macro |
| --- | --- | --- | --- |
| Accuracy | 0.915131 | 0.908368 | 0.878245 |
| Precision | 0.259513 | 0.168190 | 0.224964 |
| Recall | 0.306379 | 0.278883 | 0.278881 |
| F1 | 0.281005 | 0.209833 | 0.231946 |
| FPR | 0.050031 | 0.062917 | 0.072561 |
| ROC-AUC | 0.771821 | 0.722483 | 0.729521 |
| Average Precision | 0.235979 | 0.163090 | 0.256350 |

| Split | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Validation | 18,475 | 973 | 772 | 341 |
| Test | 43,386 | 2,913 | 1,523 | 589 |

### Test 사용자별 결과

| User | 경로 | 정상 행 | 이상 행 | TN | FP | FN | TP | Precision | Recall | F1 | FPR | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | 5 | 17031 | 466 | 15582 | 1449 | 334 | 132 | 0.083491 | 0.283262 | 0.128969 | 0.085080 | 0.732439 | 0.147153 |
| 008 | 5 | 8485 | 593 | 7789 | 696 | 420 | 173 | 0.199079 | 0.291737 | 0.236662 | 0.082027 | 0.737846 | 0.294820 |
| 013 | 4 | 18634 | 547 | 18065 | 569 | 407 | 140 | 0.197461 | 0.255941 | 0.222930 | 0.030536 | 0.732676 | 0.177741 |
| 017 | 5 | 2149 | 506 | 1950 | 199 | 362 | 144 | 0.419825 | 0.284585 | 0.339223 | 0.092601 | 0.715120 | 0.405686 |

모든 metric의 macro 유효 사용자 수는 4명이고, 이번 실제 데이터에서는 NA metric이 없습니다. 사용자별 TN/FP/FN/TP의 합이 Test 전체 confusion과 일치합니다.

### Test 이상 유형별 결과

| 유형 | 이상 행 | 탐지 행 | Recall | Error mean | Error median | Error p95 | Error p99 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| route_deviation | 378 | 81 | 0.214286 | 0.125526 | 0.030983 | 0.515572 | 1.142175 | 0.634356 | 0.016024 |
| abnormal_speed | 391 | 133 | 0.340153 | 763.889053 | 0.069508 | 8682.924805 | 11815.175781 | 0.783457 | 0.159955 |
| long_stop | 963 | 70 | 0.072690 | 0.185350 | 0.030297 | 0.227625 | 6.799963 | 0.635967 | 0.036947 |
| direction_change | 380 | 305 | 0.802632 | 1.676651 | 0.861533 | 4.083211 | 4.695685 | 0.966657 | 0.172920 |

각 ROC-AUC/AP의 정상 비교군은 동일한 Test 원본 정상 46,299행입니다. 유형별 이상 행 합은 2,112행, 탐지 행 합은 589행입니다. long_stop Recall은 0.072690으로 가장 낮고, direction_change는 0.802632입니다. abnormal_speed의 error mean은 763.889053, median은 0.069508로 극단값의 영향이 큽니다. 원본 error 및 지표는 그대로 보존했습니다.

### Stage 4 비교

| 지표 | Stage 4 | Stage 5.2 | 변화량 (5−4) |
| --- | --- | --- | --- |
| Accuracy | 0.918946 | 0.908368 | -0.010578 |
| Precision | 0.530973 | 0.168190 | -0.362784 |
| Recall | 0.689655 | 0.278883 | -0.410773 |
| F1 | 0.600000 | 0.209833 | -0.390167 |
| FPR | 0.058889 | 0.062917 | +0.004028 |
| ROC-AUC | 0.818333 | 0.722483 | -0.095850 |
| Average Precision | 0.514765 | 0.163090 | -0.351675 |

Stage 4 Test는 987행(정상 900 + 이상 87), Stage 5.2는 48,411행(정상 46,299 + 이상 2,112)입니다. 이상 비율은 각각 약 8.81%, 4.36%입니다. 이번 설정에서 F1·Recall·ROC-AUC·AP가 낮아졌으며 FPR은 소폭 증가했습니다. 평가 사용자·경로·규모·이상 비율이 달라 단일 요인의 인과 효과나 모델 우열로 해석할 수 없습니다. AP와 Precision은 이상 비율의 영향도 받습니다.

### 행 수에 따른 가중 영향

| User | 평가 행 비중 | 정상 행 비중 | 사용자 내 이상 비율 |
| --- | --- | --- | --- |
| 001 | 36.14% | 36.78% | 2.66% |
| 008 | 18.75% | 18.33% | 6.53% |
| 013 | 39.62% | 40.25% | 2.85% |
| 017 | 5.48% | 4.64% | 19.06% |

사용자 001과 013이 평가 행의 75.76%를 차지합니다. 특히 `013__20080927233805` 한 경로의 정상 행이 13,459개로 Test 정상 행의 29.07%입니다. 전체 row-level 결과에 긴 경로와 사용자의 비중이 크게 반영됩니다. 정상 행 비중이 40.25%인 사용자 013의 FPR은 3.05%로 다른 사용자보다 낮고, 전체 FPR 6.29%는 사용자 macro FPR 7.26%보다 낮습니다. 전체 AP 0.163090과 macro AP 0.256350도 다르지만 사용자별 이상 비율이 2.66~19.06%여서 AP 차이를 가중 효과만으로 설명하면 안 됩니다.

### 실행 후 검증과 발견 사항

전체 145개 테스트가 통과했습니다(신규 25개, 기존 120개; 약 31.80초). 실제 checkpoint와 scaler를 다시 불러와 Test error와 예측이 저장 CSV와 일치함을 확인했습니다. best epoch 98이 history의 Validation normal loss 최소점인 것도 확인했습니다. 입력 dataset 및 Stage 4 산출물 총 40개 파일의 실행 전후 checksum이 일치합니다. 9개 그래프를 생성했고 학습 loss·zoom 분포·사용자 FPR 그래프를 확인했습니다.

처음 그래프 테스트에서 로컬 Tk/Tcl GUI 설치 문제를 발견해 Stage 5의 PNG 저장 backend를 Agg로 고정했습니다. 모델·데이터·평가 설정에 영향을 주지 않습니다. 검증 스크립트에서 CSV로 다시 읽은 loss와 JSON loss의 최하위 소수점 차이를 확인했으며, 비교에 수치 허용오차를 적용했습니다. 기록된 원래 값은 변경하지 않았습니다.

실험상의 주요 관찰은 long_stop 탐지율 저하와 사용자별 행 수 불균형입니다. 누수·lineage·scaler·checkpoint 복원에서 발견된 오류는 없습니다. 이번 결과를 개선하기 위한 구조·feature·threshold·split 변경이나 재학습은 수행하지 않았습니다.

### Stage 5.3 전 준비할 사항

현재 Stage 5.2 CLI는 seed 42로 고정되어 있습니다. Stage 5.3에서는 여러 model seed를 지정하고 seed별 별도 경로에 실행·집계하는 진입점과 그 테스트를 추가해야 합니다. dataset·user split·합성 seed·8 feature·모델·학습 설정·95th percentile 정책은 그대로 고정하고, 각 model seed의 checkpoint·threshold·사용자별 지표와 macro 유효 사용자 수를 개별 저장해야 합니다. 평균/표준편차는 모델 seed 변화에 대한 변동성을 측정하는 용도로 해석해야 합니다.

이번 단계에서 multi-seed, baseline, window/LSTM, 성능 튜닝을 실행하지 않았으며 커밋·푸시도 하지 않았습니다.

## 결론

단일 seed 결과만으로 현재 성능 저하가 모델 구조의 안정적인 한계인지 random initialization에 의한 변동인지 구분할 수 없습니다. 따라서 Stage 5.3에서는 dataset, user split, synthetic data, feature, architecture, threshold rule을 고정한 상태에서 여러 model seed를 반복해 안정성을 평가합니다.
