# Stage 5.3: Multi-seed Stability

Stage 5.2의 unseen-user 결과가 seed 42에만 나타나는지 확인하기 위해 고정 dataset에서 model seed 7, 21, 42, 100, 2026을 비교합니다. Dataset·user split·synthetic data·8 feature·architecture·학습 설정·threshold rule을 변경하지 않습니다.

## 실행

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_multi_seed_stability
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

CLI는 dataset 경로와 output root만 받습니다. seed 목록과 hyperparameter를 변경하는 옵션은 없습니다. Stage 5.2의 실행 함수 이름을 유지하고 `experiment_stage='5.3'` 인자로 지정한 경우에만 고정 5개 seed와 기존 학습 설정을 허용합니다. 모델이나 feature 구현을 수정하지 않았습니다.

각 seed는 Train 정상 86,067행으로 StandardScaler를 fit하고 Autoencoder를 초기화·학습합니다. Validation 정상 19,448행의 MSE로 checkpoint를 선택하고 best checkpoint를 복원합니다. 동일 정상 행 error의 95th percentile을 threshold로 저장한 뒤 Test 48,411행을 평가합니다. Validation 합성 이상과 Test 결과는 선택이나 threshold tuning에 사용하지 않습니다. model seed 변경은 초기화와 batch shuffle의 난수를 함께 변경합니다.

고정 설정: epochs=100, batch_size=64, learning_rate=0.001, patience=15, threshold_percentile=95. Dataset은 `geolife_u000_019_first5_seed42_dedup`이며 split seed와 synthetic seed도 기존 42 그대로입니다. seed마다 dataset을 재생성하지 않습니다.

## 기존 seed 42 재사용

기존 seed 42를 재사용하기 전에 다음을 확인합니다.

- dataset ID·경로·summary SHA256·모든 CSV checksum
- user split과 synthetic manifest
- 모든 학습 설정·feature 순서·architecture·device·thread 수·라이브러리 버전
- scaler fit 행 수와 Train에서 다시 계산한 mean/variance/scale
- history 최소 Validation normal loss와 best epoch
- 복원 checkpoint의 Validation/Test error·예측·lineage와 저장 CSV 일치
- Validation 정상 95th percentile threshold
- 전체·사용자별·macro·이상 유형별 지표 재계산 일치
- 모델·지표·9개 그래프 파일의 완결성

호환되지 않으면 재사용하거나 덮어쓰지 않고 실패합니다. 호환되면 재사용 여부와 전체 artifact checksum을 `multi_seed_summary.json`에 기록하고 실행 끝에 다시 확인합니다. 기존 stage metadata `5-B`는 역사적 alias로 인식하며 기존 파일은 변경하지 않습니다.

다른 seed의 결과나 multi_seed 집계 경로가 이미 있으면 학습 전에 실패합니다. 일부 결과를 조용히 덮어쓰거나 기존 결과를 삭제하지 않습니다.

## 집계

5개 seed 각각의 checkpoint·loss·threshold·Test Accuracy/Precision/Recall/F1/FPR/ROC-AUC/AP·TN/FP/FN/TP와 사용자 macro 지표를 기록합니다. 사용자 001/008/013/017별 F1/Recall/FPR/ROC-AUC/AP 및 이상 유형별 Recall도 기록합니다.

평균, **sample standard deviation (ddof=1)**, min, max를 계산합니다. 정의되지 않은 metric은 NA로 남기고 유효 seed 수를 표시합니다. sample std는 유효 seed가 2개 미만이면 NA입니다. 사용자 macro는 먼저 각 seed에서 사용자별 동일 가중 평균을 계산하고, 그 macro 값을 seed 간 집계합니다. 이상 유형의 다른 label을 정상으로 바꾸지 않습니다.

seed 42의 값·평균과의 차이·sample std 기준 위치·오름차순 숫자 순위를 저장합니다. FPR은 낮은 값이, 다른 성능 지표는 높은 값이 유리하므로 순위 방향을 해석할 때 구분합니다. seed별 성능으로 모델이나 threshold를 다시 선택하지 않습니다.

## 산출물

각 seed: 기존 `models/stage5/<dataset_id>/seed_<seed>/`, `outputs/metrics/stage5/<dataset_id>/seed_<seed>/`, `outputs/figures/stage5/<dataset_id>/seed_<seed>/` 구조를 사용합니다.

집계: `outputs/metrics/stage5/<dataset_id>/multi_seed/`

- `seed_results.csv`, `seed_summary.csv`
- `per_user_seed_results.csv`, `per_user_seed_summary.csv`
- `per_anomaly_type_seed_results.csv`, `per_anomaly_type_seed_summary.csv`
- `multi_seed_summary.json`
- `logs/seed_7.log`, `logs/seed_21.log`, `logs/seed_100.log`, `logs/seed_2026.log`

그래프: `outputs/figures/stage5/<dataset_id>/multi_seed/`

- `f1_by_seed.png`, `roc_auc_by_seed.png`, `average_precision_by_seed.png`
- `threshold_by_seed.png`, `anomaly_recall_by_seed.png`, `per_user_f1_by_seed.png`

## 검증

신규 Stage 5.3 테스트는 19개입니다. 실제 end-to-end fixture에서 고정 5개 seed를 실행하고 seed 42 재사용·파일 보존·집계 결과를 확인합니다. checksum·split·합성 manifest·출력 분리·덮어쓰기 거부·고정 설정·기존 결과 호환·mean·ddof=1·사용자별/유형별 집계·NA 지원 수와 순위를 검증합니다. Stage 1~5.2의 기존 145개 테스트를 유지합니다.

## 실제 결과

실행일: 2026-10-02 (Asia/Seoul). 5개 seed를 집계했고 seed 42만 기존 Stage 5.2 결과를 재사용했습니다. 신규 학습 seed는 7, 21, 100, 2026입니다. Dataset과 기존 seed 42의 모델 4개·metrics 14개·figure 9개 파일은 변경되지 않았습니다.

### Seed별 결과

| Seed | 재사용 | Actual epochs | Best epoch | Best Val normal loss | Threshold | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | AP | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 아니오 | 100 | 98 | 0.020183 | 0.036446 | 0.905476 | 0.212954 | 0.432765 | 0.285447 | 0.072961 | 0.771787 | 0.174938 | 42921 | 3378 | 1198 | 914 |
| 21 | 아니오 | 100 | 90 | 0.014988 | 0.030056 | 0.894611 | 0.193648 | 0.447443 | 0.270309 | 0.084991 | 0.795780 | 0.191415 | 42364 | 3935 | 1167 | 945 |
| 42 | 예 | 100 | 98 | 0.044703 | 0.144247 | 0.908368 | 0.168190 | 0.278883 | 0.209833 | 0.062917 | 0.722483 | 0.163090 | 43386 | 2913 | 1523 | 589 |
| 100 | 아니오 | 100 | 91 | 0.048207 | 0.181777 | 0.896387 | 0.131846 | 0.246212 | 0.171731 | 0.073954 | 0.725758 | 0.122395 | 42875 | 3424 | 1592 | 520 |
| 2026 | 아니오 | 98 | 83 | 0.052206 | 0.172337 | 0.903803 | 0.162377 | 0.289773 | 0.208128 | 0.068187 | 0.722437 | 0.132122 | 43142 | 3157 | 1500 | 612 |

Train/Validation/Test 행 수는 모든 seed에서 86,067 / 20,561 / 48,411로 동일합니다. Scaler fit은 86,067행, checkpoint/threshold용 Validation 정상은 19,448행입니다. 기존 seed 42의 best epoch 98, threshold 0.144246757030과 기존 지표를 그대로 유지했습니다.

### Test 전체 안정성

| Metric | Mean ± sample std | Min | Max | 유효 seed |
| --- | --- | --- | --- | --- |
| Accuracy | 0.901729 ± 0.005950 | 0.894611 | 0.908368 | 5 |
| Precision | 0.173803 ± 0.031021 | 0.131846 | 0.212954 | 5 |
| Recall | 0.339015 ± 0.093807 | 0.246212 | 0.447443 | 5 |
| F1 | 0.229089 ± 0.047369 | 0.171731 | 0.285447 | 5 |
| FPR | 0.072602 ± 0.008197 | 0.062917 | 0.084991 | 5 |
| ROC-AUC | 0.747649 ± 0.034086 | 0.722437 | 0.795780 | 5 |
| AP | 0.156792 ± 0.028981 | 0.122395 | 0.191415 | 5 |

### User macro 안정성

| User macro metric | Mean ± sample std | Min | Max |
| --- | --- | --- | --- |
| Precision | 0.236984 ± 0.011892 | 0.224964 | 0.255744 |
| Recall | 0.339050 ± 0.093216 | 0.246971 | 0.447132 |
| F1 | 0.255925 ± 0.041377 | 0.211169 | 0.305665 |
| FPR | 0.081341 ± 0.011921 | 0.072561 | 0.098425 |
| ROC-AUC | 0.754161 ± 0.029143 | 0.726384 | 0.789356 |
| AP | 0.273485 ± 0.032944 | 0.244132 | 0.314279 |

### 이상 유형별 Recall 안정성

| Anomaly type | Recall mean ± sample std | Min | Max |
| --- | --- | --- | --- |
| abnormal_speed | 0.423529 ± 0.167988 | 0.237852 | 0.624041 |
| direction_change | 0.868947 ± 0.098419 | 0.784211 | 0.981579 |
| long_stop | 0.147040 ± 0.090365 | 0.072690 | 0.265836 |
| route_deviation | 0.207937 ± 0.037906 | 0.145503 | 0.248677 |

### 사용자별 안정성

| User | F1 mean ± std | Recall mean ± std | FPR mean ± std | ROC-AUC mean ± std | AP mean ± std |
| --- | --- | --- | --- | --- | --- |
| 001 | 0.141760 ± 0.039929 | 0.354936 ± 0.097617 | 0.101168 ± 0.015454 | 0.761972 ± 0.031943 | 0.126594 ± 0.019652 |
| 008 | 0.270859 ± 0.054971 | 0.351771 ± 0.097024 | 0.084973 ± 0.006619 | 0.761578 ± 0.036463 | 0.318317 ± 0.042429 |
| 013 | 0.244397 ± 0.049494 | 0.325777 ± 0.120882 | 0.037501 ± 0.012377 | 0.762414 ± 0.057766 | 0.219721 ± 0.070982 |
| 017 | 0.366685 ± 0.025327 | 0.323715 ± 0.058248 | 0.101722 ± 0.033730 | 0.730680 ± 0.030634 | 0.429307 ± 0.022581 |

사용자별 seed F1:

| User | 7 | 21 | 42 | 100 | 2026 |
| --- | --- | --- | --- | --- | --- |
| 001 | 0.186005 | 0.178121 | 0.128969 | 0.090044 | 0.125659 |
| 008 | 0.333741 | 0.327649 | 0.236662 | 0.221583 | 0.234659 |
| 013 | 0.313854 | 0.273595 | 0.222930 | 0.187337 | 0.224270 |
| 017 | 0.389058 | 0.396020 | 0.339223 | 0.345711 | 0.363415 |

사용자별 다른 지표의 seed 값도 `per_user_seed_results.csv`에 저장했습니다. 고정 split이므로 사용자·정상/이상 row 수는 seed마다 동일하며, 사용자별 macro 유효 사용자 수도 seed 결과 CSV에 기록했습니다.

### Seed 42의 위치와 변동성

| Metric | Seed 42 | 평균과의 차이 | 평균 대비 표준편차 단위 | 성능 순위 (1=유리) |
| --- | --- | --- | --- | --- |
| Accuracy | 0.908368 | 0.006639 | 1.115752 | 1 |
| Precision | 0.168190 | -0.005613 | -0.180955 | 3 |
| Recall | 0.278883 | -0.060133 | -0.641026 | 4 |
| F1 | 0.209833 | -0.019257 | -0.406523 | 3 |
| FPR | 0.062917 | -0.009685 | -1.181583 | 1 |
| ROC-AUC | 0.722483 | -0.025166 | -0.738305 | 4 |
| AP | 0.163090 | 0.006298 | 0.217321 | 3 |

순위는 FPR에 낮은 값 우선, 다른 지표에 높은 값 우선으로 계산했습니다. Mean/std 위치는 sample std를 사용한 기술 통계이며 통계적 유의성이나 모집단 신뢰구간이 아닙니다. Test 지표 중 절대 sample std가 가장 큰 지표는 **Recall**(0.093807)입니다. Epoch·loss·threshold는 단위가 달라 이 비교에서 제외했습니다.

### 해석과 Stage 5.4 판단

이번 고정 unseen-user dataset의 F1 범위는 0.171731~0.285447, 평균은 0.229089 ± 0.047369입니다. seed 42의 단일 결과보다 높은 seed가 있어 난수에 따른 변동이 확인되지만, 5개 seed 모두 Stage 4 F1 0.6000보다 낮습니다. 따라서 이 실험 조건의 낮은 성능을 seed 42 하나의 우연만으로 설명하기는 어렵습니다. ROC-AUC 평균은 0.747649 ± 0.034086, AP 평균은 0.156792 ± 0.028981입니다.

direction_change Recall은 모든 seed에서 0.784211~0.981579로 높은 편이었습니다. long_stop은 0.072690~0.265836, route_deviation은 0.145503~0.248677로 모든 seed에서 낮은 탐지율이 반복됐습니다. abnormal_speed Recall은 0.423529 ± 0.167988로 유형 중 seed 변동이 가장 컸습니다. Seed 42의 F1은 5개 중 3위이고 평균보다 0.019257 낮아 극단적인 outlier로 보기 어렵습니다. 실제 수치에 근거한 결론만 적용하며, 5개 seed만으로 architecture의 인과적 한계를 단정하지 않습니다. Stage 4와 데이터량·사용자·이상 비율이 달라 비교를 모델 우열로 해석하지 않습니다.

Dataset·scaler·lineage·checkpoint·threshold 검증을 통과했으므로 동일 dataset과 평가 규칙으로 Stage 5.4 baseline 비교를 설계할 수 있습니다. Seed별 최대 성능을 골라 기준값으로 삼거나 threshold·feature·architecture를 변경하지 않고, 이번 5개 seed 결과와 macro 지표를 비교 기준으로 유지해야 합니다. 이번 단계에서는 baseline, route distance feature, Window/LSTM을 구현하지 않았습니다.

### 검증·시간·문제

전체 **164개 테스트가 통과**했습니다(기존 145개 + 신규 19개; 40.555초). Dataset 전체 checksum, split manifest, synthetic manifest가 일치하며 seed 42의 기존 27개 산출물 checksum도 유지됐습니다. 신규 seed마다 저장된 checkpoint/scaler를 복원해 예측·전체/사용자/유형별 지표를 재검증했습니다. 6개의 집계 그래프와 각 신규 seed의 9개 그래프를 생성했습니다.

Stage 5.3 전체 실행 시간은 **589.79초**이며 재사용 검증·신규 4개 seed 학습·저장/검증·집계·그래프를 포함합니다. 신규 4개 seed의 학습 파이프라인 시간 합은 572.24초입니다. 기존 seed 42의 과거 실행 시간은 신규 시간에 더하지 않았습니다. seed 2026은 best epoch 83 이후 patience 15가 충족되어 98 epoch에서 early stopping했습니다. 다른 4개 seed는 100 epoch를 실행했습니다. epochs=100 상한과 patience=15 설정은 동일하게 유지했습니다. 학습 설정이나 결과를 수정한 재실험은 없습니다.

기술적 검증 오류는 발견하지 않았습니다. 성능상의 발견은 seed 변동과 사용자별/이상 유형별 차이이며 위 표에 그대로 기록했습니다. Git CLI push는 인증 대화상자가 취소되어 실패했지만 GitHub 연결 앱으로 동일 tree의 Stage 5.1/5.2 두 커밋을 게시했고 로컬 이력과 upstream을 동기화했습니다.

### Git 상태

- 브랜치: `feature/stage-5-multiuser-generalization`
- Stage 5.1: `95ee383` — `stage5: prepare multi-user dataset`
- Stage 5.2: `5a9ab2b` — `stage5: add unseen-user evaluation`
- 두 커밋의 원격 업로드와 로컬/원격 HEAD 일치를 확인했습니다. 원래 로컬 커밋은 `stage5-local-verified-d3d7923` tag로 보존했습니다.
- Stage 5.3 코드·테스트·보고서는 전체 테스트 재검증 후 `stage5: add multi-seed stability analysis`로 커밋·게시합니다. PR 생성이나 main merge는 진행하지 않습니다.
- Stage 5 완료 후 PR 제목은 `Stage 5: Multi-user generalization`이며 main에는 squash commit 하나를 남길 계획입니다.
