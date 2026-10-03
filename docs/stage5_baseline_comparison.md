# Stage 5.4 Baseline Comparison

## 사전 고정 실험 설계 (구현·Test baseline 점수 확인 전 기록)

2026-10-02. Stage 5.3은 `ff5ca3e` (`stage5: add multi-seed stability analysis`)로 동일 tree 게시를 완료했다. Stage 5.4 실험 완료 후 사용자 요청에 따라 재검증·커밋·게시하며 PR/merge는 진행하지 않는다.

고정 dataset `geolife_u000_019_first5_seed42_dedup`의 Train 정상 86,067행, Validation 정상 19,448행/이상 1,113행, Test 정상 46,299행/이상 2,112행을 그대로 사용한다. Test users는 001/008/013/017이다. 사용자 split, source lineage, trajectory dedup, synthetic anomaly manifest를 변경하지 않는다. 8 feature 순서는 time_diff_sec, distance_m, speed_mps, acceleration_mps2, direction_change_deg, stop_duration_sec, bearing_sin, bearing_cos이다.

### Statistical Rule의 정확한 정의

각 feature j에 대해 Train 원본 정상 값으로만 median m_j, Q25_j, Q75_j, population std sigma_j(ddof=0)를 계산한다. 모든 quantile은 NumPy linear interpolation이다.

- IQR_j=Q75_j-Q25_j가 양수이면 s_j=IQR_j/1.349.
- IQR_j=0이고 sigma_j>0이면 s_j=sigma_j. 드문 정지처럼 중앙 50%가 같은 값인 feature도 삭제하지 않는다.
- 둘 다 0이면 s_j=1e-9*max(1,abs(m_j)). Train 상수 feature의 변화도 점수에 반영하며 0으로 나누지 않는다. 이 수치 안정화 규칙은 Test 관찰 전에 고정한다.
- 행 x의 score=max_j(abs(x_j-m_j)/s_j), j는 **8개 모두**. 별도 feature weight, quality-rule penalty, threshold clip은 없다.

bearing_sin/cos는 각도를 연속적인 단위원 좌표로 표현하므로 0/360도 경계의 불연속을 피한다. 두 좌표 모두 같은 방식으로 정규화해 최대 편차에 포함한다. 이 좌표별 통계는 Train 이동 방향 분포를 반영하지만 회전 불변 score나 경로 이탈 거리를 뜻하지 않는다. 방향 선호가 다른 정상 사용자도 점수가 높을 수 있다는 제한을 해석에 포함한다. Rule은 deterministic 단일 실행이며 5-seed 결과로 복제하지 않는다. Rule의 robust scale과 다른 모델의 StandardScaler는 방법 자체의 차이로 기록한다.

### Isolation Forest의 고정 설정

scikit-learn 1.9.1. n_estimators=200, max_samples='auto'(min(256, Train rows)), contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, verbose=0, warm_start=False. random_state만 7/21/42/100/2026으로 바꾼다.

Train 정상으로만 StandardScaler를 fit한다. Stage 5.2와 같은 float32 입력/변환 함수를 사용한다. tree의 축별 split은 양의 affine scaling에서 대체로 같은 구간을 표현하므로 scaling으로 성능 향상을 기대하는 선택이 아니다. 입력 처리의 일관성과 fit 출처 검증을 위해 사용하며 부동소수점 차이는 가능하다. 다른 데이터로 scaler를 refit하지 않는다.

score=-IsolationForest.score_samples(scaled_x). 높은 값이 이상이다. predict()/decision_function()의 내부 offset을 최종 판단에 사용하지 않는다. contamination은 Test prevalence와 무관한 'auto'로 고정한다. 공식 API의 score_samples는 낮을수록 이상인 점수다: [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

### Threshold·AE 재사용·평가

신규 baseline threshold=np.percentile(Validation 원본 정상 score,95,method='linear'). 예측은 score>threshold. Validation 이상/Test는 fit이나 threshold 계산에 전달하지 않는다. 동점으로 실제 Validation FPR이 정확히 5%보다 작을 수 있다. Test FPR은 사용자 분포 차이로 5%가 아닐 수 있다.

AE는 Stage 5.3의 5개 seed 산출물을 checksum 검증 후 읽기만 한다. 기존 threshold 및 예측을 사용하며 threshold percentile을 다시 계산하거나 모델 추론/학습을 실행하지 않는다. 저장된 Test 예측의 평가 지표를 대조하는 작업만 허용한다. 점수 단위는 Rule standardized maximum deviation, IF negative normality score, AE standardized-feature reconstruction MSE로 서로 달라 raw score/threshold 크기로 모델을 비교하지 않는다.

Accuracy/Precision/Recall/F1/FPR/ROC-AUC/AP와 TN/FP/FN/TP, 사용자별 6 metric 및 동일 가중 macro, 이상 유형별 count/detected count/Recall/score mean/median/p95/p99/ROC-AUC/AP를 저장한다. 유형별 ROC/AP의 음성은 동일 Test 원본 정상이고 다른 이상 유형은 제외한다. IF/AE는 5 seeds의 mean/sample std(ddof=1)/min/max, Rule은 단일 값(std=NA, 실행 수 1)으로 기록한다. NA를 0으로 대체하지 않고 정의된 실행/사용자 수를 표시한다.

사전 공정성 검사에서 dataset checksum, Train/Val normal/Test row identity·feature·lineage, schema, users, synthetic/dedup manifest, AE config의 Train-only fit/Val-normal p95 출처를 대조한다. 모든 AE 산출물 및 dataset의 실행 전후 checksum을 보존한다. 산출물이 이미 있으면 덮어쓰지 않고 실패한다. Test 결과를 보고 공식·feature·hyperparameter를 변경하지 않는다.

실행: `python -m src.compare_stage5_baselines`

산출물: `outputs/metrics/stage5/<dataset_id>/baseline_comparison/`, 모델 `models/stage5/<dataset_id>/baselines/`, 그래프 `outputs/figures/stage5/<dataset_id>/baseline_comparison/`. IF는 seed별 하위 디렉터리로 분리한다.
## 실제 결과와 검증

실행일: 2026-10-02 (Asia/Seoul). 사전 고정 설정을 한 번 실행했다. Test 결과에 따른 score·feature·threshold percentile·hyperparameter 수정이나 재실험은 없다. AE 5개 seed는 원래 저장된 점수·예측·threshold를 읽었다. AE 재학습·모델 추론·threshold 재계산을 수행하지 않았다.

### 공정성·leakage

| Split | Users | Trajectories | Normal rows | Anomaly rows | Total |
| --- | --- | --- | --- | --- | --- |
| Train | 12 | 51 | 86067 | 0 | 86067 |
| Validation | 4 | 12 | 19448 | 1113 | 20561 |
| Test | 4 | 19 | 46299 | 2112 | 48411 |

동일 dataset·행 순서·8 feature·Test 사용자·source lineage·합성 manifest·dedup 정책을 확인했다. Train/Validation normal/Test의 identity+feature+lineage digest를 기록했다. 모든 baseline fit은 Train 원본 정상 86,067행이며 threshold는 Validation 원본 정상 19,448행이다. IF 각 tree의 내부 subsample은 사전 고정 `max_samples=auto`에 따라 256행이다. 전체 Train 86,067행이 fit 입력이고 이 subsampling은 IF 알고리즘의 고정 설정이다.

각 IF scaler의 mean/var/scale은 기존 AE Train 전용 scaler와 **정확히 일치**했다. Rule의 center는 전체 Train 정상 median과 정확히 일치했다. Rule의 stop_duration_sec는 IQR=0이므로 사전 정의한 population std fallback을 사용했고 다른 7 feature는 IQR scale을 사용했다.

Validation FPR은 신규 6회 모두 0.050031이다. Rule threshold는 13.022437059583535이다. IF threshold는 아래 seed 표에 저장했다. AE threshold는 Stage 5.3의 값을 그대로 사용했다. Synthetic/Test label은 fit·threshold에 사용하지 않았다. 조건 위반은 0건이며 robust scale/StandardScaler·점수 단위·실행 수의 방법별 차이를 명시했다.

실행 전후 dataset 전체 checksum, 기존 AE **127개 산출물** (seed 42는 27개, 나머지 각 25개), Stage 5.3 집계 파일 및 기존 Stage 4/dataset 보호 파일 40개의 checksum이 유지됐다. Cross-split user/source/sample/fingerprint leakage는 0이며 Train 014의 동일 split duplicate 1개 제외 정책도 유지됐다.

공정성 증거: [comparison_validation.json](../outputs/metrics/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/comparison_validation.json), 독립 재검증: [final_verification.json](../outputs/metrics/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/final_verification.json).

### 전체 Test 지표

Rule은 deterministic 단일 값이고 std는 NA이다. IF/AE는 5 seeds의 mean ± sample std(ddof=1)이며 신뢰구간 또는 유의성 검정이 아니다.

| Metric | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| Accuracy | 0.908843 | 0.905873 ± 0.001772 | 0.901729 ± 0.005950 |
| Precision | 0.199373 | 0.201639 ± 0.004704 | 0.173803 ± 0.031021 |
| Recall | 0.361269 | 0.391004 ± 0.009190 | 0.339015 ± 0.093807 |
| F1 | 0.256946 | 0.266040 ± 0.005419 | 0.229089 ± 0.047369 |
| FPR | 0.066179 | 0.070641 ± 0.001954 | 0.072602 ± 0.008197 |
| ROC-AUC | 0.669346 | 0.746730 ± 0.006199 | 0.747649 ± 0.034086 |
| AP | 0.138804 | 0.210599 ± 0.015466 | 0.156792 ± 0.028981 |
| TN | 43235.000000 | 43028.400000 ± 90.469884 | 42937.600000 ± 379.490843 |
| FP | 3064.000000 | 3270.600000 ± 90.469884 | 3361.400000 ± 379.490843 |
| FN | 1349.000000 | 1286.200000 ± 19.408761 | 1396.000000 ± 198.119913 |
| TP | 763.000000 | 825.800000 ± 19.408761 | 716.000000 ± 198.119913 |

TN/FP/FN/TP의 소수점은 seed 평균이다. 실제 seed별 count는 정수다.

IF/AE의 min/max:

| Model | Metric | Min | Max | Defined runs |
| --- | --- | --- | --- | --- |
| Isolation Forest | Accuracy | 0.903452 | 0.908347 | 5 |
| Isolation Forest | Precision | 0.194709 | 0.205995 | 5 |
| Isolation Forest | Recall | 0.383523 | 0.406723 | 5 |
| Isolation Forest | F1 | 0.259036 | 0.273480 | 5 |
| Isolation Forest | FPR | 0.067712 | 0.072982 | 5 |
| Isolation Forest | ROC-AUC | 0.738936 | 0.754789 | 5 |
| Isolation Forest | AP | 0.193005 | 0.231282 | 5 |
| Isolation Forest | TN | 42920.000000 | 43164.000000 | 5 |
| Isolation Forest | FP | 3135.000000 | 3379.000000 | 5 |
| Isolation Forest | FN | 1253.000000 | 1302.000000 | 5 |
| Isolation Forest | TP | 810.000000 | 859.000000 | 5 |
| Autoencoder | Accuracy | 0.894611 | 0.908368 | 5 |
| Autoencoder | Precision | 0.131846 | 0.212954 | 5 |
| Autoencoder | Recall | 0.246212 | 0.447443 | 5 |
| Autoencoder | F1 | 0.171731 | 0.285447 | 5 |
| Autoencoder | FPR | 0.062917 | 0.084991 | 5 |
| Autoencoder | ROC-AUC | 0.722437 | 0.795780 | 5 |
| Autoencoder | AP | 0.122395 | 0.191415 | 5 |
| Autoencoder | TN | 42364.000000 | 43386.000000 | 5 |
| Autoencoder | FP | 2913.000000 | 3935.000000 | 5 |
| Autoencoder | FN | 1167.000000 | 1592.000000 | 5 |
| Autoencoder | TP | 520.000000 | 945.000000 | 5 |

### 실행별 전체 Test 결과

| Model | Seed | Threshold | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | AP | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Statistical Rule | 단일 | 13.022437 | 0.908843 | 0.199373 | 0.361269 | 0.256946 | 0.066179 | 0.669346 | 0.138804 | 43235 | 3064 | 1349 | 763 |
| Isolation Forest | 7 | 0.529314 | 0.908347 | 0.205323 | 0.383523 | 0.267459 | 0.067712 | 0.754789 | 0.231282 | 43164 | 3135 | 1302 | 810 |
| Isolation Forest | 21 | 0.522600 | 0.905393 | 0.199171 | 0.386837 | 0.262955 | 0.070952 | 0.748726 | 0.193005 | 43014 | 3285 | 1295 | 817 |
| Isolation Forest | 42 | 0.518532 | 0.905724 | 0.205995 | 0.406723 | 0.273480 | 0.071513 | 0.738936 | 0.220458 | 42988 | 3311 | 1253 | 859 |
| Isolation Forest | 100 | 0.525601 | 0.903452 | 0.194709 | 0.386837 | 0.259036 | 0.072982 | 0.748880 | 0.199744 | 42920 | 3379 | 1295 | 817 |
| Isolation Forest | 2026 | 0.522690 | 0.906447 | 0.202998 | 0.391098 | 0.267271 | 0.070045 | 0.742321 | 0.208505 | 43056 | 3243 | 1286 | 826 |
| Autoencoder | 7 | 0.036446 | 0.905476 | 0.212954 | 0.432765 | 0.285447 | 0.072961 | 0.771787 | 0.174938 | 42921 | 3378 | 1198 | 914 |
| Autoencoder | 21 | 0.030056 | 0.894611 | 0.193648 | 0.447443 | 0.270309 | 0.084991 | 0.795780 | 0.191415 | 42364 | 3935 | 1167 | 945 |
| Autoencoder | 42 | 0.144247 | 0.908368 | 0.168190 | 0.278883 | 0.209833 | 0.062917 | 0.722483 | 0.163090 | 43386 | 2913 | 1523 | 589 |
| Autoencoder | 100 | 0.181777 | 0.896387 | 0.131846 | 0.246212 | 0.171731 | 0.073954 | 0.725758 | 0.122395 | 42875 | 3424 | 1592 | 520 |
| Autoencoder | 2026 | 0.172337 | 0.903803 | 0.162377 | 0.289773 | 0.208128 | 0.068187 | 0.722437 | 0.132122 | 43142 | 3157 | 1500 | 612 |

### User macro 비교

각 seed 안에서 Test 사용자 4명의 정의된 metric을 동일 가중 평균한 뒤 seed 간 집계했다. 이번 실제 실행의 모든 요청 metric은 4명 모두 정의됐다.

| Macro metric | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| Precision | 0.304676 | 0.369979 ± 0.011052 | 0.236984 ± 0.011892 |
| Recall | 0.363612 | 0.393664 ± 0.008959 | 0.339050 ± 0.093216 |
| F1 | 0.300037 | 0.337003 ± 0.006735 | 0.255925 ± 0.041377 |
| FPR | 0.068850 | 0.064865 ± 0.002236 | 0.081341 ± 0.011921 |
| ROC-AUC | 0.676768 | 0.764535 ± 0.004294 | 0.754161 ± 0.029143 |
| AP | 0.217836 | 0.313453 ± 0.007598 | 0.273485 ± 0.032944 |

### 사용자별 비교

Precision:

| Test user | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| 001 | 0.093645 | 0.082328 ± 0.001648 | 0.088770 ± 0.025328 |
| 008 | 0.236321 | 0.335032 ± 0.008375 | 0.221198 ± 0.034704 |
| 013 | 0.338739 | 0.407327 ± 0.021009 | 0.201323 ± 0.025414 |
| 017 | 0.550000 | 0.655228 ± 0.024784 | 0.436644 ± 0.043299 |

Recall:

| Test user | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| 001 | 0.420601 | 0.469957 ± 0.008848 | 0.354936 ± 0.097617 |
| 008 | 0.342327 | 0.394266 ± 0.016497 | 0.351771 ± 0.097024 |
| 013 | 0.343693 | 0.329799 ± 0.005106 | 0.325777 ± 0.120882 |
| 017 | 0.347826 | 0.380632 ± 0.006643 | 0.323715 ± 0.058248 |

F1:

| Test user | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| 001 | 0.153185 | 0.140105 ± 0.002588 | 0.141760 ± 0.039929 |
| 008 | 0.279614 | 0.362176 ± 0.010852 | 0.270859 ± 0.054971 |
| 013 | 0.341198 | 0.364283 ± 0.008955 | 0.244397 ± 0.049494 |
| 017 | 0.426150 | 0.481446 ± 0.010844 | 0.366685 ± 0.025327 |

FPR:

| Test user | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| 001 | 0.111385 | 0.143362 ± 0.003069 | 0.101168 ± 0.015454 |
| 008 | 0.077313 | 0.054685 ± 0.001710 | 0.084973 ± 0.006619 |
| 013 | 0.019695 | 0.014135 ± 0.001209 | 0.037501 ± 0.012377 |
| 017 | 0.067008 | 0.047278 ± 0.004931 | 0.101722 ± 0.033730 |

ROC-AUC:

| Test user | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| 001 | 0.707726 | 0.756359 ± 0.003682 | 0.761972 ± 0.031943 |
| 008 | 0.648177 | 0.769018 ± 0.008676 | 0.761578 ± 0.036463 |
| 013 | 0.656998 | 0.716229 ± 0.012599 | 0.762414 ± 0.057766 |
| 017 | 0.694172 | 0.816537 ± 0.006189 | 0.730680 ± 0.030634 |

AP:

| Test user | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| 001 | 0.126200 | 0.111433 ± 0.013605 | 0.126594 ± 0.019652 |
| 008 | 0.195426 | 0.319682 ± 0.010448 | 0.318317 ± 0.042429 |
| 013 | 0.169395 | 0.273039 ± 0.009184 | 0.219721 ± 0.070982 |
| 017 | 0.380323 | 0.549658 ± 0.010917 | 0.429307 ± 0.022581 |

### 이상 유형별 비교

ROC-AUC/AP의 음성은 모든 방식에 동일한 Test 원본 정상 46,299행이다. 해당 유형의 synthetic 이상만 양성으로 포함하며 다른 이상 유형은 제외한다.

| Type | Actual anomaly rows | Statistical Rule detected | Isolation Forest detected | Autoencoder detected |
| --- | --- | --- | --- | --- |
| route_deviation | 378 | 73.000000 | 48.600000 ± 3.646917 | 78.600000 ± 14.328294 |
| abnormal_speed | 391 | 63.000000 | 55.000000 ± 16.598193 | 165.600000 ± 65.683331 |
| long_stop | 963 | 326.000000 | 347.200000 ± 1.483240 | 141.600000 ± 87.021836 |
| direction_change | 380 | 301.000000 | 375.000000 ± 1.000000 | 330.200000 ± 37.399198 |

Recall:

| Type | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| route_deviation | 0.193122 | 0.128571 ± 0.009648 | 0.207937 ± 0.037906 |
| abnormal_speed | 0.161125 | 0.140665 ± 0.042451 | 0.423529 ± 0.167988 |
| long_stop | 0.338525 | 0.360540 ± 0.001540 | 0.147040 ± 0.090365 |
| direction_change | 0.792105 | 0.986842 ± 0.002632 | 0.868947 ± 0.098419 |

ROC-AUC:

| Type | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| route_deviation | 0.628291 | 0.601612 ± 0.005600 | 0.631110 ± 0.045058 |
| abnormal_speed | 0.613895 | 0.683816 ± 0.011804 | 0.830628 ± 0.052143 |
| long_stop | 0.596321 | 0.736165 ± 0.014428 | 0.673632 ± 0.041473 |
| direction_change | 0.952300 | 0.982595 ± 0.004022 | 0.965768 ± 0.009819 |

AP:

| Type | Statistical Rule | Isolation Forest | Autoencoder |
| --- | --- | --- | --- |
| route_deviation | 0.014690 | 0.012860 ± 0.000961 | 0.014475 ± 0.002256 |
| abnormal_speed | 0.147961 | 0.015212 ± 0.000681 | 0.135728 ± 0.028494 |
| long_stop | 0.039000 | 0.056390 ± 0.003744 | 0.038137 ± 0.009323 |
| direction_change | 0.117803 | 0.527501 ± 0.058777 | 0.140923 ± 0.027163 |

Score 통계 (모델 간 raw score 단위 비교 금지):

| Model | Type | Score mean | Score median | Score p95 | Score p99 |
| --- | --- | --- | --- | --- | --- |
| Statistical Rule | route_deviation | 11.126978 | 3.853614 | 46.285134 | 143.012649 |
| Statistical Rule | abnormal_speed | 3260.428172 | 2.223333 | 55262.027738 | 55262.029865 |
| Statistical Rule | long_stop | 8.146891 | 3.452329 | 14.252419 | 56.061990 |
| Statistical Rule | direction_change | 47.719114 | 14.089638 | 243.308853 | 261.957685 |
| Isolation Forest | route_deviation | 0.456043 ± 0.002784 | 0.439296 ± 0.003679 | 0.569448 ± 0.004827 | 0.623402 ± 0.005886 |
| Isolation Forest | abnormal_speed | 0.464761 ± 0.003140 | 0.445512 ± 0.005060 | 0.539717 ± 0.005797 | 0.575780 ± 0.007194 |
| Isolation Forest | long_stop | 0.486166 ± 0.004416 | 0.452741 ± 0.004982 | 0.589342 ± 0.008167 | 0.658007 ± 0.008582 |
| Isolation Forest | direction_change | 0.621353 ± 0.005332 | 0.616675 ± 0.007241 | 0.700031 ± 0.007050 | 0.710466 ± 0.008426 |
| Autoencoder | route_deviation | 0.095831 ± 0.046028 | 0.022803 ± 0.009758 | 0.405001 ± 0.220290 | 1.059950 ± 0.268882 |
| Autoencoder | abnormal_speed | 1037.152149 ± 607.660088 | 0.069941 ± 0.021157 | 16672.680880 ± 11326.343821 | 17369.025620 ± 10917.833160 |
| Autoencoder | long_stop | 0.163946 ± 0.063193 | 0.027271 ± 0.006050 | 0.233796 ± 0.094417 | 3.785848 ± 2.379905 |
| Autoencoder | direction_change | 1.692904 ± 0.216911 | 0.854178 ± 0.211702 | 5.168666 ± 2.019211 | 6.160598 ± 2.216347 |

사용자/유형/전체의 mean·sample std·min·max·defined run count는 summary CSV에 모두 저장했다. Rule의 std를 0으로 채우거나 5개 seed로 복제하지 않았다. 점수가 threshold와 같은 경우의 정상 판정을 유지하도록 신규 prediction CSV는 round-trip float parser로 읽는다.

### 산출물·그래프

Metrics 경로: `outputs/metrics/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/`

- `baseline_results.csv`, `baseline_summary.csv`
- `per_user_baseline_metrics.csv`, `per_user_baseline_summary.csv`
- `per_anomaly_type_baseline_metrics.csv`, `per_anomaly_type_baseline_summary.csv`
- `baseline_comparison.json`, `comparison_validation.json`, `experiment_design.json`, `final_verification.json`
- `statistical_rule/`, `isolation_forest/seed_{7,21,42,100,2026}/`의 Validation/Test 예측, 지표, 사용자/macro/유형 결과

Model 경로: `models/stage5/geolife_u000_019_first5_seed42_dedup/baselines/`에 Rule parameters/config와 IF seed별 model/scaler/config를 저장했다. AE 디렉터리는 읽기 전용으로 보존했다.

- [model_f1_comparison.png](../outputs/figures/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/model_f1_comparison.png)

- [model_roc_auc_comparison.png](../outputs/figures/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/model_roc_auc_comparison.png)

- [model_average_precision_comparison.png](../outputs/figures/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/model_average_precision_comparison.png)

- [model_fpr_comparison.png](../outputs/figures/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/model_fpr_comparison.png)

- [anomaly_recall_by_model.png](../outputs/figures/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/anomaly_recall_by_model.png)

- [per_user_f1_by_model.png](../outputs/figures/stage5/geolife_u000_019_first5_seed42_dedup/baseline_comparison/per_user_f1_by_model.png)


6개 그래프는 Rule 단일 막대와 IF/AE mean ± sample std를 표시한다. F1 및 유형별 Recall 그래프를 직접 열어 label/오차막대/가독성을 확인했다.

### 테스트·시간

`python -m unittest discover -s tests -v` (`.venv` Python 사용): **188개 모두 통과** (기존 164 + 신규 24; 최종 74.139초). 기존 Stage 1~5.3 테스트는 수정하지 않았다. Train 정상 전용 fit, Test 입력 거부, Validation anomaly 제외, p95/strict comparison, 8 feature, score 방향, 사용자/macro/유형 평가, seed sample std, AE 무학습/무추론/무threshold 재계산 및 checksum 보존, NA 처리, score CSV 동점 보존을 검증했다.

초기 검증에서는 큰 synthetic fixture score를 절대 소수점으로 비교한 테스트가 부동소수점 오차로 실패했다. 상대 오차 검사와 동일 round-trip parser로 검증을 맞춘 후 전체 suite를 재실행했다. Detector 공식·feature weight·IF hyperparameter는 변경하지 않았다.

실제 Stage 5.4 전체 시간은 **33.03초**, Rule/IF 6회 파이프라인 시간 합은 **19.45초**다. 전체 시간은 데이터/AE 파일 확인·baseline fit/scoring·CSV 저장/검증·집계/그래프를 포함한다. 별도 최종 독립 검증과 단위 테스트 시간은 실험 시간에 포함하지 않는다. AE 과거 학습 시간은 더하지 않았다. CLI 실행 로그는 `outputs/metrics/stage54_experiment.log`, 테스트 로그는 `outputs/metrics/stage54_tests.log`다.

### 주요 관찰과 가능한 설명

**관찰:** IF의 평균 F1은 0.266040으로 AE 0.229089보다 0.036951 높다. Recall은 0.391004 대 0.339015, AP는 0.210599 대 0.156792, FPR은 0.070641 대 0.072602다. ROC-AUC는 0.746730 대 0.747649로 비슷하다. IF F1 sample std는 0.005419로 AE 0.047369보다 작다. 고정 dataset의 model seed 변동을 나타내며 새 사용자 집단 간 분산을 측정한 결과는 아니다.

**관찰:** Rule F1 0.256946은 AE 평균보다 높지만 ROC-AUC 0.669346과 AP 0.138804는 AE보다 낮다. 하나의 operating point에서의 탐지 성능과 전체 score ranking은 같은 결론을 주지 않는다. 복잡한 AE가 이번 조건의 전체 지표에서 일관된 우위를 제공했다고 보기는 어렵다.

**관찰:** direction_change Recall은 Rule 0.792105, IF 0.986842, AE 0.868947이다. 세 방식에서 상대적으로 잘 탐지된다. route_deviation은 Rule 0.193122, IF 0.128571, AE 0.207937이며 ROC-AUC도 약 0.60~0.63으로 세 방식 모두 어려움을 보인다.

**관찰:** long_stop Recall은 Rule 0.338525, IF 0.360540, AE 0.147040이다. 세 방식 모두 과반을 놓치지만 baseline은 AE보다 상당히 높은 탐지율이다. abnormal_speed는 AE 0.423529로 Rule 0.161125 및 IF 0.140665보다 높으며 AE의 seed std 0.167988은 크다. 유형에 따라 방식의 강점이 달라 전체 F1만으로 모델을 선택하지 않는다.

**관찰:** user macro F1은 Rule 0.300037, IF 0.337003 ± 0.006735, AE 0.255925 ± 0.041377이다. IF의 사용자 008/013/017 평균 F1은 AE보다 높고 사용자 001은 IF 0.140105 대 AE 0.141760으로 비슷하다. IF의 사용자 001 FPR은 0.143362 ± 0.003069로 나머지 사용자보다 높다. 동일 Validation 정상 5% 정책에도 모든 모델의 전체 Test FPR은 약 6.6~7.3%이며 사용자별 차이가 있다.

**가능한 설명:** route_deviation의 공통 약점은 현재 point-level 8 feature가 경로 문맥을 충분히 표현하지 못할 가능성을 제시한다. 실제 feature overlap과 synthetic route 생성의 동작을 확인하는 후속 실험이 필요하다. 이 비교만으로 표현력 한계를 인과적으로 입증한 것은 아니다.

**가능한 설명:** AE long_stop 약점과 abnormal_speed 강점은 표준화된 복원 MSE, checkpoint 선택, feature별 복원 반응 또는 모델 방법의 차이를 후속 실험 대상으로 제시한다. 특정 architecture가 원인이라고 단정할 수 없다. IF의 안정성·유형 차이도 tree subsampling/score 형태의 영향 가능성을 검증해야 한다.

원본 정상 label은 연구 가정이고 이상은 synthetic이다. 한 고정 user split과 생성 방식에서의 결과다. 5개 model seed를 독립 사용자 dataset 반복으로 취급하거나 성능 차이를 통계적 유의성으로 해석하지 않는다.

### Stage 5 최종 결론 후보

1. 누출 없는 user-disjoint dataset과 deterministic dedup에서 unseen-user 평가는 Stage 4 조건보다 어렵고, seed 42 하나의 우연으로 낮은 AE 성능을 설명하기 어렵다. Stage 4와 데이터 조건이 다르므로 차이를 architecture의 인과 효과로 해석하지 않는다.

2. 동일 8 feature·Train 정상 fit·Validation 정상 p95 조건에서 AE가 baseline보다 일관된 전체 우위를 보이지 않았다. IF의 평균 F1/AP와 seed 안정성, AE의 abnormal_speed 탐지라는 서로 다른 특성이 확인됐다. 최고 seed나 최고 모델을 최종 배포 후보로 선정하지 않는다.

3. route_deviation 공통 약점, long_stop의 AE 추가 약점, direction_change의 상대적 용이성 및 사용자별 False Positive 차이를 다음 단계 문제로 명확히 남긴다.

### Stage 6에서 먼저 개선할 문제 후보

1. **route_deviation의 표현과 생성 label 검증**: frozen 예측에서 FN의 현재 feature 분포·경로 문맥을 점검하고, 이후 승인된 별도 단계에서 route/context feature 또는 시계열 표현을 하나씩 추가하는 통제 실험을 설계한다. 사용자·synthetic 조건을 바꾸는 실험과 표현을 바꾸는 실험을 분리해야 한다.

2. **long_stop의 AE 복원 반응**: 현재 feature별 reconstruction contribution과 정상/이상 score 분포를 먼저 확인하고 학습 목적/표현/모델 변경을 하나씩 분리한 ablation 후보를 만든다. 높은 abnormal_speed Recall이 함께 유지되는지도 평가한다.

3. **사용자별 calibration과 FPR 편차**: 특히 001의 높은 FPR 및 Validation→Test operating point 이동을 분석한다. 후속 calibration 설계는 Train/Validation 사용자만 사용하고 새 Test 결과로 threshold를 조절하지 않는다.

이번 단계에서 route-aware feature, Window, LSTM, 새로운 threshold, feature 삭제/추가, AE 변경이나 tuning을 구현하지 않았다. Stage 6 실행은 사용자의 다음 결정에 따른다.

### Git 상태

- Stage 5.3: `ff5ca3e` — `stage5: add multi-seed stability analysis`. CLI push 인증 실패 후 사용자 승인된 GitHub 연결 앱으로 동일 blob/tree를 기존 branch에 게시했다. 로컬과 원격 HEAD가 일치한다. 원본 로컬 commit은 `stage53-local-verified-40fe589` tag로 보존했다.

- Stage 5.4: 소스·테스트·보고서·README 4개 파일을 재검증 후 `stage5: compare rule isolation forest and autoencoder`로 커밋·게시한다. PR 생성이나 merge는 진행하지 않는다. 데이터/모델/실험 산출물은 기존 ignore 정책을 유지한다.
