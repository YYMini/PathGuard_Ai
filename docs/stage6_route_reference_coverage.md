# Stage 6.2 — Train-only Route Reference Coverage & Abstention Audit

## 사전 분석 계약 (2026-10-04)

목적은 unseen-user 정상 위치에 대한 Train-only point reference의 coverage와 synthetic route 분리력을 검증하는 것입니다. 기존 Stage 1~6.1 코드는 변경하지 않으며 모델 학습/추론, scaler/threshold 재계산, synthetic/label/split 변경, production feature 추가를 하지 않습니다. Stage 6.3 구현과 commit/push/PR/merge도 하지 않습니다.

Reference는 frozen Train의 품질 유효 원본 정상 좌표만 사용합니다. User/split/trajectory manifest와 exact duplicate 제외를 검증하고 BallTree haversine으로 lat/lon radians를 질의합니다. EARTH_RADIUS_M=6371000이며 leaf_size=40은 계산 설정으로 고정합니다. Train 좌표 index는 Train만으로 구성하고 데이터/metadata는 정렬해 재현성을 유지합니다. 동일 좌표 reference tie는 lexical lineage를 우선합니다. 좌표 중복 자체는 trajectory fingerprint 중복이 아니므로 제거하지 않습니다.

Coverage 반경은 요청된 10/25/50/100/200/500/1000m, abstention 후보는 25/50/100/200/500m로 고정합니다. 모든 반경을 보고하며 Test 결과로 최적 반경을 선택하지 않습니다. Validation 원본 정상에도 같은 반경을 기술적으로 적용하되 selection/calibration은 하지 않습니다. 높은 거리 score를 더 route-deviation-like로 정의하고 ROC-AUC/AP만 계산합니다. 원본 값은 제거/clip하지 않고 그래프 축에만 log scale을 적용할 수 있습니다.

Source-matched 비교는 (source_trajectory_id, source_point_index)로 synthetic point를 frozen 정상 point에 연결하고 user/split/timestamp를 확인합니다. Delta는 synthetic minus original reference distance입니다. Feature 동일/zero displacement는 Stage 6.1의 사전 tolerance와 재계산을 대조합니다. Saved manifest에는 point별 configured amplitude가 없으므로 random seed로 anomaly를 재생성하지 않습니다. 규칙의 100~300m range와 실제 변위만 보고합니다.

Covered synthetic subset은 **원본 source distance<=C**로 정의합니다. 이는 분석용 oracle 조건이고 실제 synthetic 입력에서 source 원본은 관측되지 않습니다. Synthetic 거리 자체로 gate하면 탐지 대상 offset도 coverage 밖으로 제거할 수 있습니다. 그러므로 이 분석은 deployable abstention detector 성능이 아니며 실제 gate 설계는 후속 작업입니다. single-class/빈 subset ROC/AP는 NA와 사유를 기록합니다. 데이터의 point/trajectory 내 상관으로 row-level 지표를 독립 표본의 추정치로 해석하지 않습니다.

Haversine 좌표 순서와 radians 규칙은 [scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html)를 따릅니다. nearest **point**는 continuous route/polyline, 도로 적합성, 개인 습관 경로를 나타내지 않습니다.

## 완료 결과 (2026-10-05)

판정은 **B + D가 주된 결과**입니다. Global Train point reference는 unseen Test 정상 위치 전체를 설명하지 못했고, 사용자별 coverage 격차가 큽니다. 원본 source가 reference 근처라는 조건에서는 A에 해당하는 분리 신호가 관찰되지만, 이는 audit oracle 조건이며 배포 정책의 성능이 아닙니다. 모델에 거리 feature를 즉시 추가할 근거로 사용하지 않습니다.

전체 **240개 테스트 PASS**(기존 208 + 신규 32, 63.140초), 실제 audit **10.493040초**입니다. Reference 86,067 point·12명·51개 trajectory, Test 정상 46,299행, route anomaly 378행, source pair 378개로 실제 counts가 예상과 일치합니다. 코드에서 이 숫자로 강제 filtering하지 않았습니다.

## Train-only reference와 누수 방지

Train user는 000/003/005/006/007/009/010/012/014/015/016/019입니다. Frozen train.csv를 읽고 user split 및 trajectory manifest와 대조합니다. source 품질·training eligibility·finite coordinates를 확인하며 synthetic/low-quality/excluded duplicate 및 다른 split 사용자 유입 시 실패합니다. 동일 fingerprint의 eligible Train source도 실패합니다. Reference 내 low-quality/synthetic/duplicate trajectory/Validation 또는 Test user는 모두 0입니다.

동일 좌표 반복 행은 5,677개이며 정상 정지/공유 위치일 수 있으므로 삭제하지 않습니다. 좌표가 같은 reference의 metadata는 lexical lineage 첫 행으로 canonicalize합니다. 서로 다른 좌표의 같은 거리 tie는 저장된 sklearn 버전과 정렬된 입력에서 BallTree 순서를 따릅니다. 이런 tie 때문에 nearest user 빈도를 개인 경로의 유일한 설명으로 해석하면 안 됩니다. nearest user·trajectory·point index는 분석용 metadata입니다.

Stage 5 dataset/models/scaler/predictions/metrics/figures 235개와 Stage 6.1 산출물 18개, 총 **253개 파일을 작업 전후 checksum 비교**했고 변화는 0개입니다. user/source/sample/fingerprint cross-split leakage도 모두 0입니다. 기존 source 및 Stage 1~6.1 테스트는 수정하지 않았습니다. snapshot은 `outputs/metrics/stage62_protected_snapshot.json`입니다.

## Test 정상 reference 거리와 coverage

전체 Test 정상 거리(m):

- count: 46299
- mean: 5093.376602
- median: 441.214175
- min: 0.034429
- max: 54441.843391
- p25: 21.253367
- p50: 441.214175
- p75: 4521.288655
- p90: 9907.484227
- p95: 37061.894007
- p99: 50663.998036

| radius_m | normal_count | normal_covered_count | normal_coverage_pct |
| --- | --- | --- | --- |
| 10.000000 | 46299.000000 | 7560.000000 | 16.328646 |
| 25.000000 | 46299.000000 | 12318.000000 | 26.605326 |
| 50.000000 | 46299.000000 | 14768.000000 | 31.897017 |
| 100.000000 | 46299.000000 | 16915.000000 | 36.534266 |
| 200.000000 | 46299.000000 | 19101.000000 | 41.255751 |
| 500.000000 | 46299.000000 | 23714.000000 | 51.219249 |
| 1000.000000 | 46299.000000 | 28149.000000 | 60.798289 |

100m coverage는 **36.5343%**, 500m는 **51.2192%**, 1000m도 **60.7983%**입니다. p95는 37.06km, max는 54.44km입니다. Extreme 정상 값을 제거하지 않았고 figure의 symlog 축만 적용했습니다. 이런 정상의 거리 큰 tail 때문에 큰 거리 자체를 행동 이상으로 사용할 수 없습니다.

## 사용자별 geographic coverage

| user_id | normal_count | normal_median | normal_p90 | normal_p95 | coverage_25m_pct | coverage_50m_pct | coverage_100m_pct | coverage_200m_pct | coverage_500m_pct | route_count | route_distance_median | source_matched_delta_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | 17031 | 639.829379 | 8583.254623 | 9069.659432 | 19.423404 | 22.470789 | 26.410663 | 31.689273 | 42.792555 | 88 | 660.647040 | 35.599199 |
| 008 | 8485 | 19.728066 | 695.179278 | 1731.410662 | 55.474367 | 66.458456 | 76.676488 | 84.832057 | 89.204478 | 93 | 61.394075 | 29.706233 |
| 013 | 18634 | 2438.810085 | 40898.930376 | 48910.955401 | 18.138886 | 23.059998 | 26.113556 | 28.936353 | 38.365354 | 79 | 158.775763 | 73.208989 |
| 017 | 2149 | 144.480301 | 811.130515 | 954.344961 | 42.950209 | 46.765938 | 48.627268 | 51.838064 | 79.478827 | 118 | 83.147925 | 37.407828 |

100m coverage는 user 001/008/013/017에서 **26.4107/76.6765/26.1136/48.6273%**입니다. 최대 격차는 50.5629 percentage points입니다. 008은 상대적으로 covered지만 001/013은 정상 위치 다수가 reference에서 멉니다. User 013 정상 거리 median 2.44km와 p95 48.91km는 사용자마다 활동 지역과 reference 구성의 차이가 크다는 관찰입니다. 개인 행동 특성이나 오탐 원인을 인과적으로 증명하지 않습니다.

## Test route distance와 분리 지표

Route synthetic 378행 거리(m):

- count: 378
- mean: 573.590151
- median: 118.161124
- min: 1.636108
- max: 4610.273713
- p25: 47.588582
- p50: 118.161124
- p75: 370.114734
- p90: 962.211849
- p95: 4456.283171
- p99: 4563.573820

전체 normal 46,299 vs route 378의 거리 score는 **ROC-AUC 0.399946529**, **AP 0.006115867**입니다. Positive prevalence/AP random baseline은 **0.008098207**입니다. 큰 거리=더 route-like라는 사전 방향을 유지했습니다. Test 결과를 보고 score 방향을 뒤집지 않았고 classifier threshold도 만들지 않았습니다.

Matched source original의 거리 median은 **14.888071m**로 전체 정상 median **441.214175m**보다 훨씬 작습니다. Synthetic 구간은 실제 frozen random point segment 선택의 부분집합이고, 사용자별 anomaly/normal row 비중도 다릅니다. 따라서 전체 cohort 차이와 주입 효과를 분리해야 합니다. 전체 ROC<0.5를 spatial 정보가 원천적으로 무의미하다는 결론으로 해석하지 않습니다.

사용자별 전체 분리 지표:

| user_id | full_roc_auc | full_average_precision | full_prevalence |
| --- | --- | --- | --- |
| 001 | 0.470210 | 0.004554 | 0.005140 |
| 008 | 0.639490 | 0.014245 | 0.010842 |
| 013 | 0.389750 | 0.003227 | 0.004222 |
| 017 | 0.458246 | 0.045273 | 0.052051 |

## Source-matched delta와 실제 변위

같은 source lineage의 synthetic minus original reference 거리(m):

- count: 378
- mean: 41.737666
- median: 39.200099
- min: -194.751839
- max: 213.768691
- p25: 0.000000
- p50: 39.200099
- p75: 92.885331
- p90: 142.998638
- p95: 168.865935
- p99: 206.039806
- delta_gt_0_fraction: 0.695767
- delta_ge_10m_fraction: 0.648148
- delta_ge_25m_fraction: 0.563492
- delta_ge_50m_fraction: 0.460317
- delta_ge_100m_fraction: 0.235450
- delta_ge_200m_fraction: 0.013228

Positive/zero/negative delta는 **263/38/77행**, 비율 **69.5767/10.0529/20.3704%**입니다. 25/50/100/200m 이상 증가는 **56.3492/46.0317/23.5450/1.3228%**입니다. 실제 displacement median **118.010872m**보다 reference delta median **39.200099m**가 작습니다. Offset 방향에 따라 다른 Train point에 가까워지거나 reference의 빈 영역으로 이동할 수 있기 때문에 displacement가 그대로 reference distance 증가로 전달되지 않습니다.

Actual displacement와 reference delta의 Spearman rho는 **0.473419788**입니다. 모든 pair에서 `abs(delta reference distance) <= actual displacement + 1e-5m`를 확인했습니다. 동일 spherical metric에서 reference set까지의 거리 함수가 만족하는 관계입니다.

Configured amplitude는 기존 generator 코드의 **100~300m 범위**만 확인 가능합니다. Saved manifest에는 sample별 실제 amplitude가 없어 requested amplitude vs actual displacement 상관을 계산하지 않았습니다. seed replay나 synthetic 재생성도 하지 않았습니다. CSV에는 configured_amplitude_m=NA 및 사유를 기록했습니다.

## Coverage-aware / abstention 조건부 감사

Synthetic의 covered 여부는 **원본 source reference distance<=C**로 정했습니다. Synthetic query 거리 자체는 gate에 사용하지 않습니다. Threshold 기반 Accuracy/F1/Recall은 계산하거나 최적화하지 않았습니다.

| radius_m | normal_covered_count | normal_coverage_pct | route_source_covered_count | route_source_coverage_pct | normal_abstained_count | route_source_abstained_count | roc_auc | average_precision | prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25.000000 | 12318.000000 | 26.605326 | 211.000000 | 55.820106 | 33981.000000 | 167.000000 | 0.895896 | 0.760729 | 0.016841 |
| 50.000000 | 14768.000000 | 31.897017 | 222.000000 | 58.730159 | 31531.000000 | 156.000000 | 0.858343 | 0.629293 | 0.014810 |
| 100.000000 | 16915.000000 | 36.534266 | 236.000000 | 62.433862 | 29384.000000 | 142.000000 | 0.817581 | 0.380996 | 0.013760 |
| 200.000000 | 19101.000000 | 41.255751 | 251.000000 | 66.402116 | 27198.000000 | 127.000000 | 0.741128 | 0.041958 | 0.012970 |
| 500.000000 | 23714.000000 | 51.219249 | 304.000000 | 80.423280 | 22585.000000 | 74.000000 | 0.656032 | 0.058464 | 0.012657 |

반경을 늘리면 coverage는 증가하지만 분리력은 대체로 약해집니다. AP는 prevalence와 rank tail에 민감하며 단조 감소하지 않습니다(200m→500m). 25m 조건의 ROC-AUC/AP **0.895896/0.760729**는 전체 모델 성능이 아니며, 정상의 73.3947%와 anomaly source의 44.1799%는 이 조건 밖입니다. Test에서 25m를 최적 policy로 채택하지 않았습니다. 모든 요청 반경을 그대로 보고합니다.

**배포 한계:** 실제 입력에는 synthetic 원본 source가 없으므로 source-conditioned gate를 직접 사용할 수 없습니다. Synthetic point 자체가 coverage 밖으로 이동하면 naive current-distance gate가 탐지 대상 이탈을 abstain할 수 있습니다. 이 단계의 값은 reference가 제공하는 조건부 분리 신호의 감사 결과이며 deployable abstention 성능이나 selective classification 성능이 아닙니다.

사용자별 모든 반경의 조건부 결과:

| user_id | radius_m | normal_coverage_pct | route_source_covered_count | route_source_coverage_pct | roc_auc | average_precision | prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | 25 | 19.423404 | 14 | 15.909091 | 0.896226 | 0.649197 | 0.004214 |
| 001 | 50 | 22.470789 | 14 | 15.909091 | 0.835847 | 0.264718 | 0.003645 |
| 001 | 100 | 26.410663 | 14 | 15.909091 | 0.725322 | 0.006287 | 0.003103 |
| 001 | 200 | 31.689273 | 14 | 15.909091 | 0.604503 | 0.003284 | 0.002587 |
| 001 | 500 | 42.792555 | 45 | 51.136364 | 0.796719 | 0.298379 | 0.006137 |
| 008 | 25 | 55.474367 | 55 | 59.139785 | 0.882701 | 0.695117 | 0.011550 |
| 008 | 50 | 66.458456 | 64 | 68.817204 | 0.857336 | 0.561771 | 0.011222 |
| 008 | 100 | 76.676488 | 78 | 83.870968 | 0.823295 | 0.297695 | 0.011847 |
| 008 | 200 | 84.832057 | 93 | 100.000000 | 0.753831 | 0.030760 | 0.012755 |
| 008 | 500 | 89.204478 | 93 | 100.000000 | 0.716881 | 0.021775 | 0.012138 |
| 013 | 25 | 18.138886 | 47 | 59.493671 | 0.911513 | 0.836485 | 0.013715 |
| 013 | 50 | 23.059998 | 48 | 60.759494 | 0.869497 | 0.758620 | 0.011047 |
| 013 | 100 | 26.113556 | 48 | 60.759494 | 0.844230 | 0.581585 | 0.009768 |
| 013 | 200 | 28.936353 | 48 | 60.759494 | 0.787857 | 0.039025 | 0.008824 |
| 013 | 500 | 38.365354 | 48 | 60.759494 | 0.594226 | 0.008171 | 0.006669 |
| 017 | 25 | 42.950209 | 95 | 80.508475 | 0.912665 | 0.827242 | 0.093320 |
| 017 | 50 | 46.765938 | 96 | 81.355932 | 0.890786 | 0.757075 | 0.087193 |
| 017 | 100 | 48.627268 | 96 | 81.355932 | 0.874970 | 0.647909 | 0.084137 |
| 017 | 200 | 51.838064 | 96 | 81.355932 | 0.830145 | 0.300055 | 0.079339 |
| 017 | 500 | 79.478827 | 118 | 100.000000 | 0.576564 | 0.069893 | 0.064622 |

같은 25m 조건에서도 anomaly source coverage는 user 001/008/013/017에서 **15.9091/59.1398/59.4937/80.5085%**입니다. Global 평균만으로 사용자별 정보 가용성을 설명할 수 없습니다. User 001의 25~200m covered anomaly는 14개뿐으로 적고 같은 point들이 반복 집계되므로 metric 안정성이나 독립 표본 근거가 아닙니다.

## Validation 정상의 고정 반경 결과

Validation은 reference에 포함하지 않고 같은 index에 질의만 했습니다. Normal 19,448개에 같은 사전 반경을 적용했습니다. parameter/threshold/reference 보정에는 사용하지 않았습니다.

| radius_m | normal_count | normal_covered_count | normal_coverage_pct |
| --- | --- | --- | --- |
| 10.000000 | 19448.000000 | 2321.000000 | 11.934389 |
| 25.000000 | 19448.000000 | 3532.000000 | 18.161251 |
| 50.000000 | 19448.000000 | 4205.000000 | 21.621761 |
| 100.000000 | 19448.000000 | 4637.000000 | 23.843069 |
| 200.000000 | 19448.000000 | 5193.000000 | 26.701974 |
| 500.000000 | 19448.000000 | 11030.000000 | 56.715343 |
| 1000.000000 | 19448.000000 | 14950.000000 | 76.871658 |

Validation normal coverage@100m도 **23.8431%**로 낮습니다. 향후 coverage policy를 연구한다면 Train/Validation에서 사전 고정하고 새로운 holdout에서 평가해야 합니다. 이번에 확인한 Test 결과로 그 policy를 선택하면 현재 Test는 개발 정보가 됩니다.

## Nearest Train user 집중

Nearest reference user 빈도는 모든 Test 정상 point를 집계했습니다. 각 Test user의 최대 비중 Train user는:

| Test user | nearest Train user | point 수 | 비율(%) |
| --- | --- | ---: | ---: |
| 001 | 007 | 5124 | 30.0863 |
| 008 | 014 | 2745 | 32.3512 |
| 013 | 012 | 7369 | 39.5460 |
| 017 | 014 | 845 | 39.3206 |

전체 빈도는 `nearest_reference_user_summary.csv`에 저장했습니다. 일부 Train 사용자의 경로가 reference의 지역적 support를 많이 담당합니다. 이것만으로 그 사용자가 결과를 인과적으로 지배한다고 입증하지 않으며, Train-user 제거 실험이나 coverage 지역별 추가 감사가 필요합니다.

## Stage 6.1 label boundary 연계

- Zero-displacement **38행**: 실제 좌표 변위와 reference delta가 모두 정확히 0m입니다. Original/synthetic reference distance median은 둘 다 **22.499917m**입니다.
- Identical-feature **19행**: 위 38행의 subset이며 실제 변위/reference delta 모두 0m입니다. Original/synthetic median은 둘 다 **21.923252m**입니다.
- 이번 frozen 데이터의 19행은 좌표가 이동했는데 local feature만 같았던 사례가 아닙니다. 테스트에서는 그런 경우와 zero-coordinate case를 별도로 구분하도록 검증했습니다. Stage 6.1의 200m longitude 이동 반례는 별개의 Train-only 반례입니다.
- Label=1이어도 source에 비해 spatial distance가 달라지지 않는 point가 실제 존재합니다. 끝점 incoming feature가 변할 수 있다는 Stage 6.1의 해석은 그대로 유지하고 label은 수정하지 않았습니다.

각 point 좌표·lineage·원본/합성 거리·delta·flag는 `label_boundary_spatial_audit.csv`에 저장했습니다.

## Case 판정과 한계

**Case B:** global normal coverage가 작습니다. 특히 25~200m에서 26.6~41.3%이며 전체 거리 score는 정상의 geographic tail을 anomaly-like로 순위화합니다. Global Train point reference를 unseen-user 행동 이상 reference로 바로 사용할 수 없습니다.

**Case D:** 사용자별 coverage 격차가 큽니다. Coverage-aware 판단과 out-of-support 상태를 분리할 필요성이 관찰됩니다. 실제 abstention gate의 설계·검증은 아직 없습니다.

**Case A의 조건부 신호:** 원본이 reference 가까이 있다는 oracle 조건에서 분리력이 확인됩니다. 전체 Test 정상의 높은 coverage라는 A의 전제는 충족되지 않으므로 전역 A로 판정하지 않습니다.

**Case C:** 일부 사용자·넓은 반경에서 분포 중첩이 남지만, 전체 사용자에 대해 '높은 coverage인데 거리 표현만 부족'이라고 일반화할 수 없습니다. 현재 evidence만으로 nearest point의 구조적 부족과 geographic support 부족을 완전히 분리하지 못합니다.

Nearest point distance는 continuous polyline/centerline/road 거리, 개인 정상 경로, 경로 방향·진행 순서를 나타내지 않습니다. Sparse sampling과 다른 경로로의 접근도 영향을 줍니다. Point-level ROC/AP는 trajectory 내 상관과 사용자별 가중치의 영향을 받습니다. Confidence interval, 독립 재표본, map truth, 정상 경로 ground truth가 없으며 인과적 원인별 기여율도 추정하지 않았습니다.

## Stage 6.3 제안 (설계만)

1. Global nearest Train distance를 곧바로 AE input에 추가하지 않습니다. 현재 전역 score가 geographic novelty와 행동 이탈을 혼합하기 때문입니다.
2. 다음 실험은 **추론 때 관측 가능한 route context와 coverage gate의 식별성 감사**를 우선 제안합니다. 관측 prefix에서 reference support를 확정한 뒤 이후 이탈을 보는 설계와 external map/personal opt-in history의 정보 조건을 구분합니다. 미래 point, Test 정상 경로, synthetic 원본을 oracle reference로 쓰지 않아야 합니다.
3. Train-only polyline/segment reference는 point sampling 오차를 줄일 후보지만 unknown region을 해결한다는 보장은 없습니다. 지도 기반 geographic coverage와 habitual-route anomaly도 구분해야 합니다.
4. 개인 reference가 도움이 될 가능성은 있지만 unseen-user 조건에서 사전 개인 정상 데이터는 현재 없습니다. 사용자별 reference를 쓰려면 enrollment/history 가정을 명시한 별도 평가가 필요합니다.
5. Sequence/window 모델은 route 진행 맥락을 다룰 후보이며 현재 feature의 절대 offset 비식별성을 자동 해결하지 않습니다. 따라서 이번 결과만으로 spatial feature보다 sequence 학습을 먼저 시작할 근거는 없습니다. 먼저 정보 가용성과 gate를 검증하고 새로운 holdout을 마련하는 편이 타당합니다.

Stage 6.3 코드, route feature model input, 모델 학습 및 threshold 변경은 구현하지 않았습니다.

## 코드, 테스트, 재현 및 산출물

코드: `src/audit_route_reference_coverage.py`, 테스트: `tests/test_route_reference_coverage.py`.

```powershell
python -m unittest discover -s tests -v
python -m src.audit_route_reference_coverage
```

기존 Stage 6.2 output directory가 있으면 덮어쓰기를 거부합니다. 보호 snapshot이 없거나 Stage 5/6.1 지정 root가 빠지면 실패합니다. 테스트 fixture는 임시 dataset이며 실제 frozen 파일에는 synthetic 생성이나 모델 fit을 하지 않습니다. 신규 32개 테스트는 reference contamination/품질/중복, distance/radians, pairing/경계, coverage/사용자, ROC/AP, protected artifacts와 금지 모델 호출, NA/덮어쓰기 방지를 검증합니다.

Metrics: `outputs/metrics/stage6/route_reference_coverage/`의 필수 8 CSV, `route_reference_audit.json`, `stage6_2_verification.json`, 독립 검증 `stage6_2_final_checks.json` 및 Validation 거리/coverage·사용자별 abstention CSV.

Figures: `outputs/figures/stage6/route_reference_coverage/`:

1. `normal_vs_route_distance_distribution.png`
2. `source_matched_delta_distribution.png`
3. `normal_coverage_by_radius.png`
4. `coverage_by_user.png`
5. `spatial_distance_roc.png`
6. `abstention_coverage_curve.png`

독립 검증은 frozen 좌표·nearest metadata의 Haversine 거리, 65개 고정 query의 전체 Train brute-force 최솟값, 모든 coverage 및 global/covered/user ROC/AP, pair delta, boundary 및 output hash를 확인했습니다. Brute-force 최대 거리 오차는 **3.64e-12m**입니다. 테스트 로그는 `outputs/metrics/stage62_tests.log`, 실제 실행 로그는 `outputs/metrics/stage62_experiment.log`입니다.

Git HEAD는 Stage 6.1 `2b2c1f5ceb594ccc6dbc3d5eff48b8ca9cc6fcde` 그대로이며 Stage 6.2는 미커밋/미푸시 상태입니다. PR/main merge도 하지 않았습니다.
