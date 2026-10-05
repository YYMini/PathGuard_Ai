# Stage 6.1 Route Representation & Synthetic Label Audit

## 사전 분석 범위와 방법

2026-10-02. Stage 5.4 게시 commit은 `19c603a79cbee8996152eb3f06f53c5c8901c29e`이며 로컬/원격 HEAD 일치 후 분석을 시작했다. 모델 재학습·재추론·threshold/scaler 재생성·파라미터 선택을 하지 않는다. 고정 dataset과 Stage 5 산출물 235개 파일의 사전 checksum을 사용하고 실행 전후 검증한다. Stage 6.1은 별도 파일에만 기록하며 커밋/push/PR/merge하지 않는다.

원본과 저장된 route_deviation 전체 synthetic copy를 `(source_trajectory_id, source_point_index)`로 1:1 정렬한다. Validation/Test의 모든 label=1 point를 감사하며 기존 evaluation에 포함되지 않은 source-quality 제외 point도 구분해 기록한다. 모델 비교 및 주 분석은 고정 Test evaluation의 378개 route point를 사용한다. 생성 규칙은 읽기만 하며 합성 데이터를 다시 생성하지 않는다.

미터 변위는 기존 Earth radius 6,371,000m의 Haversine으로 계산한다. 이전/다음 이웃은 제외된 point를 건너뛰지 않는 full synthetic copy에서 가져온다. 기존 8 feature와 bearing_deg의 원본/합성 값을 저장하고 delta를 계산한다. bearing delta는 [-180,180) signed circular difference와 absolute difference를 모두 기록한다. 전체 copy의 label 밖 feature 변화도 기록한다. 저장된 feature를 좌표·timestamp에서 재계산하는 감사는 허용하며 모델 추론은 실행하지 않는다.

분포: Train original normal / Test original normal / Test route anomaly / route와 같은 source point의 원본 / Validation route anomaly. 사용자별 동일 통계를 추가한다. n, mean, median, sample std(ddof=1), IQR, p01/p05/p25/p50/p75/p95/p99/min/max를 기록한다. robust scale은 Stage 5.4 Rule과 동일한 Train IQR/1.349 → population std → constant floor 규칙이다. median effect=abs(cohort median−Train median)/Train scale. Pair delta effect는 같은 point의 synthetic−original 변화이며 사용자 분포 차이와 구분한다.

각 feature의 Train empirical percentile은 midrank `100*(#Train<x + 0.5*#Train==x)/N`이다. p01~p99 내부 여부는 Train 기준의 기술적 분포 진단이며 anomaly 판정 threshold가 아니다. 8개 marginal interval 내부라고 joint distribution에서 정상임을 뜻하지 않는다. near-identical feature는 사전 고정 numerical tolerance atol=1e-6, rtol=1e-8로 판정하고 zero displacement tolerance는 1e-6m다. 이 값으로 모델이나 threshold를 선택하지 않는다.

GPS noise의 ground truth가 없으므로 Train 정상의 시간 가중 prev→next 선형 보간 residual을 **noise proxy**로만 보고한다. 전후/현재 source-quality가 모두 유효하고 양쪽 이동 거리<=3m·속도<=0.5m/s인 stationary triplet subset도 별도로 보고한다. 이 residual에는 움직임·회전·sampling 영향이 섞이며 실제 GPS error라고 단정하지 않는다. 합성 speed>50m/s 및 distance>1000m & dt<=60s는 기존 quality 기준의 물리적 검토 지표다. acceleration은 Train abs(acceleration) p99 초과를 기술적 비교로 사용하며 보편적인 물리 한계로 해석하지 않는다.

200m 위치 이동의 정보 소실은 **Train의 첫 source trajectory**를 메모리에서 일정 longitude rotation한 대조 예제로 검증한다. 첫 point의 great-circle 변위가 정확히 200m가 되는 회전 각을 사용한다. 이는 구면의 isometry이므로 모든 상대 movement feature가 이론상 동일하다. 재계산한 8 feature와 실제 변위를 보고하며 새 anomaly dataset/production feature/모델 점수를 만들지 않는다.

Candidate route representation은 설계와 leakage 조건만 검토한다. Test 사용자의 정상 경로를 미리 알거나 synthetic 원본을 추론 시 reference로 사용하는 candidate는 현재 unseen-user 시나리오에서 제외한다. 모든 score와 Recall은 기존 CSV의 저장된 값을 읽고 유형별 기존 Stage 5.4 지표와 대조한다. 분석 결과를 기반으로 Stage 6.2 후보를 제안하되 이번 단계에서 변경을 구현하지 않는다.
## 완료 결과와 검증

Stage 5.4는 전체 188개 테스트와 독립 지표 검증 후 기존 feature branch에 게시했습니다. 최종 local/remote SHA는 `19c603a79cbee8996152eb3f06f53c5c8901c29e`로 일치합니다. CLI 인증 실패 후 허용된 GitHub connector로 동일 tree를 게시하고 fetch/diff로 확인했습니다. tree는 `ffb5a8cbecbe688e568378f88e27b9b3ca9242d9`입니다. PR/main merge/squash는 하지 않았습니다.

Stage 6.1은 `src/audit_route_representation.py`와 20개 신규 테스트로 구현했습니다. 기존 188개를 포함한 **208개 테스트 PASS**(55.487초), 실제 audit **16.931초**입니다. 데이터 및 Stage 5 모델·metrics·figures **235개 보호 파일 checksum이 모두 유지**됐습니다. 사용자/source/sample/fingerprint cross-split leakage는 모두 0입니다. 기존 AE 학습·추론·threshold/scaler 재계산, IF 튜닝, Test 기반 파라미터 선택은 하지 않았습니다. Stage 6.1 코드·문서는 커밋/푸시하지 않았습니다.

신규 테스트는 source pairing/lineage, 실제 metre 변위, feature delta, midrank percentile와 tie, Train-only reference, 사용자 집계, label 경계, 200m 반례, checksum/누수, 출력 덮어쓰기 거부, 금지된 모델 호출, 필수 산출물을 검증합니다. 전체 명령은 `python -m unittest discover -s tests -v`입니다. 로그는 `outputs/metrics/stage61_tests.log`, 독립 검증은 `stage6_1_final_checks.json`입니다.

## Synthetic 생성 규칙

기존 `src/synthetic_anomalies.py`의 route 규칙은 길이 10~30 point 구간을 고르고 trajectory 첫/마지막 point를 제외합니다. 시작 index와 길이는 source별 고정 seed로 결정됩니다. seed는 `[42, source_trajectory_id, anomaly_type, 1]`의 JSON SHA256 앞 8 byte이며 manifest에 기록됩니다. amplitude A는 uniform(100,300)m, weight는 `sin(linspace(0,pi,size))`입니다. 구간 양 끝 좌표 offset은 0에 가깝습니다.

시작→끝 chord의 heading은 `atan2(delta_lat, delta_lon*cos(start_lat))`이고 north/east offset은 각각 `-cos(heading)*A*weight`, `sin(heading)*A*weight`입니다. latitude에는 north/111320, longitude에는 east/(111320*cos(updated_lat))를 더합니다. timestamp와 altitude는 유지하고 전체 copy의 이동 feature를 재계산합니다. point index 기준 곡선이므로 시간 기준 속도·가속도 연속성을 보장하지 않습니다.

label은 offset 크기와 무관하게 구간 전체에 1을 부여합니다. source-quality flag를 상속하며 synthetic finite coordinate 검사는 물리적 속도 한계를 검사하지 않습니다. evaluation은 label=1이면서 source-quality가 유효한 point만 포함합니다. 전체 31개 구간 597 labelled point 중 Validation 218→217, Test 379→378이 evaluation에 포함됩니다.

## 좌표 변위, 경계와 물리적 타당성

Test route 378개에서 변위 mean **117.087695m**, median **118.010872m**, std **73.445588m**, IQR **115.352829m**, min/max **0/286.787280m**, p95 **241.784993m**, p99 **268.148379m**입니다. 19개 구간의 지속 시간 median **49초**, 범위 **13~217초**입니다.

- zero displacement label=1: **38/378 (10.0529%)**. 각 구간 양 끝입니다.
- 8개 feature가 원본과 tolerance 내 동일: **19/378 (5.0265%)**. 시작점 label의 per-point 의미가 특히 모호합니다. 끝점은 incoming feature가 변할 수 있어 38개 전체를 잘못된 label로 단정하지 않습니다.
- Test full synthetic copy에서 label=0인데 feature가 변한 point: **30개**. 좌표가 변한 label=0 point는 0개입니다. lag/bearing/stop carry의 경계 효과이며, 현재 evaluation은 synthetic label=0 copy를 제외하므로 실제 정상 negative가 오염됐다고 주장하지 않습니다.
- synthetic speed>50m/s: **3개**(paired original 0개). distance>1000m & dt<=60s: **0개**.
- abs(acceleration)>Train p99=4.780469m/s²: **34개**(paired original 8개). 이는 기술적 비교이며 교통수단 공통 물리 한계가 아닙니다.

Train valid triplet 85,312개의 보간 residual median/p99는 **1.198663/21.461018m**, stationary subset 8,442개는 **0.318150/1.719014m**입니다. route 변위는 이 단기 proxy보다 대체로 크지만 residual에는 실제 움직임·회전·sampling이 섞입니다. 공통 위치 bias와 상관된 GPS 오차를 측정하지 못하므로 true GPS noise 또는 정상 경로 이탈의 근거로 사용할 수 없습니다. 현재 generator는 detour 모양 coordinate perturbation이며 도로, 사용자 습관 경로, 이동 목적에 기반한 실제 route departure를 검증하지 않습니다. teleport가 지배적이라는 증거도 없습니다.

## Feature 분포와 paired 변화

아래 표의 std는 sample ddof=1입니다. 전체 사용자 및 Validation 통계와 각 point의 Train percentile은 CSV에 보존했습니다. Test normal과 paired original을 함께 비교하므로 cohort 구성 차이와 injection 자체의 변화를 구분할 수 있습니다.

### train_normal

| feature | n | mean | median | std | iqr | p01 | p05 | p25 | p50 | p75 | p95 | p99 | min | max | standardized_median_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_diff_sec | 86067 | 4.29984 | 5 | 9.21616 | 3 | 1 | 1 | 2 | 5 | 5 | 5 | 11 | 1 | 298 | 0 |
| distance_m | 86067 | 25.5384 | 11.6712 | 56.3314 | 20.9381 | 0 | 0.420211 | 5.20031 | 11.6712 | 26.1384 | 94.0105 | 155.154 | 0 | 6430.02 | 0 |
| speed_mps | 86067 | 8.85298 | 3.62931 | 10.6081 | 13.3337 | 0 | 0.112096 | 1.07431 | 3.62931 | 14.408 | 30.8065 | 45.4471 | 0 | 49.9394 | 0 |
| acceleration_mps2 | 86067 | 0.0100207 | -0.00133947 | 1.66486 | 0.24411 | -2.65768 | -0.769905 | -0.120019 | -0.00133947 | 0.124091 | 0.774534 | 3.09539 | -259.675 | 47.2775 | 0 |
| direction_change_deg | 86067 | 17.8864 | 4.59051 | 32.9973 | 16.6026 | 0 | 0 | 0.788817 | 4.59051 | 17.3914 | 94.3882 | 166.133 | 0 | 180 | 0 |
| stop_duration_sec | 86067 | 3.26323 | 0 | 15.6417 | 0 | 0 | 0 | 0 | 0 | 0 | 18 | 65 | 0 | 553 | 0 |
| bearing_sin | 86067 | 0.0808389 | 0.0646884 | 0.682302 | 1.26313 | -0.999851 | -0.993134 | -0.497377 | 0.0646884 | 0.765756 | 0.996806 | 0.999912 | -1 | 1 | 0 |
| bearing_cos | 86067 | 0.0097711 | 0.000873369 | 0.726529 | 1.50592 | -0.999895 | -0.996538 | -0.746701 | 0.000873369 | 0.759223 | 0.998176 | 0.999916 | -1 | 1 | 0 |

### test_normal

| feature | n | mean | median | std | iqr | p01 | p05 | p25 | p50 | p75 | p95 | p99 | min | max | standardized_median_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_diff_sec | 46299 | 2.96019 | 2 | 7.24577 | 4 | 1 | 1 | 1 | 2 | 5 | 5 | 7 | 1 | 297 | 1.349 |
| distance_m | 46299 | 10.1584 | 7.51259 | 24.8502 | 9.71294 | 0 | 0.085163 | 3.34564 | 7.51259 | 13.0586 | 30.1147 | 31.2529 | 0 | 3829.72 | 0.267934 |
| speed_mps | 46299 | 7.54902 | 3.48424 | 9.19538 | 10.1752 | 0 | 0.0170326 | 0.859017 | 3.48424 | 11.0342 | 30.046 | 31.1358 | 0 | 49.9936 | 0.014677 |
| acceleration_mps2 | 46299 | 0.026469 | -2.17417e-05 | 1.95422 | 0.337083 | -2.48497 | -0.988117 | -0.167704 | -2.17417e-05 | 0.169379 | 1.06399 | 3.44342 | -261.651 | 42.1097 | 0.00728205 |
| direction_change_deg | 46299 | 13.5899 | 2.92455 | 27.7048 | 12.5004 | 0 | 0 | 0.476182 | 2.92455 | 12.9766 | 66.282 | 155.805 | 0 | 180 | 0.135363 |
| stop_duration_sec | 46299 | 43.4906 | 0 | 289.336 | 0 | 0 | 0 | 0 | 0 | 0 | 66 | 1520.1 | 0 | 3835 | 0 |
| bearing_sin | 46299 | 0.1491 | 0.167785 | 0.729705 | 1.4197 | -0.999953 | -0.997947 | -0.486103 | 0.167785 | 0.933594 | 0.999499 | 0.999993 | -1 | 1 | 0.110105 |
| bearing_cos | 46299 | -0.0791598 | -0.0395072 | 0.662611 | 1.10793 | -0.99996 | -0.998638 | -0.724647 | -0.0395072 | 0.383279 | 0.990919 | 0.999866 | -1 | 1 | 0.0361728 |

### test_route_deviation

| feature | n | mean | median | std | iqr | p01 | p05 | p25 | p50 | p75 | p95 | p99 | min | max | standardized_median_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_diff_sec | 378 | 3.37831 | 3 | 3.66752 | 2 | 1 | 1 | 2 | 3 | 4 | 5 | 18.23 | 1 | 49 | 0.899333 |
| distance_m | 378 | 23.3979 | 20.6676 | 13.7073 | 18.2424 | 1.29773 | 5.17719 | 13.016 | 20.6676 | 31.2584 | 48.3166 | 61.0397 | 0 | 73.7456 | 0.579621 |
| speed_mps | 378 | 10.7325 | 7.24339 | 9.83399 | 10.7354 | 0.333678 | 1.15275 | 3.96615 | 7.24339 | 14.7016 | 30.2072 | 47.7047 | 0 | 50.4037 | 0.365644 |
| acceleration_mps2 | 378 | 0.59869 | -0.00361513 | 4.60832 | 1.16907 | -8.57399 | -3.03235 | -0.455023 | -0.00361513 | 0.714044 | 6.2065 | 25.8777 | -24.8539 | 38.0183 | 0.0125757 |
| direction_change_deg | 378 | 17.1059 | 6.08519 | 29.5139 | 14.8775 | 0.00288261 | 0.250073 | 1.79421 | 6.08519 | 16.6717 | 81.0537 | 142.918 | 0 | 179.018 | 0.121447 |
| stop_duration_sec | 378 | 0.0793651 | 0 | 0.749215 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3.23 | 0 | 12 | 0 |
| bearing_sin | 378 | -0.0703852 | -0.21045 | 0.742076 | 1.57616 | -0.999498 | -0.988783 | -0.824667 | -0.21045 | 0.751495 | 0.985569 | 0.999793 | -0.999952 | 0.999998 | 0.293842 |
| bearing_cos | 378 | -0.136226 | -0.285167 | 0.654524 | 1.13361 | -0.997932 | -0.982163 | -0.704695 | -0.285167 | 0.428911 | 0.953752 | 0.991566 | -0.999998 | 0.997078 | 0.256234 |

### paired_test_source_original

| feature | n | mean | median | std | iqr | p01 | p05 | p25 | p50 | p75 | p95 | p99 | min | max | standardized_median_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_diff_sec | 378 | 3.37831 | 3 | 3.66752 | 2 | 1 | 1 | 2 | 3 | 4 | 5 | 18.23 | 1 | 49 | 0.899333 |
| distance_m | 378 | 9.49147 | 10.155 | 6.95714 | 7.05975 | 0 | 0.280682 | 5.21097 | 10.155 | 12.2707 | 20.2931 | 29.2219 | 0 | 76.4703 | 0.0976851 |
| speed_mps | 378 | 4.83717 | 3.14325 | 5.83243 | 4.03381 | 0 | 0.0935607 | 1.30173 | 3.14325 | 5.33554 | 19.382 | 23.047 | 0 | 47.0209 | 0.0491761 |
| acceleration_mps2 | 378 | 0.0710051 | -0.0093869 | 1.98849 | 0.380425 | -4.20584 | -1.07317 | -0.174738 | -0.0093869 | 0.205687 | 1.60112 | 5.54142 | -23.7114 | 21.5308 | 0.0444717 |
| direction_change_deg | 378 | 17.0313 | 6.9514 | 27.0279 | 18.4403 | 0 | 0 | 1.98472 | 6.9514 | 20.425 | 68.5499 | 144.164 | 0 | 172.338 | 0.191828 |
| stop_duration_sec | 378 | 1.64021 | 0 | 5.81708 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 33.69 | 0 | 42 | 0 |
| bearing_sin | 378 | -0.0475358 | 1.22465e-16 | 0.602957 | 0.958309 | -0.999966 | -0.999136 | -0.569607 | 1.22465e-16 | 0.388701 | 0.938965 | 0.997498 | -0.999986 | 0.999615 | 0.0690858 |
| bearing_cos | 378 | -0.187034 | -0.305152 | 0.775729 | 1.65343 | -1 | -0.999445 | -0.966879 | -0.305152 | 0.686551 | 0.993593 | 0.99961 | -1 | 1 | 0.274136 |

### validation_route_deviation

| feature | n | mean | median | std | iqr | p01 | p05 | p25 | p50 | p75 | p95 | p99 | min | max | standardized_median_effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_diff_sec | 217 | 4.58525 | 5 | 5.61194 | 3 | 1 | 1 | 2 | 5 | 5 | 5 | 27.84 | 1 | 70 | 0 |
| distance_m | 217 | 25.8923 | 22.8711 | 21.8385 | 19.4964 | 0.657104 | 4.63013 | 14.3875 | 22.8711 | 33.8839 | 52.533 | 61.8844 | 0 | 269.669 | 0.721586 |
| speed_mps | 217 | 8.71127 | 6.0444 | 8.22163 | 7.19775 | 0.131421 | 0.780194 | 3.14841 | 6.0444 | 10.3462 | 25.9449 | 36.4406 | 0 | 44.5756 | 0.24434 |
| acceleration_mps2 | 217 | 0.32899 | 0.0225543 | 3.6222 | 0.848395 | -7.97241 | -3.11353 | -0.306883 | 0.0225543 | 0.541513 | 4.13229 | 15.0732 | -23.7555 | 22.191 | 0.132042 |
| direction_change_deg | 217 | 22.5656 | 9.49951 | 32.507 | 19.3373 | 0.00452654 | 0.248701 | 3.50758 | 9.49951 | 22.8449 | 89.3063 | 151.048 | 0 | 168.388 | 0.398868 |
| stop_duration_sec | 217 | 0.718894 | 0 | 7.29214 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13.88 | 0 | 105 | 0 |
| bearing_sin | 217 | -0.0133246 | -0.0399216 | 0.654787 | 1.31211 | -0.999479 | -0.967326 | -0.648916 | -0.0399216 | 0.663192 | 0.965939 | 0.999066 | -0.999964 | 0.999708 | 0.111721 |
| bearing_cos | 217 | 0.0743055 | 0.208991 | 0.755088 | 1.51461 | -0.998581 | -0.97361 | -0.691699 | 0.208991 | 0.822908 | 0.99715 | 0.999758 | -0.999942 | 1 | 0.186431 |

Test route **307/378 (81.2169%)**는 8 feature 모두 Train 각 feature의 p01~p99 안에 있습니다. 이는 marginal overlap이며 joint 정상 여부를 증명하지 않습니다. 모든 feature의 cohort median effect는 1 미만입니다. time_diff는 주입 전후 동일하므로 median 차이는 sampling/cohort 차이입니다. signed acceleration median은 큰 양·음 tail을 감춥니다.

| Feature | paired median absolute delta | Train scale로 나눈 median delta | p01~p99 안 point 수 |
| --- | ---: | ---: | ---: |
| time_diff_sec | 0 | 0 | 372 |
| distance_m | 9.764698 | 0.629121 | 378 |
| speed_mps | 3.614927 | 0.365730 | 372 |
| acceleration_mps2 | 0.443459 | 2.450644 | 322 |
| direction_change_deg | 6.761402 | 0.549379 | 376 |
| stop_duration_sec | 0 | 0 | 378 |
| bearing_sin | 0.650890 | 0.695137 | 375 |
| bearing_cos | 0.512573 | 0.459160 | 376 |

각 point의 최대 paired scaled delta median은 **3.356699**, p95는 **31.542551**입니다. 따라서 실제 anomaly 전체가 기존 feature에서 거의 동일하다는 결론은 틀립니다.

## 200m 공간 정보 소실 반례

Train 첫 trajectory `000__20081023025304`(908 point)를 메모리에서 일정 longitude **0.0023474361238106528°**만큼 회전했습니다. anchor 변위는 **200m**, 전체 point 범위는 **199.927598~200.004174m**입니다. latitude 및 모든 delta longitude가 동일하므로 spherical Haversine 거리와 bearing, timestamp 기반 speed/acceleration, 방향 변화와 stop이 이론상 동일합니다.

실측 최대 absolute delta는 distance **6.50e-9m**, speed **3.05e-9m/s**, acceleration **1.52e-9m/s²**, direction **2.62e-7°**, sin/cos **2.14e-9/1.83e-9**, time/stop **0**으로 사전 tolerance를 모두 만족했습니다. 이는 동일 8 feature만으로 순수 절대 위치 이동을 식별할 수 없다는 반례입니다. 실제 generator의 국소 굽힘이 모두 동일하다는 뜻이나 특정 모델 구조가 실패 원인이라는 뜻은 아닙니다. 반례 데이터는 저장하거나 모델에 입력하지 않았습니다.

## 사용자별 route 결과

Rule은 단일 deterministic 결과, IF/AE는 5 seeds mean ± sample std입니다. score는 모델별 단위가 달라 서로 절대 비교하지 않습니다.

| user | points/segments | displacement median(m) | Rule Recall | IF Recall | AE Recall | 8 marginal 안 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 001 | 88/5 | 107.889169 | 0.318182 | 0.250000 ± 0.011364 | 0.270455 ± 0.047808 | 59 |
| 008 | 93/5 | 110.695568 | 0.118280 | 0.081720 ± 0.016307 | 0.163441 ± 0.063795 | 83 |
| 013 | 79/4 | 143.231619 | 0.215190 | 0.139241 ± 0.008951 | 0.253165 ± 0.057313 | 65 |
| 017 | 118/5 | 113.755652 | 0.144068 | 0.067797 ± 0.008475 | 0.166102 ± 0.030906 | 100 |

사용자 순서 001/008/013/017의 mean score는 Rule **17.039266/7.415497/16.777494/5.859991**, IF **0.467496/0.450375/0.469669/0.442847**, AE **0.122105/0.058364/0.131967/0.081572**입니다. 전체 route Recall은 Rule **0.193122**, IF **0.128571 ± 0.009648**, AE **0.207937 ± 0.037906**입니다. 사용자별 모든 feature 분포, score·Recall의 mean/std/min/max, detected count는 summary/seed CSV에 있습니다. 비슷한 변위 median에도 Recall이 달라 절대 변위만으로 사용자 차이를 설명할 수 없습니다.

## 저장된 score에 대한 관찰

seed 평균 score와 변위의 Spearman rho는 Rule **0.016619**, IF **−0.064437**, AE **−0.043238**입니다. 최대 paired feature delta와 score의 rho는 각각 **0.765853/0.756649/0.708095**입니다. threshold나 모델은 변경하지 않았습니다.

| Subset | n | Rule Recall | IF mean Recall | AE mean Recall |
| --- | ---: | ---: | ---: | ---: |
| 모든 marginal p01~p99 안 | 307 | 0.035831 | 0.015635 | 0.092508 |
| 하나 이상 밖 | 71 | 0.873239 | 0.616901 | 0.707042 |
| 원본과 8 feature 동일 | 19 | 0 | 0 | 0.021053 |

이는 현재 점수가 feature 변화와 연관된다는 관찰입니다. 공간 정보 부재의 전체 Recall 기여율, joint separability, 더 좋은 기존 feature 모델의 가능성을 인과적으로 판정하지는 못합니다.

## Route representation 후보와 leakage 조건

아래는 설계 검토만 했으며 production feature는 추가하지 않았습니다. 현재 Test 사용자는 unseen user이므로 자신의 정상 경로를 이미 알고 있다고 가정할 수 없습니다. synthetic source 원본은 audit 정렬용 oracle이며 추론 feature reference로 사용할 수 없습니다.

| 후보 | 필요한 reference | unseen-user 한계와 안전 조건 |
| --- | --- | --- |
| centerline 거리 | Train-only 경로 centerline 또는 사전 외부 지도 | 개인 정상 경로 아님; coverage 밖 abstention 필요 |
| nearest Train point 거리 | Train 정상 좌표 | sampling density/지역 coverage confound; Train만 index 구축 |
| 정상 trajectory와 거리 | Train trajectory library | 새로운 지역의 정상 경로도 멀어짐; Test 자체 정상/reference 사용 금지 |
| local route deviation | Train-only local road/route segments 또는 사전 지도 | baseline 후보 정의와 coverage를 Train/Validation에서 고정 |
| route density | Train-only spatial density | 희귀 정상 지역 오탐; bandwidth를 Test로 결정 금지 |
| nearest segment 거리 | Train polyline segments | 직선 연결 오류/지도 mismatch; temporal continuity 별도 검토 |
| trajectory-relative displacement | 관측 prefix/local frame | 절대 offset은 상쇄됨; 미래/전체 정상 trajectory oracle 금지 |

모든 index·reference·normalization은 Train으로 구축하고, Validation 정상으로 coverage/abstention 가능성을 검증한 후 고정해야 합니다. Validation/Test 정상 좌표를 reference에 합치거나 Test 사용자 습관 경로를 몰래 이용하면 현재 실험과 다른 정보 조건입니다. 외부 지도는 사전 가용성을 명시해야 하고 map-matching 자체가 정상 경로를 뜻하지 않습니다.

## 판정과 Stage 6.2 제안

판정은 **D: 여러 문제가 함께 관찰됨**입니다. (1) 절대 위치 offset에 대한 기존 표현의 비식별성, (2) 많은 marginal 분포 overlap과 낮은 저장 Recall, (3) zero-offset/동일-feature 시작 label 및 경계 효과가 확인됐습니다. A처럼 전체 feature가 거의 안 변한다고 할 수 없고, B의 충분한 모델 학습 가능성이나 C의 label 문제가 주원인이라는 인과적 주장은 검증되지 않았습니다. 원인별 기여 크기는 여전히 불확실합니다.

Stage 6.2는 **A: route-relative representation의 Train-only reference 및 coverage/abstention feasibility 검증**을 우선 제안합니다. 가장 직접적인 구조적 증거는 200m 반례입니다. unseen-user 정상 지역 coverage를 확인하지 않고 거리 feature를 즉시 넣는 것은 권하지 않습니다. B generator 개선은 endpoint/연속성 label 계약을 별도로 정리할 가치가 있지만 동일-feature 5.0%만으로 전체 저 Recall을 설명하지 못합니다. C stop reconstruction, D user 001 FPR은 이번 route evidence에서 우선순위가 낮습니다. 이번 작업에서는 어떤 Stage 6.2 후보도 구현하지 않았습니다.

## 산출물과 재현

Metrics: `outputs/metrics/stage6/geolife_u000_019_first5_seed42_dedup/route_audit/`.
필수 `route_deviation_samples.csv`, `route_feature_summary.csv`, `route_feature_percentiles.csv`, `route_user_summary.csv`, `synthetic_route_audit.json`, `stage6_1_verification.json` 외에 segment/score/seed metrics/seed summary, `route_diagnostic_evidence.json`, 독립 검증 `stage6_1_final_checks.json`을 보존했습니다.

Figures: `outputs/figures/stage6/geolife_u000_019_first5_seed42_dedup/route_audit/`.

1. `route_displacement_distribution.png`
2. `normal_route_feature_distributions.png`
3. `feature_effect_size_comparison.png`
4. `route_score_distributions_by_model.png`
5. `route_recall_by_user.png`
6. `displacement_vs_anomaly_score.png`

```powershell
python -m src.audit_route_representation
```

이미 생성된 output directory가 있으면 덮어쓰지 않고 실패합니다. 사전 보호 snapshot `outputs/metrics/stage54_frozen_artifacts.json`을 요구합니다. runtime report, source/input/output SHA256, 235개 보호 checksum, lineage 검증은 JSON에 기록했습니다. 산출물은 `.gitignore` 정책상 Git에 넣지 않았습니다. 문서 작성만으로 테스트를 재실행하지 않았으며 tested source hash가 유지됨을 마지막에 확인합니다.

## Git checkpoint 분리 (2026-10-04)

위의 미커밋 상태 설명은 audit 완료 시점의 기록입니다. 이후 Stage 6.1의 4개 파일을 별도 백업 및 untracked 포함 stash로 보존했습니다. Stage 5의 188개 테스트를 재실행한 뒤 PR #4를 squash merge했습니다. main commit은 `f2e7eeed3f9cb4ffef9b635ca4205dda6954fab0`이며 Stage 5 head와 tree가 동일합니다. 최신 main에서 `feature/stage-6-route-context`를 만들고 모든 파일의 텍스트를 백업과 대조해 충돌 없이 복원했습니다. Stage 6.1만 검증·checkpoint 게시하며 Stage 6 PR/main merge나 Stage 6.2 구현은 하지 않습니다. 실행 결과와 최종 SHA는 `outputs/metrics/stage61_git_checkpoint.json`에 기록합니다.
