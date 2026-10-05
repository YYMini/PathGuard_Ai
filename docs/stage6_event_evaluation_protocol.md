# Stage 6.6 — Segment / Event Evaluation Protocol Audit

기존 point Recall 12.86~20.79%와 event detection 89.47~92.63%는 같은 저장 prediction을 서로 다른 단위로 평가한 결과다. Event의 최초 반응은 대체로 빠르지만 coverage가 낮고 정상 trajectory 19개 전부에 오경보가 있다. 높은 EDR만으로 운영상 유용한 detector라고 결론낼 수 없다.

이 보고서와 event metric은 **supplementary evaluation**이다. Stage 5 Accuracy/Precision/Recall/F1/FPR/ROC-AUC/AP, dataset, 모델, scaler, label, score, threshold를 보존했다. 재학습·재추론·threshold 선택·smoothing·voting·hysteresis·최소 길이 최적화·synthetic 생성은 수행하지 않았다.

Rule은 1개 deterministic 결과이고 IF/AE는 seeds **7, 21, 42, 100, 2026** 각각을 평가했다. 모델 표는 seed별 통계의 평균이며 표준편차는 **sample std, ddof=1**이다. 특히 mean of seed medians/p90를 pooled median/p90로 해석하면 안 된다. 5개 seed는 같은 19개 event의 반복 평가이지 독립 event 95개가 아니다.

## 평가 계약

- Positive: 기존 route synthetic sample의 고정 label segment 한 개 = event 한 개. 모든 **19개** 유지; Tier 4 포함을 이유로 제거하지 않는다. 시작/끝 source index와 timestamp는 full copy/manifest로 검증한다.
- Full geometry는 label **379행**, 저장 prediction universe는 품질 평가 대상 **378행**이다. 제외된 1행은 첫 event 내부의 quality 제외 point이며 prediction을 추정하지 않는다.
- `event_point_coverage = 저장 TP / full labelled length`는 요청된 원래 label geometry 분모를 사용한다. `evaluated_point_coverage = TP / 예측 존재 label 수`도 저장한다. 전자는 **관측된 positive 비율**로 unknown 1행을 negative로 판정했다는 뜻이 아니다.
- 기존 point Recall은 `sum(TP)/sum(evaluated length)`이며 evaluated coverage를 길이 가중하면 정확히 일치한다. Full coverage를 길이 가중하면 분모 379인 별도 값이다. Event coverage의 단순 평균/median은 point Recall과 다르다.
- Detected는 **event 안에서** positive가 한 번 이상 존재하는 것. 최초 index offset / `max(full length-1,1)`으로 relative position을 구한다. Miss는 `NA + missed_event`; delay 요약은 detected-only이다. Early 10/25/50%는 사전 고정, 분모 all events, miss는 false이다.
- Run은 smoothing 없는 consecutive positive이다. Negative, source index hole, trajectory/sample/event 경계, 기존 `LONG_GAP_THRESHOLD_SEC=300` 초과 interval에서 분리한다. Unknown point나 filtered row 사이를 이어 붙이지 않는다.
- Negative: **original Test normal 46,299행**, 원본 trajectory 19개만 사용한다. Synthetic copied normal은 prediction이 저장되지 않아 이 scope에 넣지 않는다.
- Time exposure는 양 끝이 평가 대상 정상인 adjacent original index 중 `0 < timestamp delta <= 300s` interval의 합이며 저장 `time_diff_sec`와 일치 검증한다. Per-hour는 이 제한된 유효 관측 노출에 대한 raw run burden이고 실제 notification 빈도를 의미하지 않는다.
- Original normal과 별도 synthetic event copy를 artificial stream으로 연결하지 않는다. Fully scored causal combined stream이 없어 **event precision / alarm-event matching = NA**이다. Positive EDR와 negative burden을 별도로 보고한다.

## 요청된 42개 결과

### 1. Stage 6.5 테스트 재검증

`python -m unittest discover -s tests -v`: **347 PASS**, 108.186초. 관련 변경은 README와 Stage 6.5 module/test/doc 네 파일뿐이었다. 선택 stage 후 commit했다.

### 2. Stage 6.5 commit SHA

게시 commit: `07c0e6136a35704ba0f5cdac2da988109c0f36cb`. Message: `stage6: audit synthetic route label identifiability`. 동일 tree: `30c0b22bafefd6f29adb7fbe0bce584a961f20fb`. 원래 검증 로컬 commit `ac765a7`은 tag `stage65-local-verified-ac765a7`로 보존했다.

### 3. Push 결과

CLI push는 credentials 부재로 실패했다. 명시적으로 허용한 GitHub connector를 이용해 로컬 commit의 정확한 blob으로 동일 tree를 생성하고 현재 feature branch ref를 non-force 업데이트했다. PR/main merge는 없다.

### 4. Local / remote HEAD

게시 후 `HEAD == origin/feature/stage-6-route-context == 07c0e6136a35704ba0f5cdac2da988109c0f36cb`. tree diff 0을 확인한 후 local HEAD를 게시 commit에 정렬했다. Stage 6.6은 이 HEAD 위의 미커밋 작업이다.

### 5. Checkpoint checksum

Stage 6.5 보호 **459개 변경 0**을 재검증했다. 기존 snapshot을 그대로 검증한 뒤 Stage 6.5 output/figure 및 신규 source/test/doc을 추가해 Stage 6.6 보호 **495개** snapshot을 만들었다. README는 진행 상태 갱신을 허용하는 기존 예외이며 보호 대상 source/test/doc은 변경하지 않았다.

### 6. Positive route event 수

기존 full synthetic copy와 manifest의 **19 samples / 19 positive events**. Users 001/008/013/017의 event 수는 **5/5/4/5**. Detector-event rows **209=19×11**. Boundary 재선정 및 event 제외는 0이다.

### 7. Event 길이 / 시간 통계

Full length: min **11**, median **20**, mean **19.9474**, max **30** points. Duration: min **13**, median **49**, mean **64.3684**, max **217** seconds; duration NA event **0**. Length p33/p66 = **16.88/22**, groups short/medium/long **6/7/6**. Duration p33/p66 = **46.76/71.68s**, groups **6/6/7**. Geometry quantile은 detector 성능 최적화와 무관한 기술통계다.

| Model | Length group | Events | EDR | Median delay points | Median coverage |
| --- | --- | --- | --- | --- | --- |
| AE | long | 6 | 76.67% | 8.20 | 7.90% |
| AE | medium | 7 | 100.00% | 1.00 | 23.27% |
| AE | short | 6 | 96.67% | 1.00 | 23.81% |
| IF | long | 6 | 80.00% | 1.00 | 5.57% |
| IF | medium | 7 | 97.14% | 1.00 | 10.18% |
| IF | short | 6 | 100.00% | 1.00 | 13.39% |
| Rule | long | 6 | 83.33% | 1.00 | 10.87% |
| Rule | medium | 7 | 100.00% | 1.00 | 15.00% |
| Rule | short | 6 | 83.33% | 1.00 | 29.02% |

| Model | Duration group | Events | EDR | Median delay seconds | Median coverage |
| --- | --- | --- | --- | --- | --- |
| AE | long | 7 | 80.00% | 49.20 | 9.37% |
| AE | medium | 6 | 96.67% | 3.90 | 18.52% |
| AE | short | 6 | 100.00% | 1.60 | 34.57% |
| IF | long | 7 | 80.00% | 32.20 | 4.65% |
| IF | medium | 6 | 100.00% | 3.50 | 11.02% |
| IF | short | 6 | 100.00% | 1.50 | 16.99% |
| Rule | long | 7 | 85.71% | 54.00 | 9.09% |
| Rule | medium | 6 | 83.33% | 4.00 | 18.86% |
| Rule | short | 6 | 100.00% | 1.50 | 29.02% |

### 8. Rule EDR

**17/19 = 89.4737%**. Deterministic single result, seed/std **NA**.

### 9. IF EDR mean ± sample std

**92.6316% ± 2.8828 percentage points**, min/max **89.4737/94.7368%**. Seeds 7/21/42/100/2026 detected events **18/17/17/18/18**.

### 10. AE EDR mean ± sample std

**91.5789% ± 6.0009 percentage points**, min/max **84.2105/100%**. Seeds 7/21/42/100/2026 detected events **18/19/17/16/17**.

### 11. Early detection @10/25/50%

전체 event 분모, fixed inclusive cutoff. Miss 포함; model/seed별 결과는 `event_detection_by_seed.csv`.

| Model | @10% | @25% | @50% |
| --- | --- | --- | --- |
| Rule | 63.16% | 68.42% | 68.42% |
| IF | 80.00% | 80.00% | 80.00% |
| AE | 71.58% | 74.74% | 75.79% |

### 12. Detection delay

모든 run의 detected-only median은 **1 point / 3 seconds**. 그러나 늦은 탐지 tail이 있으므로 median만 보면 부족하다. Delay는 observable intervention 이전 sine-zero label start에서 측정한다. 모든 event start는 Tier 4여서 1-point delay는 최초 좌표 변화에 대한 지연과 같지 않다. Miss를 큰 지연 값으로 대체하지 않았다.

| Model | Unit | Mean | Median | p75 | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| Rule | delay_points | 5.41 | 1.00 | 4.00 | 18.60 | 21.40 |
| Rule | delay_seconds | 20.59 | 3.00 | 42.00 | 67.60 | 73.20 |
| IF | delay_points | 3.78 | 1.00 | 1.00 | 11.92 | 20.06 |
| IF | delay_seconds | 19.03 | 3.00 | 5.00 | 59.14 | 96.91 |
| AE | delay_points | 4.18 | 1.00 | 4.00 | 14.14 | 16.68 |
| AE | delay_seconds | 14.91 | 3.00 | 16.60 | 52.36 | 58.57 |

### 13. Event coverage

Full-labelled coverage의 모델별 median(5 seed는 seed median 평균) **Rule 15.00%, IF 8.61%, AE 19.72%**. Unknown 1행의 caveat를 유지하며 evaluable coverage도 함께 제공한다. Miss coverage는 0.

| Model | mean | median | p25 | p75 | p90 |
| --- | --- | --- | --- | --- | --- |
| Rule | 21.77% | 15.00% | 8.89% | 27.53% | 42.59% |
| IF | 15.04% | 8.61% | 5.94% | 14.82% | 26.70% |
| AE | 22.51% | 19.72% | 10.03% | 28.28% | 44.05% |

### 14. Longest run / fragmentation

Run longest ratio는 full event length 분모. Average run length는 TP/run count이며 zero-run event는 NA이다.

| Model | Median longest (points) | Median longest ratio | Mean runs / event | Total positive runs |
| --- | --- | --- | --- | --- |
| Rule | 2.00 | 9.09% | 2.05 | 39.00 |
| IF | 1.00 | 5.58% | 1.64 | 31.20 |
| AE | 2.00 | 10.97% | 2.26 | 43.00 |

### 15. Rule normal FP / FPR

FP **3064.00**, FPR **6.62%**.

### 16. IF normal FP / FPR

FP **3270.60**, FPR **7.06%**. 5-seed mean이며 개별 정수 FP는 아래 표에 있다. 저장 Stage 5 FP/FPR와 각 run에서 정확히 일치한다.

### 17. AE normal FP / FPR

FP **3361.40**, FPR **7.26%**. 5-seed mean이며 개별 정수 FP는 아래 표에 있다. 저장 Stage 5 FP/FPR와 각 run에서 정확히 일치한다.

### 18. False alert run count

Rule **1,075**, IF **849.2 ± 24.2631**, AE **1,612.6 ± 279.6423** (mean ± sample std). No smoothing; 동일 FP point 수여도 fragmentation으로 run 수가 달라진다.

| Model | Seed | Normal FP | FPR | Alert runs | FP / 1000 |
| --- | --- | --- | --- | --- | --- |
| AE | 7 | 3378 | 7.30% | 1738 | 72.96 |
| AE | 21 | 3935 | 8.50% | 2002 | 84.99 |
| AE | 42 | 2913 | 6.29% | 1556 | 62.92 |
| AE | 100 | 3424 | 7.40% | 1247 | 73.95 |
| AE | 2026 | 3157 | 6.82% | 1520 | 68.19 |
| IF | 7 | 3135 | 6.77% | 811 | 67.71 |
| IF | 21 | 3285 | 7.10% | 854 | 70.95 |
| IF | 42 | 3311 | 7.15% | 874 | 71.51 |
| IF | 100 | 3379 | 7.30% | 864 | 72.98 |
| IF | 2026 | 3243 | 7.00% | 843 | 70.04 |
| Rule | NA | 3064 | 6.62% | 1075 | 66.18 |

### 19. False alert runs per trajectory

| Model | Runs / trajectory |
| --- | --- |
| Rule | 56.58 |
| IF | 44.69 |
| AE | 84.87 |

### 20. FP per 1,000 original normal points

| Model | FP / 1000 |
| --- | --- |
| Rule | 66.18 |
| IF | 70.64 |
| AE | 72.60 |

### 21. Normal trajectory any-alert %

모든 **11개 detector run**, 정상 trajectory **19개 모두**에서 ≥1 positive: **100%**. Long trajectory에서 any-alert는 노출 길이 의존성이 크므로 이를 단독 metric으로 선택하지 않는다.

### 22. Time exposure / per-hour metric

각 detector run에 유효 exposure **136,658 seconds = 37.9606h**, counted intervals **46,212**. 평가 행 내부 인접 후보 46,280개 중 source hole/long-gap link **68개** 제외. Full original 46,391행에서 last-first span을 합하면 **336,621s**로 과대 노출된다. 92 quality-excluded source points의 interval을 건너 잇지 않았다. Runs/hour는 이 관측 subset의 raw alert run 빈도이며 production notification rate가 아니다.

| Model | Raw runs / valid observation hour |
| --- | --- |
| Rule | 28.32 |
| IF | 22.37 |
| AE | 42.48 |

### 23. 사용자별 EDR

Seed 평균; 사용자별 원본 event 수 5/5/4/5를 보존.

| Model | User | Events | EDR | Early @25% | Median delay points | Median delay s | Median coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AE | 001 | 5 | 84.00% | 56.00% | 2.40 | 8.10 | 15.00% |
| AE | 008 | 5 | 84.00% | 80.00% | 1.00 | 2.60 | 21.82% |
| AE | 013 | 4 | 100.00% | 95.00% | 1.10 | 2.70 | 23.19% |
| AE | 017 | 5 | 100.00% | 72.00% | 1.00 | 3.80 | 18.09% |
| IF | 001 | 5 | 92.00% | 76.00% | 1.00 | 3.80 | 9.33% |
| IF | 008 | 5 | 100.00% | 100.00% | 1.00 | 2.00 | 8.61% |
| IF | 013 | 4 | 100.00% | 100.00% | 1.00 | 2.50 | 16.56% |
| IF | 017 | 5 | 80.00% | 48.00% | 6.90 | 37.40 | 8.29% |
| Rule | 001 | 5 | 100.00% | 60.00% | 1.00 | 5.00 | 15.00% |
| Rule | 008 | 5 | 60.00% | 60.00% | 1.00 | 2.00 | 13.64% |
| Rule | 013 | 4 | 100.00% | 100.00% | 1.00 | 2.50 | 22.82% |
| Rule | 017 | 5 | 100.00% | 60.00% | 2.00 | 42.00 | 13.04% |

### 24. 사용자별 delay

23번 표의 각 user/seed detected-only median을 집계했다. IF user 017은 median delay 평균 **6.9 points / 37.4s**로 global median 1/3보다 늦다. Rule user 017은 **2 points / 42s**. AE user 001 median 평균 **2.4 points / 8.1s**. Median-of-user와 global detected-event median은 서로 다른 집계다.

### 25. 사용자별 false alert burden

| Model | User | FP / 1000 | Raw alert runs | Any-alert trajectory % |
| --- | --- | --- | --- | --- |
| AE | 001 | 101.17 | 413.40 | 100.00 |
| AE | 008 | 84.97 | 492.80 | 100.00 |
| AE | 013 | 37.50 | 552.60 | 100.00 |
| AE | 017 | 101.72 | 153.80 | 100.00 |
| IF | 001 | 143.36 | 322.20 | 100.00 |
| IF | 008 | 54.68 | 266.80 | 100.00 |
| IF | 013 | 14.14 | 187.40 | 100.00 |
| IF | 017 | 47.28 | 72.80 | 100.00 |
| Rule | 001 | 111.39 | 310.00 | 100.00 |
| Rule | 008 | 77.31 | 391.00 | 100.00 |
| Rule | 013 | 19.70 | 269.00 | 100.00 |
| Rule | 017 | 67.01 | 105.00 | 100.00 |

User-macro EDR는 **Rule 90.00%, IF 93.00%, AE 92.00%**. Macro early@25%는 **70.00/81.00/75.75%**. Macro FP/1000은 **68.8503/64.8650/81.3413**. User 013의 정상 point 수가 커서 micro와 macro burden이 다르다; `macro_user_event_metrics.csv`를 함께 보고한다.

### 26. IF seed stability

| Metric | Mean | Sample std (ddof=1) | Min | Max |
| --- | --- | --- | --- | --- |
| event_detection_rate | 92.63% | 2.88% | 89.47% | 94.74% |
| early_detection_25_rate | 80.00% | 4.40% | 73.68% | 84.21% |
| delay_points_median | 1.00 | 0.00 | 1.00 | 1.00 |
| delay_seconds_median | 3.00 | 0.00 | 3.00 | 3.00 |
| event_point_coverage_median | 8.61% | 1.08% | 6.67% | 9.09% |
| false_alert_run_count | 849.20 | 24.26 | 811.00 | 874.00 |
| fp_per_1000_normal_points | 70.64 | 1.95 | 67.71 | 72.98 |

비율 metric의 std 표기는 percentage points. 통계는 매 seed 결과에서 계산했고 Test cutoff 선택은 없다.

### 27. AE seed stability

| Metric | Mean | Sample std (ddof=1) | Min | Max |
| --- | --- | --- | --- | --- |
| event_detection_rate | 91.58% | 6.00% | 84.21% | 100.00% |
| early_detection_25_rate | 74.74% | 11.41% | 57.89% | 89.47% |
| delay_points_median | 1.00 | 0.00 | 1.00 | 1.00 |
| delay_seconds_median | 3.00 | 0.00 | 3.00 | 3.00 |
| event_point_coverage_median | 19.72% | 5.21% | 13.04% | 26.67% |
| false_alert_run_count | 1612.60 | 279.64 | 1247.00 | 2002.00 |
| fp_per_1000_normal_points | 72.60 | 8.20 | 62.92 | 84.99 |

비율 metric의 std 표기는 percentage points. 통계는 매 seed 결과에서 계산했고 Test cutoff 선택은 없다.

### 28. Point Recall vs EDR

| Model | Official route point Recall | Supplementary EDR |
| --- | --- | --- |
| Rule | 19.31% | 89.47% |
| IF | 12.86% | 92.63% |
| AE | 20.79% | 91.58% |

Point Recall은 모든 평가 label 중 탐지 비율이고 EDR는 event 내 단 한 번의 positive로 충분하다. 따라서 어느 단위가 정답인지 단일 수치로 결정하지 않는다.

### 29. EDR vs early detection

EDR와 early@25% 차이는 Rule **21.05pp**, IF **12.63pp**, AE **16.84pp**. Detected-only relative position >50%인 late events는 Rule **4/17**, IF **12/88**, AE **15/87** (IF/AE pooled event-seed descriptive counts). 전체 탐지가 후반에 집중한 것은 아니지만 늦은 tail이 존재한다.

### 30. EDR vs event coverage

EDR가 90% 가까워도 median coverage **8.61~19.72%**, longest positive run median **1~2 points**이다. Event 발생 감지는 비교적 성공하지만 anomaly 전체를 지속 추적했다는 근거는 약하다. Coverage/run 정보를 EDR 옆에 유지해야 한다.

### 31. EDR vs false alerts

정상 FP **66.18~72.60/1000**, raw runs **44.69~84.87/trajectory**, any-alert **100%**. 허용 burden budget을 사후 Test에서 만들지 않는다. 이 관측은 positive-only EDR를 운영상 usefulness로 곧바로 해석할 수 없음을 보여준다. EDR 기반 detector 단순 순위 결론은 내리지 않는다.

### 32. Tier 4 boundary 포함 event 비교

**contains Tier 4 = 19**, **without Tier 4 = 0**. Fully observable event subgroup의 EDR/delay/coverage는 **NA + no_events_without_tier4_boundary**. Tier 4의 event effect는 이 데이터에서 비교 추정할 수 없다. Evaluated tiers **340/19/0/19**, observable **359/378**. Full geometry의 context 미감사 quality row **1**은 tier로 추정하지 않았다.

### 33. 신규 테스트

신규 **43 PASS**, focused 1.629초. Sample/manifest boundary, any-positive/miss, first index/relative/delay, inclusive fixed cutoffs, full/evaluable coverage, run/fragmentation/holes, tiers/exact join, seeds/Rule, trajectory/user aggregation, FP1000, frozen threshold/no mutation, NaN, exposure/gap, checksum, overwrite refusal을 검증했다. AST로 training/inference/threshold 선택/rolling 호출 부재도 검사한다.

### 34. 전체 테스트

`python -m unittest discover -s tests -v`: **390 PASS**, 96.571초. 기존 Stage 1~6.5 **347개**를 모두 유지했다.

### 35. Protected checksum / 독립 검증

작업 전 snapshot **495개**, 작업 후 **changed 0**. 별도 verifier는 Stage 6.6 evaluator를 import하지 않고 raw full Test / manifest / source normal / 저장 prediction에서 **209 event**, **509,289 normal prediction**, **209 trajectory result**, **13,384 raw alert runs**, **156 seed aggregate**를 재계산했다. 모든 output/figure/source checksum도 확인했다.

### 36. Official Stage 5 metric unchanged

Stage 5 dataset/models/scalers/predictions/metric 파일은 snapshot hash 그대로이다. 11개 run에서 official route point Recall과 original-normal FP/FPR를 직접 비교했고 Stage 6.5 event detection/delay와 일치했다. 기존 Accuracy/Precision/Recall/F1/FPR/ROC-AUC/AP를 덮어쓰지 않았다.

### 37. Runtime

실제 Stage 6.6 audit **16.0374초**. Data/prediction 읽기, metric 및 8개 figure 생성 포함. 모델 training/inference **없음**. Test 및 independent verifier runtime은 audit runtime과 분리한다.

### 38. Case W~AC 판정

**Y + Z + AC**, **X는 일부 user/event tail에서만 지지**. 이는 사전 numeric 운영 budget 없이 결과를 기술한 판단이며 최적 detector/threshold 선택 기준이 아니다.

| Case | 판정 | 근거 |
| --- | --- | --- |
| W | 확인 못 함 | EDR/중앙 지연은 양호하지만 낮은 운영 부담을 입증 못 함 |
| X | 국소적으로 관측 | late detected tails, user 017 지연; 전체 후반 집중 아님 |
| Y | 지지 | 높은 EDR와 동시에 정상 any-alert100%, 다수 raw runs |
| Z | 지지 | coverage median8.61~19.72%, longest run1~2points |
| AA | 확인 못 함 | early event signal은 있지만 낮은 false burden 입증 못 함 |
| AB | 전체 결과는 지지 안 함 | 높은 전체 EDR/early signal; 일부 긴 event/user 취약 |
| AC | 지지 | user001/013 burden 차이, AE EDR/early seed 변동 |

### 39. Point-only 평가의 한계

Point Recall은 sparse early event 반응과 miss/late/fragmented 반응을 섞는다. EDR 역시 한 번의 spike로 성공 처리해 지속성·delay·정상 부담을 숨긴다. 장기 logging gap이 큰 time denominator나 user point imbalance를 방치하면 alert burden도 왜곡된다.

### 40. Event supplementary metric 채택 가치

추가 가치가 있다. 같은 prediction의 event 발생 감지, 조기성, coverage/run 지속성, 정상 부담을 분리해 사용자 목적에 연결한다. 공식 point metric과 함께 보고하며 high EDR를 point metric 개선이나 production event precision으로 바꾸어 부르지 않는다.

### 41. 추천 dual protocol

후보 **A** point+EDR는 발생 여부를 보여주지만 delay/burden/coverage를 생략한다. **B** point+EDR+delay+false burden을 기본으로, **C**의 coverage+fixed early+per-exposure를 함께 확장하는 것이 적합하다.

발생 여부: EDR와 miss counts. 빠른 탐지: fixed early10/25/50 및 detected-only delay distribution+miss count. 지속성: coverage/longest run/fragmentation. 정상 부담: FP/FPR+run/trajectory+FP1000, 유효 exposure가 있을 때만 run/hour. User imbalance: per-user 및 equal-user macro. Stability: seed별 값과 sample std/min/max. 이 권고는 Stage 5 공식 protocol을 소급 변경하지 않는다.

### 42. Stage 6.7 최우선 후보

**A: event-aware protocol을 prospective하게 명세·확정**하는 것이 우선이다. 다음으로 부담 개선 필요 시 **B: Validation-only alert policy 연구**를 고려한다. 알림 단위/운영 budget/continuous-stream matching/exposure/quality-hole 처리 및 seed/user 보고를 먼저 고정하고 Test에서 tuning하지 않는다. C limited route/context, D sequence/window, E cold-start/abstention, F synthetic label refinement는 대안으로 남긴다. 이번 작업에서 Stage 6.7은 구현하지 않았다.

## 결과 파일과 재현

Code: [audit_event_evaluation_protocol.py](../src/audit_event_evaluation_protocol.py), tests: [test_event_evaluation_protocol.py](../tests/test_event_evaluation_protocol.py).

```powershell
python -m src.audit_event_evaluation_protocol
python -m unittest discover -s tests -v
```

Audit는 기존 output 디렉터리가 있으면 덮어쓰기를 거부한다. 현재 실행 결과는 이미 존재하므로 같은 root 재실행은 의도적으로 실패한다. Fresh workspace/output root에는 보호 snapshot과 frozen Stage 5·6.5 artifact가 있어야 한다.

Metrics: `outputs/metrics/stage6/event_evaluation/`의 **17 CSV + event_protocol_summary.json**. 최소 요청 13 CSV 외에 event catalog, duration, seed stability, user macro를 추가했다. Independent verifier와 checkpoint/checksum/test log는 ignored `outputs/metrics/`에 별도 보관했다.

Figures: `outputs/figures/stage6/event_evaluation/`의 요청 **8 PNG**. EDR/bar error bars는 seed sample std이며 Rule는 단일 값이다. Delay/coverage 분포의 IF/AE는 동일 event의 seed별 관측을 pooled해 보여준 descriptive figure이다.

## 한계 및 최종 Git 상태

19 synthetic events / 4 unseen users로 결과의 범위가 제한된다. Seeds는 사건의 독립 표본이 아니다. 품질 제외 label 1행의 detector decision을 알 수 없어 full coverage는 observed-positive fraction으로 제한된다. Synthetic copies의 unlabelled rows에는 저장 prediction이 없어 fully scored causal stream이 없고 event precision은 NA이다. Short raw alert runs가 사용자에게 실제로 전달될 notification 수라고 해석할 수 없다. 지속 tracking metric과 run/hour는 정한 품질/노출 범위에 종속된다. Tier 4 없는 event가 없어 Tier 4 효과의 비교/인과 추론은 불가능하다.

Branch `feature/stage-6-route-context`; Stage 6.5만 commit/push. **Stage 6.6 미커밋/미푸시**, Stage 6 PR 없음, main merge 없음, Stage 6.7 구현 없음. Protected 변경 0; README와 Stage 6.6 신규 module/test/doc만 tracked working-tree 변경 대상이다.
