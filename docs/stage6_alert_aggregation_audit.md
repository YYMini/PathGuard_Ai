# Stage 6.7 — Validation-only Alert Aggregation & Temporal Policy Audit

Validation에서만 선택한 **G0_C60**을 세 detector family 전체에 lock했다. Test 오경보 notification은 family 평균 수의 비율 기준 **Rule 49.02%, IF 39.52%, AE 57.27% 감소**했다. 그러나 Notification EDR과 Early25도 감소했고 정상 trajectory의 any-notification은 여전히 100%였다. 선택된 gate가 RAW여서 detector의 point/gate 지표와 false gate run은 변하지 않았다. Cooldown만으로 안정적인 운영상 해법을 입증하지 못했다.

**연구 제한:** policy parameter 선택에는 Test를 사용하지 않았지만 candidate family는 Stage 6.6 Test의 높은 alert burden을 본 뒤 설계했다. 이 Test는 **policy parameter selection에 쓰지 않은 held-out application**이다. 완전한 untouched final benchmark가 아니다. 향후 새 unseen-user confirmatory cohort가 필요하다.

## 실행 계약과 재현

Candidate config를 Validation prediction/metric 읽기 전에 동결했다. Gate **G0 RAW, G1 consecutive2, G2 consecutive3, G3 consecutive5, G4 vote2/3, G5 vote3/5, G6 vote5/10**와 cooldown **0/60/300 seconds**의 **21개 policy**만 사용했다. Candidate 추가/변경, seed/user별 다른 policy 선택, Test policy search/oracle-best/reselection은 수행하지 않았다.

1. Gate는 현재를 포함한 trailing full window만 사용한다. Warm-up 동안 false. Source trajectory/sample 시작, missing source index, 기존 >300s long gap에서 gate history/run을 reset한다.
2. Notification candidate는 새 gate positive run의 첫 point이다. 마지막 emitted notification에서 cooldown 이상 경과했을 때만 emit한다. Suppressed candidate는 clock을 갱신하지 않으며 같은 positive run 중간에 deferred notification을 새로 만들지 않는다.
3. Cooldown은 trajectory/sample마다 reset하고 quality hole에서는 직전 emitted 시간만 유지한다. Gate와 raw detector는 cooldown으로 변경하지 않는다.
4. Event label onset에서 gate/cooldown을 reset하지 않는다. 기존 full synthetic copy의 pre-event prefix와 event end까지 replay한다. Prefix의 GPS/time와 **8개 feature를 원본에 exact join/equality 검증**한 후 동일 입력의 기존 normal prediction을 재사용한다. 변경된 copied negatives나 unscored post-event point의 decision을 만들지 않는다.
5. Primary EDR는 **event 안의 실제 notification** 유무이다. Gate EDR/coverage와 notification occurrence coverage를 따로 저장한다. Full label과 evaluation row 수가 다르면 두 분모를 기록한다. 품질 제외 prediction은 unknown이며 impute/bridge하지 않는다.
6. Early10/25/50은 고정 relative position cutoff이며 분모는 전체 event. Miss는 NA delay; delay distribution은 detected-only. Full label 시작점의 Tier4에 대한 의미는 Stage6.5/6.6 제한을 계승한다.
7. Normal burden은 original evaluation normals만 사용한다. 정상/별도 copy를 하나의 artificial production stream으로 합치지 않는다. Notification event precision은 계산하지 않는다. Per-hour는 양 끝 quality-valid인 adjacent index 중 0<delta<=300s exposure의 합을 사용한다.
8. Selection은 family 평균 Notification EDR≥raw의90%, Early25≥raw의80%; IF/AE 최소4/5 seeds에서 EDR≥각 raw의85%를 요구한다. Eligible에서 FP notification1000, notification/trajectory, median delay(NA last), EDR descending, 고정 complexity 순서의 lexicographic 최소를 선택한다. Weighted utility가 없다.

Complexity tie-break: **G0 < G1 < G4 < G2 < G5 < G3 < G6**; 동일 gate는 **C0 < C60 < C300**. Rule는 deterministic single result/std NA. IF/AE는 seeds **7,21,42,100,2026**, mean/sample std(ddof=1)/min/max를 보고한다. Mean of seed medians/p90와 pooled event median/p90는 서로 다른 집계이다.

```powershell
python -m src.audit_alert_aggregation --phase validation-select
python -m src.audit_alert_aggregation --phase test-evaluate --locked-policy outputs/metrics/stage6/alert_aggregation/locked_alert_policies.json
python -m unittest discover -s tests -v
```

이미 실행된 selection/lock/Test output은 overwrite를 거부한다. Test phase는 canonical lock 경로와 hash를 요구하며 candidate grid를 받지 않는다. Prediction artifact 11개가 두 split 모두 존재해 이번 실행의 frozen inference/training은 0이다.

## 요청된 57개 완료 항목

### 1. Stage 6.6 390 tests 재검증

`python -m unittest discover -s tests -v`: **390 PASS**, **114.450초**. 관련 네 파일만 selective staging했으며 `git add .`는 쓰지 않았다.

### 2. Stage 6.6 commit SHA

게시 SHA `cd5bbf07e603357aefa36d507a3d8c57dbfdb751`. Message: `stage6: audit event evaluation protocol`. Tree `0df318dac25d94689cec43d428f95811d99ad399`. 원래 로컬 검증 commit `6a3ee3f`는 tag `stage66-local-verified-6a3ee3f`로 보존했다.

### 3. Push 결과

CLI는 credentials 부재로 실패했다. 허용된 GitHub connector로 commit blob과 동일 tree를 만들고 current Stage6 branch ref를 non-force 게시했다. PR/main merge 없음.

### 4. Local / remote HEAD

`HEAD == origin/feature/stage-6-route-context == cd5bbf07e603357aefa36d507a3d8c57dbfdb751`. 동일 tree diff0 확인 후 로컬 HEAD를 게시 commit에 맞췄다. Stage6.7은 미커밋 작업이다.

### 5. Checkpoint protected checksum

Stage6.6 기준 **495개 변경0**. Stage6.6 산출물17CSV+1JSON,8figures,신규source/test/doc3개를 추가해 Stage6.7 snapshot은 **524개(495+29)**이다. 숫자를 강제로 맞추지 않았으며 README만 기존 진행상태 갱신 예외이다.

### 6. Validation normal rows

기존 frozen evaluation original normal **19,448행**. Split/quality rule과 user split을 유지했다.

### 7. Validation route event count

Manifest/full-copy boundary에서 실제 발견 **12개**. 코드에 event 수를 하드코딩하지 않았다. 모든 기존 segment 유지; `validation_event_catalog.csv`에 full/evaluated geometry를 저장했다.

### 8. Validation prediction artifact 존재 여부

Rule1, IF5, AE5의 기존 `validation_predictions.csv` **11개 모두 존재**. CSV와 frozen 평가 row identity/features, stored score>stored threshold decision을 검증했다.

### 9. Frozen inference 수행 여부

**없음**. Validation/Test 모두 기존 prediction 재사용. Model weights/scaler load, fit/partial_fit/retrain/calibration/percentile/threshold recomputation 없음.

### 10. Artifact checksum

Validation prediction/config **22개** hash를 `validation_input_provenance.json` 및 lock에 기록했다. Test prediction/config22개는 `test_input_provenance.json`. 기존 models/scalers/weights/datasets/predictions/official metrics 전체는 protected524 snapshot으로 검증했다. Prefix equality 검증은 데이터 대응/입력 확인이며 새 inference가 아니다.

### 11. Candidate gate 7개

G0/G1/G2/G3/G4/G5/G6의 정의는 위 실행 계약과 `configs/stage67_alert_policy_candidates.json`에 고정했다. Voting은 현재 raw가0이어도 trailing vote 조건을 충족하면 positive일 수 있다. 미래/centered window 없음.

### 12. Cooldown 3개

**0/60/300s**. Inclusive elapsed≥cooldown. Suppressed candidate는 last emitted time을 바꾸지 않는다.

### 13. 총 candidate 21개

Family당 **21개**, family mean grid **63행**, seed grid **231행=21×11**. Test는 RAW/LOCKED **22개 detector-policy 결과**만 계산했다.

### 14. Candidate manifest SHA

Candidate config SHA256 **`c54ba694eff56a008c95c7c8fefc6391217b674d6a56af62350f629d7f4c62c9`**. Manifest는 `2026-10-05T09:24:33.685022+00:00`에 생성했으며 initial freeze receipt와 lock/Test 후 hash가 같다. `candidate_policy_manifest.json`의 경로·hash·count가 동결 근거이다.

### 15. Rule raw Validation baseline

| Raw Validation metric | Value | Sample std |
| --- | --- | --- |
| Notification EDR | 66.67% | NA |
| Early25 | 58.33% | NA |
| False notifications | 650.00 | NA |
| False notifications / 1000 | 33.42 | NA |
| Median delay points | 1.00 | NA |
| Median gate coverage | 8.57% | NA |
| Normal any-notification % | 91.67 | NA |

IF/AE std는 ratio metric에서 percentage points, Rule std는NA. Early10/50, time delay, notification coverage, gate runs는 전체 grid에 저장했다.

### 16. IF raw Validation baseline

| Raw Validation metric | Value | Sample std |
| --- | --- | --- |
| Notification EDR | 66.67% | 0.00% |
| Early25 | 66.67% | 0.00% |
| False notifications | 613.80 | 24.39 |
| False notifications / 1000 | 31.56 | 1.25 |
| Median delay points | 1.00 | 0.00 |
| Median gate coverage | 6.77% | 0.39% |
| Normal any-notification % | 100.00 | 0.00 |

IF/AE std는 ratio metric에서 percentage points, Rule std는NA. Early10/50, time delay, notification coverage, gate runs는 전체 grid에 저장했다.

### 17. AE raw Validation baseline

| Raw Validation metric | Value | Sample std |
| --- | --- | --- |
| Notification EDR | 85.00% | 14.91% |
| Early25 | 75.00% | 11.79% |
| False notifications | 642.80 | 50.63 |
| False notifications / 1000 | 33.05 | 2.60 |
| Median delay points | 1.00 | 0.00 |
| Median gate coverage | 15.82% | 8.75% |
| Normal any-notification % | 98.33 | 3.73 |

IF/AE std는 ratio metric에서 percentage points, Rule std는NA. Early10/50, time delay, notification coverage, gate runs는 전체 grid에 저장했다.

### 18. Rule selected policy

**G0_C60**. Family/user/seed 공통 RAW gate +60s notification cooldown. Gate persistence/voting은 선택되지 않았다.

### 19. IF selected policy

**G0_C60**. Family/user/seed 공통 RAW gate +60s notification cooldown. Gate persistence/voting은 선택되지 않았다.

### 20. AE selected policy

**G0_C60**. Family/user/seed 공통 RAW gate +60s notification cooldown. Gate persistence/voting은 선택되지 않았다.

### 21. Eligibility 근거

| Family | EDR retention | Early25 retention | Seed guard passes | Eligible candidates |
| --- | --- | --- | --- | --- |
| Rule | 100.00% | 100.00% | 1 | 2 |
| IF | 97.50% | 85.00% | 5 | 2 |
| AE | 90.20% | 84.44% | 5 | 2 |

Eligible는 각 family에서 **G0_C0/G0_C60 두 개**뿐이었다. 고정 objective에서 G0_C60의 false notification1000이 낮아 선택했다. C300과 persistence/voting 후보의 탈락 이유는 grid에 기록했다. Rule에는 4/5 guard를 적용하지 않는다.

### 22. Validation false notification 감소

| Family | Raw count → locked mean count | Reduction (ratio of family means) |
| --- | --- | --- |
| Rule | 650.00 → 394.00 | 39.38% |
| IF | 613.80 → 399.40 | 34.93% |
| AE | 642.80 → 343.00 | 46.64% |

### 23. Validation EDR retention

Rule **100%**, IF **97.50%**, AE **90.20%**. 고정90% relative guard를 모두 충족했다.

### 24. Validation Early25 retention

Rule **100%**, IF **85.00%**, AE **84.44%**. 고정80% relative guard를 모두 충족했다. Test에서 재선택하지 않았다.

### 25. Locked policy SHA

**`082919ec25ae9d4c6f1cb0170142511b64aaf3d3ef14487be32c7b6bce83cebb`**. Lock에는 family별 concrete policy, Validation-only selection metric,22개 prediction/config hash,candidate manifest/config hash,code hash,timestamp,Stage6.6 Git SHA가 있다. Lock hash는 별도 verification receipt와 대조했고 Test 종료 후 동일함을 확인했다.

### 26. Stage6.6 Test raw metric 재현

**11개 detector run /209 event 모두 일치**. Raw point Recall, gate EDR/Early25/coverage/delay, normal FP/FPR/false gate run/exposure를 strict tolerance로 비교했다. 이 데이터에서는 RAW notification EDR/Early25/delay도 기존 Stage6.6과 일치한다. 일반적으로 pre-event ongoing positive run이 있으면 notification EDR와 gate EDR가 달라질 수 있어 구현/테스트는 두 단위를 구분한다.

### 27. Rule Test RAW → LOCKED EDR

Primary Notification EDR **89.47% → 73.68%**. Gate EDR **89.47% → 89.47%**. Family-wide fixed G0_C60; detector threshold은 동일하다.

### 28. IF Test RAW → LOCKED EDR

Primary Notification EDR **92.63% → 82.11%**. Gate EDR **92.63% → 92.63%**. Family-wide fixed G0_C60; detector threshold은 동일하다.

### 29. AE Test RAW → LOCKED EDR

Primary Notification EDR **91.58% → 75.79%**. Gate EDR **91.58% → 91.58%**. Family-wide fixed G0_C60; detector threshold은 동일하다.

### 30. Early25 변화

| Model | Notification Early25 |
| --- | --- |
| Rule | 68.42% → 42.11% |
| IF | 80.00% → 65.26% |
| AE | 74.74% → 52.63% |

### 31. Median / p90 delay 변화

Detected-only seed 통계의 평균. Miss가 늘어 비교 집단이 바뀌므로 matched detected events의 paired delay도 따로 제공한다.

| Model | points median | points p90 | seconds median | seconds p90 |
| --- | --- | --- | --- | --- |
| Rule | 1.00 → 1.00 | 18.60 → 20.70 | 3.00 → 4.50 | 67.60 → 81.20 |
| IF | 1.00 → 1.00 | 11.92 → 17.72 | 3.00 → 3.30 | 59.14 → 76.22 |
| AE | 1.00 → 1.10 | 14.14 → 18.28 | 3.00 → 3.60 | 52.36 → 62.98 |

공통 detected event의 평균 paired 증가: Rule **1.50points/12.71s**, IF **0.78points/5.41s**, AE **1.26points/4.25s** (IF/AE event-seed descriptive pooling). `test_event_delay_pair_comparison.csv`에서 NA miss와 lost notification event를 유지했다.

### 32. Event coverage 변화

| Model | Gate coverage median | Notification occurrence coverage median |
| --- | --- | --- |
| Rule | 15.00% → 15.00% | 9.09% → 4.55% |
| IF | 8.61% → 8.61% | 8.37% → 4.91% |
| AE | 19.72% → 19.72% | 11.75% → 5.00% |

G0 gate는 raw point와 같아 coverage가 변하지 않는다. Notification coverage는 발생 횟수/full labelled points 비율이며 지속 tracking coverage가 아니다. Cooldown으로 줄어든 notification coverage를 detector 성능 악화라고 해석하지 않는다. Notification occurrence와 gate 지속성은 별도로 보고한다. Full/evaluable 분모도 보존했다.

### 33. False notifications /1000 RAW → LOCKED

| Model | Notifications /1000 normal | Notification count |
| --- | --- | --- |
| Rule | 23.22 → 11.84 | 1075.00 → 548.00 |
| IF | 18.34 → 11.09 | 849.20 → 513.60 |
| AE | 34.83 → 14.88 | 1612.60 → 689.00 |

### 34. False gate run RAW → LOCKED

| Model | False gate runs |
| --- | --- |
| Rule | 1075.00 → 1075.00 |
| IF | 849.20 → 849.20 |
| AE | 1612.60 → 1612.60 |

Raw FP points와 aggregated positive points도 G0 선택으로 변하지 않는다. Gate run reduction은 **0%**. 실제 줄어든 것은 emitted notifications이며 raw point FP/1000 **66.18/70.64/72.60**을 알림 rate와 혼동하지 않는다.

### 35. Normal any-notification trajectory %

Test 정상19개 trajectory 모두 RAW/LOCKED에서 notification≥1: **100%**. Locked false notification/trajectory **Rule28.84, IF27.03, AE36.26**. 유효 exposure **136,658s=37.9606h**에서 locked notifications/hour **14.44/13.53/18.15**. Raw는28.32/22.37/42.48 per valid hour. Per-hour는 제한된 관측 노출의 빈도이다.

### 36. False notification reduction %

| Family | Ratio of mean counts (%) | Mean of per-seed ratios (%) |
| --- | --- | --- |
| Rule | 49.02 | 49.02 |
| IF | 39.52 | 39.51 |
| AE | 57.27 | 56.45 |

Primary family reduction은 요청 식 `1 - mean(locked count)/mean(raw count)`이다. Stability 표는 seed마다 같은 식을 계산한 뒤 평균/std를 구하므로 특히 AE에서 수치가 다르다. 어떤 aggregation을 썼는지 명시했고 뒤섞지 않았다.

### 37. Gate 때문에 놓친 event 수

선택된 policy의 추가 gate-induced miss **0** (11개 detector run 모두). Persistence/voting 후보는 Validation에서 eligibility를 만족하지 못했지만 Test에서는 평가하지 않았다. Warm-up/consecutive/vote/quality-hole suppression reason은 unit tests와 Validation event grid에서 검증했다.

### 38. Cooldown 때문에 놓친 event 수

추가 miss: **Rule3**, IF seeds별 **3/2/2/2/1** (mean2,총10 event-seed), AE **5/3/2/2/3** (mean3,총15 event-seed). 모든 추가 miss의 suppressing notification은 event 이전 정상 prefix에서 발생했다. 서로 다른 seed의 같은 event를 독립 사건으로 세지 않는다.

### 39. Delay 증가 원인

Cooldown-induced delayed events: **Rule2**, IF 총4 event-seed(mean0.8), AE 총6(mean1.2). 추가 miss+delay flags **Rule5/IF14/AE21**은 pre-event false notification suppression과 일치한다. Locked gate가 RAW여서 persistence delay가 아니라 이미 활성화된 cooldown 영향이다. 이 데이터의 carried gate-run without event notification은0이다.

### 40. 사용자별 EDR 변화

| Family | User | RAW → LOCKED EDR | RAW → LOCKED Early25 |
| --- | --- | --- | --- |
| Rule | 001 | 100.00% → 100.00% | 60.00% → 60.00% |
| Rule | 008 | 60.00% → 40.00% | 60.00% → 20.00% |
| Rule | 013 | 100.00% → 50.00% | 100.00% → 50.00% |
| Rule | 017 | 100.00% → 100.00% | 60.00% → 40.00% |
| IF | 001 | 92.00% → 92.00% | 76.00% → 76.00% |
| IF | 008 | 100.00% → 80.00% | 100.00% → 72.00% |
| IF | 013 | 100.00% → 75.00% | 100.00% → 75.00% |
| IF | 017 | 80.00% → 80.00% | 48.00% → 40.00% |
| AE | 001 | 84.00% → 80.00% | 56.00% → 48.00% |
| AE | 008 | 84.00% → 68.00% | 80.00% → 48.00% |
| AE | 013 | 100.00% → 50.00% | 95.00% → 45.00% |
| AE | 017 | 100.00% → 100.00% | 72.00% → 68.00% |

### 41. 사용자별 false alert 감소

| Family | User | Notifications/1000 RAW → LOCKED | Mean count RAW → LOCKED | Ratio-of-mean reduction |
| --- | --- | --- | --- | --- |
| Rule | 001 | 18.20 → 11.16 | 310.00 → 190.00 | 38.71% |
| Rule | 008 | 46.08 → 20.15 | 391.00 → 171.00 | 56.27% |
| Rule | 013 | 14.44 → 6.49 | 269.00 → 121.00 | 55.02% |
| Rule | 017 | 48.86 → 30.71 | 105.00 → 66.00 | 37.14% |
| IF | 001 | 18.92 → 12.11 | 322.20 → 206.20 | 36.00% |
| IF | 008 | 31.44 → 16.15 | 266.80 → 137.00 | 48.65% |
| IF | 013 | 10.06 → 6.06 | 187.40 → 113.00 | 39.70% |
| IF | 017 | 33.88 → 26.71 | 72.80 → 57.40 | 21.15% |
| AE | 001 | 24.27 → 13.45 | 413.40 → 229.00 | 44.61% |
| AE | 008 | 58.08 → 21.19 | 492.80 → 179.80 | 63.51% |
| AE | 013 | 29.66 → 10.42 | 552.60 → 194.20 | 64.86% |
| AE | 017 | 71.57 → 40.02 | 153.80 → 86.00 | 44.08% |

### 42. 사용자별 편차

User013 Notification EDR는 **Rule1→0.5, IF1→0.75, AE1→0.5**지만 user001은 **Rule1→1, IF.92→.92, AE.84→.80**이다. User017 locked notification/1000은 **30.71/26.71/40.02**로 global 평균보다 높다. 동일 policy를 모두에게 적용한 차이이며 user별 재선택을 하지 않았다. User/seed 수가 작고 event 길이/노출 분포 차이가 있어 인과 기여율은 주장하지 않는다.

### 43. IF locked seed stability

| Locked metric | Mean | Sample std ddof=1 | Min | Max |
| --- | --- | --- | --- | --- |
| gate_coverage_median | 8.61% | 1.08% | 6.67% | 9.09% |
| notification_event_detection_rate | 82.11% | 4.71% | 78.95% | 89.47% |
| notification_early25_rate | 65.26% | 4.71% | 57.89% | 68.42% |
| notification_delay_points_median | 1.00 | 0.00 | 1.00 | 1.00 |
| notification_delay_seconds_median | 3.30 | 0.45 | 3.00 | 4.00 |
| notification_coverage_median | 4.91% | 0.20% | 4.55% | 5.00% |
| false_notifications_per_1000_normal_points | 11.09 | 0.23 | 10.82 | 11.34 |

| Per-seed ratio metric (%) | Mean | Sample std | Min | Max |
| --- | --- | --- | --- | --- |
| false_notification_reduction_pct | 39.51 | 0.75 | 38.22 | 40.09 |
| edr_retention_pct | 88.63 | 3.94 | 83.33 | 94.44 |
| early25_retention_pct | 81.63 | 5.10 | 75.00 | 86.67 |

### 44. AE locked seed stability

| Locked metric | Mean | Sample std ddof=1 | Min | Max |
| --- | --- | --- | --- | --- |
| gate_coverage_median | 19.72% | 5.21% | 13.04% | 26.67% |
| notification_event_detection_rate | 75.79% | 6.00% | 68.42% | 84.21% |
| notification_early25_rate | 52.63% | 6.45% | 42.11% | 57.89% |
| notification_delay_points_median | 1.10 | 0.22 | 1.00 | 1.50 |
| notification_delay_seconds_median | 3.60 | 0.65 | 3.00 | 4.50 |
| notification_coverage_median | 5.00% | 0.00% | 5.00% | 5.00% |
| false_notifications_per_1000_normal_points | 14.88 | 0.95 | 13.87 | 16.46 |

| Per-seed ratio metric (%) | Mean | Sample std | Min | Max |
| --- | --- | --- | --- | --- |
| false_notification_reduction_pct | 56.45 | 6.22 | 48.52 | 65.93 |
| edr_retention_pct | 82.90 | 6.44 | 72.22 | 88.24 |
| early25_retention_pct | 70.98 | 7.32 | 58.82 | 78.57 |

### 45. 신규 테스트

**68 PASS**, focused5.834초. 요청 causal gate7개/no future/window/reset/cooldown/emission/suppression/event/coverage/false burden/선택 guard/lexicographic/lock/hash/Test rejection/regression/immutable artifact/NaN 범위를 포함한다. 추가 exact prefix reuse, 변경된 prefix reuse 실패, pre-event ongoing run 단위 구분을 검증했다.

### 46. 전체 테스트

`python -m unittest discover -s tests -v`: **458 PASS**, **103.798초**. 기존 Stage1~6.6 **390개**를 모두 유지했다.

### 47. Protected checksum

작업 전 snapshot **524개**, 작업 후 **changed0**. Stage5 dataset/user split/labels/model/scaler/threshold/predictions/official metrics와 Stage6.1~6.6 output/source/test/doc을 유지했다.

### 48. Candidate / lock hash verification

Candidateconfig와 candidate manifest는 initial receipt까지 대조했고 lock hash는별도receipt,Validation/source artifact hashes,Test 시작 전/후 검증을 통과했다. Verifier가 chronology와 family당21 candidate/1 lock/selected eligible/lexicographic reproducibility를 독립 재계산했다. 모든 figure/output hash도 일치했다.

### 49. Test leakage 여부

Parameter selection leakage **없음**. Pure Validation selector는 Test/mixed/untagged data를 거부한다. Test evaluator는 typed lock만 받고 grid를 거부하며 search/rank/optimization 호출이 없다. Selection/lock은 Validation artifact와 metric만 기록한다. Candidate family design은 Stage6.6 Test 관찰 이후이므로 **post-hoc design limitation은 존재**한다. 이를 untouched-final Test라고 표현하지 않는다.

### 50. Runtime

Validation selection **74.4220s**, locked Test **33.7824s**, 합 **108.2044s**. CSV 읽기·계산·8개 figure 포함. Tests/독립 verifier/문서작성 시간은 별도. Training/inference0.

### 51. Case AD~AK 판정

**AF + AH(탐지 retention의 제한된 재현) + AI**. AE의 tradeoff 현상도 관측됐지만 그 원인을 sparse prediction으로 단정하지 않는다. AJ는 평가된 RAW/LOCKED의 잔여 부담 범위에서만 지지된다.

| Case | 판정 | 근거 |
| --- | --- | --- |
| AD | 강한 지지 못 함 | False notification감소하지만 모든familyTest EDR retention90%미만 |
| AE | Tradeoff 관측; sparse 원인 미확인 | EDR/early하락, selectedgate는RAW이므로persistence원인아님 |
| AF | 지지 | pre-event false alert cooldown이추가miss/delay유발 |
| AG | 선택 결과 지지 못 함 | Persistence/vote eligible없음; Test후보추가계산없음 |
| AH | Retention 측면 지지 | Validation guard통과,Test모든familyEDR90%조건미달;burden감소는재현 |
| AI | 지지 | User013크게하락,user001상대유지,user017burden높음 |
| AJ | 평가 policy 범위에서지지 | Test RAW/LOCKED anynotification100%;전체21Test후보주장하지않음 |
| AK | 아님 | 선택은G0_C60이며G0_C0rawpolicy아님 |

### 52. Alert aggregation 실효성

Notification 빈도 제어에는 효과가 있다. 하지만 detector/gate를 개선한 것은 아니며 gate-level false runs/pointFP는 그대로다. Causal cooldown이 알림 절반 정도를 줄이는 동시에 event를 suppress할 수 있다는 한계를 실제 replay로 확인했다.

### 53. 실사용 수준으로 개선됐는지

입증하지 못했다. Locked에서도11~15 notifications/1000,27~36notifications/trajectory,14~18notifications/valid observation hour,anynotification100%가 남았다. 사용자의 운영 허용 budget을 Test에서 사후 정하지 않았으며 작은 synthetic cohort만으로 실사용 성공을 주장하지 않는다.

### 54. Event / early detection 희생

| Family | Test EDR retention (ratio of means) | Test Early25 retention (ratio of means) |
| --- | --- | --- |
| Rule | 82.35 | 61.54 |
| IF | 88.64 | 81.58 |
| AE | 82.76 | 70.42 |

미리 고정한 Validation constraint를 Test에 대한 기술적 점검으로 적용하면 모든family EDR90%목표 미달; Rule/AE Early25도80%미달. 이를 보고 policy를 바꾸지 않았다. Miss가 늘고 late tail도 커져 false alert감소만으로 성공으로 판단할 수 없다.

### 55. Universal policy 가능성

세family 모두 같은Validationpolicy를 선택했지만 Test사용자별탐지손실/부담차이가 커 범용운영policy의성공을 입증하지 못했다. User-specific policy나seed-specific선택을 도입해 Test를 개선하지 않았다. Validation의12event/4users와Test19event/4users는 작은표본이다.

### 56. 다음 Stage 최우선 후보

**B: Detector calibration / representation improvement의 제한된 Validation-only 연구**를 제안한다. Raw gate의 정상 FP/fragmentation이 그대로이고 cooldown이 true event를 suppress하므로 notification만 줄이는 finalization보다 원래 score/representation의 정상 분리 문제를 다룰 근거가 있다. 후보 C route-aware,E 새confirmatory cohort,D cold-start/abstention,F sequence는 대안이다. Stage6.8는 구현하지 않았다.

### 57. 새 untouched final cohort 필요성

필요하다. Policy family와본결과/다음설계는 이미관찰된Testcohort의영향을받았다. 향후 protocol/정책/model을새cohort평가전에고정하고 새unseenusers를confirmatoryholdout으로예약해야한다. 현재Test결과는parameterheld-out application으로만보고한다.

## Validation 전체 candidate grid

다음은 Validation-only family mean 결과이다. Test candidate grid/oracle best 표는 없다. 각 policy/seed와 event별 모든 지표는 CSV에 보관했다.

| Family | Policy | Eligible | Notification EDR | Early25 | False notifications/1000 | Rejected reason |
| --- | --- | --- | --- | --- | --- | --- |
| AE | G0_C0 | True | 85.00% | 75.00% | 33.05 |  |
| AE | G0_C300 | False | 38.33% | 28.33% | 8.76 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G0_C60 | True | 76.67% | 63.33% | 17.64 |  |
| AE | G1_C0 | False | 48.33% | 18.33% | 9.45 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G1_C300 | False | 33.33% | 13.33% | 4.77 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G1_C60 | False | 48.33% | 18.33% | 7.23 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G2_C0 | False | 25.00% | 3.33% | 3.86 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G2_C300 | False | 20.00% | 3.33% | 2.50 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G2_C60 | False | 25.00% | 3.33% | 3.37 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G3_C0 | False | 6.67% | 0.00% | 0.87 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G3_C300 | False | 6.67% | 0.00% | 0.77 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G3_C60 | False | 6.67% | 0.00% | 0.86 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G4_C0 | False | 51.67% | 21.67% | 10.68 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G4_C300 | False | 33.33% | 13.33% | 5.21 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G4_C60 | False | 51.67% | 21.67% | 8.08 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G5_C0 | False | 36.67% | 11.67% | 5.64 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G5_C300 | False | 31.67% | 10.00% | 3.32 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G5_C60 | False | 36.67% | 11.67% | 4.61 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G6_C0 | False | 16.67% | 3.33% | 2.17 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G6_C300 | False | 16.67% | 3.33% | 1.65 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| AE | G6_C60 | False | 16.67% | 3.33% | 1.97 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G0_C0 | True | 66.67% | 66.67% | 31.56 |  |
| IF | G0_C300 | False | 18.33% | 18.33% | 10.05 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G0_C60 | True | 65.00% | 56.67% | 20.54 |  |
| IF | G1_C0 | False | 18.33% | 1.67% | 7.22 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G1_C300 | False | 11.67% | 0.00% | 4.24 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G1_C60 | False | 18.33% | 1.67% | 6.07 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G2_C0 | False | 15.00% | 0.00% | 2.79 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G2_C300 | False | 8.33% | 0.00% | 2.11 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G2_C60 | False | 15.00% | 0.00% | 2.66 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G3_C0 | False | 3.33% | 1.67% | 0.93 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G3_C300 | False | 3.33% | 1.67% | 0.76 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G3_C60 | False | 3.33% | 1.67% | 0.90 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G4_C0 | False | 20.00% | 1.67% | 8.22 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G4_C300 | False | 11.67% | 0.00% | 4.73 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G4_C60 | False | 20.00% | 1.67% | 6.86 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G5_C0 | False | 16.67% | 0.00% | 4.03 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G5_C300 | False | 10.00% | 0.00% | 2.90 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G5_C60 | False | 16.67% | 0.00% | 3.81 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G6_C0 | False | 13.33% | 1.67% | 1.58 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G6_C300 | False | 13.33% | 1.67% | 1.22 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| IF | G6_C60 | False | 13.33% | 1.67% | 1.46 | mean_EDR_retention;mean_early25_retention;four_of_five_seed_EDR_guard |
| Rule | G0_C0 | True | 66.67% | 58.33% | 33.42 |  |
| Rule | G0_C300 | False | 16.67% | 16.67% | 9.72 | mean_EDR_retention;mean_early25_retention |
| Rule | G0_C60 | True | 66.67% | 58.33% | 20.26 |  |
| Rule | G1_C0 | False | 41.67% | 16.67% | 9.15 | mean_EDR_retention;mean_early25_retention |
| Rule | G1_C300 | False | 25.00% | 16.67% | 4.42 | mean_EDR_retention;mean_early25_retention |
| Rule | G1_C60 | False | 41.67% | 16.67% | 6.53 | mean_EDR_retention;mean_early25_retention |
| Rule | G2_C0 | False | 16.67% | 8.33% | 3.81 | mean_EDR_retention;mean_early25_retention |
| Rule | G2_C300 | False | 16.67% | 8.33% | 2.31 | mean_EDR_retention;mean_early25_retention |
| Rule | G2_C60 | False | 16.67% | 8.33% | 3.19 | mean_EDR_retention;mean_early25_retention |
| Rule | G3_C0 | False | 8.33% | 0.00% | 1.08 | mean_EDR_retention;mean_early25_retention |
| Rule | G3_C300 | False | 8.33% | 0.00% | 0.98 | mean_EDR_retention;mean_early25_retention |
| Rule | G3_C60 | False | 8.33% | 0.00% | 1.08 | mean_EDR_retention;mean_early25_retention |
| Rule | G4_C0 | False | 41.67% | 16.67% | 9.46 | mean_EDR_retention;mean_early25_retention |
| Rule | G4_C300 | False | 33.33% | 16.67% | 4.73 | mean_EDR_retention;mean_early25_retention |
| Rule | G4_C60 | False | 41.67% | 16.67% | 7.46 | mean_EDR_retention;mean_early25_retention |
| Rule | G5_C0 | False | 25.00% | 8.33% | 5.40 | mean_EDR_retention;mean_early25_retention |
| Rule | G5_C300 | False | 25.00% | 8.33% | 2.83 | mean_EDR_retention;mean_early25_retention |
| Rule | G5_C60 | False | 25.00% | 8.33% | 4.11 | mean_EDR_retention;mean_early25_retention |
| Rule | G6_C0 | False | 8.33% | 0.00% | 2.47 | mean_EDR_retention;mean_early25_retention |
| Rule | G6_C300 | False | 8.33% | 0.00% | 1.75 | mean_EDR_retention;mean_early25_retention |
| Rule | G6_C60 | False | 8.33% | 0.00% | 2.21 | mean_EDR_retention;mean_early25_retention |

## 파일과 최종 상태

Code: [audit_alert_aggregation.py](../src/audit_alert_aggregation.py), [alert_policy.py](../src/alert_policy.py). Config: [stage67_alert_policy_candidates.json](../configs/stage67_alert_policy_candidates.json). Tests: [test_alert_aggregation.py](../tests/test_alert_aggregation.py).

Metrics root `outputs/metrics/stage6/alert_aggregation/`: mandatory candidate manifest,validation grid/selection/events/false alerts,locked JSON,Test raw/locked/events/false/user/stability/suppression,audit JSON,final checks를 저장했다. 추가 seed grid/event catalogs/notification trace/reduction/paired delay/phase provenance를 제공한다. Figures root `outputs/figures/stage6/alert_aggregation/`: 요청된8PNG;Validation2개와Test6개를구분했다.

독립 verifier는 Stage6.7모듈을import하지않고 scalar deque와원본CSV에서 **231Validationpolicy-seed**, **2,772Validationevent**, **22Testpolicy-seed**, **418Testevent** 판단을재계산했다. Notification counts,selected eligible/lexicographic,rawStage6.6regression,seedddof1,hash/chronology/보호파일을확인했다.

Branch `feature/stage-6-route-context`. Stage6.6만checkpoint게시. **Stage6.7미커밋/미푸시**,PR없음,mainmerge없음,Stage6.8구현없음. Candidateconfig/lock은Test결과후에도그대로다.
