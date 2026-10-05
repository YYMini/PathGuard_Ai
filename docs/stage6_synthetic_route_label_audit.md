# Stage 6.5 — Synthetic Route Label Identifiability & Boundary Audit

기존 route label의 **94.97%**는 현재 좌표 또는 기존 8 feature가 달라진다. 나머지 **19행(5.03%)**은 각 segment 시작점에 있고, 당시 raw GPS prefix까지 original과 같아 현재 causal 관측으로 구분할 신호가 없다. 이 구조적 boundary 문제는 존재하지만 낮은 전체 recall의 주원인이라고 보기는 어렵다. Tier 4를 제외해도 기존 recall 변화는 약 **+0.68~1.02%p**다.

같은 저장 prediction의 event 탐지율은 **Rule 89.47%, IF 평균 92.63%, AE 평균 91.58%**로 point recall보다 훨씬 높다. 현재 모델은 대부분의 synthetic segment에서 최소 한 point를 잡지만 segment의 모든 labelled point를 잡지는 않는다. 다음 우선 후보는 **Stage 6.6 B: segment/event-level evaluation protocol**이다. Label refinement는 task 의미를 정의한 뒤 별도 실험 후보로 남긴다.

이번 분석은 label을 수정하지 않는다. 기존 공식 Stage 5 metric, CSV, model/scaler, threshold, user split, generator를 모두 보존했다. 원본 source join과 tier는 audit-only 정보이고 detector 입력에 추가하지 않았다. 실제 audit에서 generation/training/inference를 실행하지 않았다.

## 목적과 정보 구분

**Intervention**은 generator가 저장 raw 좌표/시간을 실제로 바꾼 point, **Label**은 기존 evaluation CSV의 anomaly=1, **Observable effect**는 당시의 좌표/derived feature/허용된 causal representation에서 original과 차이가 나는 상태다. 세 집합이 같다고 가정하지 않는다. Tier는 counterfactual pair의 관측 차이를 기술하며 label correctness를 자동 판정하거나 deployment에서 사용할 annotation이 아니다.

Original과 synthetic의 차이가 있다고 해서 normal-vs-anomaly 분리력이 충분하다는 뜻은 아니다. 특히 synthetic interval 소속을 정답으로 정의한다면 변위 0인 시작점도 interval label로는 타당할 수 있다. 그러나 GPS-only current/causal observation으로 그 point를 original 정상과 구분하는 point target에는 모순이 생긴다. 목표가 modified-point인지 latent interval membership인지 event detection인지 명확히 정의해야 한다.

## 완료 보고 1–45

1. **Stage 6.4 테스트 재검증:** `python -m unittest discover -s tests -v` **307 PASS**, 73.959초.

2. **Stage 6.4 commit SHA:** `246d8425b7f0358e7ac5967cbe4f31468b05787f`, 메시지 `stage6: audit trajectory window similarity`. 원래 로컬 검증 commit `b592535b53481ea8e9223822595a1f6e715c61b8`은 tag `stage64-local-verified-b592535`로 보존했다.

3. **Push:** CLI 인증 실패 후 사용자가 허용한 GitHub connector/API로 동일 tree를 게시했다. Tree SHA `c87cea382f503c7277f9e379c2299a41941c4fb4` 일치. 현재 feature branch만 fast-forward했으며 force push/PR/main merge 없음.

4. **Local/remote HEAD:** 둘 다 `246d8425b7f0358e7ac5967cbe4f31468b05787f`. Stage 6.5 code/test/doc/README 변경은 미커밋·미푸시다. 이전 Stage 6.4 보고서는 그 완료 시점의 미커밋 상태를 기록한 frozen report로 보존한다.

5. **Checkpoint checksum:** 기존 370개 보호 파일 변경 0 및 Stage 6.4 출력/실행 source/7개 figure checksum 재확인. Stage 6.5 시작 전에 Stage 6.4 출력과 기존 tracked code/test/doc 등을 포함한 **459개**를 `outputs/metrics/stage65_protected_snapshot.json`에 기록했다.

6. **Generator 규칙:** `src/synthetic_anomalies.py:generate_anomaly_sample` 및 `src/prepare_multiuser_dataset.py:generate_evaluation_synthetic`를 직접 감사했다.

- Source trajectory를 timestamp 기준 stable sort한 independent full copy 하나를 anomaly type별로 생성한다. Stage 5는 eligible Validation/Test original마다 type별 1 sample을 만들고 Train에는 만들지 않는다.
- `_segment`는 contiguous 길이 L을 정수 10..30에서 선택한다(짧은 trajectory는 n-2까지). 시작 index는 1 이상, end는 exclusive이고 마지막 trajectory point 전에 끝난다. trajectory endpoint와 segment endpoint를 혼동하지 않는다.
- `metres ~ Uniform(100,300)` 및 `weights=sin(linspace(0,pi,L))`로 displacement를 taper한다. 원본 segment 첫/마지막 좌표로 heading을 만들고 north=-cos(heading)*metres*weights, east=sin(heading)*metres*weights를 적용한다. latitude에는 north/111320, longitude에는 east/(111320*cos(updated latitude))를 더한다(cos 하한 1e-6).
- Segment 첫 weight는 0, 끝 weight는 부동소수점으로 sin(pi)에 가깝다. 실제 저장 좌표에서 끝의 작은 offset이 round되어 unchanged일 수 있다. 기존 Test에서는 모든 segment의 시작/끝 좌표가 그대로다.
- `_recalculate`는 **full copy**의 기존 이동 feature를 다시 계산한다. Label은 좌표 mask와 별개로 `[start,end-1]` 전체에 1을 붙인다. route_deviation branch는 timestamp를 수정하지 않는다.
- Feature는 causal transition이다. distance/speed는 previous→current, acceleration은 previous/current speed, bearing/direction은 현재·이전 bearing(미세 움직임에서는 carry), stop duration은 past accumulated stop이다. 다음 point의 수정이 현재 feature로 역전파되는 구현은 없다.
- Stage 5 seed는 JSON `[dataset_seed, source_id, anomaly_type, 1]` SHA256 앞 8byte 정수다. point 수가 그대로인지 확인한 뒤 ordered original `source_point_index`를 복사한다. Quality-valid flag는 원본에 연결하고 evaluation에는 original normal + valid synthetic labelled rows만 남긴다.

위 규칙은 코드와 저장 파일에서 확인한 내용이다. 이 audit에서는 RNG/generator 호출이나 synthetic 재생성이 없었다. Observed maximum displacement를 configured amplitude라고 추정하지 않았다.

7. **Intervention metadata:** manifest에 sample/source/user/split, segment start/end timestamp, labelled point count, full point count, random seed, quality-valid label count는 있다. **실제 modified mask와 configured amplitude는 없다.** Stage 6.2의 amplitude column도 NA이며 저장 amplitude 증거가 아니다.

8. **Lineage reconstruction:** `(source_trajectory_id,source_point_index)`로 full synthetic copy와 original을 exact many-to-one join했다. Evaluation join/prediction/context는 `(sample_id,source_trajectory_id,source_point_index)`로 exact 검증했다. Missing/duplicate/user mismatch는 실패한다. 기존 `original_trajectory_id` filename metadata와 새 source trajectory identifier가 충돌하지 않도록 paired identifier를 별도 이름으로 유지했다. raw latitude/longitude, Haversine coordinate displacement, timestamp delta, 8개 feature의 synthetic-original delta를 다시 계산했다.

9. **Route label 수:** evaluation **378행**, full copy **46,391행** 중 label **379행**. Source-quality-invalid labelled interior 1행은 원래 evaluation에서 제외돼 있다. 해당 row의 prediction/tier를 만들어내지 않았다. Label run과 propagation은 full copy에서 분석하고 detector/tier는 evaluation 378행에 한정한다.

10. **Coordinate displacement threshold:** 아래 모든 사전 threshold를 병렬 보고했다. cutoff 선택/label 변경 없음.

| threshold_m | comparison | count | percentage |
| --- | --- | --- | --- |
| 0 | > | 340 | 89.9471 |
| 1 | >= | 340 | 89.9471 |
| 5 | >= | 340 | 89.9471 |
| 10 | >= | 340 | 89.9471 |
| 25 | >= | 328 | 86.7725 |
| 50 | >= | 294 | 77.7778 |
| 100 | >= | 216 | 57.1429 |
| 200 | >= | 53 | 14.0212 |

11. **Zero displacement:** 38행. Coordinate zero는 ≤1e-6m, strict >0 count도 별도로 보고했다. 실제 데이터에서 strict zero와 tolerance zero의 count가 동일하다.

12. **Feature changed count:** native feature 단위로 `np.isclose(synthetic,original,atol=1e-6,rtol=1e-8)`를 사용했다. 0개=19, 1개=0, 2~3개=0, 4+개=359이다. 상세 count는 4개=3, 5개=1, 6개=311, 7개=44, 8개=0. 'all 8 changed'는 4+ category의 보조 subset이며 중복 분류로 합계에 더하지 않는다. Timestamp가 보존돼 time_diff_sec도 변하지 않아 all-eight changed는 0이다.

| feature | row_count | changed_count | changed_percentage | absolute_delta_median | absolute_delta_p90 | absolute_delta_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| time_diff_sec | 378 | 0 | 0 | 0 | 0 | 0 |
| distance_m | 378 | 358 | 94.709 | 9.7647 | 36.3914 | 40.7748 |
| speed_mps | 378 | 358 | 94.709 | 3.6149 | 13.8883 | 17.5913 |
| acceleration_mps2 | 378 | 359 | 94.9735 | 0.4435 | 3.402 | 5.7078 |
| direction_change_deg | 378 | 357 | 94.4444 | 6.7614 | 51.3999 | 79.3407 |
| stop_duration_sec | 378 | 46 | 12.1693 | 0 | 5 | 12 |
| bearing_sin | 378 | 356 | 94.1799 | 0.6509 | 1.3852 | 1.6121 |
| bearing_cos | 378 | 357 | 94.4444 | 0.5126 | 1.1764 | 1.4167 |

13. **Identical feature:** 19행, 기존 Stage 6.1~6.4 count와 일치. Tolerance를 Test 결과로 조정하지 않았다.

14. **Tier 1**, 15. **Tier 2**, 16. **Tier 3**, 17. **Tier 4**:

| identifiability_tier | row_count | percentage | coordinate_displacement_median | feature_changed_count_median |
| --- | --- | --- | --- | --- |
| tier1 | 340 | 89.9471 | 126.9077 | 6 |
| tier2 | 19 | 5.0265 | 0 | 6 |
| tier3 | 0 | 0 | NA | NA |
| tier4 | 19 | 5.0265 | 0 | 0 |

Tier 판정은 우선순위가 있는 서로 배타적 partition이다. 현재 displacement가 있으면 Tier 1, displacement가 없고 8-feature 변화가 있으면 Tier 2, 둘 다 같고 causal score 차이만 있으면 Tier 3, 그마저 모두 같으면 Tier 4다. 합계=378.

Context는 frozen Stage 6.3 prefix/combined distance 및 Stage 6.4 **W10/gap0 prefix/combined** window score다. 가장 짧은 사전 context를 대표하는 boundary inspection이며 detector config selection이 아니다. Equality는 atol=1e-5m, rtol=1e-8. 모든 378 pair에서 네 score comparison이 available이다. 결측 representation을 equality로 바꾸지 않으며, comparison이 불완전한 row를 Tier 4로 선언하면 fail-fast한다.

Tier 3가 0이라는 결과는 temporal effect가 없다는 뜻이 아니다. 모든 ending boundary는 derived feature가 먼저 변하므로 Tier 2에 배정됐고, context 변화도 함께 존재한다. Tier 4는 현재 audit 표현에서 signal이 없다는 뜻이며 잘못된 label로 자동 판정하지 않았다.

18. **Start/interior/end 분포:**

| segment_boundary | labelled_rows | displacement_median | displacement_p90 | feature_changed_count_median | tier1_count | tier2_count | tier3_count | tier4_count | tier1_pct | tier2_pct | tier3_pct | tier4_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| end | 19 | 0 | 0 | 6 | 0 | 19 | 0 | 0 | 0 | 100 | 0 | 0 |
| interior | 340 | 126.9077 | 220.3297 | 6 | 340 | 0 | 0 | 0 | 100 | 0 | 0 | 0 |
| start | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 | 0 | 100 |

19. **Tier 4 boundary 집중:** Tier 4 19개 모두 segment 시작이고 interior/end에는 0이다. 독립 검증에서 그 19개의 **current까지 전체 raw original/synthetic prefix도 동일**함을 확인했다. 이는 단지 네 score가 우연히 같은 결과보다 강한 현재 fixture의 관측 동일성 증거다. 미래 point나 onset 이후 신호로 시작점의 causal point 판정을 소급하지 않았다.

20. **38 zero-displacement:** 시작 19개는 Tier 4, 종료 19개는 Tier 2다. 종료에서는 previous displaced point에 따른 transition/derived feature 변화와 과거 synthetic window의 영향이 있다. 직접 raw intervention이 없다고 observable effect도 없다고 해석하지 않는다.

21. **19 identical-feature:** 모두 Tier 4이고 Tier 3로 이동한 행은 0이다. 기존 detector 11개 run × 19 pair에서 original 정상과 score/decision이 모두 같았다. AE의 2개 positive decision도 paired original 정상에서 같은 false positive였다. 이런 positive를 boundary 고유 신호 탐지로 해석하지 않는다.

**Full-copy label–intervention 2×2:** 기존 full copy에 unlabelled rows가 실제로 있어 table을 계산할 수 있다. Copied negatives는 evaluation metric의 정상 class로 추가하지 않았다.

| anomaly_label | coordinate_intervention_strict | row_count |
| --- | --- | --- |
| 0 | False | 46012 |
| 0 | True | 0 |
| 1 | False | 38 |
| 1 | True | 341 |

**Boundary propagation:** 다음 source point의 수정이 현재 feature를 바꾸는 효과는 없다. 이전 좌표/속도/bearing/stop carry 때문에 segment 밖 30개 unlabelled row의 feature가 달라진다. 이는 저장 label과 observable effect의 범위가 다를 수 있다는 증거다. 그 rows는 기존 prediction CSV에 synthetic negatives로 저장돼 있지 않아 새로운 prediction을 생성하지 않았다.

| scope | copied_rows | directly_modified_strict_count | feature_observable_count | feature_propagation_only_count | propagation_previous_modified_count | propagation_next_modified_count | propagation_carried_history_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 46391 | 341 | 390 | 49 | 19 | 0 | 30 |
| labelled | 379 | 341 | 360 | 19 | 19 | 0 | 0 |
| not_labelled | 46012 | 0 | 30 | 30 | 0 | 0 | 30 |

**Consecutive label runs:** 19 sample에 각 1 run, full labelled length 합 379. 평가 가능 labelled length는 378이다. `route_label_consecutive_runs.csv`에는 full/evaluation 길이, direct/feature/context observable 길이, Tier count, omitted context 길이를 저장했다. Context length는 score가 존재하는 evaluated subset에 한정하고 미평가 1행을 silent zero나 Tier 4로 바꾸지 않았다.

22. **Rule recall**, 23. **IF recall**, 24. **AE recall** by tier: Rule은 단일 run, IF/AE는 model seed 7/21/42/100/2026 mean 및 sample std(ddof=1)다. Empty Tier 3는 **NA**이며 recall 0으로 치환하지 않는다.

| model | identifiability_tier | anomaly_row_count | detected_count_mean | recall_mean | recall_sample_std | defined_seed_count |
| --- | --- | --- | --- | --- | --- | --- |
| autoencoder | tier1 | 340 | 74.8 | 0.22 | 0.044284 | 5 |
| autoencoder | tier2 | 19 | 3.4 | 0.178947 | 0.07982 | 5 |
| autoencoder | tier3 | 0 | 0 | NA | NA | 0 |
| autoencoder | tier4 | 19 | 0.4 | 0.021053 | 0.047075 | 5 |
| isolation_forest | tier1 | 340 | 46.4 | 0.136471 | 0.008969 | 5 |
| isolation_forest | tier2 | 19 | 2.2 | 0.115789 | 0.044035 | 5 |
| isolation_forest | tier3 | 0 | 0 | NA | NA | 0 |
| isolation_forest | tier4 | 19 | 0 | 0 | 0 | 5 |
| statistical_rule | tier1 | 340 | 68 | 0.2 | NA | 1 |
| statistical_rule | tier2 | 19 | 5 | 0.263158 | NA | 1 |
| statistical_rule | tier3 | 0 | 0 | NA | NA | 0 |
| statistical_rule | tier4 | 19 | 0 | 0 | NA | 1 |

**Score by tier:** 기존 Stage 6.2 global / Stage 6.3 causal / Stage 6.4 window score를 재사용해 median/p75/p90을 저장했다. Displacement가 있고 feature가 달라진다는 것이 normal class 밖에 있다는 뜻은 아니다. Source original과 달라도 기존 정상 분포에서 흔한 이동일 수 있다.

| identifiability_tier | score | count | median | p75 | p90 |
| --- | --- | --- | --- | --- | --- |
| tier1 | synthetic_reference_distance_m | 340 | 118.9806 | 369.3409 | 960.9918 |
| tier1 | stage63_prefix_synthetic_score_m | 340 | 17.7651 | 27.4159 | 38.7075 |
| tier1 | stage63_combined_synthetic_score_m | 340 | 17.7651 | 27.1332 | 37.7706 |
| tier1 | w10g0_prefix_synthetic_score_m | 340 | 175.8161 | 219.3415 | 301.3776 |
| tier1 | w10g0_combined_synthetic_score_m | 340 | 171.8242 | 217.2136 | 268.4511 |
| tier2 | synthetic_reference_distance_m | 19 | 23.0766 | 426.4433 | 1598.7175 |
| tier2 | stage63_prefix_synthetic_score_m | 19 | 21.6166 | 34.1686 | 42.3073 |
| tier2 | stage63_combined_synthetic_score_m | 19 | 15.3514 | 34.1686 | 42.3073 |
| tier2 | w10g0_prefix_synthetic_score_m | 19 | 182.123 | 252.9804 | 302.8837 |
| tier2 | w10g0_combined_synthetic_score_m | 19 | 182.123 | 247.8285 | 278.5662 |
| tier3 | synthetic_reference_distance_m | 0 | NA | NA | NA |
| tier3 | stage63_prefix_synthetic_score_m | 0 | NA | NA | NA |
| tier3 | stage63_combined_synthetic_score_m | 0 | NA | NA | NA |
| tier3 | w10g0_prefix_synthetic_score_m | 0 | NA | NA | NA |
| tier3 | w10g0_combined_synthetic_score_m | 0 | NA | NA | NA |
| tier4 | synthetic_reference_distance_m | 19 | 21.9233 | 415.6401 | 1567.3156 |
| tier4 | stage63_prefix_synthetic_score_m | 19 | 8.9336 | 11.8501 | 12.7885 |
| tier4 | stage63_combined_synthetic_score_m | 19 | 8.3466 | 11.8501 | 12.7885 |
| tier4 | w10g0_prefix_synthetic_score_m | 19 | 95.7777 | 167.6584 | 184.0262 |
| tier4 | w10g0_combined_synthetic_score_m | 19 | 70.9999 | 148.8911 | 184.0262 |

25. **Observable subset recall**, 26. **Tier 4 제외 audit-only 변화:** Observable subset=Tier 1+2+3=359행(94.97%), Tier 4=19행(5.03%). 아래는 model seed mean이다. 명칭은 **counterfactual observable-subset recall**이며 official metric이나 수정된 정답이 아니다.

| model | official_frozen_recall | observable_subset_recall | audit_delta_percentage_points | tier4_recall |
| --- | --- | --- | --- | --- |
| autoencoder | 0.207937 | 0.217827 | 0.989079 | 0.021053 |
| isolation_forest | 0.128571 | 0.135376 | 0.680462 | 0 |
| statistical_rule | 0.193122 | 0.203343 | 1.022093 | 0 |

Low point recall은 Tier 4 5.03%를 제외해도 대부분 남는다. Boundary impossibility를 무시해서는 안 되지만 이 현상만으로 detector representation/model 한계를 설명하지 않는다.

**Displacement magnitude별 recall:** 0m, (0,10), [10,25), [25,50), [50,100), [100,200), ≥200m의 모든 bin을 유지한다. (0,10)은 0행이므로 NA다. n<10은 low-support로 표시하는 사전 규칙을 뒀으며 이 데이터의 nonempty bin은 최소 12행이다. Displacement가 커진다고 recall이 단조 증가하지 않는다.

| model | displacement_bin | anomaly_row_count | recall_mean | recall_sample_std | low_support |
| --- | --- | --- | --- | --- | --- |
| autoencoder | (0,10) | 0 | NA | NA | False |
| autoencoder | [10,25) | 12 | 0.433333 | 0.136931 | False |
| autoencoder | [100,200) | 163 | 0.177914 | 0.038313 | False |
| autoencoder | [200,inf) | 53 | 0.184906 | 0.033752 | False |
| autoencoder | [25,50) | 34 | 0.323529 | 0.088235 | False |
| autoencoder | [50,100) | 78 | 0.253846 | 0.081286 | False |
| autoencoder | zero | 38 | 0.1 | 0.034312 | False |
| isolation_forest | (0,10) | 0 | NA | NA | False |
| isolation_forest | [10,25) | 12 | 0.316667 | 0.037268 | False |
| isolation_forest | [100,200) | 163 | 0.08589 | 0.008676 | False |
| isolation_forest | [200,inf) | 53 | 0.143396 | 0.028615 | False |
| isolation_forest | [25,50) | 34 | 0.347059 | 0.013153 | False |
| isolation_forest | [50,100) | 78 | 0.117949 | 0.014044 | False |
| isolation_forest | zero | 38 | 0.057895 | 0.022017 | False |
| statistical_rule | (0,10) | 0 | NA | NA | False |
| statistical_rule | [10,25) | 12 | 0.333333 | NA | False |
| statistical_rule | [100,200) | 163 | 0.122699 | NA | False |
| statistical_rule | [200,inf) | 53 | 0.264151 | NA | False |
| statistical_rule | [25,50) | 34 | 0.294118 | NA | False |
| statistical_rule | [50,100) | 78 | 0.25641 | NA | False |
| statistical_rule | zero | 38 | 0.131579 | NA | False |

**Feature-change strength:** 0/1/2~3/4+ 그룹별 frozen recall도 저장했다. 이 dataset의 1 및 2~3 그룹은 비어 있어 NA이며 0-feature 그룹=Tier 4, 4+ 그룹=observable subset이다. 작은 numerical change와 유효한 anomaly signal을 동일시하지 않는다.

27. **Synthetic sample/event 수:** 19개 sample, 19개 consecutive labelled event. 기존 evaluated labelled points만 사용한 'event 안에서 저장 positive decision 최소 1개' 기준이다.

28. **Rule event rate**, 29. **IF event rate**, 30. **AE event rate**:

| model | model_run_count | event_count | detected_events_mean | event_detection_rate_mean | event_detection_rate_sample_std | event_detection_rate_min | event_detection_rate_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| autoencoder | 5 | 19 | 17.4 | 0.915789 | 0.060009 | 0.842105 | 1 |
| isolation_forest | 5 | 19 | 17.6 | 0.926316 | 0.028828 | 0.894737 | 0.947368 |
| statistical_rule | 1 | 19 | 17 | 0.894737 | NA | 0.894737 | 0.894737 |

31. **First detection delay:** 모든 run의 detected-event median은 labelled run 시작 기준 **1 point / 3초**다. 검출하지 못한 event는 delay=NA이고 '즉시 검출'이나 0초로 넣지 않았다. Full source index/timestamp를 사용해 evaluation quality omission을 압축된 row 위치로 오해하지 않았다. Sine-zero start 다음 point가 첫 직접 intervention이므로 label start 대비 1 point delay가 관측된 intervention 대비 늦은 탐지를 뜻하지는 않는다. 모든 per-event position/delay 209행을 저장했다.

Event rate는 existing positive synthetic segment 조건부 값이다. **Negative-event false alarm, false alarms/hour, event precision을 평가한 것이 아니다.** 긴 event는 우연한 positive에 더 노출될 수 있으므로 높은 event rate만으로 service detector 성능을 주장하거나 기존 공식 metric을 대체하지 않는다.

32. **사용자별 Tier**, 33. **Observable ratio**:

| user_id | route_label_count | tier1_count | tier2_count | tier3_count | tier4_count | observable_ratio | tier4_proportion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | 88 | 78 | 5 | 0 | 5 | 0.943182 | 0.056818 |
| 008 | 93 | 83 | 5 | 0 | 5 | 0.946237 | 0.053763 |
| 013 | 79 | 71 | 4 | 0 | 4 | 0.949367 | 0.050633 |
| 017 | 118 | 108 | 5 | 0 | 5 | 0.957627 | 0.042373 |

Tier 4 비율은 4.24~5.68%로 사용자 간 편차가 작다. Observable-subset recall 및 event rate 차이는 더 크므로 이를 label identifiability 비율만으로 설명하지 않는다. 인과관계는 주장하지 않았다.

34. **사용자별 event 및 observable-subset recall:** 같은 사용자 안의 model-seed mean이다.

| user_id | statistical_rule_observable_recall_mean | isolation_forest_observable_recall_mean | autoencoder_observable_recall_mean | statistical_rule_event_detection_rate_mean | isolation_forest_event_detection_rate_mean | autoencoder_event_detection_rate_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 001 | 0.337349 | 0.26506 | 0.286747 | 1 | 0.92 | 0.84 |
| 008 | 0.125 | 0.086364 | 0.170455 | 0.6 | 1 | 0.84 |
| 013 | 0.226667 | 0.146667 | 0.264 | 1 | 1 | 1 |
| 017 | 0.150442 | 0.070796 | 0.173451 | 1 | 0.8 | 1 |

User 008의 Rule event rate는 60%이고 user 017 IF mean은 80%로, global event rate가 높다고 모든 사용자/event가 해결됐다고 해석하지 않는다. IF/AE seed variation과 sample support도 CSV에 남겼다.

**Sample별:** 19 sample 모두 Tier 4 시작점 1개씩이므로 특정 sample에만 19개 unidentifiable point가 집중된 것은 아니다. Sample별 full/evaluated length, direct/tier count, displacement median/max, observable ratio는 `route_label_per_sample.csv`에 있다. Displacement max는 관측값이고 configured amplitude가 아니다.

| user_id | source_trajectory_id | evaluated_labelled_rows | excluded_labelled_rows | directly_displaced_rows | tier1_count | tier2_count | tier3_count | tier4_count | displacement_median | displacement_max | observable_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | 001__20081023055305 | 16 | 1 | 14 | 14 | 1 | 0 | 1 | 164.668 | 260.8238 | 0.9375 |
| 001 | 001__20081023234104 | 15 | 0 | 13 | 13 | 1 | 0 | 1 | 129.3369 | 207.4403 | 0.9333 |
| 001 | 001__20081024234405 | 20 | 0 | 18 | 18 | 1 | 0 | 1 | 74.0818 | 109.381 | 0.95 |
| 001 | 001__20081025231428 | 14 | 0 | 12 | 12 | 1 | 0 | 1 | 136.3975 | 204.19 | 0.9286 |
| 001 | 001__20081026081229 | 23 | 0 | 21 | 21 | 1 | 0 | 1 | 146.7131 | 224.0374 | 0.9565 |
| 008 | 008__20081024114834 | 22 | 0 | 20 | 20 | 1 | 0 | 1 | 85.5573 | 125.4358 | 0.9545 |
| 008 | 008__20081024132624 | 15 | 0 | 13 | 13 | 1 | 0 | 1 | 118.0109 | 189.2746 | 0.9333 |
| 008 | 008__20081025041134 | 11 | 0 | 9 | 9 | 1 | 0 | 1 | 97.9723 | 166.6806 | 0.9091 |
| 008 | 008__20081025225205 | 22 | 0 | 20 | 20 | 1 | 0 | 1 | 110.6956 | 162.2912 | 0.9545 |
| 008 | 008__20081026055934 | 23 | 0 | 21 | 21 | 1 | 0 | 1 | 137.4152 | 209.8389 | 0.9565 |
| 013 | 013__20080927120819 | 28 | 0 | 26 | 26 | 1 | 0 | 1 | 134.9034 | 196.583 | 0.9643 |
| 013 | 013__20080927233805 | 20 | 0 | 18 | 18 | 1 | 0 | 1 | 181.6118 | 268.1484 | 0.95 |
| 013 | 013__20080929001436 | 13 | 0 | 11 | 11 | 1 | 0 | 1 | 139.9453 | 197.9122 | 0.9231 |
| 013 | 013__20080929090206 | 18 | 0 | 16 | 16 | 1 | 0 | 1 | 160.6425 | 237.433 | 0.9444 |
| 017 | 017__20081030092748 | 15 | 0 | 13 | 13 | 1 | 0 | 1 | 178.8091 | 286.7873 | 0.9333 |
| 017 | 017__20081030112722 | 22 | 0 | 20 | 20 | 1 | 0 | 1 | 129.5266 | 189.8994 | 0.9545 |
| 017 | 017__20081030234453 | 28 | 0 | 26 | 26 | 1 | 0 | 1 | 93.9888 | 136.9616 | 0.9643 |
| 017 | 017__20081031091917 | 23 | 0 | 21 | 21 | 1 | 0 | 1 | 171.5124 | 261.9061 | 0.9565 |
| 017 | 017__20081101020307 | 30 | 0 | 28 | 28 | 1 | 0 | 1 | 94.5158 | 137.2361 | 0.9667 |

35. **신규 테스트:** 40 PASS. Exact lineage/prediction join, filename metadata collision, Haversine/zero/38 count, feature tolerance/change/19 identical count, four Tier/exclusive/exhaustive/NA guard, segment/singleton/multiple runs, future-independent propagation, bins, recall/counterfactual/event/first delay, quality-omission delay, user/sample aggregate, immutable labels/CSV, NaN/inf, checksum/export/7 figures를 검증했다.

36. **전체 테스트:** `python -m unittest discover -s tests -v`, **347 PASS**(기존 307+신규 40), 95.779초. 프로젝트 `.venv` Python으로 실행했다. 기존 test fixture의 training tests는 유지했지만 실제 Stage 5 model을 실행/재학습하지 않았다.

37. **Checksum:** 작업 전 snapshot **459개 변경 0**. Stage 1~6.4의 data/model/scaler/prediction/metrics/figures 및 기존 code/test/doc를 보존했다. README만 현재 진행 상태로 갱신했다. 신규 Stage 6.5 출력 checksum과 실행 source hash도 검증했다.

38. **Leakage/audit-only separation:** 독립 verifier에서 378 source pair, 38 zero/19 identical, Tier 합계, 11 stored prediction run의 **4,158 pair**, **19 event×11 run=209** event count/first detection/delay를 재계산했다. 19 Tier 4의 full raw prefix 동일성을 추가 확인했다. Original·tier·manifest·label은 audit join/aggregation에만 사용했고 detector 입력·score·threshold를 만들거나 변경하지 않았다. Audit-only separation 위반 및 dataset cross-split leakage 0.

39. **Runtime:** 실제 audit **22.445초**(입력 검증, 집계, CSV/JSON/PNG 출력과 내부 checksum 포함). 후속 독립 verifier 시간은 별도다.

40. **Case P/Q/R/S/T/U/V:**

| Case | 판정 | 근거 |
| --- | --- | --- |
| P | 지지, 관측 차이와 분리 가능성은 구분 | Tier 1+2 94.97%; Tier 4 제외 후 recall 개선은 약 1%p 이내여서 낮은 point recall 대부분이 남는다. |
| Q | 현재 Tier 정의에서는 불충족 | Tier 3=0. Ending boundary의 temporal effect는 존재하지만 feature가 먼저 달라져 Tier 2다. |
| R | 제한적 지지 | 5.03% positive는 현재까지 raw prefix도 같아 causal point target에 식별 불가능성이 있다. 전체 recall 저하의 주원인은 아니다. |
| S | 강하게 지지 | Tier 4 전부 19개 segment의 start; sine-zero endpoint와 interval label 규칙에 대응한다. |
| T | 불충족 | Interior Tier 4=0. 광범위한 interior 비관측 문제는 현재 데이터에서 발견되지 않았다. |
| U | 강하게 지지 | Point recall 12.86~20.79%와 event rate 89.47~92.63% 사이에 큰 차이가 있다. 목표 범위와 평가 단위를 구분할 필요가 있다. |
| V | 전체에서는 불충족, 일부 미검출 지속 | Global event rate가 높지만 사용자/seed에 따라 60~100%이고 놓친 event가 남는다. label 문제만으로 해결됐다고 볼 수 없다. |

41. **현재 route label 핵심 문제:** smooth intervention의 실제 좌표 mask와 interval label이 다르다. Start는 현재 causal 정보에서 identical, end는 좌표 동일해도 derived feature가 변한다. Segment 밖에도 feature effect 30행이 있고, point coverage와 event presence가 다른 목적이다. Tier 4는 자동 label error가 아니다.

42. **Point-level evaluation 유지:** 기존 official metric은 그대로 유지할 수 있지만 'label된 모든 point를 맞히는 coverage 목표'라는 의미와 boundary의 observability 한계를 명시해야 한다. Point-only recall을 route event detection 실패율로 그대로 해석하지 않는다. Future event label로 earlier identical observation을 소급 평가하는 문제를 별도로 기술한다.

43. **Segment/event 평가 필요:** route anomaly의 구간 의미를 보려면 추가 event protocol이 필요하다. Positive-event rate 외에 정상 control event/window, false alarm rate, onset definition, matching/merge rules, delay, macro-user/seed support를 Train/Validation 중심으로 설계해야 한다. 이번 audit가 그 protocol을 구현하거나 threshold를 고른 것은 아니다.

44. **Label refinement 실험:** A/B/C/D/E 정책은 향후 후보이며 아래 count-only simulation만 했다. A는 unchanged 38개를 negative 후보로 둘 수 있으나 derived-observable end를 버리는 문제가 있고, B/E는 현재 audited observable 359/ignore 19를 구분한다. C는 point 수와 다른 event 단위여서 19 event로 표시한다. D는 target 설계가 아직 없어 count=NA로 두었다. 어떤 policy도 CSV에 적용하지 않았다.

| policy | positive | boundary_ignore | negative_candidate | count_unit |
| --- | --- | --- | --- | --- |
| A direct displacement only | 340 | 0 | 38 | points |
| B direct plus observable boundary | 359 | 19 | 0 | points |
| C segment/event target | 19 | 0 | 0 | events |
| D transition/window target (requires target design) | NA | NA | NA | not_defined |
| E separate unidentifiable boundary | 359 | 19 | 0 | points |

45. **Stage 6.6 최우선 후보:** **B Segment/event-level evaluation protocol**. Point와 event 목표 차이가 가장 큰 결과이고, negative controls/false-alarm 및 onset/delay를 갖춘 검증이 우선이다. 다음은 **A Synthetic label refinement experiment**, **C Window/transition target 재설계**다. D cold-start/abstention, E limited route-aware feature, F sequence model은 별도 후보이며 이번에 구현하지 않았다. 새로운 Test cutoff를 골라 label이나 threshold를 최적화하지 않았다.

## 한계

- Audit tier는 original counterfactual 비교에서 정한 관측 차이이고 deployable classifier가 아니다. Source original/미래 generator 의도는 실서비스에서 조회할 수 있다고 가정하지 않는다.
- Four context score가 동일하다는 것만으로 모든 가능한 history 표현이 동일하다는 일반 정리를 주장하지 않는다. 현재 Tier 4 19개에서는 별도 raw-prefix 검증으로 전체 causal observation도 같은 특수 상태임을 확인했다.
- Native-unit tolerance, meter tolerance 및 strict coordinate modification을 구분했다. Quantile/bin은 기술통계이고 label threshold가 아니다.
- Data는 기존 synthetic test subset 한 종류이며 실제 route departure의 다양성과 negative-event 분포를 충분히 대표하지 않는다.
- Events/points는 같은 source trajectory에 의존한다. Inferential significance, 독립 confidence interval, 사용자별 인과 원인을 주장하지 않는다.
- Quality-omitted 1행은 full-copy 구조만 감사하고 기존 없는 prediction/context를 만들지 않았다.
- Event rate는 최소 한 positive라는 비교적 쉬운 기준이므로 event precision이나 완전한 segment segmentation 성능을 증명하지 않는다.

## 산출물과 재현

Metrics: `outputs/metrics/stage6/route_label_audit/`.

필수 CSV 11개와 JSON 외에도 full-copy audit, displacement thresholds, per-seed tier/bin/feature recall, label runs/progression, intervention confusion, propagation, prediction audit pairs, event summary, score-by-tier를 저장했다. `route_label_audit.json`에 generator review/tolerance/seed/availability/원본 보존 정책과 runtime, `stage6_5_verification.json`에 신규 출력 checksum이 있다.

Independent verification: `outputs/metrics/stage65_independent_verify.py`, `stage65_independent_checks.log`, `stage65_final_checks.json`.
Checkpoint: `outputs/metrics/stage64_history_publication.json`, `stage64_prepublication_tests.log`.
Tests: `tests/test_synthetic_route_labels.py`, `outputs/metrics/stage65_all_tests.log`.

Figures: `outputs/figures/stage6/route_label_audit/`의 7개 PNG:

1. `identifiability_tier_distribution.png`
2. `coordinate_displacement_distribution.png`
3. `feature_change_count_distribution.png`
4. `detector_recall_by_tier.png`
5. `tier_distribution_by_user.png`
6. `tier_by_segment_position.png`
7. `event_detection_summary.png`

```powershell
python -m unittest discover -s tests -v
python -m src.audit_synthetic_route_labels --protected-snapshot outputs/metrics/stage65_protected_snapshot.json
```

입력/기존 audit/prediction은 read-only로 처리하고 이미 output directory가 있으면 overwrite를 거부한다. 독립 재실행은 새 output-root에 같은 frozen 선행 파일과 그 root의 보호 snapshot을 준비해야 한다.

Stage 6.5 결과는 **미커밋·미푸시**이며 PR/main merge 없음. Stage 6.6/7, generator/data/label 수정, 새 model inference/training 없음.
