# Stage 6.3 — Inference-observable Context Reference & Abstention Audit

## 사전 분석 계약 (2026-10-05)

Stage 6.2 checkpoint: `e657c8cdf5e5c2379b0c43186eac56435b64aac1`, local/remote 일치. 기존 240개 테스트와 253개 보호 checksum을 재검증한 후 게시했습니다. Stage 6.3은 별도 모듈로 구현하고 commit/push/PR/main merge하지 않습니다.

개인 reference는 같은 Test 사용자의 이미 시작된 더 이른 trajectory 중 timestamp<query time인 관측 좌표만 사용합니다. Current-prefix는 현재 관측 copy의 source_point_index<현재 index AND timestamp<현재 time입니다. 같은 timestamp는 포함하지 않습니다. Prior trajectory와 현재 trajectory가 시간상 겹치더라도 query 이후 point는 제외합니다. 새로운 시나리오마다 memory를 초기화합니다. Prior history는 frozen 이전 원본 로그 그대로라는 실험 가정이며 anomaly label로 history를 정제하지 않습니다.

원본/route copy의 전체 유효 좌표·timestamp를 replay합니다. Model evaluation 행만 최종 분석에 남기지만 reference history는 model 품질 flag나 synthetic/source anomaly label로 filtering하지 않습니다. Stream의 finite GPS와 엄격한 시간/index 순서를 검증하고, label/onset/원본 현재·미래 좌표는 deployable scorer에 전달하지 않습니다. Route copy마다 자신이 관측한 synthetic prefix가 memory에 누적됩니다. Stage 6.1/6.2 metadata는 점수 계산 후 audit pairing/progression/boundary 설명에만 사용합니다.

Stage 6.2 global 거리 CSV는 frozen static 비교 baseline으로만 재사용합니다. Timestamp<t 원칙에 맞는 **causal global Train reference**를 별도로 계산합니다. Static baseline이 과거 데이터였다는 가정은 하지 않으며, future timestamp reference가 허용되는 deployable 결과로 제시하지 않습니다. Hybrid는 causal global과 combined personal 거리의 min으로 사전 정의한 보조 분석입니다.

Reference별 N=10/25/50/100 past points와 duration=5/15/30min 후보를 모두 보고하며 Test에서 하나를 선택하지 않습니다. Distance cutoff로 support를 정의하지 않습니다. Coverage 반경은 10/25/50/100/200/500m, high distance=route-like score 방향을 고정합니다. Missing/no-history는 0 거리로 대체하지 않으며 availability와 coverage 전체 분모/available 분모를 함께 기록합니다. ROC/AP는 finite available 또는 N-supported subset만 대상으로 하고 prevalence/분모를 기록합니다.

Progression은 labelled full segment position/(length-1)로 0~1을 계산하고 early [0,1/3), middle [1/3,2/3), late [2/3,1]를 사용합니다. Onset/past-labelled count와 source matching은 audit-only이며 scorer와 support에는 입력하지 않습니다. First point와 각 bucket의 prefix distance, paired delta, prior-labelled memory count를 보고합니다. Contamination과 geometric turn/sampling/시작점의 zero-offset가 함께 영향을 줄 수 있어 bucket 감소만으로 인과성을 단정하지 않습니다.

Memory query는 Haversine EARTH_RADIUS_M=6371000m를 사용합니다. 일정 block의 과거 point만 BallTree로 구축하고 최신 과거 buffer는 직접 Haversine으로 질의해 정확한 전체 과거 nearest distance를 유지합니다. Block size는 계산 설정이며 Test로 선택하지 않습니다. Timeline, nearest reference timestamp와 index, context count/duration을 기록해 causality를 검증합니다. 미래/self reference가 들어오면 fail-fast합니다.

## 완료와 검증

Stage 6.2는 전체 **240 PASS**(51.663초)와 독립 검증 후 `e657c8cdf5e5c2379b0c43186eac56435b64aac1`로 checkpoint 게시했습니다. CLI push 인증이 없어 GitHub connector로 동일 tree `fd31fef54d93774c31526712323f9d509c715eaf`를 게시한 뒤 fetch/diff 검증으로 local/remote HEAD를 일치시켰습니다. PR/main merge는 하지 않았습니다.

Stage 6.3은 기존 240개와 신규 32개를 포함한 **272개 테스트 PASS**(52.740초)입니다. 실제 정상 46,299행·route 378행·source pair 378개·Test trajectory 19개를 분석했습니다. 전체 실행 **45.231555초**, personal reference 구성 및 두 관측 시나리오 replay/query **25.806334초**, causal global query **7.758286초**입니다. Personal 시간에는 nearest query도 포함하므로 순수 tree build 시간으로 해석하지 않습니다.

Stage 5·6.1·6.2의 **273개 보호 파일 checksum 유지**, 변화 0개입니다. Reference cross-split leakage/causality 위반도 0입니다. Personal memory의 같은 Test 사용자 과거 관측은 이번에 명시한 정보 조건이며 Train global reference나 모델 학습 데이터로 합치지 않았습니다. 모델/feature/threshold/scaler/synthetic/label/split은 변경하지 않았습니다.

## Scorer와 audit 정보 분리

Scorer 입력은 user/source identity, point index, timestamp, latitude/longitude뿐입니다. Label, source 원본의 현재/future coordinate, onset, quality flag는 입력 allowlist 밖입니다. Test model 평가에 포함된 point만 통계를 계산하되, history는 retained trajectory 전체의 finite 관측을 replay합니다. 원본 46,391관측과 각 route full copy 46,391관측을 처리했습니다. 시간상 더 늦은 원본 trajectory를 prior pool에 넣지 않았고 겹치는 prior trajectory도 query 시각 이후 point는 제외합니다.

원본 replay와 synthetic replay는 같은 과거 prior logs를 사용하되, current-prefix는 각각 실제 관측 copy 좌표를 사용합니다. 과거 anomaly를 label로 제거하지 않습니다. Prior logs가 이전 frozen 원본 관측이라는 점은 실험 가정이며 과거의 여러 synthetic anomaly가 함께 있었을 시나리오는 이번 범위 밖입니다. 따라서 현실 history가 항상 정상이라고 보증한 실험은 아닙니다.

Source-matched delta, progression 및 boundary label은 scoring 후에만 붙였습니다. N-support는 과거 관측 수만 사용하며 원본 source의 현재 위치나 current score cutoff를 사용하지 않습니다. 이는 oracle 없이 계산 가능한 support 조건이지만, 수가 많다는 사실만으로 reference가 올바른 정상 경로를 설명한다고 보증하지 않습니다.

## Reference 의미와 static baseline

- `global_train_causal`: Train 정상 좌표 중 query timestamp보다 엄격히 과거인 point만 사용.
- `prior_personal`: 같은 사용자의 더 이른 trajectory 관측 중 timestamp<t인 point.
- `causal_prefix`: 현재 copy의 earlier index AND earlier timestamp 관측.
- `combined_personal`: prior + current-prefix.
- `global_personal_min`: causal global과 combined의 min, 사전 정의 보조 분석.

Stage 6.2 static global score는 파일을 수정하지 않고 재사용했으며 ROC/AP **0.399947/0.006116**을 정확히 재현했습니다. 이 baseline은 Train timestamp가 query보다 과거인지 제한하지 않았으므로 이번의 historical-causal score와 구분합니다. Timestamp 제한을 적용한 global은 정상 median **1165.374581m**, ROC/AP **0.342083/0.005751**입니다. Stage 6.2 산출물을 재계산하거나 덮어쓰지 않았습니다.

## Context availability와 시간 span

| reference | normal_total | normal_available_count | normal_available_pct | anomaly_total | anomaly_available_count | anomaly_available_pct | normal_no_context_count | anomaly_no_context_count | normal_history_ge_300sec_pct | normal_history_ge_900sec_pct | normal_history_ge_1800sec_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_train_causal | 46299 | 46299 | 100.000000 | 378 | 378 | 100.000000 | 0 | 0 | 100.000000 | 100.000000 | 100.000000 |
| prior_personal | 46299 | 42446 | 91.678006 | 378 | 297 | 78.571429 | 3853 | 81 | 91.678006 | 91.678006 | 91.678006 |
| causal_prefix | 46299 | 46299 | 100.000000 | 378 | 378 | 100.000000 | 0 | 0 | 96.332534 | 89.328063 | 79.399123 |
| combined_personal | 46299 | 46299 | 100.000000 | 378 | 378 | 100.000000 | 0 | 0 | 98.848787 | 97.166245 | 94.885419 |
| global_personal_min | 46299 | 46299 | 100.000000 | 378 | 378 | 100.000000 | 0 | 0 | 100.000000 | 100.000000 | 100.000000 |

Prior history unavailable은 정상 **3853행(8.3220%)**, route **81행(21.4286%)**입니다. 첫 retained trajectory 4개는 `no_personal_history`로 기록했습니다. Prefix/combined availability는 평가 subset에서 100%이지만 모든 trajectory의 첫 point는 context가 없습니다. Full original/route replay 각각 19개, 합계 **38개 initial observation**은 prefix unsupported입니다. 이들은 기존 model-quality evaluation 밖이므로 평가 availability 100%가 cold-start 문제 없음을 뜻하지 않습니다.

Duration은 실제 past observations의 max timestamp−min timestamp입니다. 긴 recording gap을 연속 관측으로 해석할 수 없고 오래된 point가 유효한 반복 경로를 제공하는지도 별개입니다. 5/15/30min 후보는 선택 없이 기술적으로 보고했습니다.

사용자별 availability:

| user_id | reference | normal_available_pct | anomaly_available_pct | normal_no_context_count | anomaly_no_context_count |
| --- | --- | --- | --- | --- | --- |
| 001 | prior_personal | 94.386707 | 81.818182 | 956 | 16 |
| 001 | causal_prefix | 100.000000 | 100.000000 | 0 | 0 |
| 001 | combined_personal | 100.000000 | 100.000000 | 0 | 0 |
| 008 | prior_personal | 88.803771 | 76.344086 | 950 | 22 |
| 008 | causal_prefix | 100.000000 | 100.000000 | 0 | 0 |
| 008 | combined_personal | 100.000000 | 100.000000 | 0 | 0 |
| 013 | prior_personal | 92.336589 | 64.556962 | 1428 | 28 |
| 013 | causal_prefix | 100.000000 | 100.000000 | 0 | 0 |
| 013 | combined_personal | 100.000000 | 100.000000 | 0 | 0 |
| 017 | prior_personal | 75.849232 | 87.288136 | 519 | 15 |
| 017 | causal_prefix | 100.000000 | 100.000000 | 0 | 0 |
| 017 | combined_personal | 100.000000 | 100.000000 | 0 | 0 |

## 정상 coverage와 거리

Coverage의 전체 분모는 전체 정상 46,299개입니다. Available-row 분모 coverage는 CSV에 별도 기록했고 unsupported를 작은 거리로 채우지 않았습니다.

| reference | radius_m | normal_available_count | unsupported_count | covered_count | coverage_all_pct | coverage_available_pct |
| --- | --- | --- | --- | --- | --- | --- |
| global_train_causal | 10 | 46299 | 0 | 4823 | 10.417072 | 10.417072 |
| global_train_causal | 25 | 46299 | 0 | 7262 | 15.685004 | 15.685004 |
| global_train_causal | 50 | 46299 | 0 | 8620 | 18.618113 | 18.618113 |
| global_train_causal | 100 | 46299 | 0 | 10309 | 22.266140 | 22.266140 |
| global_train_causal | 200 | 46299 | 0 | 12642 | 27.305125 | 27.305125 |
| global_train_causal | 500 | 46299 | 0 | 16371 | 35.359295 | 35.359295 |
| prior_personal | 10 | 42446 | 3853 | 4446 | 9.602799 | 10.474485 |
| prior_personal | 25 | 42446 | 3853 | 6677 | 14.421478 | 15.730575 |
| prior_personal | 50 | 42446 | 3853 | 8525 | 18.412925 | 20.084342 |
| prior_personal | 100 | 42446 | 3853 | 10047 | 21.700253 | 23.670075 |
| prior_personal | 200 | 42446 | 3853 | 12196 | 26.341822 | 28.732978 |
| prior_personal | 500 | 42446 | 3853 | 15176 | 32.778246 | 35.753663 |
| causal_prefix | 10 | 46299 | 0 | 29333 | 63.355580 | 63.355580 |
| causal_prefix | 25 | 46299 | 0 | 41668 | 89.997624 | 89.997624 |
| causal_prefix | 50 | 46299 | 0 | 46244 | 99.881207 | 99.881207 |
| causal_prefix | 100 | 46299 | 0 | 46270 | 99.937364 | 99.937364 |
| causal_prefix | 200 | 46299 | 0 | 46282 | 99.963282 | 99.963282 |
| causal_prefix | 500 | 46299 | 0 | 46295 | 99.991361 | 99.991361 |
| combined_personal | 10 | 46299 | 0 | 30841 | 66.612670 | 66.612670 |
| combined_personal | 25 | 46299 | 0 | 41684 | 90.032182 | 90.032182 |
| combined_personal | 50 | 46299 | 0 | 46252 | 99.898486 | 99.898486 |
| combined_personal | 100 | 46299 | 0 | 46276 | 99.950323 | 99.950323 |
| combined_personal | 200 | 46299 | 0 | 46286 | 99.971922 | 99.971922 |
| combined_personal | 500 | 46299 | 0 | 46295 | 99.991361 | 99.991361 |
| global_personal_min | 10 | 46299 | 0 | 32525 | 70.249897 | 70.249897 |
| global_personal_min | 25 | 46299 | 0 | 41702 | 90.071060 | 90.071060 |
| global_personal_min | 50 | 46299 | 0 | 46266 | 99.928724 | 99.928724 |
| global_personal_min | 100 | 46299 | 0 | 46283 | 99.965442 | 99.965442 |
| global_personal_min | 200 | 46299 | 0 | 46289 | 99.978401 | 99.978401 |
| global_personal_min | 500 | 46299 | 0 | 46296 | 99.993520 | 99.993520 |

정상 거리와 route 분리 지표(available subset):

| reference | normal_count | anomaly_count | normal_median | normal_p90 | normal_p95 | route_median | roc_auc | average_precision | prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_train_causal | 46299 | 378 | 1165.374581 | 69719.252843 | 98183.179938 | 194.140989 | 0.342083 | 0.005751 | 0.008098 |
| prior_personal | 42446 | 297 | 1808.156658 | 75546.466186 | 135512.715932 | 340.663267 | 0.349816 | 0.004962 | 0.006949 |
| causal_prefix | 46299 | 378 | 6.890658 | 25.017338 | 30.083957 | 17.385259 | 0.762807 | 0.073845 | 0.008098 |
| combined_personal | 46299 | 378 | 6.309197 | 24.926927 | 30.076761 | 17.162322 | 0.768776 | 0.077439 | 0.008098 |
| global_personal_min | 46299 | 378 | 5.803436 | 24.742029 | 30.066796 | 15.163742 | 0.755839 | 0.068080 | 0.008098 |

Prefix/combined의 coverage@50m는 **99.8812/99.8985%**로 causal global **18.6181%**, static global **31.8970%**보다 높습니다. 그러나 prior만의 @50m는 전체 분모 **18.4129%**(available 분모 약 20.08%)로 좋지 않습니다. Personal history 보유율과 공간 반복 경로 support를 동일하게 볼 수 없습니다.

Prefix/combined ROC/AP **0.762807/0.073845**, **0.768776/0.077439**는 full normal/route 모두 available한 조건입니다. Prior ROC/AP **0.349816/0.004962**는 정상 42,446개와 anomaly 297개의 다른 subset이므로 직접적인 모델 우열 비교가 아닙니다. Hybrid ROC **0.755839**도 combined보다 높지 않아 단순 min 결합이 항상 개선을 뜻하지 않습니다.

## Prefix가 무엇을 설명하는가

Prefix nearest가 바로 직전 point인 비율은 정상 **82.9824%**, route **79.1005%**입니다. Prefix distance가 기존 `distance_m` feature와 수치적으로 같은 비율은 정상 **87.1142%**, route **79.1005%**이고, Spearman은 **0.962725/0.736934**입니다.

높은 prefix coverage의 상당 부분은 가까운 직전 관측으로 설명됩니다. 현재 point가 smooth하게 이동하면 그 직전 point도 함께 이동하므로 nearest-prefix 거리만으로 habitual route에서의 지속적 이탈을 식별했다고 할 수 없습니다. 특히 전체 prefix와 current point를 함께 longitude 회전하면 spherical distance가 유지되는 공간 비식별성도 남습니다. 새 spatial 정보와 기존 이동 거리 신호의 재표현을 구분하는 후속 검증이 필요합니다.

## Source-matched delta (audit only)

각 시나리오의 관측 history를 사용한 synthetic−original 점수 delta입니다. Prefix/combined는 reference 좌표 자체가 달라지므로 동일 fixed set에서의 Stage 6.2 delta와 의미가 다릅니다.

| reference | count | mean | median | p25 | p75 | p90 | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_train_causal | 378 | 24.499938 | 23.553010 | -5.044235 | 76.177387 | 117.245197 | 152.550853 | -150.896189 | 227.434710 |
| prior_personal | 297 | 54.100636 | 40.265844 | -14.238715 | 127.889093 | 187.233264 | 218.574049 | -177.770033 | 267.928684 |
| causal_prefix | 378 | 11.520268 | 8.200920 | 1.715776 | 17.541360 | 29.908202 | 36.920307 | -8.617219 | 54.991425 |
| combined_personal | 378 | 11.897872 | 8.507646 | 2.061501 | 17.636522 | 31.546130 | 37.035833 | -8.463340 | 54.991425 |
| global_personal_min | 378 | 12.174759 | 10.156745 | 2.574365 | 17.643019 | 29.290120 | 37.220289 | -10.794067 | 55.702611 |

Prior paired delta 297개는 history available pair만 포함합니다. Missing prior history 81개는 0으로 대체하지 않았습니다. Source-matched 결과는 실제 gate로 사용하지 않았습니다.

## N-support와 inference-observable abstention

모든 사전 N 후보를 보고했습니다. Current distance가 크다는 이유로 abstain하지 않습니다.

| reference | minimum_context_points | normal_supported_count | anomaly_supported_count | normal_evaluable_pct | anomaly_evaluable_pct | normal_insufficient_context_count | anomaly_insufficient_context_count | normal_median | anomaly_median | roc_auc | average_precision | prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_train_causal | 10 | 46299 | 378 | 100.000000 | 100.000000 | 0 | 0 | 1165.374581 | 194.140989 | 0.342083 | 0.005751 | 0.008098 |
| global_train_causal | 25 | 46299 | 378 | 100.000000 | 100.000000 | 0 | 0 | 1165.374581 | 194.140989 | 0.342083 | 0.005751 | 0.008098 |
| global_train_causal | 50 | 46299 | 378 | 100.000000 | 100.000000 | 0 | 0 | 1165.374581 | 194.140989 | 0.342083 | 0.005751 | 0.008098 |
| global_train_causal | 100 | 46299 | 378 | 100.000000 | 100.000000 | 0 | 0 | 1165.374581 | 194.140989 | 0.342083 | 0.005751 | 0.008098 |
| prior_personal | 10 | 42446 | 297 | 91.678006 | 78.571429 | 0 | 0 | 1808.156658 | 340.663267 | 0.349816 | 0.004962 | 0.006949 |
| prior_personal | 25 | 42446 | 297 | 91.678006 | 78.571429 | 0 | 0 | 1808.156658 | 340.663267 | 0.349816 | 0.004962 | 0.006949 |
| prior_personal | 50 | 42446 | 297 | 91.678006 | 78.571429 | 0 | 0 | 1808.156658 | 340.663267 | 0.349816 | 0.004962 | 0.006949 |
| prior_personal | 100 | 42446 | 297 | 91.678006 | 78.571429 | 0 | 0 | 1808.156658 | 340.663267 | 0.349816 | 0.004962 | 0.006949 |
| causal_prefix | 10 | 46129 | 378 | 99.632821 | 100.000000 | 170 | 0 | 6.891976 | 17.385259 | 0.762681 | 0.077292 | 0.008128 |
| causal_prefix | 25 | 45844 | 378 | 99.017257 | 100.000000 | 455 | 0 | 6.895662 | 17.385259 | 0.762232 | 0.077325 | 0.008178 |
| causal_prefix | 50 | 45370 | 363 | 97.993477 | 96.031746 | 929 | 15 | 6.892410 | 16.682320 | 0.753959 | 0.064907 | 0.007937 |
| causal_prefix | 100 | 44421 | 363 | 95.943757 | 96.031746 | 1878 | 15 | 6.832385 | 16.682320 | 0.752983 | 0.065788 | 0.008106 |
| combined_personal | 10 | 46263 | 378 | 99.922245 | 100.000000 | 36 | 0 | 6.309197 | 17.162322 | 0.768765 | 0.078494 | 0.008104 |
| combined_personal | 25 | 46203 | 378 | 99.792652 | 100.000000 | 96 | 0 | 6.305136 | 17.162322 | 0.768806 | 0.078510 | 0.008115 |
| combined_personal | 50 | 46103 | 363 | 99.576665 | 96.031746 | 196 | 15 | 6.292554 | 16.598598 | 0.761425 | 0.064992 | 0.007812 |
| combined_personal | 100 | 45903 | 363 | 99.144690 | 96.031746 | 396 | 15 | 6.256921 | 16.598598 | 0.761449 | 0.065825 | 0.007846 |
| global_personal_min | 10 | 46299 | 378 | 100.000000 | 100.000000 | 0 | 0 | 5.803436 | 15.163742 | 0.755839 | 0.068080 | 0.008098 |
| global_personal_min | 25 | 46299 | 378 | 100.000000 | 100.000000 | 0 | 0 | 5.803436 | 15.163742 | 0.755839 | 0.068080 | 0.008098 |
| global_personal_min | 50 | 46299 | 378 | 100.000000 | 100.000000 | 0 | 0 | 5.803436 | 15.163742 | 0.755839 | 0.068080 | 0.008098 |
| global_personal_min | 100 | 46299 | 378 | 100.000000 | 100.000000 | 0 | 0 | 5.803436 | 15.163742 | 0.755839 | 0.068080 | 0.008098 |

Prefix N=10→100에서 normal evaluable은 **99.6328→95.9438%**, anomaly는 **100→96.0317%**입니다. Combined는 **99.9222→99.1447%**, anomaly **100→96.0317%**입니다. ROC 변화는 작고 N을 늘리는 것만으로 representation이 크게 개선됐다는 증거는 없습니다. Global/hybrid는 global past count가 커서 모든 N을 통과하며 personal context가 충분하다는 뜻은 아닙니다. Prior는 available history가 이미 100개를 넘어서 N 결과가 동일했습니다.

N availability support는 oracle 없이 가능한 warm-up/abstention 후보지만 geographic support, context 신뢰성, contamination 또는 route suitability를 판정하지 않습니다. 이번에는 N 선택·decision threshold 최적화·최종 selective detector를 만들지 않았습니다.

## Anomaly progression과 contamination

Bucket은 full labelled segment 기준이며 early/middle/late 평가 point 수는 **127/118/133**입니다. 이 label 기반 partition은 사후 audit에만 사용합니다.

| progression_bucket | reference | point_count | available_count | distance_median | distance_p90 | source_matched_delta_median | past_labelled_context_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early | prior_personal | 127 | 100 | 252.332367 | 3022.842206 | 28.504036 | 3.000000 |
| early | causal_prefix | 127 | 127 | 19.074483 | 42.519397 | 10.666046 | 3.000000 |
| early | combined_personal | 127 | 127 | 19.074483 | 42.519397 | 10.666046 | 3.000000 |
| middle | prior_personal | 118 | 93 | 403.926207 | 3602.658370 | 123.030334 | 10.000000 |
| middle | causal_prefix | 118 | 118 | 12.507359 | 23.046153 | 3.282350 | 10.000000 |
| middle | combined_personal | 118 | 118 | 12.507359 | 23.046153 | 4.315589 | 10.000000 |
| late | prior_personal | 133 | 104 | 360.055596 | 2953.803014 | 22.534399 | 17.000000 |
| late | causal_prefix | 133 | 133 | 21.522236 | 38.543177 | 10.800312 | 17.000000 |
| late | combined_personal | 133 | 133 | 21.088857 | 37.606778 | 10.800312 | 17.000000 |

Prefix median은 **19.074483→12.507359→21.522236m**, paired delta median은 **10.666046→3.282350→10.800312m**입니다. Early→middle 감소는 관찰됐지만 late에서 다시 증가하므로 '후반으로 갈수록 score가 감소'하는 Case H는 충족하지 않습니다.

Past-labelled prefix count median은 **3→10→17**로 증가했고 nearest prefix가 과거 labelled point였던 비율은 **62.9921→99.1525→87.2180%**입니다. 따라서 reference memory에 anomalous observations가 누적된다는 사실은 확인됐습니다. 그러나 sinusoidal 주입 형태, sampling/turn geometry, zero-offset endpoints도 score에 영향을 주므로 감소 원인을 contamination 하나로 단정할 수 없습니다. Label-oracle clean memory를 deployable comparator로 사용하지 않았습니다.

첫 segment point 19개는 원본과 동일 좌표/feature/past prefix이므로 모든 available reference의 paired delta가 0입니다. First-point CSV와 full bucket CSV를 별도로 보존했습니다.

## Boundary point의 context delta

Zero-displacement 38행:

- Causal global: 38개 모두 delta=0.
- Prior: available 30개 모두 delta=0, 나머지 8개 unsupported.
- Prefix: **23개 delta=0**, **15개 nonzero**; mean **7.142882m**, max **54.991425m**.
- Combined: mean **6.170633m**, max **54.991425m**.

Current endpoint 좌표가 같아도 이전 synthetic prefix는 원본 prefix와 달라 history-dependent score가 달라질 수 있습니다. 이는 현재 좌표 displacement를 새로 발견했다는 의미가 아닙니다.

Identical-feature 19행은 모든 available reference의 delta가 **0m**입니다. Prior는 15개 available, 4개 unsupported입니다. Label은 변경하지 않았습니다.

## 사용자별 결과

Reference별 availability, 정상 median/p90/p95, route median, ROC/AP:

| user_id | reference | normal_available_pct | anomaly_available_pct | normal_median | normal_p90 | normal_p95 | route_median | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | global_train_causal | 100.000000 | 100.000000 | 902.581981 | 9777.965230 | 10541.091590 | 671.062471 | 0.410856 | 0.004163 |
| 001 | prior_personal | 94.386707 | 81.818182 | 2616.312785 | 11608.004155 | 12484.173594 | 167.375409 | 0.297932 | 0.003103 |
| 001 | causal_prefix | 100.000000 | 100.000000 | 5.743501 | 14.577976 | 17.758948 | 17.860927 | 0.835263 | 0.188786 |
| 001 | combined_personal | 100.000000 | 100.000000 | 5.035384 | 13.690153 | 16.978004 | 17.501275 | 0.843919 | 0.205365 |
| 001 | global_personal_min | 100.000000 | 100.000000 | 4.563767 | 12.686514 | 15.340700 | 16.600980 | 0.854973 | 0.213041 |
| 008 | global_train_causal | 100.000000 | 100.000000 | 49.808213 | 1014.416309 | 1731.410662 | 81.872684 | 0.542753 | 0.010872 |
| 008 | prior_personal | 88.803771 | 76.344086 | 415.483451 | 2795.384074 | 3888.268070 | 478.735497 | 0.619752 | 0.012512 |
| 008 | causal_prefix | 100.000000 | 100.000000 | 5.996518 | 15.794878 | 19.601364 | 18.697079 | 0.826789 | 0.195319 |
| 008 | combined_personal | 100.000000 | 100.000000 | 5.712322 | 14.439309 | 17.714742 | 18.697079 | 0.839343 | 0.218237 |
| 008 | global_personal_min | 100.000000 | 100.000000 | 5.273040 | 12.700707 | 16.332618 | 15.963478 | 0.829368 | 0.179013 |
| 013 | global_train_causal | 100.000000 | 100.000000 | 6556.872893 | 98265.458967 | 101265.510134 | 1771.431354 | 0.437707 | 0.004156 |
| 013 | prior_personal | 92.336589 | 64.556962 | 4701.317316 | 146581.274314 | 152145.458827 | 225.217933 | 0.215701 | 0.001880 |
| 013 | causal_prefix | 100.000000 | 100.000000 | 8.011174 | 30.336843 | 30.827832 | 14.268417 | 0.576668 | 0.024101 |
| 013 | combined_personal | 100.000000 | 100.000000 | 7.928687 | 30.336374 | 30.826926 | 14.268417 | 0.577953 | 0.025183 |
| 013 | global_personal_min | 100.000000 | 100.000000 | 7.927353 | 30.335791 | 30.826031 | 14.268417 | 0.578138 | 0.029173 |
| 017 | global_train_causal | 100.000000 | 100.000000 | 144.480301 | 811.130515 | 954.344961 | 83.147925 | 0.458246 | 0.045273 |
| 017 | prior_personal | 75.849232 | 87.288136 | 308.130557 | 5131.100915 | 5323.423062 | 394.987278 | 0.580886 | 0.065231 |
| 017 | causal_prefix | 100.000000 | 100.000000 | 10.567799 | 14.640449 | 16.679616 | 17.661047 | 0.840927 | 0.323333 |
| 017 | combined_personal | 100.000000 | 100.000000 | 9.869800 | 14.388191 | 16.500517 | 17.478414 | 0.853564 | 0.328811 |
| 017 | global_personal_min | 100.000000 | 100.000000 | 7.458045 | 12.477711 | 14.129089 | 14.581521 | 0.830520 | 0.336124 |

사용자별 coverage@25/50/100/200m:

| user_id | reference | coverage_25m_pct | coverage_50m_pct | coverage_100m_pct | coverage_200m_pct |
| --- | --- | --- | --- | --- | --- |
| 001 | causal_prefix | 99.718161 | 99.917797 | 99.964770 | 99.970642 |
| 001 | combined_personal | 99.771006 | 99.935412 | 99.970642 | 99.976513 |
| 001 | global_personal_min | 99.788621 | 99.953027 | 99.976513 | 99.982385 |
| 001 | global_train_causal | 16.287946 | 19.258998 | 23.650989 | 29.716400 |
| 001 | prior_personal | 20.433327 | 24.402560 | 26.275615 | 29.211438 |
| 008 | causal_prefix | 99.622864 | 99.858574 | 99.929287 | 99.976429 |
| 008 | combined_personal | 99.670006 | 99.882145 | 99.964643 | 99.988214 |
| 008 | global_personal_min | 99.717148 | 99.917501 | 99.976429 | 99.988214 |
| 008 | global_train_causal | 41.649971 | 50.029464 | 57.560401 | 67.613435 |
| 008 | prior_personal | 20.659988 | 26.104891 | 30.642310 | 35.627578 |
| 013 | causal_prefix | 75.711066 | 99.935602 | 99.957068 | 99.967801 |
| 013 | combined_personal | 75.721799 | 99.946335 | 99.967801 | 99.978534 |
| 013 | global_personal_min | 75.743265 | 99.957068 | 99.973167 | 99.978534 |
| 013 | global_train_causal | 0.166363 | 0.482988 | 1.889020 | 3.917570 |
| 013 | prior_personal | 5.049909 | 8.747451 | 12.970913 | 19.174627 |
| 017 | causal_prefix | 98.836668 | 99.208934 | 99.581201 | 99.813867 |
| 017 | combined_personal | 98.883201 | 99.255468 | 99.581201 | 99.813867 |
| 017 | global_personal_min | 99.208934 | 99.534667 | 99.767334 | 99.906933 |
| 017 | global_train_causal | 42.950209 | 46.765938 | 48.627268 | 51.838064 |
| 017 | prior_personal | 23.406235 | 24.383434 | 25.825966 | 29.083295 |

Combined ROC-AUC는 user 001/008/013/017에서 **0.843919/0.839343/0.577953/0.853564**입니다. User 013은 coverage가 높아도 분리력이 낮아 high coverage와 high separation을 동일시할 수 없습니다. 사용자별 prevalence도 달라 AP의 절대 비교는 주의가 필요합니다. 모든 N의 사용자 support rate/ROC/AP는 `support_analysis.csv`에 저장했습니다.

## Prior trajectory와 cold start 기록

| user_id | source_trajectory_id | trajectory_order | previous_trajectory_count | previous_history_logged_points | past_history_points_at_start | past_history_duration_at_start_sec | history_status_at_start |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | 001__20081023055305 | 0 | 0 | 0 | 0 | 0.000000 | no_personal_history |
| 001 | 001__20081023234104 | 1 | 1 | 961 | 961 | 22283.000000 | available |
| 001 | 001__20081024234405 | 2 | 2 | 3089 | 3089 | 88965.000000 | available |
| 001 | 001__20081025231428 | 3 | 3 | 10164 | 10164 | 193016.000000 | available |
| 001 | 001__20081026081229 | 4 | 4 | 13840 | 13840 | 263710.000000 | available |
| 008 | 008__20081024114834 | 0 | 0 | 0 | 0 | 0.000000 | no_personal_history |
| 008 | 008__20081024132624 | 1 | 1 | 955 | 955 | 5867.000000 | available |
| 008 | 008__20081025041134 | 2 | 2 | 2053 | 2053 | 8130.000000 | available |
| 008 | 008__20081025225205 | 3 | 3 | 5516 | 5516 | 86316.000000 | available |
| 008 | 008__20081026055934 | 4 | 4 | 6215 | 6215 | 127195.000000 | available |
| 013 | 013__20080927120819 | 0 | 0 | 0 | 0 | 0.000000 | no_personal_history |
| 013 | 013__20080927233805 | 1 | 1 | 1430 | 1430 | 2570.000000 | available |
| 013 | 013__20080929001436 | 2 | 2 | 14896 | 14896 | 78127.000000 | available |
| 013 | 013__20080929090206 | 3 | 3 | 16165 | 16165 | 139508.000000 | available |
| 017 | 017__20081030092748 | 0 | 0 | 0 | 0 | 0.000000 | no_personal_history |
| 017 | 017__20081030112722 | 1 | 1 | 521 | 521 | 4256.000000 | available |
| 017 | 017__20081030234453 | 2 | 2 | 891 | 891 | 11931.000000 | available |
| 017 | 017__20081031091917 | 3 | 3 | 1130 | 1130 | 52372.000000 | available |
| 017 | 017__20081101020307 | 4 | 4 | 1519 | 1519 | 87343.000000 | available |

순서는 시작 timestamp, 동률이면 source ID지만 동일 사용자 동일 시작 시각은 ambiguous order로 실패합니다. Prior index에는 현재/future trajectory가 없으며 이전 trajectory가 query 시각을 넘어 지속되면 그 future portion은 제외합니다. 삭제를 통해 first trajectory나 unsupported point를 숨기지 않았습니다.

## Case E/F/G/H/I 판정

- **E의 제한적 신호:** prefix/combined가 정상 coverage와 ROC/AP를 global보다 개선했습니다. 다만 nearest-prefix가 기존 step distance를 많이 재표현하므로 개인 정상 route가 식별됐다고 결론내리지 않습니다.
- **F의 일부 evidence:** 특히 user 013은 높은 prefix coverage에도 ROC≈0.58입니다. 단순 nearest context 이상의 shape/sequence 정보가 필요할 가능성은 있으나 모델 대안을 검증하지 않았습니다.
- **G는 cold-start subset에 해당:** prior history 없는 정상 8.32%, anomaly 21.43%와 초기 prefix warm-up이 존재합니다. 전체 personal availability가 낮다고 일반화하지 않습니다.
- **H의 단조 late 감소는 미확인:** past anomaly memory 누적은 확인됐지만 late score가 다시 증가했습니다. Contamination의 causal 기여량은 미검증입니다.
- **I는 확인:** 사용자별 분리력 차이가 큽니다. 일률적 representation/policy의 유효성을 보장하지 않습니다.

## Stage 6.4 제안 — B를 우선 (구현하지 않음)

**B: Trajectory/window similarity representation의 식별성 audit**를 우선 제안합니다. Prefix 거리의 87.1%가 기존 정상 step-distance와 같고 사용자별 분리 차이가 남으므로, 먼저 관측 가능한 shape/context가 기존 8 feature보다 추가 정보를 제공하는지 검증해야 합니다. 이는 Window 모델 학습을 즉시 시작하자는 결론이 아닙니다. Future/label/source oracle를 제외한 시나리오와 별도 holdout 계약을 먼저 고정하는 설계 단계가 필요합니다.

A personalized distance feature 추가는 아직 habitual-route validity를 확인하지 못해 보류합니다. C memory contamination control은 실제 anomaly observation이 memory에 들어가므로 후보지만 H의 단조 감소가 확인되지 않아 주요 원인으로 확정하지 않습니다. D warm-up/abstention은 서비스 정보 조건에 필요하지만 N만으로 geographic support를 보장하지 않습니다. E label refinement는 19개 동일-point boundary 문제를 별도로 다룰 가치가 있습니다. 어떤 Stage 6.4 코드나 새 모델도 구현하지 않았습니다.

Personal Test history를 쓰는 것은 Stage 5/6.2 cold-start Train-only 평가보다 더 많은 정보를 허용한 조건입니다. 이 차이를 모델 성능 개선이나 architecture 우승으로 해석하지 않습니다. 관측 로그가 chronological하다는 사실과 행동 anomaly의 인과적 식별은 별개입니다.

## 재현과 산출물

코드: `src/audit_context_reference.py`; 테스트: `tests/test_context_reference.py`.

```powershell
python -m unittest discover -s tests -v
python -m src.audit_context_reference
```

기존 출력 directory가 있으면 덮어쓰지 않고 실패합니다. `outputs/metrics/stage63_protected_snapshot.json`으로 Stage 5/6.1/6.2 root를 검증합니다. 신규 테스트는 past-only, no-self, future/current trajectory 차단, user role, timestamp/index order, overlapping prior, no-history/warm-up, label/source mutation invariance, support/coverage/user metrics, boundary, brute-force buffer/tree, immutable outputs와 금지 모델 호출을 검증합니다.

Metrics: `outputs/metrics/stage6/context_reference/`에 필수 8 CSV 및 `context_reference_audit.json`, trajectory history, boundary, first-point, paired delta summary, verification, 독립 final checks, signal diagnostics를 저장했습니다.

Figures: `outputs/figures/stage6/context_reference/`:

1. `coverage_by_reference_type.png`
2. `normal_vs_route_context_distance.png`
3. `context_availability_by_user.png`
4. `roc_auc_by_reference_type.png`
5. `support_vs_separation.png`
6. `anomaly_progression_distance.png`
7. `per_user_context_coverage.png`

독립 검증은 normal/synthetic 각 32개, 총 **64개 query**의 전체 허용 pool을 raw frozen 로그에서 재구성해 count/duration/nearest distance를 brute-force로 확인했습니다. 최대 거리 오차 **2.91e-11m**입니다. 모든 saved nearest timestamp<t, prefix index<current, 모든 support/user ROC/AP와 coverage 및 paired delta, output hash와 273개 보호 checksum도 검증했습니다. 로그는 `outputs/metrics/stage63_tests.log`, `outputs/metrics/stage63_experiment.log`입니다.

Stage 6.3은 미커밋/미푸시입니다. HEAD는 Stage 6.2 checkpoint 그대로이며 PR/main merge하지 않았습니다.
