# Final research summary

## 연구 질문과 문제 정의

학습한 사용자의 정상 GPS movement만으로 다른 사용자의 synthetic anomaly를 탐지할 수 있는가? Route segment에 대한 point detection이 낮아도 event와 실제 notification의 의미 있는 탐지는 가능한가? 사용자 이동 다양성과 false alerts까지 고려할 때 시스템의 적용 범위는 어디인가?

PathGuard_Ai는 GPS movement feature 기반 row anomaly detector를 만들고, 관찰된 일반화 문제를 단계별 반증 가능한 분석으로 조사했다. 최종 결론의 범위는 synthetic GPS 이상과 offline alert replay다. 실제 위험·범죄·실종 행동으로 확장한 주장은 하지 않는다.

## 데이터와 baseline

Stage1–3에서 GeoLife loading/visualization, elapsed time/distance/speed/acceleration/direction/stop/bearing 표현, quality filtering 및4종 synthetic anomaly 생성을 구현했다. Stage5는 users000–019의 고정 user split을 사용했다. Dedup 후 Train12명51trajectory86,067normal, Validation4명12trajectory19,448normal/1,113anomaly, Test4명19trajectory46,299normal/2,112anomaly다. Scaler와 model fitting은 Train만, checkpoint/threshold는 Validation만 사용한다.

Statistical Rule과 Isolation Forest는 Autoencoder보다 복잡도가 낮은 비교 기준이다. 같은 features/data semantics로 baseline과 AE를 비교해 모델 복잡도만으로 성능을 정당화하지 않았다.

## Autoencoder와 unseen-user 결과

Stage4의 row AE는8→16→8→4 encoder,4→8→16→8 decoder, hidden ReLU, reconstruction MSE를 사용했다. Adam/lr0.001, batch64, max100epochs, early-stop patience15의 고정 학습 설정을 적용했다. Train-only StandardScaler와 Validation normal loss/checkpoint/95percentile threshold를 사용했다.

Stage5.2 canonical seed42 Test F1약0.2098/AUC0.7225는 사용자 분리 후 일반화 한계를 드러냈다. Stage5.3에서 seed7/21/42/100/2026를 모두 보고했으며 AE F1은0.2291±0.0474, AUC0.7476±0.0341이었다. 좋은 seed 하나를 대표 결과로 택하지 않았다. Stage5.4의 baseline 비교는 안정성과 단순 detector의 경쟁력을 함께 확인했다.

## Representation과 spatial route reference 조사

Stage6.1에서 기존8개 movement feature가 absolute location이나 개인 경로 정체성을 직접 담지 않는다는 translation counterexample를 구성했다. Route anomaly의 약81.22%가 Train marginal p01–p99 안에 있었지만, 이는 feature joint normality의 증명이 아니다.

Stage6.2에서 Train normal GPS86,067개를 global spatial BallTree reference로 사용했다. Test normal median nearest distance약441m,50mcoverage약31.9%로 unseen-user coverage가 충분하지 않았다. Route score AUC0.3999/AP0.0061은 이 고정 reference의 실패 결과다. 공간 정보 전체가 본질적으로 쓸모없다는 결론은 내리지 않았다.

## Causal context와 window similarity

Stage6.3은 inference-time에 얻을 수 있는 과거·prefix reference를 평가했다. Combined score AUC0.7688/AP0.0774의 신호가 있었지만 normal prefix에서도 약87.11%가 등간격 거리 관계를 보여 개인의 habitual route 이해를 증명하지 못했다.

Stage6.4는 고정 window sizes/gaps/reference modes의36조합을 평가했다. Discrete Fréchet distance와 stride10을 사용하고 미래나 overlap context를 금지했다. W10/G0 combined AUC약0.6862/AP0.0123과 일부 짧은 문맥 신호는 보였지만 robust route detection을 해결하지 못했다. Final에는 route/context/window model을 추가하지 않았다.

## Label identifiability

Stage6.5에서 route label378개를 boundary evidence tiers340/19/0/19개로 분류했다. Full labels379개와 quality-valid378개의 분모를 구분했다. Sine deviation의 zero-amplitude boundary가 weakly identifiable label을 만들 수 있지만 Tier4를 제외해도 recall 상승은약0.68–1.02pp에 그쳤다. 이 경계 문제만으로 낮은 route recall의 대부분을 설명하지 못했고 공식 label을 변경하지 않았다.

## Point, event, alert의 차이

Stage6.6의19route events에서 낮은 point recall(Rule19.31%, IF12.86%, AE20.79%)과 높은 any-point EDR(89.47/92.63/91.58%)이 동시에 나타났다. 많은 event가 sparse spike 하나로 탐지돼 높은 EDR이 지속적인 이상 인식과 같지 않았다. 원본 normal trajectory 모두에서 raw alert가 발생했다.

Stage6.7에서는 Validation에서 G0_C60을 고정했다. Cooldown은 알림 수를 줄였지만 일부 이벤트 알림을 놓치거나 늦췄다. 기존 Test 분석이 policy family 설계에 선행했으므로 이 결과를 exploratory로 남기고, 별도의 unseen-user Final로 확인하기로 했다.

## Confirmatory final holdout

Protocol commit3b267f7c9d6593facc4135c4992d5f2c1144d519를 cohort 처리 전에 게시했다. Users020–039 filename first5는100개, 기존quality를 통과한84개이며16개 quality제외/0개 duplicate제외다. User021의 zero eligible을 그대로 유지해 actual support19명이다. 기존 user overlap0, remaining duplicates0, lineage0.

기존 Rule1개/IF5개/AE5개와 scaler/threshold/features/generator/G0_C60을 동결해223,289normal/9,263anomaly rows를 한 번 평가했다. Normal copied portions는 중복 집계하지 않았다. First successful results를 freeze하고 metric-producing source를 수정하지 않았다.

## 최종 결과와 해석

| Detector | Pooled F1 | ROC-AUC | Raw point route EDR | Locked notification EDR |
| --- | --- | --- | --- | --- |
| Rule | 0.2841 | 0.6857 | 0.8095 | 0.5833 |
| IF | 0.3067±0.0032 | 0.7370±0.0053 | 0.8357±0.0155 | 0.5714±0.0429 |
| AE | 0.2364±0.0453 | 0.7529±0.0486 | 0.9000±0.0707 | 0.6452±0.0330 |

±는5seed sampleSD, Rule은 단일 결과다. IF는 F1과 안정성에서 유리하고 AE는 ranking/raw event EDR에서 상대적으로 높지만 precision과 seed stability는 낮다. Final–Stage5 F1 delta는Rule+0.0272, IF+0.0407, AE+0.0073이지만 동일 detector와 다른 cohort의 차이로 해석한다. IF AUC/AP는 소폭 하락했다.

Raw notification과 locked notification은 any-point event와 다른 단위다. Locked false notifications/1000normal은15.48/15.50/16.77이며 normal any-notification은100/100/99.76%다. Cooldown은 notification을49.64/42.71/58.26% 줄이는 동시에 notification EDR/Early@25도 낮췄다. 이러한 부작용과 route point recall/coverage의 한계가 새 사용자에서도 재현됐다.

## 실험 규율과 재현성

전체543tests, 신규85, 기존458regression을 유지했다. Protected578files changed0, independent verification은저장prediction2,558,072행과scalar event/normal replay를검증했다. 7개 figures는frozen summaryCSV만읽어생성했다. Demo는 canonical seed42의 기존 artifact inference이며 실제 GPS 기록을 포함하지 않는 인공 trajectory 예제를 사용한다. Final cohort는 예제에 사용하지 않는다.

실패한 접근을 삭제하지 않고 dataset/model/threshold/policy/label과 seed를 보존했다. 최종 결과가 이미 관찰된 뒤에는 새로운 cohort를 confirmatory로 가장하지 않는다.

## 한계와 향후 연구

Synthetic label은 실제 위험행동 ground truth가 아니다. GeoLife의 역사적·지역적 범위, 작은 cohort, 개인 경로 다양성, zero-eligible 사용자, 큰 정상 false-alert burden이 결과 해석을 제한한다. Event precision 및 실제 online 배포 성능은 검증하지 않았다.

Sequence encoder/self-supervised embedding, user adaptation, road/map context, real labels, calibration, 별도Validation을 둔 suppression과 큰 외부dataset은 후속 연구 후보로만 기록했다. [상세 결과](stage7_final_validation.md)와 [재현 문서](reproducibility.md), [한국어 portfolio](portfolio_summary_ko.md)에 근거를 남기고 이번 Stage1–7 연구 구현을 완료한다.
