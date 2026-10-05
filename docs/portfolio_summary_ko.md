# PathGuard_Ai 연구 포트폴리오 요약

## 프로젝트 배경과 연구 질문

GPS 위치 기록은 사람이 언제 어디서 어떻게 이동했는지를 담지만, 일반적인 이동과 비정상 이동을 구분하는 일은 단순한 속도 임계값만으로 해결하기 어렵다. 같은 속도나 회전도 사용자, 교통수단, 지역과 이동 목적에 따라 의미가 달라진다. 정상 이동을 학습한 모델이 새로운 사용자를 만났을 때도 유효한지, 한 지점에서 이상을 예측한 결과가 실제 알림으로 연결될 때 어떤 비용이 생기는지를 확인하고자 PathGuard_Ai를 구성했다.

처음에는 GPS movement feature와 Autoencoder reconstruction error로 이상을 탐지하는 파이프라인을 만드는 데서 출발했다. 이후 사용자 분리 평가에서 낮은 성능과 큰 변동성이 드러나면서 연구 질문을 “모델을 구현할 수 있는가”에서 “어떤 표현과 평가 조건에서 일반화가 실패하며, 그 실패를 어떻게 설명할 수 있는가”로 구체화했다. 프로젝트는 데이터 처리부터 탐색 실험, 원인 분석, 새 사용자에 대한 사전 동결 최종 검증까지 Stage1–7로 진행했다.

이 문서는 구현한 기능 목록보다 실험 설계와 의사결정, 실패한 가설을 확인한 과정을 설명한다. 사용한 synthetic anomaly는 실제 범죄나 실종 행동의 label이 아니며, 실제 위험 탐지 서비스의 정확도나 안전성을 입증한 것으로 해석하지 않는다.

## 수행 역할과 기술 스택

프로젝트 수행 범위는 GPS ingestion/시각화, feature engineering, 품질 검사와 exact trajectory deduplication, synthetic anomaly 생성 및 lineage 관리, baseline과 PyTorch Autoencoder 구현, user split과 multi-seed 실험, causal route/context 분석, event/alert 평가, 최종 protocol 동결, 테스트와 재현 문서 작성이다. Git/GitHub의 Stage 단위 branch·PR·squash merge로 연구 코드와 결과 해석의 변경을 관리했다.

실제 사용한 기술은 Python, Pandas, NumPy, scikit-learn, PyTorch, Folium, Matplotlib, Numba다. 데이터 처리와 metric 계산은 Pandas/NumPy/scikit-learn, Autoencoder는 PyTorch, route 시각화는 Folium, 결과 figure는 Matplotlib, window distance 계산 가속은 Numba를 사용했다. 이 프로젝트에서 사용하지 않은 배포 플랫폼이나 UI framework는 성과에 포함하지 않았다.

## 데이터 설계와 누수 방지

Microsoft GeoLife GPS trajectories를 사용하고 Stage5/6에서는 users000–019를 고정 후보로 구성했다. 사용자 기반으로 Train12명, Validation4명, Test4명으로 분리해 같은 사람의 이동 습관이 학습과 평가에 섞이는 위험을 줄였다. 품질 검사와 dedup 후 Train은51trajectory/86,067정상행, Validation은12trajectory/19,448정상행과1,113이상행, Test는19trajectory/46,299정상행과2,112이상행이다.

Exact duplicate는 fingerprint로 찾고 동일 split 안에서는 정렬상 첫 trajectory를 유지했다. 제외한 trajectory와 이유, 행 수를 manifest와 summary에 남겼으며 split 간 duplicate는 error로 처리했다. Synthetic anomaly는 정상 source의 품질 검증과 deduplication 뒤 생성했다. Original normal은 한 번 평가하고 synthetic copy의 label=1구간만 anomaly로 집계해 copied normal이 여러 번 정상 분모에 반영되지 않도록 했다.

Final cohort는 기존 사용자를 재사용하지 않고 users020–039 각각의 filename lexical first5를 사전에 고정했다. 20명 모두 원본이 존재해100trajectory가 선택됐고, 기존 quality 기준으로84개가 적격이었다. User021의 first5는 모두 품질 조건을 통과하지 못해 실제 metric을 계산할 수 없었다. 이를 더 나은 사용자나 여섯 번째 trajectory로 교체하지 않고 NA로 기록했다. 따라서 후보20명과 실제 평가 가능19명을 구분한다. 최종 정상223,289행, 이상9,263행, synthetic336samples, route84events이고 기존 사용자 overlap과 duplicate 제외, lineage error는 모두0이었다.

## 모델 구현과 사용자 일반화 문제

시간차, 거리, 속도, 가속도, 방향 변화, 정지 시간, bearing sine/cosine의8개 movement feature를 사용했다. Row Autoencoder는8→16→8→4 bottleneck encoder와4→8→16→8 decoder로 구성하고 reconstruction MSE를 anomaly score로 사용했다. Scaler는 Train에서만 fit하고, checkpoint와95percentile threshold는 Validation normal에서 정했다. Statistical Rule과 Isolation Forest도 같은 feature와 평가 의미를 사용해 비교했다.

Stage5의 canonical seed42 Autoencoder Test F1은약0.2098, AUC는0.7225로 나타났다. 모델을 구현했다는 사실과 사용자 일반화가 충분하다는 결론 사이에 차이가 있음을 확인했다. 특정 seed의 결과가 우연히 좋아졌을 가능성을 조사하기 위해 seed7/21/42/100/2026를 모두 실행했다. AE F1은0.2291±0.0474, AUC는0.7476±0.0341이었다. 이때 ±는 표본 표준편차이며, 좋은 seed만 선택하는 대신 전체 변동성을 보고했다.

Baseline 비교도 이 판단에 필요했다. 높은 모델 복잡도가 곧 좋은 탐지 성능을 뜻하지 않았고, 비교적 단순한 detector가 안정적인 F1을 보일 수 있었다. 이후 연구는 새 모델을 계속 추가하기보다 기존 표현이 놓치는 정보와 평가 단위를 분석하는 방향으로 진행했다.

## 실패한 접근과 원인 분석

첫 번째 가설은 기존 movement representation에 경로 정보가 부족하다는 것이었다. 동일한 이동을 공간적으로 평행 이동해도 movement feature가 거의 같아지는 counterexample를 확인했다. 이는 absolute route identity를 표현하지 못하는 구조적 한계를 보여준다. 또한 많은 route anomaly가 Train feature의 개별 범위 안에 있었지만, 이를 모든 feature의 joint normality로 확대 해석하지 않았다.

다음으로 Train normal GPS를 global spatial reference로 만들었다. 그러나 unseen Test normal도 학습 위치에서 멀리 떨어진 경우가 많았고 median nearest distance는약441m였다. Route detection AUC0.3999/AP0.0061은 기대에 못 미쳤다. 이 실패를 숨기지 않고 학습 사용자와 평가 사용자의 공간 coverage 차이가 reference score를 교란한다는 증거로 기록했다. 동시에 “공간 정보를 쓰면 항상 실패한다”는 일반적인 결론까지 내리지는 않았다.

Inference 시점에 이용 가능한 과거와 prefix를 이용한 causal context도 조사했다. 일부 ranking signal은 있었지만 normal에서 일정 step distance가 흔하게 나타나 개인의 habitual route를 이해했다고 말할 수 없었다. Window similarity 실험에서는 과거의 비중첩 window, 사전 고정한 길이·gap과 Discrete Fréchet distance를 사용했다. 일부 짧은 문맥 신호에도 불구하고36조합의 전체 결과는 robust route detection을 해결하지 못했다. 결과를 본 뒤 Final feature나 window를 선택해 끼워 넣지 않았다.

Label 문제도 별도로 점검했다. Synthetic route deviation의 sine boundary에서 변위가0에 가까워 label=1이어도 movement 변화가 거의 없을 수 있었다. Evidence tier를 나눠 경계를 분석했으나 가장 약한 tier를 제외해도 recall 상승은약0.68–1.02percentage points에 그쳤다. 경계 label의 불명확성이 존재하지만, 낮은 route recall 대부분의 설명은 아니라는 결론이었다. 분석을 위해 공식 label을 수정하지 않았다.

## Point·event·notification의 구분

Point recall이 낮은데도 event detection이 높을 수 있다는 사실이 중요한 전환점이었다. 한 anomaly segment에서 한 점이라도 positive가 나오면 any-point event는 detected로 집계된다. Stage6에서 낮은 route point recall과약90%수준의 event detection이 동시에 나타났고, event coverage는 낮아 sparse spike 하나가 결과를 이끌 수 있었다. 높은 EDR을 지속적 이상 인식이나 실제 알림 precision과 혼동하지 않도록 평가 단위를 나눴다.

원본 정상 trajectory에서 발생하는 false alert도 함께 계산했다. 전체 정상 trajectory에 raw alert가 발생하는 조건에서60초 cooldown은 알림 수를 줄일 수 있었지만, event onset 이전의 false notification이 실제 event notification을 억제하거나 지연시키기도 했다. Validation에서 선택한 G0_C60을 동결하고 detection loss, early detection, delay, notification burden을 동시에 평가했다. Alert 수 감소만을 개선으로 소개하지 않았다.

## 사전 동결한 최종 검증

기존 Stage5 Test와 Stage6 결과는 이미 여러 번 관찰한 exploratory evidence다. Final에서는 protocol,8 features, generator/config/seed,11개 model/scaler/threshold, G0_C60을 cohort 처리 전에 commit하고 원격에 게시했다. 그 뒤 새 사용자 cohort를 생성하고 기존 artifact를 load해 한 번 평가했다. 재학습, scaler fit, threshold 재계산, policy 재선택은 없었다.

첫 성공 결과를 manifest로 동결하고 이후 metric-producing code/config를 수정하지 않았다. Result file, source, model, dataset SHA256과 실행 시각을 남겼으며, 낮은 결과를 이유로 사용자나 trajectory를 교체하지 않았다. 이 과정은 좋은 수치를 만드는 작업보다 연구 절차와 해석의 경계를 검토 가능하게 만드는 작업이었다.

| Detector | Final pooled F1 | ROC-AUC | Raw point route EDR | Locked notification EDR |
| --- | --- | --- | --- | --- |
| Rule | 0.2841 | 0.6857 | 0.8095 | 0.5833 |
| Isolation Forest | 0.3067±0.0032 | 0.7370±0.0053 | 0.8357±0.0155 | 0.5714±0.0429 |
| Autoencoder | 0.2364±0.0453 | 0.7529±0.0486 | 0.9000±0.0707 | 0.6452±0.0330 |

IF는 F1과 seed stability에서 상대적으로 유리했고 AE는 ranking과 raw event EDR이 높았으나 낮은 precision과 큰 변동성을 보였다. Stage5보다 일부 Final metric이 높아졌지만 같은 model을 서로 다른 cohort에 적용한 차이이므로 모델 품질 향상으로 주장하지 않는다.

Cooldown은 false notification counts를 Rule49.64%, IF42.71%, AE58.26% 줄였다. 그러나 locked false notifications/1000normal은약15.48/15.50/16.77이고 정상 trajectory의 any-notification은100/100/99.76%였다. Early detection도 하락했다. 결국 새 사용자에서 일부 discriminative signal이 재현됐지만, route 표현과 운영 가능한 alert specificity의 한계도 함께 재현됐다.

## 재현성, 검증과 배운 점

전체543개 테스트 중 신규85개를 추가하고 기존458개 regression을 유지했다. Stage1–6 보호파일578개는 checksum 변화0개였다. 별도의 독립 verifier가 저장된 prediction2,558,072행, user별 결과, type별 통계, scalar event/normal alert replay와 model/protocol/result hash를 다시 확인했다. 이 검증은 새로운 scoring run이 아니라 저장된 첫 결과의 일관성 검사다.

최종7개 figure는 동결된 summaryCSV에서 생성했다. Demo는 기존 Rule/IF/AE를 load하고 CSV validation, feature engineering, score/prediction, G0_C60 notification까지 실행한다. 공개 예제는 실제 이동 기록을 복사하지 않은 인공 trajectory이며, seed42는 기존 canonical seed다. Final cohort는 데모 예제에 사용하지 않는다. README, 재현 문서, detailed report와 공개 summary/manifest를 정리해 다른 사람이 결과와 전제 조건을 함께 검토할 수 있도록 했다.

이 프로젝트에서 배운 핵심은 모델 성능과 연구 증거를 구분하는 방법이다. Baseline과 seed 비교 없이 한 숫자를 대표 결과로 삼으면 변동성과 사용자 차이를 놓칠 수 있다. Causal context를 지키지 않은 route reference는 inference 조건을 벗어날 수 있으며, point/event/notification의 서로 다른 단위를 섞으면 좋은 EDR 뒤에 false-alert 부담이 가려진다. 실패한 가설을 보존하고 label, representation, coverage와 policy의 영향을 분리하는 것이 후속 연구의 출발점이 된다.

## 한계와 향후 연구

Synthetic GPS anomaly는 실제 위험행동 ground truth가 아니고 GeoLife의 시간적·지역적 범위도 제한적이다. Final 후보20명 중19명만 적격 trajectory를 가지며, 사용자별 이동 목적과 교통수단의 차이를 충분히 설명하지 못한다. 높은 event detection으로 실제 event precision, online deployment 또는 안전성을 증명하지 않았다. Binary artifacts와 원본 data는 저장소 밖의 보존 bundle이 필요해 fresh clone만으로 전체 실험을 재현할 수 없다는 조건도 명시했다.

향후 연구 후보는 sequence/temporal representation, self-supervised embedding, per-user adaptation, road/map context, real anomaly labels, calibration, 독립 Validation을 통한 alert suppression과 더 큰 외부dataset이다. 이번 Stage7에서 새 실험을 시작하지 않고 Future Work로 기록했다. PathGuard_Ai의 최종 성과는 GPS 이상탐지 pipeline과 함께, 새로운 사용자에서 재현된 부분과 실패한 부분을 사전 동결 protocol로 구분해 설명할 수 있는 연구 기록이다.
