# Stage 5.1: Multi-user Dataset Preparation

사용자 단위 GeoLife 데이터 준비 단계입니다. Stage 1~4 함수를 재사용하며 모델 학습은 실행하지 않습니다.

## 실행

```powershell
.\.venv\Scripts\python.exe -m src.prepare_multiuser_dataset
.\.venv\Scripts\python.exe -m src.prepare_multiuser_dataset --validate-existing data/processed/stage5/geolife_u000_019_first5_seed42_dedup
```

기본 선택은 사용자 `000~019`의 파일명순 첫 5개 경로입니다. 제외 경로를 대체하지 않습니다. `--users`, `--files-per-user`, `--split-seed`, `--synthetic-seed`, `--dataset-id`, `--output-root`를 지정할 수 있습니다. 기존 dataset 디렉터리는 덮어쓰지 않습니다. 재생성은 새 dataset ID를 사용합니다.

## ID와 원본 품질 연결

`user_id`는 세 자리 문자열입니다. `trajectory_id`와 `source_trajectory_id`는 `user_id__원본파일명` 형식이고 `original_trajectory_id`는 파일명을 보존합니다. `source_point_index`는 정제·정렬 후 경로별 0부터 시작합니다. 경로별 특징 계산을 호출해 사용자·경로 경계를 넘어 계산하지 않습니다.

합성은 좌표·시간 변형 후 기존 Stage 2 특징을 재계산하고 원본 point index를 유지합니다. 원본 품질 연결 키는 `(source_trajectory_id, source_point_index)`입니다. timestamp를 사용하지 않아 abnormal_speed의 시간 변경 후에도 연결이 유지됩니다. 원본 키 중복, sample 내부의 같은 원본 키 중복, 누락·존재하지 않는 원본·사용자 불일치는 거부합니다. 서로 다른 sample의 원본 키 공유는 허용합니다.

`source_quality_valid`는 원본의 `is_low_quality == False` 및 `is_training_eligible == True`입니다. 첫 행은 특징 초기값이므로 False입니다. 합성의 품질 플래그·reason은 원본 측정 품질입니다. `synthetic_value_valid`는 생성 결과의 유한한 특징·좌표 범위·비음수 거리/속도 등 수치 유효성입니다. 원본에도 공통 스키마를 위해 True를 저장합니다. 합성 속도가 크다는 이유로 원본 품질을 다시 판정하지 않습니다.

정렬 전 timestamp 역전·중복·유효하지 않은 좌표/시간을 입력 manifest에 집계합니다. Stage 1 정제와 Stage 3 품질 기준은 변경하지 않습니다.

## 저장 파일

`data/processed/stage5/<dataset_id>/` 아래에 저장하며 Git에서 제외합니다.

- `input_manifest.csv`: 입력 파일, SHA256, 원본/정제 행 수, 시간 역전·중복, fingerprint, 품질 적격 여부와 중복 제외 후 적격 여부. split, 중복 여부, 유지 경로, 제외 사유와 제외 행 수도 기록합니다.
- `user_quality_summary.csv`, `trajectory_quality_summary.csv`: 사용자·경로별 품질과 제외 사유. 경로별 manifest에는 fingerprint, split, 유지/제외 ID와 제외 행 수를 보존합니다.
- `user_split_manifest.csv`: seed로 생성한 사용자 분할.
- `synthetic_anomaly_manifest.csv`: 원본·유형·sample별 seed와 이상 구간.
- `quality_checked.csv`: 부적격 경로를 포함한 전체 품질 검사 결과.
- `source_normal.csv`: 적격 원본 전체와 lineage·품질 metadata.
- `train.csv`: 적격 원본 중 유효 정상 행만.
- `validation.csv`, `test.csv`: 감사·원본 연결 검증용 전체 원본 + 전체 합성 복사본.
- `validation_evaluation.csv`, `test_evaluation.csv`: 유효 원본 정상 + 원본 품질 유효 합성 label=1 구간. 합성 정상 복사 구간 제외.
- `leakage_report.json`, `dataset_summary.json`: 누수·lineage·fingerprint·CSV 검증, 중복 처리 내역, 처리 후 규모·시간·환경·checksum.
- `duplicate_report.json`: 중복 fingerprint, 사용자/경로 ID, split, 유지/제외 경로, 제외 사유, 정제 행 수 및 모델용 정상 행 감소분.

CSV에 소수초를 보존합니다. `read_dataset_csv()`로 ID·boolean·혼합 정밀도 timestamp를 읽습니다. 저장 후 ID·lineage·timestamp 및 전체 검증을 반복합니다. `--validate-existing`는 checksum, split seed, lineage, 평가용 CSV와 deterministic 중복 제외 기록을 읽기 전용으로 재검증합니다.

Stage 5.2는 명시적 lineage와 평가용 CSV를 사용해야 합니다. Stage 4의 위치 기반 `_attach_source_quality()`를 평가용 CSV에 그대로 적용하면 안 됩니다. 기존 8개 모델 feature allowlist를 저장하며 bearing sin/cos와 scaler는 Stage 5.2에서 적용합니다.

## 분할·합성·중복 정책

사용자 정렬 후 NumPy default_rng(split_seed)로 섞어 60/20/20으로 나눕니다. 품질 적격 경로가 없는 사용자를 자동 제외하지 않고 실패합니다. 중복 제외 때문에 모델용 경로가 0개가 되더라도 사용자 split은 유지합니다. Train 합성은 생성하지 않습니다. Validation/Test의 적격 경로마다 route_deviation, abnormal_speed, long_stop, direction_change를 1개씩 생성합니다.

합성 sample seed는 SHA256(JSON [synthetic_seed, global_source_id, anomaly_type, sample_number])의 앞 8바이트입니다. 입력 순서와 모델 seed에 독립적이며 manifest에 기록합니다.

fingerprint는 시간 정렬 후 위도·경도 7자리와 시작 이후 경과 ns로 계산합니다. 사용자·고도·절대 시작 날짜는 제외합니다. 이 정의의 exact duplicate 검사이며 모든 근사 경로 복제를 탐지하지는 않습니다.

`src/trajectory_deduplication.py`의 `deduplicate_trajectory_manifest()`가 Train/Validation/Test에 동일한 정책을 적용합니다. 동일 split의 동일 fingerprint는 `(original_trajectory_id, trajectory_id)`를 문자열 오름차순 정렬해 첫 번째만 유지합니다. 나머지는 `exact_duplicate_trajectory`로 표시하고 모델용 적격 경로에서 제외합니다. 성능이나 입력 순서는 유지 기준에 영향을 주지 않습니다. 사용자 split을 정한 뒤, 모델용 데이터 선택과 synthetic anomaly 생성 전에 적용합니다.

중복 원본은 `quality_checked.csv`와 입력/경로 manifest에 남습니다. 제외 내역은 `dataset_summary.json`, `leakage_report.json`, `duplicate_report.json`에도 기록합니다. 정제 경로 전체의 제외 행 수와 품질 필터 후 모델용 정상 행 감소분을 구분합니다.

split 간 동일 fingerprint가 있으면 품질 적격 여부와 관계없이 `FingerprintLeakageError`로 실패합니다. 자동 제외나 split 변경은 하지 않습니다. 오류의 입력 manifest와 leakage report를 저장하고 합성 및 split CSV 생성 전에 중단합니다. `ready_for_stage5_b`는 구조·split 검사 상태이며 모델 성능이나 연구 타당성을 보장하지 않습니다.

## 실제 실행 결과

2026-10-02 (Asia/Seoul), `geolife_u000_019_first5_seed42_dedup` 중복 처리 후 재생성 결과입니다. 이전 `geolife_u000_019_first5_seed42` dataset은 그대로 보존했습니다.

| 항목 | 결과 |
| --- | ---: |
| 사용자 | 20 |
| 입력 경로 | 100 |
| 원본 행 | 155,794 |
| 정제 후 행 | 155,743 |
| 중복 처리 전 품질 적격 경로 | 83 |
| 중복 처리 후 모델용 적격 경로 | 82 |
| 제외 경로 | 18 (품질 17 + exact duplicate 1) |
| 중복 제외 정제 행 | 2,491 |
| 중복 제외 모델용 정상 행 | 2,482 |
| 적격 정상 행 | 151,814 |
| 실행 시간 | 40.22초 |
| CSV 크기 | 240,704,765 bytes (약 229.6 MiB) |

| Split | 사용자 | 경로 | 유효 정상 | 합성 sample | 유효 합성 이상 | 평가 총 행 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | 000, 003, 005, 006, 007, 009, 010, 012, 014, 015, 016, 019 | 51 | 86,067 | 0 | 0 | 86,067 |
| validation | 002, 004, 011, 018 | 12 | 19,448 | 48 | 1,113 | 20,561 |
| test | 001, 008, 013, 017 | 19 | 46,299 | 76 | 2,112 | 48,411 |

Validation 합성 이상은 품질 필터 전 1,117행, 이후 1,113행입니다. Test는 2,115행에서 2,112행입니다. 원본 품질 불량은 615행이며 long_gap 340, unrealistic_speed 258, gps_jump 43, invalid_time 0입니다. 플래그는 중복될 수 있습니다. 입력 중복 timestamp 51행은 Stage 1 규칙으로 정제했습니다.

사용자·원본 경로·sample overlap, global ID collision, lineage 누락·잘못된 중복, NaN/inf는 모두 0입니다. CSV round-trip, leading zero, 저장 파일 checksum 검증을 통과했습니다.

중복 fingerprint 그룹은 1개(2개 경로), split 간 중복 그룹은 0개입니다. Train 사용자 `014`의 `014__20081020054500`을 유지하고 `014__20081020134500`을 `exact_duplicate_trajectory`로 제외했습니다. fingerprint는 `b6994392c67f9f32fd50db05a0c27e55ebce7c1317e993d0dd88835a1330ec15`입니다. 제외 경로의 정제 행은 2,491개이고 모델용 정상 행은 2,482개입니다. Train은 기존 52경로·88,549행에서 51경로·86,067행으로 변경됐습니다. 전체 모델용 경로는 83개에서 82개, 정상 행은 154,296개에서 151,814개로 감소했습니다.

모델용 데이터에 남은 중복 fingerprint는 0개입니다. 기존 사용자 split과 Validation/Test 데이터 및 synthetic manifest가 바이트 단위로 동일함을 확인했습니다. 새 dataset의 읽기 전용 재검증도 통과했습니다.

### 제외 경로

| 사용자 | 원본 경로 | point 수 | 제외 사유 |
| --- | --- | ---: | --- |
| 000 | 20081027115449 | 50 | point_count<100 |
| 004 | 20081024155859 | 76 | point_count<100 |
| 007 | 20081025142200 | 15 | point_count<100 |
| 007 | 20081026160935 | 93 | point_count<100 |
| 011 | 20080926111008 | 56 | point_count<100 |
| 011 | 20080927055038 | 26 | point_count<100 |
| 011 | 20080927114921 | 16 | point_count<100 |
| 011 | 20080927120956 | 38 | point_count<100 |
| 012 | 20080928021931 | 90 | point_count<100 |
| 013 | 20080929224234 | 13 | point_count<100 |
| 014 | 20081020134500 | 2,491 | exact_duplicate_trajectory |
| 015 | 20081020092056 | 76 | point_count<100 |
| 015 | 20081023024930 | 63 | point_count<100 |
| 016 | 20081027122004 | 7 | point_count<100 |
| 016 | 20081027122027 | 59 | point_count<100 |
| 018 | 20081104093602 | 30 | point_count<100;low_quality_ratio>0.05 |
| 018 | 20081105014016 | 36 | point_count<100 |
| 018 | 20081105033455 | 13 | point_count<100 |

## 테스트와 다음 단계

기존 Stage 5.1 39개에 중복 정책 테스트 13개를 추가했습니다. same-split 탐지, 입력 순서와 무관한 유지 선택, 제외 사유·행 수 기록, Validation/Test 적용, cross-split 실패, CSV round-trip fingerprint 보존, 합성 전 제외 및 저장 결과 재검증을 확인합니다. 기존 Stage 1~4 68개와 Stage 5.1 52개를 포함해 전체 **120개가 통과**했습니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

중복 정책을 적용한 고정 dataset을 준비했습니다. 다음 Stage 5.2는 기존 8 feature·모델 구조를 유지하며, 별도 학습 요청 후 시작합니다. 이번 단계에서는 학습·모델 평가·multi-seed·baseline·window 모델을 실행하지 않습니다. 원본 정상과 합성 이상은 연구상 가정·실험 라벨이며 실제 위험 판정 성능을 의미하지 않습니다.
