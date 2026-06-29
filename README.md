# PathGuard_Ai

PathGuard_Ai는 GPS 이동 데이터를 기반으로 이동 패턴과 이상행동 탐지를 연구하는 포트폴리오용 Python 프로젝트입니다.

1단계에서는 Microsoft GeoLife GPS Trajectories의 `.plt` 파일을 정제된 CSV로 변환하고, 이동 경로를 OpenStreetMap 지도에 시각화했습니다.

2단계에서는 trajectory별 GPS 이동 특징을 생성하고, 계산 결과와 데이터 품질을 시계열 그래프 및 Folium 지도에 시각화했습니다.

## 프로젝트 구조

```text
PathGuard_Ai/
├─ data/
│  ├─ raw/
│  │  └─ geolife/
│  │     └─ Data/
│  └─ processed/
├─ docs/
│  └─ images/
│     └─ stage2/
│        └─ 20081026134407/
├─ outputs/
│  ├─ figures/
│  └─ maps/
├─ src/
│  ├─ __init__.py
│  ├─ feature_engineering.py
│  ├─ load_geolife.py
│  ├─ visualize_features.py
│  └─ visualize_route.py
├─ tests/
│  ├─ test_feature_engineering.py
│  └─ test_visualize_features.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 설치 방법

Windows PowerShell에서 프로젝트 루트 `PathGuard_Ai`로 이동한 뒤 아래 명령을 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell 실행 정책으로 가상환경 활성화가 차단되면 현재 세션에 한해 다음 명령을 먼저 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## GeoLife 데이터 배치

GeoLife GPS Trajectories 데이터를 내려받아 압축을 푼 뒤, `Data` 폴더가 다음 위치에 오도록 배치합니다.

```text
data/raw/geolife/Data/
└─ 000/
   └─ Trajectory/
      ├─ 20081023025304.plt
      ├─ 20081024020959.plt
      └─ ...
```

GeoLife 원본 데이터는 용량과 라이선스 관리를 위해 Git 추적 대상에서 제외됩니다.

---

# Stage 1. GeoLife 데이터 로딩 및 경로 시각화

## GPS 데이터 변환

사용자 `000`의 파일명 기준 첫 5개 `.plt` 파일을 읽고 검증한 뒤 하나의 CSV로 저장합니다.

```powershell
python -m src.load_geolife
```

생성 파일:

```text
data/processed/gps_raw.csv
```

다음 항목을 검증합니다.

* 위도·경도 범위
* timestamp 변환 여부
* 중복된 trajectory와 timestamp 조합
* trajectory별 GPS 포인트 수
* 경로별 시작 및 종료 시간

## GPS 이동 경로 시각화

CSV의 첫 번째 trajectory를 지도에 표시합니다.

```powershell
python -m src.visualize_route
```

특정 `trajectory_id`를 선택할 수도 있습니다.

```powershell
python -m src.visualize_route --trajectory-id 20081026134407
```

생성 파일:

```text
outputs/maps/route_<trajectory_id>.html
```

예를 들어 위 명령은 다음 파일을 생성합니다.

```text
outputs/maps/route_20081026134407.html
```

존재하지 않는 ID를 입력하면 사용할 수 있는 trajectory ID 목록이 출력됩니다.

생성된 HTML 파일에서는 다음 내용을 확인할 수 있습니다.

* OpenStreetMap 배경 지도
* 전체 GPS 이동 경로
* 출발 지점
* 도착 지점
* 경로 전체 범위 자동 조정

지도 타일을 불러오려면 인터넷 연결이 필요합니다.

---

# Stage 2. GPS 이동 특징 전처리 및 시각화

## GPS 이동 특징 생성

`data/processed/gps_raw.csv`를 읽어 각 `trajectory_id` 안에서만 이동 특징을 계산합니다.

| 특징                     | 설명                       | 단위   |
| ---------------------- | ------------------------ | ---- |
| `time_diff_sec`        | 이전 GPS 포인트와의 시간 차이       | 초    |
| `distance_m`           | Haversine 공식으로 계산한 이동 거리 | m    |
| `speed_mps`            | 이동 거리 ÷ 시간 차이            | m/s  |
| `acceleration_mps2`    | 속도 변화 ÷ 시간 차이            | m/s² |
| `bearing_deg`          | 북쪽을 0°로 한 초기 방위각         | °    |
| `direction_change_deg` | 이전 방위각과의 최소 각도 차이        | °    |
| `stop_duration_sec`    | 연속 정지 구간의 누적 시간          | 초    |

모든 계산은 trajectory별로 독립적으로 수행합니다. 한 경로의 마지막 GPS 포인트와 다른 경로의 첫 번째 포인트가 연결되지 않도록 처리했습니다.

각 trajectory의 첫 번째 포인트는 이전 위치가 없으므로 계산된 특징값을 0으로 초기화합니다.

## 전처리 기준

이동 거리가 1m 미만이면 GPS 오차로 인한 방위각 흔들림을 줄이기 위해 직전 유효 방위각을 유지합니다.

다음 조건을 모두 만족하면 해당 구간을 정지 상태로 처리합니다.

```text
이동 거리 ≤ 3m
이동 속도 ≤ 0.5m/s
```

정지 조건이 연속되면 `time_diff_sec`를 누적하고, 다시 이동하면 `stop_duration_sec`를 0으로 초기화합니다.

## 전처리 실행

기본 입출력 경로로 실행합니다.

```powershell
python -m src.feature_engineering
```

입력과 출력 경로를 직접 지정할 수도 있습니다.

```powershell
python -m src.feature_engineering `
  --input data/processed/gps_raw.csv `
  --output data/processed/gps_features.csv
```

생성 파일:

```text
data/processed/gps_features.csv
```

## 전처리 검증 결과

GeoLife 사용자 `000`의 trajectory 5개를 대상으로 실행한 결과입니다.

| 항목            |     결과 |
| ------------- | -----: |
| 입력 행 수        |  3,424 |
| 출력 행 수        |  3,424 |
| Trajectory 수  |      5 |
| 유효하지 않은 시간 간격 |      0 |
| NaN 및 무한대     |      0 |
| 전체 단위 테스트     | 17개 통과 |

전체 데이터에서는 다음과 같은 극단값을 확인했습니다.

* `speed_mps` 50m/s 초과: 1개
* 최대 GPS 기록 간격: 18,453초
* 전체 최대 속도: 60.5829m/s

해당 값은 GPS 좌표 오류, 장시간 기록 공백, 이동 수단 변경 또는 실제 이동 상황일 수 있습니다.

Stage 2에서는 이를 자동 삭제하거나 이상행동으로 판정하지 않고, 이후 학습 데이터 구성 단계에서 별도의 품질 기준을 적용할 수 있도록 유지했습니다.

---

## 이동 특징 시각화

전처리된 이동 특징을 다음 시계열 그래프로 확인할 수 있습니다.

* 속도
* 가속도
* 방향 변화량
* 정지 시간
* GPS 기록 간격

옵션을 생략하면 CSV의 첫 번째 trajectory를 사용합니다.

```powershell
python -m src.visualize_features
```

특정 trajectory를 선택하려면 다음과 같이 실행합니다.

```powershell
python -m src.visualize_features --trajectory-id 20081026134407
```

입력 CSV를 직접 지정할 수도 있습니다.

```powershell
python -m src.visualize_features `
  --input data/processed/gps_features.csv `
  --trajectory-id 20081026134407
```

일반 실행 결과는 다음 폴더에 저장됩니다.

```text
outputs/figures/<trajectory_id>/
```

생성 파일:

```text
speed_timeline.png
acceleration_timeline.png
direction_change_timeline.png
stop_duration_timeline.png
time_gap_timeline.png
feature_summary.csv
```

## 특징 지도

전처리된 특징을 이동 경로와 함께 확인할 수 있는 Folium 지도를 생성합니다.

```text
outputs/maps/feature_route_<trajectory_id>.html
```

특징 지도에서는 다음 내용을 확인할 수 있습니다.

* 전체 이동 경로
* 출발 및 도착 지점
* 선택한 trajectory ID
* GPS 포인트 수
* 기록 시작 및 종료 시간
* 고속도 확인 지점
* 장시간 GPS 기록 공백 지점
* 장시간 정지 지점

표시되는 기준은 데이터 탐색을 위한 참고 기준이며 위험 또는 이상행동 판정 기준이 아닙니다.

```text
속도 > 50m/s
GPS 기록 간격 > 300초
누적 정지 시간 ≥ 60초
```

---

## 포트폴리오 결과 내보내기

GitHub README에서 확인할 PNG와 요약 CSV를 생성하려면 `--export-portfolio` 옵션을 사용합니다.

```powershell
python -m src.visualize_features `
  --trajectory-id 20081026134407 `
  --export-portfolio
```

포트폴리오 결과 저장 위치:

```text
docs/images/stage2/20081026134407/
```

`outputs/`에 생성되는 일반 실행 결과는 Git 추적 대상에서 제외되며, `docs/images/stage2/`의 대표 이미지와 요약 CSV는 Git에서 추적됩니다.

---

## 대표 경로 시각화

### 속도 변화

전체 속도와 50m/s 데이터 품질 참고선을 비교합니다.

![Speed timeline](docs/images/stage2/20081026134407/speed_timeline.png)

### 가속도 변화

양수·음수 가속도와 최솟값·최댓값 지점을 확인합니다.

![Acceleration timeline](docs/images/stage2/20081026134407/acceleration_timeline.png)

### 방향 변화량

0~180도 범위의 방향 변화와 변화량이 큰 상위 지점을 확인합니다.

![Direction change](docs/images/stage2/20081026134407/direction_change_timeline.png)

### 정지 시간

정지 조건이 연속될 때 누적되는 정지 시간을 확인합니다.

![Stop duration](docs/images/stage2/20081026134407/stop_duration_timeline.png)

### GPS 기록 간격

GPS 포인트 사이의 측정 간격과 가장 긴 기록 공백을 확인합니다.

![Time gap](docs/images/stage2/20081026134407/time_gap_timeline.png)

---

## 대표 경로 분석 결과

분석 대상:

```text
trajectory_id: 20081026134407
```

| 항목           |          결과 |
| ------------ | ----------: |
| GPS 포인트      |        745개 |
| 기록 시간        |      4,800초 |
| 총 이동 거리      | 18,648.325m |
| 평균 포인트 속도    |    2.931m/s |
| 전체 경로 평균 속도  |  약 3.885m/s |
| 최대 속도        |   36.647m/s |
| 최소 가속도       |  -5.784m/s² |
| 최대 가속도       |   6.056m/s² |
| 최대 방향 변화량    |    173.886° |
| 최대 누적 정지 시간  |         35초 |
| 최대 GPS 기록 간격 |        275초 |

`평균 포인트 속도`는 각 GPS 포인트에서 계산된 속도의 산술평균이며, `전체 경로 평균 속도`는 총 이동 거리를 전체 기록 시간으로 나눈 값입니다.

그래프를 통해 경로 후반부에서 상대적으로 높은 속도 구간이 나타나며, 일부 시점에서 큰 방향 변화와 급가속·급감속이 발생한 것을 확인했습니다.

이 경로에서는 다음 참고 조건을 만족하는 지점이 없었습니다.

```text
속도 > 50m/s
GPS 기록 간격 > 300초
누적 정지 시간 ≥ 60초
```

따라서 실제 데이터에 존재하지 않는 품질 확인 지점은 그래프와 지도에 임의로 표시하지 않았습니다.

전체 요약 수치는 다음 파일에서 확인할 수 있습니다.

```text
docs/images/stage2/20081026134407/feature_summary.csv
```

---

## 단위 테스트

다음 명령으로 전체 테스트를 실행합니다.

```powershell
python -m unittest discover -s tests
```

총 17개의 테스트를 통해 다음 항목을 검증했습니다.

* Haversine 거리 계산
* 동서남북 방향의 방위각 계산
* 최소 방향 변화량 계산
* 정지 시간 누적 및 초기화
* trajectory 간 계산 분리
* NaN 및 무한대 방지
* trajectory 선택 및 오류 처리
* 특징 요약 통계 계산
* PNG 그래프 생성
* Folium 지도 HTML 생성
* 포트폴리오 결과 내보내기
* 시각화 과정에서 원본 DataFrame 유지

---

## 생성 데이터 안내

다음 파일은 프로그램 실행 과정에서 생성되며 Git 저장소에는 포함되지 않습니다.

```text
data/processed/gps_raw.csv
data/processed/gps_features.csv
outputs/figures/
outputs/maps/*.html
```

GeoLife 원본 데이터와 Python 가상환경도 `.gitignore`를 통해 Git 추적 대상에서 제외됩니다.

대표 포트폴리오 이미지는 다음 경로에 저장되며 GitHub에서 확인할 수 있습니다.

```text
docs/images/stage2/
```

---

## 현재 구현 범위

현재까지 다음 기능을 구현했습니다.

* GeoLife `.plt` 파일 로딩 및 검증
* 원본 GPS 데이터의 CSV 변환
* OpenStreetMap 기반 이동 경로 시각화
* trajectory별 GPS 이동 특징 생성
* 거리·속도·가속도 계산
* 방위각과 방향 변화량 계산
* 연속 정지 시간 계산
* GPS 데이터 품질 검증
* 이동 특징 시계열 그래프 생성
* 데이터 품질 확인 지점 Folium 지도 표시
* 경로별 요약 CSV 생성
* 총 17개의 단위 테스트

아직 다음 기능은 구현하지 않았습니다.

* 정상·이상 라벨 생성
* 규칙 기반 위험 판정
* 이상 점수 계산
* 학습 데이터 분할
* PyTorch
* Autoencoder
* 머신러닝 학습
* Streamlit UI
* Android GPS 연동

현재 표시되는 속도, 기록 공백, 정지 시간 기준은 데이터를 탐색하기 위한 참고 기준이며 위험 또는 이상행동을 의미하지 않습니다.
