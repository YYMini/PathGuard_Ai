# PathGuard_Ai

PathGuard_Ai는 GPS 이동 데이터를 바탕으로 이상행동 탐지를 연구하는 포트폴리오용 Python 프로젝트입니다. 현재 1단계에서는 Microsoft GeoLife GPS Trajectories의 `.plt` 파일을 정제된 CSV로 합치고, 첫 번째 이동 경로를 OpenStreetMap 지도에 시각화합니다.

## 프로젝트 구조

```text
PathGuard_Ai/
├─ data/
│  ├─ raw/geolife/
│  └─ processed/
├─ outputs/maps/
├─ src/
│  ├─ __init__.py
│  ├─ load_geolife.py
│  └─ visualize_route.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 설치 방법 (Windows PowerShell)

프로젝트 루트 `PathGuard_Ai`에서 아래 명령을 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell 실행 정책 때문에 활성화가 차단되면 현재 세션에 한해 다음 명령을 먼저 실행할 수 있습니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## GeoLife 데이터 배치

GeoLife 데이터를 직접 내려받아 압축을 푼 뒤, `Data` 폴더가 다음 위치에 오도록 배치합니다.

```text
data/raw/geolife/Data/
└─ 000/
   └─ Trajectory/
      ├─ 20081023025304.plt
      ├─ 20081024020959.plt
      └─ ...
```

원본 데이터는 용량과 라이선스 관리를 위해 Git 추적 대상에서 제외됩니다.

## 실행 방법

사용자 `000`의 파일명 기준 첫 5개 `.plt` 파일을 읽고 검증한 뒤 CSV로 저장합니다.

```powershell
python -m src.load_geolife
```

생성 파일: `data/processed/gps_raw.csv`

이어서 CSV의 첫 번째 경로를 지도에 표시합니다.

```powershell
python -m src.visualize_route
```

특정 `trajectory_id`를 선택하려면 다음과 같이 실행합니다. ID는 문자열로 처리되므로 원래 형식이 유지됩니다.

```powershell
python -m src.visualize_route --trajectory-id 20081026134407
```

생성 파일: `outputs/maps/route_<trajectory_id>.html`

예를 들어 위 명령은 `outputs/maps/route_20081026134407.html`을 생성합니다. 존재하지 않는 ID를 입력하면 사용 가능한 전체 경로 ID 목록이 출력됩니다.

HTML 파일을 브라우저에서 열면 OpenStreetMap 위에 전체 이동선과 출발·도착 지점이 표시됩니다. 지도 타일을 불러오려면 인터넷 연결이 필요합니다.

## 현재 구현 범위

이번 단계는 GeoLife 데이터 로딩, 검증, CSV 변환, Folium 경로 시각화까지만 포함합니다. AI 이상행동 탐지 모델, PyTorch, Streamlit, Android 앱은 아직 구현하지 않았습니다.
