> [!NOTE]
> 이 프로젝트는 AI(Antigravity)에 의해 설계 및 생성된 자동화 도난 탐지 시스템의 **프로토타입**입니다. 실제 환경에서의 정확도는 조명, 카메라 각도 및 학습 데이터의 특성에 따라 달라질 수 있습니다.

# Automated Theft Detection System

이 프로젝트는 YOLOv11과 CLIP 모델을 결합하여 실시간 영상 데이터에서 고착된 물건의 도난을 탐지하고 분석하는 파이프라인을 구현합니다.

## 📥 Input & 📤 Output 정의

시스템의 원활한 작동을 위한 입력값과 도출되는 최종 결과물은 다음과 같습니다.

### 📥 System Input
1.  **영상 데이터**: 실시간 스트리밍 또는 저장된 파일(`.mp4`, `.mov` 등)
2.  **객체 정의**: YOLO 추적용 클래스(`VALID_LOST_ITEMS`) 및 CLIP 분석용 태그 리스트
3.  **임계값 설정**: 
    - `stationary_threshold`: 물건이 고착되었다고 판단할 시간 (프레임 단위)
    - `proximity_pixels`: 인물과 물건 사이의 접촉/근접 인정 거리

### 📤 System Output
1.  **이미지 증거물 (in `output/` folder)**:
    - `stolen_item_baseline_*.jpg`: 도난당하기 전, 물건의 깨끗한 고착 상태 이미지 (정밀 분석용)
    - `theft_moment_*.jpg`: 도난이 발생한 시점의 전체 상황을 담은 스냅샷
2.  **AI 분석 데이터**:
    - **Category**: CLIP이 판별할 물건의 종류 (예: Smartphone, Laptop)
    - **Color**: 물건의 주요 색상 정밀 추출
    - **Feature Vector**: 512차원의 이미지 특징 벡터 (추후 DB 검색 및 유사도 비교용)
3.  **실시간 모니터링 로그**: 도난 탐지 시점, 객체 ID, 발생 시간 등을 콘솔에 기록

## 🚀 파이프라인 작동 과정 (Pipeline Workflow)

### 1. 실시간 객체 추적 (Object Tracking)
*   **YOLOv11s** 모델을 사용하여 영상 내의 `person` 및 지정된 `VALID_LOST_ITEMS`(가방, 우산, 핸드폰, 노트북 등)를 실시간으로 탐지합니다.
*   `persist=True` 설정을 통해 객체가 사라졌다 다시 나타나도 동일한 ID를 유지하도록 추적합니다.

### 2. 고착 물건 식별 (Stationary Object Identification)
*   특정 물건이 영상 내에서 일정 시간(기본 50프레임, 약 2.5초) 동안 움직이지 않으면 시스템은 이를 **'Stationary(고착)'** 상태로 등록합니다.
*   고착 상태가 확정되는 순간, 해당 물건의 깨끗한 상태인 **Baseline Snapshot**을 별도로 저장합니다.

### 3. 도난 탐지 로직 (Theft Detection Logic)
*   **근접도 감시 (Proximity Check)**: 고착된 물건 주변에 인물이 접근하거나 접촉(Bounding Box 중첩 또는 일정 거리 이내)하는지를 상시 감시하며 이력을 기록합니다.
*   **실종 판별 (Disappearance)**: 고착 상태였던 물건이 갑자기 추적 범위에서 사라질 경우를 감지합니다.
*   **경보 발생 (Alert)**: 물건이 사라졌을 때, **직전에 인물과의 접촉 이력**이 있었다면 이를 '도난'으로 간주하여 도난 순간의 장면과 물건 정보를 저장합니다.

### 4. 도난 객체 정밀 분석 (AI Interaction & Analysis)
*   **카테고리 분류 (CLIP Category)**: 저장된 Baseline 이미지를 CLIP 모델에 입력하여 물건의 상세 종류(Smartphone, Laptop, Wallet 등)를 확률 기반으로 분류합니다.
*   **색상 분석 (CLIP Color)**: 물건의 색상 프롬프트를 통해 특징적인 색상을 추출합니다.
*   **벡터 추출 (Feature Extraction)**: 이미지에서 512차원(또는 모델 설정에 따른) 특징 벡터를 추출하여, 추후 데이터베이스 검색이나 유사도 비교에 활용할 수 있도록 합니다.

## 🔍 주요 설정 및 식별 대상

본 시스템은 두 단계의 모델을 거쳐 객체를 식별합니다.

### 1. YOLOv11 탐지 및 추적 대상
영상의 실시간 모니터링 과정에서 추적되는 주요 대상입니다.
*   **기본 탐지**: `person` (인물 - 도난 주체 확인용)
*   **추적 소지품 (`VALID_LOST_ITEMS`)**: 
    - `backpack` (배낭), `umbrella` (우산), `handbag` (핸드백), `bottle` (병)
    - `cup` (컵), `cell phone` (핸드폰), `book` (책), `laptop` (노트북)

### 2. CLIP 정밀 분석 태그
도난 사건 발생 시, 저장된 스냅샷을 기반으로 수행되는 정밀 분석 항목입니다.

| 구분 | 태그 목록 (Labels) |
| :--- | :--- |
| **카테고리** | `smartphone`, `laptop`, `earphones`, `bag`, `wallet`, `credit card`, `student ID card`, `textbook`, `notebook`, `umbrella`, `glasses` |
| **색상** | `black`, `white`, `gray`, `red`, `blue`, `green`, `yellow`, `brown`, `pink`, `purple`, `orange`, `beige`, `transparent` |

## 📂 프로젝트 구조 및 모듈

코드는 기능별로 패키지화되어 관리됩니다.

*   `main.py`: 프로그램의 진입점. 전체 시나리오를 제어합니다.
*   `config.py`: 모델 ID, 파일 경로, 분석 태그 등 전역 설정을 관리합니다.
*   **`core/`** (핵심 로직)
    - `detector.py`: 고착 상태 판정 및 도난 탐지 알고리즘 (`TheftDetector`)
    - `processor.py`: 영상 프레임 분석 및 트래킹 루프 서비스
*   **`models/`** (AI 모델)
    - `loader.py`: YOLOv11 및 CLIP 모델 로딩
    - `analyzer.py`: CLIP 기반의 카테고리/색상 분석 및 벡터 추출
*   **`output/`**: 도난 상황 발생 시 생성되는 스냅샷 이미지가 저장되는 디렉토리 (자동 생성)

## 📦 실행 방법
1. 필요한 패키지 설치: `pip install -r requirements.txt`
2. 시스템 실행: `python main.py`
