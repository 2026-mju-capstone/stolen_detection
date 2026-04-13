# Automated Theft Detection System

> [!NOTE]
> 이 프로젝트는 AI(Antigravity)에 의해 설계 및 생성된 자동화 도난 탐지 시스템의 프로토타입입니다. 실제 환경에서의 정확도는 조명, 카메라 각도 및 학습 데이터의 특성에 따라 달라질 수 있습니다.

이 프로젝트는 YOLOv11과 CLIP 모델을 결합하여 실시간 영상 데이터에서 고착된 물건의 도난을 탐지하고 분석하는 시스템입니다.

## 1. Input & Output 정의

### System Input
1.  **영상 데이터**: 실시간 스트리밍 또는 저장된 파일(.mp4, .mov 등)
2.  **객체 정의**: YOLO 추적용 클래스(VALID_LOST_ITEMS) 및 CLIP 분석용 태그 리스트
3.  **임계값 설정**: 
    - `stationary_threshold`: 물건이 고착되었다고 판단할 시간 (프레임 단위)
    - `proximity_pixels`: 인물과 물건 사이의 접촉 인정 거리

### System Output
1.  **이미지 증거물 (in output/ folder)**:
    - `stolen_item_baseline_*.jpg`: 도난 전, 물건의 깨끗한 고착 상태 이미지
    - `theft_moment_*.jpg`: 도난 발생 시점의 전체 상황 스냅샷
2.  **AI 분석 데이터**:
    - **Category & Color**: CLIP이 판별한 물건의 종류 및 주요 색상
    - **Feature Vector**: 512차원의 이미지 특징 추출값 (Similarity Search용)
3.  **실시간 모니터링 로그**: 도난 탐지 시점, 객체 ID, 발생 시간 등의 콘솔 기록

## 2. 시스템 작동 프로세스 (System Pipeline)

영상 유입부터 최종 분석까지의 단계별 메커니즘은 다음과 같습니다.

### 단계 1: 실시간 객체 추적 및 상태 감시
- **객체 추적**: YOLOv11s를 사용하여 인물(person)과 주요 소지품을 실시간으로 추적합니다.
- **고착 상태 판정**: 객체의 이동 거리가 50px 미만인 상태가 50프레임 이상 유지되면 '고착(Stationary)' 상태로 전환합니다.
- **데이터 백업**: 고착 확정 시 즉시 해당 객체의 원본 크롭 이미지를 Baseline으로 저장합니다.

### 단계 2: 접촉 및 부재 감지
- **근접 이력 관리**: 인물이 물건에 인접하거나 접촉할 경우 `near_history` 카운터를 활성화합니다. 이 이력은 인물이 물건 주변을 떠난 후에도 일정 시간 유지됩니다.
- **객체 유실 감지**: 추적 중이던 고착 객체가 영상에서 사라지면(10프레임 이상) 이를 인지합니다.

### 단계 3: 도난 확정 및 현장 기록
- **판정 조건**: 1) 객체가 고착 상태였는가, 2) 실종 직전 인물과의 접촉 이력이 있었는가를 검증합니다.
- **사건 기록**: 위 조건 충족 시 도난으로 판단, 현재 영상 프레임을 '도난 순간'으로 캡처하여 증거를 확보합니다.

### 단계 4: 사후 정밀 분석 및 벡터 추출
- **AI 추론**: CLIP 모델을 통해 미리 저장된 Baseline 이미지를 분석합니다.
- **속성 정의**: 텍스트 프롬프트 매칭을 통해 물건의 상세 카테고리와 대표 색상을 추출합니다.
- **특징 벡터화**: 이미지를 정규화된 512차원 벡터로 변환하여 출력하며, 모든 분석 결과를 로그로 출력합니다.

## 3. 주요 식별 대상

### YOLOv11 추적 대상
*   **기본 탐지**: `person` (인물 - 도난 주체 확인용)
*   **추적 소지품 (VALID_LOST_ITEMS)**: 
    - `backpack` (배낭), `umbrella` (우산), `handbag` (핸드백), `bottle` (병)
    - `cup` (컵), `cell phone` (핸드폰), `book` (책), `laptop` (노트북)

### CLIP 정밀 분석 태그

| 구분 | 태그 목록 (Labels) |
| :--- | :--- |
| **카테고리** | `smartphone`, `laptop`, `earphones`, `bag`, `wallet`, `credit card`, `student ID card`, `textbook`, `notebook`, `umbrella`, `glasses` |
| **색상** | `black`, `white`, `gray`, `red`, `blue`, `green`, `yellow`, `brown`, `pink`, `purple`, `orange`, `beige`, `transparent` |

## 4. 프로젝트 구조 및 모듈

- `main.py`: 프로그램 진입점 및 전체 시나리오 제어
- `config.py`: 모델 경로 및 전역 설정 관리
- **core/**: 도난 탐지 알고리즘(`detector.py`) 및 영상 처리 서비스(`processor.py`)
- **models/**: 모델 로더(`loader.py`) 및 CLIP 분석기(`analyzer.py`)
- **output/**: 도난 스냅샷 저장 디렉토리

## 5. 실행 방법
1. 패키지 설치: `pip install -r requirements.txt`
2. 시스템 실행: `python main.py`
