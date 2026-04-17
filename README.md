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
4.  **실시간 텔레메트리 (UI 활성 시)**: 
    - `Frame`: 현재 프레임 / 전체 프레임 (진행도 표시)
    - `Avg FPS`: 전체 소요 시간 대비 평균 처리 속도

## 2. 시스템 작동 프로세스 (System Pipeline)

영상 유입부터 최종 분석까지의 단계별 메커니즘은 다음과 같습니다.

### 단계 1: 실시간 객체 추적 및 상태 감시
- **객체 추적**: YOLOv11s를 사용하여 인물(person)과 주요 소지품을 실시간으로 추적합니다.
- **고착 상태 판정**: 객체의 이동 거리가 50px 미만인 상태가 50프레임 이상 유지되면 '고착(Stationary)' 상태로 전환합니다.
- **데이터 백업**: 고착 확정 시 즉시 해당 객체의 원본 크롭 이미지를 Baseline으로 저장합니다.

### 단계 2: 소유권 추적 및 접촉 감지
- **Monitoring (추적 및 소유권 확인)**: 
    - 물건이 **처음 감지되는 순간(First Appearance)**에만 근처의 인물을 '소유자(Owner)'로 등록합니다.
    - 처음 등장 시 주변에 사람이 없다면 해당 물건은 무소유 상태로 간주합니다.
    - YOLOv11을 이용해 물건의 위치 이동을 실시간 추적합니다.
    - 50프레임 이상 고정된 물건은 'Stationary(고정 상태)'로 분류하고 기준 이미지를 저장합니다.
- **근접 이력 관리**: 인물이 물건에 인접하거나 접촉할 경우 `near_history` 카운터를 활성화합니다. 이 이력은 인물이 물건 주변을 떠난 후에도 일정 시간 유지됩니다.
- **객체 유실 감지**: 추적 중이던 고착 객체가 영상에서 사라지면(10프레임 이상) 이를 인지합니다.

### 단계 3: 지연 검증 및 신뢰도 기반 도난 판정
- **다중 프레임 검증 (Verification)**: 객체 유실 감지 시 즉시 경보를 울리지 않고, `VERIFICATION_FRAMES`(기본 30프레임) 동안 상태를 재확인하여 일시적인 시야 가림을 필터링합니다. 이때 물건이 사라진 **최초의 순간(Frame 1)**을 증거 프레임으로 미리 확보하여, 검증 완료 시점에 범인의 모습이 담긴 정확한 증거물을 저장합니다.
- **신뢰도 점수 산출 로직 (Scoring Algorithm)**:
    - 시스템은 다음의 가중치를 합산하여 `Confidence Score` (0.0~1.0)를 계산합니다.
    - **소유권 원칙 (Owner Retrieval)**: 물건을 가져간 사람이 시스템에 등록된 최초 소유자(Owner)인 경우, 점수를 즉시 **0.0**으로 강제 설정하여 오탐을 원천 차단합니다.
    - **접촉 이력 가중치 (Contact, +0.3)**: 소유자가 아닌 인물이 물건 주변에 머물렀거나 물리적 접촉(Touching)이 발생한 기록이 있을 경우 부여됩니다.
    - **소유권 명확성 가중치 (Owner Clarity, +0.5)**: 도난당한 물건이 명확하게 특정 인물에 의해 소유되고 있었을 경우(주인 없는 물건이 아닐 경우) 도난 판단 확률을 높입니다.
    - **고착 상태 확신도 (Stationary Degree, +0.2)**: 물건이 오랫동안(100프레임 이상) 움직이지 않고 고정되어 있었을수록 도난 판단 근거가 강화됩니다.
    - **도망 행동 감지 보너스 (Fleeing Bonus, +0.2)**: 물건 실종 직후, 주변 인물의 이동 속도가 `FLEEING_SPEED_THRESHOLD`를 초과하여 급격히 빨라질 경우 부여됩니다.
- **최종 판정**: 합산된 점수가 `THEFT_CONFIDENCE_THRESHOLD`를 넘길 경우에만 `TheftLogger`를 통해 사건을 기록하고 알람을 발생시킵니다.

### 단계 4: 사후 정밀 분석 및 벡터 추출
- **AI 추론**: CLIP 모델을 통해 미리 저장된 Baseline 이미지를 분석합니다.
- **속성 정의**: 텍스트 프롬프트 매칭을 통해 물건의 상세 카테고리와 대표 색상을 추출합니다.
- **특징 벡터화**: 이미지를 정규화된 512차원 벡터로 변환하여 출력하며, 모든 분석 결과를 로그로 출력합니다.

## 3. 주요 식별 대상

### YOLOv11 추적 대상
*   **기본 탐지**: `person` (인물 - 도난 주체 확인용)
*   **추적 소지품 (VALID_LOST_ITEMS)**: 
    - `backpack` (배낭), `umbrella` (우산), `handbag` (핸드백), `bottle` (병)
    - `cup` (컵), `cell phone` (핸드폰), `book` (책)

### CLIP 정밀 분석 태그

| 구분 | 태그 목록 (Labels) |
| :--- | :--- |
| **카테고리** | `smartphone`, `earphones`, `bag`, `wallet`, `credit card`, `student ID card`, `textbook`, `notebook`, `umbrella`, `water bottle`, `pencil case`, `plush toy` |
| **색상** | `black`, `white`, `gray`, `red`, `blue`, `green`, `yellow`, `brown`, `pink`, `purple`, `orange`, `beige` |

## 4. 프로젝트 구조 및 모듈

- `main.py`: 프로그램 진입점 및 전체 시나리오 제어
- `config.py`: 모델 경로 및 전역 설정 관리 (`SHOW_UI` 스위치 포함)
- **core/**: 도난 탐지 알고리즘(`detector.py`), 영상 처리 서비스(`processor.py`), 사건 로그 기록기(`logger.py`)
- **models/**: 모델 로더(`loader.py`) 및 CLIP 분석기(`analyzer.py`)
- **output/**: 도난 스냅샷 및 `theft_log.json` 기록 디렉토리

## 5. 실행 방법
1. 패키지 설치: `pip install -r requirements.txt`
2. 시스템 실행: `python main.py`
