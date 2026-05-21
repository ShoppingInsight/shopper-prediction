# DataMiningTerm
한성대학교 컴퓨터공학부 데이터마이닝 텀프로젝트 (기말 대체)
---

## 개발 환경 세팅

**Python 3.12** 기준입니다.

### Windows (PowerShell)

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> `Activate.ps1` 실행 시 권한 오류가 나면:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### macOS / Linux

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.12가 없다면 [python.org](https://www.python.org/downloads/release/python-3128/) 또는 `winget install Python.Python.3.12` (Windows) 로 설치.

---

## 주제 관리 보드 (`topics/`)

후보 주제를 카드 형태로 정리·비교하기 위한 정적 웹페이지. 별도 백엔드 없이 브라우저 LocalStorage + JSON 파일로 동작한다.

### 실행 방법 (VS Code Live Server 권장)

1. VS Code 확장에서 **Live Server** (Ritwick Dey) 설치
2. `topics/index.html` 우클릭 → **Open with Live Server**
3. 브라우저가 `http://127.0.0.1:5500/topics/` 로 자동 오픈

> `file://` 로 직접 열면 `data/topics.json` 로딩이 CORS로 실패한다. 최초 1회만 Live Server로 띄우면 이후엔 LocalStorage에 캐싱되어 오프라인에서도 동작.

대안으로 Python 내장 서버를 써도 된다.

```powershell
python -m http.server 8000
# 브라우저에서 http://localhost:8000/topics/
```

### 기능

- **카드 그리드**: 제목·상태·카테고리·문제 정의·모델·담당자 한눈에
- **필터/검색**: 상태(후보/검토중/채택/폐기), 카테고리(분류/회귀/비지도), 텍스트 검색
- **편집기**: 카드 클릭으로 수정, `+ 새 주제` 버튼으로 추가, 모달 내 삭제
- **JSON 내보내기/불러오기**: 팀원 간 데이터 공유용
- **초기화**: LocalStorage 비우고 `topics/data/topics.json` 원본으로 복귀

### 팀 협업 흐름

1. 각자 브라우저에서 주제 수정 → `JSON 내보내기` 로 파일 다운로드
2. 받은 파일을 `topics/data/topics.json` 에 덮어쓰고 git 커밋·푸시
3. 다른 팀원은 `git pull` 후 페이지에서 `초기화` 클릭하여 최신본 로드

### 파일 구조

```
topics/
├── index.html      # 페이지 마크업
├── style.css       # 스타일 (다크/라이트 자동)
├── script.js       # 렌더링·필터·편집·import/export
└── data/
    └── topics.json # 주제 데이터 (git 추적 대상)
```

---

## 프로젝트 개요

머신러닝 또는 딥러닝 기법을 활용하여 실제 데이터 기반 문제를 정의하고 해결하는 전 과정을 수행한다.
단순 결과 산출이 아닌, **수업에서 학습한 기법을 얼마나 정확히 이해하고 적절하게 적용했는지**를 중심으로 평가한다.

---

## 주제 조건

- 자유 주제 (허용 범위 내)
- 동일 문제에 **최소 3개 이상**의 ML/DL 모델 적용 및 성능·특성 비교 분석 필수

### 허용 기법 (수업 범위 내)

| 유형 | 기법 |
|------|------|
| 회귀 | 선형 회귀, 다중 선형 회귀, Ridge, Lasso, 로지스틱 회귀, K-NN 회귀 |
| 분류 | K-NN, Decision Tree, Random Forest, ANN/DNN, CNN, RNN |
| 비지도 | 군집화(Clustering), 차원 축소(PCA) |
| 기타 | 하이퍼파라미터 튜닝 |

### 알고리즘 설명 자료

- 문서형 정리: [`ALGORITHMS.md`](ALGORITHMS.md)
- 삼성 중고가 예측 모델링 계획: [`references/samsung-used-price-modeling-plan.md`](references/samsung-used-price-modeling-plan.md)

---

## 필수 수행 단계

1. **문제 정의**
2. **데이터 수집** — 크롤링, 수업 데이터, 공개 데이터셋 모두 허용
3. **데이터 전처리**
4. **모델 설계 및 적용**
5. **성능 평가**
6. **결과 분석 및 해석** (인사이트 도출)

---

## 제출물

| 항목 | 요건 |
|------|------|
| 결과 보고서 | 10~15페이지, 그래프·결과·해석 포함 필수 |
| 코드 | Python 기반, 실행 가능 상태 |
| 발표 동영상 | 최종 발표 영상 |
| 기여도 | 팀원별 기여도(%) 및 역할 분담 내역 |

---

## 평가 기준

| 항목 | 비중 |
|------|------|
| 문제 정의의 적절성 | 20% |
| 데이터 처리 및 전처리 | 20% |
| 모델 설계 및 적용 | 20% |
| 수업 내용의 이해 및 적용 | 20% |
| 성능 평가 및 결과 해석 | 20% |

> 높은 정확도보다 **모델 원리 이해, 선택 근거, 결과 해석**이 더 중요하게 평가됨

