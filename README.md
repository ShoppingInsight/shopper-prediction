# DataMiningTerm
한성대학교 컴퓨터공학부 데이터마이닝 텀프로젝트 (기말 대체)
2026.06.24 까지 
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

- 문서형 정리: [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md)

---

## 디렉토리 구조

| 경로 | 설명 |
|---|---|
| `data/` | 원본 온라인 쇼핑 세션 CSV 데이터셋 |
| `docs/` | 알고리즘, 전처리, 수업 참고 자료 문서 |
| `models/knn/` | KNN 구매 확률 예측 노트북과 발표 자료 |
| `models/random_forest/` | Random Forest 분석 노트북과 발표 자료 |
| `models/clustering/` | K-Means 행동 기반 방문자 세분화(비지도) 노트북 |
| `notebooks/` | 모델 공통 전처리 실행 노트북 |

---

## 필수 수행 단계
0. 1, (2, 3), 4, (5, 6) 이 한세트임 
1. **문제 정의**
2. **데이터 수집** — 크롤링, 수업 데이터, 공개 데이터셋 모두 허용
3. **데이터 전처리**
4. **모델 설계 및 적용**    >> % 정도의 결과 (성능, 정확도) 
5. **성능 평가**            
6. **결과 분석 및 해석**    >> ~한 특성에 따른 (근본적인 원리, 이해) + 인사이트 도출


---

## 제출물

| 항목 | 요건 |
|------|------|
| 결과 보고서 | 10~15페이지, 그래프·결과·해석 포함 필수 |
| 코드 | Python 기반, 실행 가능 상태 |
| 발표 동영상 | 최종 발표 영상(5~10분정도, 별도의 ppt 가능) |
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

