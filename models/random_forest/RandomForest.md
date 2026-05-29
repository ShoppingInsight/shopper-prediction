# 온라인 쇼핑객 Revenue 예측 - Random Forest 분석 자료

> **한 줄 요약**: `data/online_shoppers_intention.csv` 데이터셋의 방문 행동 정보를 활용해 구매 전환 여부(`Revenue`)를 예측하고, Logistic Regression, Decision Tree, Random Forest를 비교한 뒤 Random Forest를 중심으로 성능과 변수 중요도를 해석했다.

## 폴더 구성

| 파일 | 내용 |
|---|---|
| `RandomForest.ipynb` | EDA, 전처리, 모델 학습, GridSearchCV 튜닝, 성능 평가를 포함한 실행 노트북 |
| `presentation.html` | 노트북 실제 실행 결과와 평가 지표를 해석한 HTML 보고서 |
| `random_forest_guide.html` | Random Forest 원리와 이번 분석 결과를 쉽게 설명한 학습용 HTML |
| `generate_notebook.py` | `RandomForest.ipynb`를 재생성하는 스크립트 |

## 분석 목표

이번 분석의 중심 목표는 특정 변수 하나의 유무를 검증하는 것이 아니라, 주어진 온라인 쇼핑객 CSV 데이터셋을 이용해 방문 세션이 실제 구매로 이어지는지 예측하는 것이다.

핵심 질문은 다음과 같다.

> **"방문자의 세션 행동 정보와 방문 환경 정보를 이용해 `Revenue=True`를 얼마나 잘 예측할 수 있는가?"**

## 데이터 개요

| 항목 | 내용 |
|---|---|
| 데이터 파일 | `data/online_shoppers_intention.csv` |
| 데이터 수 | 12,330개 방문 세션 |
| 원본 컬럼 수 | 18개 |
| 타겟 변수 | `Revenue` |
| 구매 비율 | 약 15.5% |

구매 클래스가 전체의 약 15.5%인 불균형 분류 문제이므로, 단순 정확도만으로 모델을 평가하지 않고 Precision, Recall, F1-Score, ROC-AUC를 함께 확인했다.

## 적용 모델

1. **Logistic Regression**
   - 선형 기준 모델로 사용했다.
   - 클래스 불균형을 고려하기 위해 `class_weight='balanced'`를 적용했다.

2. **Decision Tree**
   - 조건 기반 분류 규칙을 확인하기 위한 해석 가능한 모델로 사용했다.
   - 과적합을 줄이기 위해 `max_depth=5`로 제한했다.

3. **Random Forest**
   - 여러 Decision Tree를 앙상블하여 예측 안정성을 높이는 메인 모델로 사용했다.
   - `GridSearchCV`와 F1-Score 기준으로 하이퍼파라미터를 튜닝했다.

## PageValues 비교의 의미

`PageValues`는 구매 전환과 매우 강하게 연결되는 변수이므로, 전체 변수 모델과 `PageValues` 제외 비교용 모델을 함께 구성했다.

이 비교의 목적은 `PageValues` 유무 자체를 프로젝트의 메인 주제로 삼는 것이 아니라, 다음 두 가지를 확인하는 것이다.

1. 전체 CSV 데이터셋에서 구매 전환을 설명하는 강한 신호가 무엇인지 확인한다.
2. `PageValues` 없이도 상품 페이지 체류 시간, 이탈률, 반송률 같은 행동 지표가 `Revenue` 예측에 얼마나 기여하는지 확인한다.

## 주요 결과 해석

- 전체 변수 Random Forest는 가장 높은 ROC-AUC와 F1-Score를 보이며 `Revenue` 예측 성능이 가장 우수했다.
- `PageValues` 제외 모델은 정확도는 낮아졌지만, 튜닝 후 구매자 재현율이 크게 개선되었다.
- 변수 중요도에서는 전체 변수 모델에서 `PageValues`가 가장 강한 신호로 나타났고, 제외 모델에서는 `ProductRelated_Duration`, `ExitRates`, `ProductRelated`, `BounceRates` 등이 주요 행동 지표로 나타났다.

## 발표 흐름

1. 온라인 쇼핑객 구매 전환 예측 문제 정의
2. 데이터셋 구조와 타겟 불균형 확인
3. 범주형 변수 인코딩과 수치형 변수 스케일링
4. Logistic Regression, Decision Tree, Random Forest 비교
5. Random Forest 하이퍼파라미터 튜닝
6. Accuracy, Precision, Recall, F1-Score, ROC-AUC 해석
7. 혼동 행렬과 ROC Curve 해석
8. 변수 중요도를 통한 구매 전환 인사이트 도출
9. 최종 결론과 비즈니스 활용 방향

## 실행 방법

저장소 루트의 `data/online_shoppers_intention.csv`를 기준으로 실행한다.

```bash
pip install -r requirements.txt
jupyter notebook models/random_forest/RandomForest.ipynb
```

노트북을 다시 생성하려면 다음 명령을 실행한다.

```bash
cd models/random_forest
python generate_notebook.py
```
