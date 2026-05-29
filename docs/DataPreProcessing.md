# 온라인 쇼핑 세션 데이터 전처리 - KNN / Decision Tree / Random Forest

> **한 줄 요약**: `online_shoppers_intention.csv`를 **공통 전처리(로드→결측치→타깃→split) → 모델별 분기(KNN / Tree)** 의 2단 구조로 처리한다. KNN은 거리, Tree는 분할로 데이터를 보기 때문에 같은 데이터에 다른 전처리를 적용해야 공정 비교가 된다.

실행 노트북: [`notebooks/preprocessing.ipynb`](../notebooks/preprocessing.ipynb)

---

## 1. 데이터 개요

| 항목 | 값 |
|---|---|
| 출처 | UCI `online_shoppers_intention.csv` |
| 크기 | 12,330 세션 × 18 컬럼 |
| 타깃 | `Revenue` (구매 True/False) |
| 양성 비율 | 15.47% (불균형) |
| 결측치 | 없음 |

---

## 2. 공통 전처리

세 모델이 같은 train/test 쌍을 공유하도록 split까지만 공통으로 수행. fit이 필요한 변환은 분기 이후에 두어 누수 방지.

```python
df = pd.read_csv('../data/online_shoppers_intention.csv')
df.isnull().sum().sum()        # 0
y = df['Revenue'].astype(int)
X = df.drop(columns=['Revenue'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 9,864 / 2,466, 양성 비율 양쪽 모두 ~15.5%
```

---

## 3. 모델별로 갈리는 이유

| 구분 | KNN | DT / RF |
|---|---|---|
| 데이터를 보는 방식 | 좌표 공간의 점 | 변수별 임계값 분할 |
| 스케일·분포·차원 의존 | **있음** | 없음 (순서만 봄, 단조 불변) |
| ID 변수 | 거리에 잡음 → 제거 | 트리가 자동 판단 → 유지 |
| 전략 | **차원 최소화** | **정보 최대 보존** |

DT와 RF는 같은 분할 기반이라 입력 파이프라인을 공유.

---

## 4. KNN 파이프라인

### 4-1. 컬럼 선택 (제거 6개 + 압축 1개)

| 처리 | 컬럼 | 이유 |
|---|---|---|
| 제거 | `Month` | 1-12 거리=11이 부자연스러움. 신호는 다른 행동 지표에 간접 반영 |
| 제거 | `OperatingSystems`, `Browser`, `Region`, `TrafficType` | ID 정수에 거리 부여하면 잡음 |
| 압축 | `VisitorType` → `is_new_visitor` Boolean | One-Hot(+3차원) 대신 가장 강한 신호(New 24.9% vs Returning 13.9%) 한 비트로 |

### 4-2. PageValues 누수 판단

| 단계 | 결과 |
|---|---|
| 1. GA 정의 | `transactionRevenue / uniquePageviews` — 거래 후 역계산 |
| 2. 분포 검증 | `=0` 구매율 3.7% vs `>0` 56.5% (노트북 출력) — 분리 비정상 |
| 3. 실험 검증 | 제거 시 KNN AUC 0.88 → 0.74 (KNN.md §9-1) |
| 4. 시점 검증 | GA는 세션 종료 후 집계 → 실시간 예측 불가 |

**결정**: 메인에서 제거, 비교용으로만 병기.

### 4-3. 변환 + 오버샘플링

```python
# log1p: long-tail을 압축해 거리 독점 방지
X[skewed] = np.log1p(X[skewed])
# StandardScaler: 단위 균등화 (fit은 train만)
scaler = StandardScaler().fit(X_tr); X_tr_s = scaler.transform(X_tr)
# SMOTE: 양성 합성으로 균형 (test엔 적용 X)
X_tr_res, y_tr_res = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr_s, y_tr)
```

| 결정 | 이유 |
|---|---|
| log1p | 체류 시간/페이지 카운트의 long-tail이 거리 계산을 지배하지 않도록 |
| StandardScaler | 단위가 다른 변수가 거리에 균등 기여 |
| SMOTE 위치 = 스케일링 이후 | SMOTE는 거리로 합성 → 스케일 안 맞으면 합성 위치가 큰 단위 변수에 끌림 |
| SMOTE 적용 = train만 | test는 실제 분포(15.5%)에서 평가해야 정직 |

### 4-4. 최종 11개 컬럼

수치형 9개(`Administrative*`, `Informational*`, `ProductRelated*`, `BounceRates`, `ExitRates`, `SpecialDay`) + Boolean 2개(`Weekend`, `is_new_visitor`).

원본 17개 → 11개로 차원 축소. SMOTE 후 학습셋 16,676 / test 2,466.

---

## 5. Tree (DT / RF) 파이프라인

### 5-1. PageValues 제거 (KNN과 동일)

같은 데이터 정의 위에서 비교하기 위해 동일하게 처리.

### 5-2. 파생 변수 4개

트리는 변수 간 비율/곱을 직접 표현 못 함 → 명시적으로 만들어 분할 깊이를 줄임.

| 변수 | 정의 |
|---|---|
| `total_pages` | `Administrative + Informational + ProductRelated` |
| `total_duration` | 세 카테고리 체류 시간 합 |
| `avg_time_per_product` | `ProductRelated_Duration / (ProductRelated + 1)` |
| `bounce_exit_ratio` | `BounceRates / (ExitRates + 1e-6)` |

### 5-3. 인코딩과 ID 처리

```python
X = pd.get_dummies(X, columns=['Month','VisitorType'], drop_first=False)
# ID 변수(OperatingSystems, Browser, Region, TrafficType)는 유지
```

- One-Hot으로 차원이 늘어도 트리는 정보 이득으로 무용 변수 자동 무시
- log/스케일링 생략 — 분할은 단조 변환 불변

### 5-4. 불균형 처리

| 위치 | 방법 |
|---|---|
| 학습 시 | `class_weight='balanced'` |
| 운영 시 | 임계값 튜닝 (Precision↑ ↔ Recall↑) |
| SMOTE | 권장 안 함 (트리는 양성 영역에서 이미 분할 가능, 합성 노이즈만 늘림) |

### 5-5. 학습 후 2차 가지치기 (선택)

```python
imp = pd.Series(rf.feature_importances_, index=X_train.columns)
keep = imp[imp >= 0.005].index
# 핵심 변수만 남기고 재학습 → 단순화 + 해석성 향상
```

### 5-6. 최종 ~31개 컬럼

원본 17개 - PageValues + 파생 4개 + Month One-Hot 10개 + VisitorType One-Hot 3개 ≈ 31개.

---

## 6. 요약 비교

| 항목 | KNN | Tree (DT/RF) |
|---|---|---|
| 전략 | 차원 최소화 | 정보 최대 보존 |
| 인코딩 | Ordinal/Boolean 압축 | One-Hot |
| ID 변수 | 제거 | 유지 |
| log1p / Scaler | 적용 | 생략 |
| 불균형 | SMOTE (train만) | `class_weight='balanced'` + 임계값 |
| PageValues | 제거 (누수) | 제거 (동일 판단) |
| 최종 컬럼 수 | 11 | ~31 |
