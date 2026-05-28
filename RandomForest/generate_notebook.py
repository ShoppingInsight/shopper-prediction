import nbformat as nbf
from pathlib import Path

def build_notebook():
    nb = nbf.v4.new_notebook()
    
    # ----------------------------------------------------
    # 1. 문서 제목 및 개요 (Markdown)
    # ----------------------------------------------------
    m1 = """# 📊 [데이터마이닝 텀프로젝트] 온라인 쇼핑객 CSV 데이터셋 기반 Revenue 예측 및 분석

## 📝 프로젝트 개요
* **교과목**: 데이터마이닝 (텀프로젝트 기말 대체)
* **주제**: 온라인 쇼핑몰 방문 세션 데이터로 구매 전환 여부(`Revenue`)를 예측하고 주요 패턴을 해석
* **핵심 목표: Revenue 예측**: CSV 데이터셋 자체를 탐색하고, 전처리와 모델 비교를 통해 구매 전환 예측 모델을 구축
* **보조 점검**: `PageValues`는 구매 여부와 관련성이 큰 변수이므로 포함/제외 결과를 비교해 모델 해석의 신뢰성을 확인
* **구현 알고리즘**: 3종 분류 알고리즘 비교 (Logistic Regression, Decision Tree, Random Forest)
* **최적화**: 메인 모델인 **Random Forest**에 대해 GridSearchCV를 이용한 하이퍼파라미터 최적화 수행
"""
    
    # ----------------------------------------------------
    # 2. 문제 정의 (Markdown)
    # ----------------------------------------------------
    m2 = """## 1. 문제 정의 및 분석 배경
온라인 쇼핑몰 환경에서 방문 고객의 이탈을 막고 실제 구매(`Revenue = True`)로 전환시키는 것은 마케팅 비용 효율화의 핵심입니다. 
본 분석의 목적은 주어진 `online_shoppers_intention.csv` 데이터셋에 포함된 세션 행동 정보, 방문 환경 정보, 방문자 특성을 활용해 구매 전환 여부를 예측하는 데이터마이닝 모델을 개발하고 해석하는 것입니다.

분석 흐름은 먼저 데이터의 구조와 변수 분포를 파악하고, 범주형/수치형 변수를 전처리한 뒤, 여러 분류 모델의 성능을 비교하는 방식으로 구성합니다. 또한 `PageValues`는 구매 여부와 강한 관련성을 보이는 변수이므로 PageValues는 보조적으로 점검하여, 전체 변수를 사용했을 때와 해당 변수를 제외했을 때 예측 성능과 변수 중요도가 어떻게 달라지는지 함께 확인합니다.
"""

    # ----------------------------------------------------
    # 3. 환경 설정 및 데이터 로드 (Code)
    # ----------------------------------------------------
    c3 = """# 1. 라이브러리 및 데이터 로드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 경고 무시 및 시각화 기본 한글 설정
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
csv_candidates = [
    Path('online_shoppers_intention.csv'),
    Path('..') / 'online_shoppers_intention.csv'
]
csv_path = next((path for path in csv_candidates if path.exists()), None)
if csv_path is None:
    raise FileNotFoundError('online_shoppers_intention.csv 파일을 현재 폴더 또는 상위 폴더에서 찾을 수 없습니다.')

df = pd.read_csv(csv_path)
print(f"데이터셋 형태: {df.shape}")
df.head()
"""

    c4 = """# 데이터 기본 정보 확인
print("=== 데이터 요약 정보 ===")
print(df.info())
print("\\n=== 수치형 데이터 기술통계 ===")
print(df.describe())
"""

    # ----------------------------------------------------
    # 4. 탐색적 데이터 분석 (EDA) 및 시각화 (Markdown & Code)
    # ----------------------------------------------------
    m5 = """## 2. 탐색적 데이터 분석 (EDA) 및 시각화
주요 변수들의 분포 및 타겟 변수(`Revenue`)와의 관계를 시각적으로 정밀하게 살펴봅니다."""

    c6 = """# 2.1 타겟 변수 Revenue 불균형도 분석
plt.figure(figsize=(12, 5))

# 파이차트 시각화
plt.subplot(1, 2, 1)
df['Revenue'].value_counts().plot.pie(
    explode=[0, 0.15], 
    autopct='%1.1f%%', 
    shadow=True, 
    startangle=90, 
    colors=['#5470c6', '#91cc75']
)
plt.title('구매 전환 여부 (Revenue) 비율')
plt.ylabel('')

# 바차트 시각화
plt.subplot(1, 2, 2)
sns.countplot(x='Revenue', data=df, palette='Set2')
plt.title('구매 전환 여부 (Revenue) 빈도 수')
plt.xlabel('Revenue (구매 여부)')
plt.ylabel('방문 세션 수')

plt.tight_layout()
plt.show()

# 실제 카운트 값 출력
print("Revenue 빈도 상세:")
print(df['Revenue'].value_counts())
"""

    c7 = """# 2.2 PageValues와 Revenue 간의 상관성 분석
plt.figure(figsize=(10, 6))
sns.boxplot(x='Revenue', y='PageValues', data=df, palette='Set3')
plt.title('구매 전환 여부에 따른 PageValues 분포 비교 (Log Scale 적용)')
plt.yscale('symlog') # 편차가 크므로 심로그 스케일 적용
plt.xlabel('Revenue (구매 여부)')
plt.ylabel('PageValues (페이지 가치, Log Scale)')
plt.show()

# 두 그룹의 PageValues 기술통계 비교
print("구매 전환 실패 그룹의 PageValues 기술통계:")
print(df[df['Revenue'] == False]['PageValues'].describe())
print("\\n구매 전환 성공 그룹의 PageValues 기술통계:")
print(df[df['Revenue'] == True]['PageValues'].describe())
"""

    c8 = """# 2.3 방문자 유형(VisitorType) 및 주말 여부(Weekend) 세그먼트별 구매 전환 비율
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(x='VisitorType', hue='Revenue', data=df, ax=axes[0], palette='pastel')
axes[0].set_title('방문자 유형별 구매 전환 빈도')
axes[0].set_xlabel('방문자 유형')
axes[0].set_ylabel('세션 수')

sns.countplot(x='Weekend', hue='Revenue', data=df, ax=axes[1], palette='pastel')
axes[1].set_title('주말 방문 여부별 구매 전환 빈도')
axes[1].set_xlabel('주말 여부')
axes[1].set_ylabel('세션 수')

plt.tight_layout()
plt.show()
"""

    # ----------------------------------------------------
    # 5. 데이터 전처리 및 실험 분할 (Markdown & Code)
    # ----------------------------------------------------
    m9 = """## 3. 데이터 전처리 및 모델 입력 데이터셋 구성
모델 학습을 위해 범주형 변수를 변환하고 연속형 변수를 정규화한 뒤, 
기본적으로 전체 변수를 활용한 예측 데이터를 구성합니다. 추가로 `PageValues` 영향도를 확인하기 위해 해당 변수를 제외한 비교용 데이터셋도 함께 구성합니다."""

    c10 = """# 3.1 범주형 데이터 인코딩 및 Boolean 매핑
df_prep = df.copy()

# Boolean 피처 0과 1로 변환
df_prep['Weekend'] = df_prep['Weekend'].astype(int)
df_prep['Revenue'] = df_prep['Revenue'].astype(int)

# 범주형 데이터 원핫 인코딩 수행
categorical_cols = ['Month', 'VisitorType']
df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=False)

# One-hot 인코딩으로 생성된 bool 피처들도 0/1 정수로 변환
for col in df_prep.columns:
    if df_prep[col].dtype == 'bool':
        df_prep[col] = df_prep[col].astype(int)

print(f"전처리 후 피처 차원: {df_prep.shape}")
df_prep.head()
"""

    c11 = """# 3.2 수치형 변수 Standard Scaling 및 실험군 분할
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 독립변수(X)와 종속변수(y) 분리
y = df_prep['Revenue']
X_all = df_prep.drop(columns=['Revenue'])

# 1) 전체 변수 데이터셋: PageValues 포함 17개 피처
X_a = X_all.copy()

# 2) 비교용 데이터셋: PageValues 제외 16개 피처
X_b = X_all.drop(columns=['PageValues'])

# 연속형(Numerical) 변수 정규화 대상 선정
num_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'SpecialDay']

# 전체 변수 데이터셋 전용 연속형 컬럼 (PageValues 포함)
num_cols_a = num_cols + ['PageValues']
# 비교용 데이터셋 전용 연속형 컬럼 (PageValues 제외)
num_cols_b = num_cols

# 각각의 독립변수에 대해 독립적으로 스케일러 피팅 및 변형
scaler_a = StandardScaler()
X_a[num_cols_a] = scaler_a.fit_transform(X_a[num_cols_a])

scaler_b = StandardScaler()
X_b[num_cols_b] = scaler_b.fit_transform(X_b[num_cols_b])

print("데이터 스케일링 및 전처리 완료.")
"""

    c12 = """# 3.3 Stratified Split을 적용한 Train/Test 데이터셋 분할 (80:20)
# 데이터 불균형 상태이므로 stratify=y 옵션으로 비율을 균등 배분합니다.
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y, test_size=0.2, random_state=42, stratify=y)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y, test_size=0.2, random_state=42, stratify=y)

print(f"[전체 변수 데이터셋] Train: {X_train_a.shape}, Test: {X_test_a.shape}")
print(f"[PageValues 제외 비교용 데이터셋] Train: {X_train_b.shape}, Test: {X_test_b.shape}")
"""

    # ----------------------------------------------------
    # 6. 모델 설계 및 적용 (Markdown & Code)
    # ----------------------------------------------------
    m13 = """## 4. 분류 모델 설계 및 학습
`Revenue` 예측을 위해 Logistic Regression, Decision Tree, Random Forest의 3가지 분류기를 적용합니다. 전체 변수 데이터셋을 중심으로 성능을 확인하고, `PageValues`를 제외한 비교용 데이터셋을 함께 학습해 특정 변수 의존도를 점검합니다.

### ⚙️ 학습 모델 개요
1. **Logistic Regression (로지스틱 회귀)**: 선형 관계를 기반으로 확률을 시그모이드 함수로 압축하여 예측하는 기준 모델입니다. L2 Regularization을 활성화하여 과적합을 방지합니다.
2. **Decision Tree (의사결정나무)**: 설명력과 가독성이 매우 뛰어난 비모수적 모델입니다. 지나치게 깊어져 발생하는 과적합을 제어하기 위해 `max_depth=5`로 제한합니다.
3. **Random Forest (랜덤 포레스트)**: 수많은 의사결정나무를 앙상블(Bagging)하여 분산을 줄이고 예측 정확도를 높인 강력한 기법입니다. 데이터 불균형 제어를 위해 `class_weight='balanced'`를 적용합니다.
"""

    c14 = """from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 3종 모델 초기화 (일관된 random_state 유지)
lr_model_a = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
dt_model_a = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
rf_model_a = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

lr_model_b = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
dt_model_b = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
rf_model_b = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 전체 변수 데이터셋 기본 학습 진행
print("=== [전체 변수 데이터셋: PageValues 포함] 기본 3종 모델 학습 시작 ===")
lr_model_a.fit(X_train_a, y_train_a)
dt_model_a.fit(X_train_a, y_train_a)
rf_model_a.fit(X_train_a, y_train_a)
print("전체 변수 데이터셋 모델 학습 완료.")

# PageValues 제외 비교용 데이터셋 기본 학습 진행
print("\\n=== [PageValues 제외 비교용 데이터셋] 기본 3종 모델 학습 시작 ===")
lr_model_b.fit(X_train_b, y_train_b)
dt_model_b.fit(X_train_b, y_train_b)
rf_model_b.fit(X_train_b, y_train_b)
print("PageValues 제외 비교용 데이터셋 모델 학습 완료.")
"""

    # ----------------------------------------------------
    # 7. 하이퍼파라미터 튜닝 (Markdown & Code)
    # ----------------------------------------------------
    m15 = """## 5. Random Forest 하이퍼파라미터 최적화 (GridSearchCV)
본 분석의 주요 예측 모델인 Random Forest에 대해 교차 검증 기반 그리드 서치를 수행하여 `Revenue` 예측 성능을 개선합니다."""

    c16 = """from sklearn.model_selection import GridSearchCV

# 그리드 서치 탐색 파라미터 격자 정의
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [2, 4]
}

# 1) 전체 변수 데이터셋 Random Forest 튜닝
print("=== [전체 변수 데이터셋] Random Forest Grid Search 시작 (Stratified 5-Fold) ===")
grid_rf_a = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_rf_a.fit(X_train_a, y_train_a)
best_rf_a = grid_rf_a.best_estimator_
print(f"전체 변수 데이터셋 최적 파라미터: {grid_rf_a.best_params_}")
print(f"전체 변수 데이터셋 최적 Train F1-Score: {grid_rf_a.best_score_:.4f}")

# 2) PageValues 제외 비교용 데이터셋 Random Forest 튜닝
print("\\n=== [PageValues 제외 비교용 데이터셋] Random Forest Grid Search 시작 (Stratified 5-Fold) ===")
grid_rf_b = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_rf_b.fit(X_train_b, y_train_b)
best_rf_b = grid_rf_b.best_estimator_
print(f"PageValues 제외 비교용 데이터셋 최적 파라미터: {grid_rf_b.best_params_}")
print(f"PageValues 제외 비교용 데이터셋 최적 Train F1-Score: {grid_rf_b.best_score_:.4f}")
"""

    # ----------------------------------------------------
    # 8. 성능 평가 및 비교 분석 (Markdown & Code)
    # ----------------------------------------------------
    m17 = """## 6. Revenue 예측 성능 평가 및 변수 영향 분석
테스트 데이터셋을 기준으로 모든 모델의 `Revenue` 예측 능력을 Accuracy, Precision, Recall, F1-Score, ROC-AUC로 평가합니다. 
이후 Random Forest의 혼동 행렬, ROC Curve, 변수 중요도를 확인해 어떤 정보가 구매 전환 예측에 크게 기여했는지 해석합니다."""

    c18 = """# 6.1 성능 지표 측정 함수 정의 및 일괄 스코어링
def evaluate_models(models_dict, X_test, y_test, case_name):
    records = []
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        # 확률 추출 (ROC-AUC 계산용)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        records.append({
            '실험군': case_name,
            '모델명': model_name,
            '정확도(Accuracy)': acc,
            '정밀도(Precision)': prec,
            '재현율(Recall)': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
    return pd.DataFrame(records)

# Case A 모델 딕셔너리
models_a = {
    'Logistic Regression': lr_model_a,
    'Decision Tree (depth=5)': dt_model_a,
    'Random Forest (기본)': rf_model_a,
    'Random Forest (최적화)': best_rf_a
}

# Case B 모델 딕셔너리
models_b = {
    'Logistic Regression': lr_model_b,
    'Decision Tree (depth=5)': dt_model_b,
    'Random Forest (기본)': rf_model_b,
    'Random Forest (최적화)': best_rf_b
}

# 스코어 측정
results_a = evaluate_models(models_a, X_test_a, y_test_a, '전체 변수 데이터셋')
results_b = evaluate_models(models_b, X_test_b, y_test_b, 'PageValues 제외 비교용')

# 종합 성능 비교표 출력
results_total = pd.concat([results_a, results_b], ignore_index=True)
results_total
"""

    c19 = """# 6.2 혼동 행렬(Confusion Matrix) 비교 시각화
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 전체 변수 데이터셋 최종 모델
y_pred_a = best_rf_a.predict(X_test_a)
cm_a = confusion_matrix(y_test_a, y_pred_a)
sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['비구매(0)', '구매(1)'], yticklabels=['비구매(0)', '구매(1)'])
axes[0].set_title('전체 변수 데이터셋 최적 Random Forest 오차행렬')
axes[0].set_xlabel('예측값 (Predicted)')
axes[0].set_ylabel('실제값 (Actual)')

# PageValues 제외 비교용 데이터셋 최종 모델
y_pred_b = best_rf_b.predict(X_test_b)
cm_b = confusion_matrix(y_test_b, y_pred_b)
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['비구매(0)', '구매(1)'], yticklabels=['비구매(0)', '구매(1)'])
axes[1].set_title('PageValues 제외 비교용 최적 Random Forest 오차행렬')
axes[1].set_xlabel('예측값 (Predicted)')
axes[1].set_ylabel('실제값 (Actual)')

plt.tight_layout()
plt.show()
"""

    c20 = """# 6.3 ROC-AUC 커브 오버랩 비교 시각화
from sklearn.metrics import roc_curve

plt.figure(figsize=(10, 7))

# 전체 변수 데이터셋 RF 최적화
y_prob_a = best_rf_a.predict_proba(X_test_a)[:, 1]
fpr_a, tpr_a, _ = roc_curve(y_test_a, y_prob_a)
auc_a = roc_auc_score(y_test_a, y_prob_a)
plt.plot(fpr_a, tpr_a, label=f'전체 변수 데이터셋 RF (AUC = {auc_a:.4f})', color='#1f77b4', lw=2)

# PageValues 제외 비교용 데이터셋 RF 최적화
y_prob_b = best_rf_b.predict_proba(X_test_b)[:, 1]
fpr_b, tpr_b, _ = roc_curve(y_test_b, y_prob_b)
auc_b = roc_auc_score(y_test_b, y_prob_b)
plt.plot(fpr_b, tpr_b, label=f'PageValues 제외 비교용 RF (AUC = {auc_b:.4f})', color='#ff7f0e', lw=2)

# 기준 대각선
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.title('최종 Random Forest 모델 ROC Curve 비교')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
"""

    c21 = """# 6.4 변수 중요도(Feature Importance) 변화 정밀 비교
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 전체 변수 데이터셋 피처 중요도
importances_a = best_rf_a.feature_importances_
indices_a = np.argsort(importances_a)[::-1]
features_a = X_train_a.columns

sns.barplot(x=importances_a[indices_a][:10], y=features_a[indices_a][:10], ax=axes[0], palette='viridis')
axes[0].set_title('전체 변수 데이터셋 상위 10개 피처 중요도')
axes[0].set_xlabel('중요도 (Importance Score)')

# PageValues 제외 비교용 데이터셋 피처 중요도
importances_b = best_rf_b.feature_importances_
indices_b = np.argsort(importances_b)[::-1]
features_b = X_train_b.columns

sns.barplot(x=importances_b[indices_b][:10], y=features_b[indices_b][:10], ax=axes[1], palette='plasma')
axes[1].set_title('PageValues 제외 비교용 상위 10개 피처 중요도')
axes[1].set_xlabel('중요도 (Importance Score)')

plt.tight_layout()
plt.show()
"""

    # ----------------------------------------------------
    # 9. 결과 고찰 및 인사이트 도출 (Markdown)
    # ----------------------------------------------------
    m22 = """## 7. 분석 결과 고찰 및 비즈니스 인사이트 도출

### 🔎 Revenue 예측 모델 관점의 핵심 해석
* **모델 비교 결과**: Logistic Regression, Decision Tree, Random Forest를 비교한 결과, 비선형 관계와 변수 간 상호작용을 반영할 수 있는 Random Forest가 `Revenue` 예측에 가장 적합한 모델로 확인됩니다.
* **데이터셋 특성**: `Revenue=True` 비율이 낮은 불균형 데이터이므로 단순 정확도만으로 모델을 판단하기 어렵습니다. 따라서 구매 고객을 얼마나 잘 찾아내는지 확인하기 위해 Precision, Recall, F1-Score, ROC-AUC를 함께 해석해야 합니다.
* **PageValues 해석**: `PageValues`는 구매 전환과 강하게 연결된 변수이므로 전체 변수 모델의 성능을 크게 높입니다. 다만 본 프로젝트의 중심은 이 변수 하나의 유무가 아니라, 주어진 CSV 데이터셋의 여러 행동/방문 특성을 활용해 `Revenue`를 예측하고 그 결과를 해석하는 것입니다.
* **현실 적용 관점**: `PageValues`를 제외한 비교용 모델은 구매 직전 또는 사후 성격이 강한 정보를 덜 사용했을 때의 예측력을 보여줍니다. 따라서 두 결과를 함께 보면 데이터셋 안에서 어떤 변수가 강력한 신호인지, 그리고 어떤 행동 지표가 독립적으로 구매 예측에 기여하는지 구분할 수 있습니다.

### 💡 실무 마케팅 및 사이트 사용성(UX) 개선 액션 아이템
변수 중요도와 EDA 결과를 바탕으로 구매 전환 가능성과 관련이 큰 행동 지표를 중심으로 비즈니스 전략을 제안합니다.

1. **`ProductRelated_Duration` (상품 관련 페이지 체류 시간)**:
   * **인사이트**: 고객이 상품 관련 정보와 세부 설명에 깊게 체류할수록 구매 전환 확률이 높아집니다.
   * **마케팅 액션**: 체류 시간이 길지만 아직 결제하지 않고 이탈하려는 조짐을 보이는 고객에게 실시간 장바구니 할인 쿠폰이나 한정 수량 알림을 팝업으로 제공하여 결제를 유도합니다.

2. **`ExitRates` / `BounceRates` (이탈률 및 반송률)**:
   * **인사이트**: 이탈률이 높은 특정 페이지나 카테고리는 사용자의 흐름을 방해하는 장애물이 존재함을 암시합니다.
   * **UX 개선**: 해당 페이지들의 로딩 속도 개선, 직관적인 UI 재배치, 결제 단계 단순화 프로세스(One-click Checkout)를 설계하여 이탈 허들을 최소화합니다.

3. **`Administrative_Duration` (관리/고객센터/회원 정보 페이지 체류 시간)**:
   * **인사이트**: 관리 페이지나 고객 안내 페이지에서 지나치게 오랜 시간을 보내는 것은 결제 절차 및 회원 가입에서의 애로사항을 뜻할 수 있습니다.
   * **사용성 개선**: 카카오페이/네이버페이 등의 간편 가입 및 간편 결제를 도입하여 번거로운 회원 프로세스를 간소화합니다.
"""

    # 노트북 빌드
    nb['cells'] = [
        nbf.v4.new_markdown_cell(m1),
        nbf.v4.new_markdown_cell(m2),
        nbf.v4.new_code_cell(c3),
        nbf.v4.new_code_cell(c4),
        nbf.v4.new_markdown_cell(m5),
        nbf.v4.new_code_cell(c6),
        nbf.v4.new_code_cell(c7),
        nbf.v4.new_code_cell(c8),
        nbf.v4.new_markdown_cell(m9),
        nbf.v4.new_code_cell(c10),
        nbf.v4.new_code_cell(c11),
        nbf.v4.new_code_cell(c12),
        nbf.v4.new_markdown_cell(m13),
        nbf.v4.new_code_cell(c14),
        nbf.v4.new_markdown_cell(m15),
        nbf.v4.new_code_cell(c16),
        nbf.v4.new_markdown_cell(m17),
        nbf.v4.new_code_cell(c18),
        nbf.v4.new_code_cell(c19),
        nbf.v4.new_code_cell(c20),
        nbf.v4.new_code_cell(c21),
        nbf.v4.new_markdown_cell(m22)
    ]
    
    output_filename = "RandomForest.ipynb"
    with open(output_filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Jupyter Notebook '{output_filename}' 생성 완료.")

if __name__ == "__main__":
    build_notebook()
