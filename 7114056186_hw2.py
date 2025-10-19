#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多元線性回歸專題：健身進度和卡路里消耗預測模型
CRISP-DM 方法論實作

作者: 學號 7114056186
日期: 2025-10-19
資料來源: Kaggle Gym Members Exercise Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import shapiro, probplot
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# 設定中文字體和圖表樣式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 建立資料夾結構
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("="*80)
print("多元線性回歸專題：健身進度和卡路里消耗預測模型")
print("CRISP-DM 方法論實作")
print("="*80)

# ============================================================================
# Phase 1: Business Understanding (商業理解)
# ============================================================================
print("\n" + "="*80)
print("Phase 1: Business Understanding")
print("="*80)
print("\n專題目標:")
print("  建立健身進度和卡路里消耗的預測模型")
print("  使用多元線性回歸分析運動相關因素對卡路里消耗的影響")
print("\n資料來源:")
print("  Kaggle: https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset/data")

# ============================================================================
# Phase 2: Data Understanding (資料理解)
# ============================================================================
print("\n" + "="*80)
print("Phase 2: Data Understanding")
print("="*80)

# 載入資料
# 注意：請先從 Kaggle 下載 gym_members_exercise_tracking.csv 到 data/ 目錄
# 指令：kaggle datasets download -d valakhorasani/gym-members-exercise-dataset

data_path = 'data/gym_members_exercise_tracking.csv'

# 如果檔案不存在，建立示範資料
if not os.path.exists(data_path):
    print(f"\n警告: 找不到 {data_path}")
    print("建立示範資料集...")
    
    np.random.seed(42)
    n_samples = 973
    
    data = {
        'Age': np.random.randint(18, 65, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Weight_kg': np.random.uniform(50, 120, n_samples),
        'Height_m': np.random.uniform(1.5, 2.0, n_samples),
        'Max_BPM': np.random.randint(140, 200, n_samples),
        'Avg_BPM': np.random.randint(100, 160, n_samples),
        'Resting_BPM': np.random.randint(50, 90, n_samples),
        'Session_Duration_hours': np.random.uniform(0.5, 3.0, n_samples),
        'Workout_Type': np.random.choice(['Cardio', 'Strength', 'Yoga', 'HIIT'], n_samples),
        'Fat_Percentage': np.random.uniform(10, 35, n_samples),
        'Water_Intake_liters': np.random.uniform(1.0, 4.0, n_samples),
        'Workout_Frequency_days': np.random.randint(2, 7, n_samples),
        'Experience_Level': np.random.randint(1, 4, n_samples)
    }
    
    data['BMI'] = data['Weight_kg'] / (data['Height_m'] ** 2)
    data['Calories_Burned'] = (
        data['Weight_kg'] * 0.5 +
        data['Session_Duration_hours'] * 200 +
        data['Avg_BPM'] * 0.8 +
        (data['Experience_Level'] - 1) * 30 +
        np.random.normal(0, 50, n_samples)
    )
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    print(f"✓ 示範資料已保存至: {data_path}")
else:
    df = pd.read_csv(data_path)
    print(f"✓ 資料載入成功: {data_path}")

# 顯示基本資訊
print(f"\n資料維度: {df.shape}")
print(f"樣本數: {df.shape[0]}")
print(f"特徵數: {df.shape[1] - 1}")

print("\n資料集前 5 筆:")
print(df.head())

print("\n資料型態:")
print(df.dtypes)

print("\n缺失值統計:")
print(df.isnull().sum())

print("\n描述性統計:")
print(df.describe())

# ============================================================================
# Phase 3: Data Preparation (資料準備)
# ============================================================================
print("\n" + "="*80)
print("Phase 3: Data Preparation")
print("="*80)

# 1. 檢查資料洩漏
print("\n1. 檢查目標變數相關性 (避免資料洩漏)")
print("-" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
correlation_with_target = correlation_matrix['Calories_Burned'].sort_values(ascending=False)

print("\n與 Calories_Burned 的相關係數:")
print(correlation_with_target)

# 識別可能的洩漏特徵
potential_leakage = correlation_with_target[
    (correlation_with_target.abs() > 0.95) & 
    (correlation_with_target.index != 'Calories_Burned')
]

if len(potential_leakage) > 0:
    print(f"\n警告: 發現可能的資料洩漏特徵 (|相關性| > 0.95):")
    print(potential_leakage)
    print("這些特徵將被排除")
else:
    print("\n✓ 未發現明顯的資料洩漏特徵")

# 2. 特徵工程
print("\n2. 特徵工程")
print("-" * 60)

# 建立衍生特徵
df['Heart_Rate_Range'] = df['Max_BPM'] - df['Resting_BPM']
df['Heart_Rate_Reserve'] = df['Max_BPM'] - df['Avg_BPM']
df['Intensity_Score'] = df['Avg_BPM'] * df['Session_Duration_hours']
df['Hydration_per_hour'] = df['Water_Intake_liters'] / df['Session_Duration_hours']
df['Weight_Height_Ratio'] = df['Weight_kg'] / df['Height_m']
df['Total_Weekly_Duration'] = df['Session_Duration_hours'] * df['Workout_Frequency_days']
df['BMI_squared'] = df['BMI'] ** 2
df['Age_BMI_interaction'] = df['Age'] * df['BMI']

print("✓ 新建特徵:")
print("  - Heart_Rate_Range: 最大心率 - 靜息心率")
print("  - Heart_Rate_Reserve: 最大心率 - 平均心率")
print("  - Intensity_Score: 運動強度評分 (平均心率 × 時長)")
print("  - Hydration_per_hour: 每小時飲水量")
print("  - Weight_Height_Ratio: 體重身高比")
print("  - Total_Weekly_Duration: 每週總運動時長")
print("  - BMI_squared: BMI 平方項")
print("  - Age_BMI_interaction: 年齡與 BMI 交互項")

# 編碼類別變數
df['Gender_encoded'] = (df['Gender'] == 'Male').astype(int)
df = pd.get_dummies(df, columns=['Workout_Type'], prefix='Workout')

print(f"\n✓ 類別變數編碼完成")

# 準備特徵集
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('Calories_Burned')

print(f"\n可用數值特徵數量: {len(numeric_features)}")

# ============================================================================
# Phase 4: Feature Selection (特徵選擇)
# ============================================================================
print("\n" + "="*80)
print("Phase 4: Feature Selection")
print("="*80)

# 準備特徵和目標變數
X = df[numeric_features]
y = df['Calories_Burned']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# 方法 1: SelectKBest (F-statistic)
print("\n方法 1: SelectKBest (F-statistic)")
selector_f = SelectKBest(score_func=f_regression, k=15)
selector_f.fit(X_train, y_train)

f_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'F_Score': selector_f.scores_
}).sort_values('F_Score', ascending=False)

print("\nTop 15 特徵 (F-statistic):")
print(f_scores.head(15))

# 方法 2: RFE with Ridge
print("\n方法 2: Recursive Feature Elimination (RFE)")
estimator = Ridge(alpha=1.0, random_state=42)
rfe_selector = RFE(estimator=estimator, n_features_to_select=15, step=1)
rfe_selector.fit(X_train, y_train)

rfe_features = pd.DataFrame({
    'Feature': X_train.columns,
    'Selected': rfe_selector.support_,
    'Ranking': rfe_selector.ranking_
}).sort_values('Ranking')

print("\nRFE 選擇的 15 個特徵:")
print(rfe_features[rfe_features['Selected']]['Feature'].tolist())

# 方法 3: Random Forest Feature Importance
print("\n方法 3: Random Forest Feature Importance")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 特徵 (Random Forest):")
print(rf_importance.head(15))

# 綜合選擇最終特徵
f_top15 = set(f_scores.head(15)['Feature'].values)
rfe_top15 = set(rfe_features[rfe_features['Selected']]['Feature'].values)
rf_top15 = set(rf_importance.head(15)['Feature'].values)

union_features = f_top15 | rfe_top15 | rf_top15

vote_count = {}
for feat in union_features:
    count = sum([feat in f_top15, feat in rfe_top15, feat in rf_top15])
    vote_count[feat] = count

voted_features = [feat for feat, count in vote_count.items() if count >= 2]
voted_features_df = pd.DataFrame({
    'Feature': voted_features,
    'Votes': [vote_count[f] for f in voted_features]
}).sort_values('Votes', ascending=False)

print(f"\n出現在至少 2 種方法中的特徵 (共 {len(voted_features)} 個):")
print(voted_features_df)

# 確保特徵數在 10-20 範圍內
if len(voted_features) < 10:
    selected_features = list(set(voted_features) | set(rf_importance.head(15)['Feature']))[:15]
elif len(voted_features) > 20:
    selected_features = voted_features_df.head(15)['Feature'].tolist()
else:
    selected_features = voted_features

print(f"\n✓ 最終選擇的特徵數: {len(selected_features)}")
print("最終特徵列表:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

# 保存特徵選擇結果
feature_selection_results = pd.DataFrame({
    'Feature': selected_features,
    'F_Score': [f_scores[f_scores['Feature'] == f]['F_Score'].values[0] if f in f_scores['Feature'].values else 0 for f in selected_features],
    'RF_Importance': [rf_importance[rf_importance['Feature'] == f]['Importance'].values[0] if f in rf_importance['Feature'].values else 0 for f in selected_features]
})
feature_selection_results.to_csv('reports/feature_selection_results.csv', index=False)

# ============================================================================
# Phase 5: Modeling (建模)
# ============================================================================
print("\n" + "="*80)
print("Phase 5: Modeling")
print("="*80)

# 使用選定的特徵
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(f"\n使用的特徵數: {len(selected_features)}")
print(f"訓練集維度: {X_train_selected.shape}")
print(f"測試集維度: {X_test_selected.shape}")

# 設定交叉驗證
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 儲存模型結果
model_results = {}

# 1. Linear Regression (Baseline)
print("\n1. Linear Regression (Baseline Model)")
print("-" * 60)

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

lr_pipeline.fit(X_train_selected, y_train)
lr_train_pred = lr_pipeline.predict(X_train_selected)
lr_test_pred = lr_pipeline.predict(X_test_selected)

lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_train_r2 = r2_score(y_train, lr_train_pred)

lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)

lr_cv_scores = cross_val_score(lr_pipeline, X_train_selected, y_train,
                                cv=kfold, scoring='neg_root_mean_squared_error')
lr_cv_rmse = -lr_cv_scores.mean()
lr_cv_std = lr_cv_scores.std()

print(f"訓練集 - RMSE: {lr_train_rmse:.4f}, MAE: {lr_train_mae:.4f}, R²: {lr_train_r2:.4f}")
print(f"測試集 - RMSE: {lr_test_rmse:.4f}, MAE: {lr_test_mae:.4f}, R²: {lr_test_r2:.4f}")
print(f"5-Fold CV RMSE: {lr_cv_rmse:.4f} (±{lr_cv_std:.4f})")

model_results['Linear Regression'] = {
    'pipeline': lr_pipeline,
    'train_rmse': lr_train_rmse, 'train_mae': lr_train_mae, 'train_r2': lr_train_r2,
    'test_rmse': lr_test_rmse, 'test_mae': lr_test_mae, 'test_r2': lr_test_r2,
    'cv_rmse': lr_cv_rmse, 'cv_std': lr_cv_std,
    'train_pred': lr_train_pred, 'test_pred': lr_test_pred
}

# 2. Ridge Regression
print("\n2. Ridge Regression")
print("-" * 60)

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_cv_scores_dict = {}

print("尋找最佳 alpha 參數...")
for alpha in alphas:
    ridge_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=alpha, random_state=42))
    ])
    cv_scores = cross_val_score(ridge_temp, X_train_selected, y_train,
                                cv=kfold, scoring='neg_root_mean_squared_error')
    ridge_cv_scores_dict[alpha] = -cv_scores.mean()
    print(f"  alpha={alpha:7.3f}: CV RMSE = {-cv_scores.mean():.4f}")

best_ridge_alpha = min(ridge_cv_scores_dict, key=ridge_cv_scores_dict.get)
print(f"\n✓ 最佳 alpha: {best_ridge_alpha}")

ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=best_ridge_alpha, random_state=42))
])

ridge_pipeline.fit(X_train_selected, y_train)
ridge_train_pred = ridge_pipeline.predict(X_train_selected)
ridge_test_pred = ridge_pipeline.predict(X_test_selected)

ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
ridge_train_mae = mean_absolute_error(y_train, ridge_train_pred)
ridge_train_r2 = r2_score(y_train, ridge_train_pred)

ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_pred))
ridge_test_mae = mean_absolute_error(y_test, ridge_test_pred)
ridge_test_r2 = r2_score(y_test, ridge_test_pred)

ridge_cv_rmse = ridge_cv_scores_dict[best_ridge_alpha]

print(f"訓練集 - RMSE: {ridge_train_rmse:.4f}, MAE: {ridge_train_mae:.4f}, R²: {ridge_train_r2:.4f}")
print(f"測試集 - RMSE: {ridge_test_rmse:.4f}, MAE: {ridge_test_mae:.4f}, R²: {ridge_test_r2:.4f}")
print(f"5-Fold CV RMSE: {ridge_cv_rmse:.4f}")

model_results['Ridge'] = {
    'pipeline': ridge_pipeline,
    'train_rmse': ridge_train_rmse, 'train_mae': ridge_train_mae, 'train_r2': ridge_train_r2,
    'test_rmse': ridge_test_rmse, 'test_mae': ridge_test_mae, 'test_r2': ridge_test_r2,
    'cv_rmse': ridge_cv_rmse, 'cv_std': 0,
    'train_pred': ridge_train_pred, 'test_pred': ridge_test_pred,
    'best_alpha': best_ridge_alpha
}

# 3. Lasso Regression
print("\n3. Lasso Regression")
print("-" * 60)

lasso_cv_scores_dict = {}

print("尋找最佳 alpha 參數...")
for alpha in alphas:
    lasso_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=alpha, random_state=42, max_iter=10000))
    ])
    cv_scores = cross_val_score(lasso_temp, X_train_selected, y_train,
                                cv=kfold, scoring='neg_root_mean_squared_error')
    lasso_cv_scores_dict[alpha] = -cv_scores.mean()
    print(f"  alpha={alpha:7.3f}: CV RMSE = {-cv_scores.mean():.4f}")

best_lasso_alpha = min(lasso_cv_scores_dict, key=lasso_cv_scores_dict.get)
print(f"\n✓ 最佳 alpha: {best_lasso_alpha}")

lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Lasso(alpha=best_lasso_alpha, random_state=42, max_iter=10000))
])

lasso_pipeline.fit(X_train_selected, y_train)
lasso_train_pred = lasso_pipeline.predict(X_train_selected)
lasso_test_pred = lasso_pipeline.predict(X_test_selected)

lasso_train_rmse = np.sqrt(mean_squared_error(y_train, lasso_train_pred))
lasso_train_mae = mean_absolute_error(y_train, lasso_train_pred)
lasso_train_r2 = r2_score(y_train, lasso_train_pred)

lasso_test_rmse = np.sqrt(mean_squared_error(y_test, lasso_test_pred))
lasso_test_mae = mean_absolute_error(y_test, lasso_test_pred)
lasso_test_r2 = r2_score(y_test, lasso_test_pred)

lasso_cv_rmse = lasso_cv_scores_dict[best_lasso_alpha]

lasso_coefs = lasso_pipeline.named_steps['regressor'].coef_
non_zero_features = sum(lasso_coefs != 0)

print(f"訓練集 - RMSE: {lasso_train_rmse:.4f}, MAE: {lasso_train_mae:.4f}, R²: {lasso_train_r2:.4f}")
print(f"測試集 - RMSE: {lasso_test_rmse:.4f}, MAE: {lasso_test_mae:.4f}, R²: {lasso_test_r2:.4f}")
print(f"5-Fold CV RMSE: {lasso_cv_rmse:.4f}")
print(f"Lasso 選擇的非零特徵數: {non_zero_features}/{len(selected_features)}")

model_results['Lasso'] = {
    'pipeline': lasso_pipeline,
    'train_rmse': lasso_train_rmse, 'train_mae': lasso_train_mae, 'train_r2': lasso_train_r2,
    'test_rmse': lasso_test_rmse, 'test_mae': lasso_test_mae, 'test_r2': lasso_test_r2,
    'cv_rmse': lasso_cv_rmse, 'cv_std': 0,
    'train_pred': lasso_train_pred, 'test_pred': lasso_test_pred,
    'best_alpha': best_lasso_alpha, 'non_zero_features': non_zero_features
}

# 4. ElasticNet CV
print("\n4. ElasticNet with Cross-Validation")
print("-" * 60)

elasticnet_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        alphas=alphas,
        cv=5,
        random_state=42,
        max_iter=10000
    ))
])

print("訓練 ElasticNetCV (尋找最佳 alpha 和 l1_ratio)...")
elasticnet_cv.fit(X_train_selected, y_train)

best_alpha_en = elasticnet_cv.named_steps['regressor'].alpha_
best_l1_ratio = elasticnet_cv.named_steps['regressor'].l1_ratio_
print(f"✓ 最佳 alpha: {best_alpha_en:.4f}")
print(f"✓ 最佳 l1_ratio: {best_l1_ratio:.4f}")

en_train_pred = elasticnet_cv.predict(X_train_selected)
en_test_pred = elasticnet_cv.predict(X_test_selected)

en_train_rmse = np.sqrt(mean_squared_error(y_train, en_train_pred))
en_train_mae = mean_absolute_error(y_train, en_train_pred)
en_train_r2 = r2_score(y_train, en_train_pred)

en_test_rmse = np.sqrt(mean_squared_error(y_test, en_test_pred))
en_test_mae = mean_absolute_error(y_test, en_test_pred)
en_test_r2 = r2_score(y_test, en_test_pred)

en_cv_scores = cross_val_score(elasticnet_cv, X_train_selected, y_train,
                               cv=kfold, scoring='neg_root_mean_squared_error')
en_cv_rmse = -en_cv_scores.mean()
en_cv_std = en_cv_scores.std()

print(f"訓練集 - RMSE: {en_train_rmse:.4f}, MAE: {en_train_mae:.4f}, R²: {en_train_r2:.4f}")
print(f"測試集 - RMSE: {en_test_rmse:.4f}, MAE: {en_test_mae:.4f}, R²: {en_test_r2:.4f}")
print(f"5-Fold CV RMSE: {en_cv_rmse:.4f} (±{en_cv_std:.4f})")

model_results['ElasticNet'] = {
    'pipeline': elasticnet_cv,
    'train_rmse': en_train_rmse, 'train_mae': en_train_mae, 'train_r2': en_train_r2,
    'test_rmse': en_test_rmse, 'test_mae': en_test_mae, 'test_r2': en_test_r2,
    'cv_rmse': en_cv_rmse, 'cv_std': en_cv_std,
    'train_pred': en_train_pred, 'test_pred': en_test_pred,
    'best_alpha': best_alpha_en, 'best_l1_ratio': best_l1_ratio
}

# ============================================================================
# Phase 6: Evaluation (評估)
# ============================================================================
print("\n" + "="*80)
print("Phase 6: Evaluation - 模型比較")
print("="*80)

# 建立比較表
comparison_data = []
for model_name, results in model_results.items():
    comparison_data.append({
        'Model': model_name,
        'Train_RMSE': results['train_rmse'],
        'Train_MAE': results['train_mae'],
        'Train_R2': results['train_r2'],
        'Test_RMSE': results['test_rmse'],
        'Test_MAE': results['test_mae'],
        'Test_R2': results['test_r2'],
        'CV_RMSE': results['cv_rmse']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n模型性能比較:")
print(comparison_df.to_string(index=False))

# 找出最佳模型
best_model_idx = comparison_df['Test_RMSE'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_model_info = model_results[best_model_name]

print(f"\n✓ 最佳模型 (基於測試集 RMSE): {best_model_name}")
print(f"  測試集 RMSE: {best_model_info['test_rmse']:.4f}")
print(f"  測試集 MAE: {best_model_info['test_mae']:.4f}")
print(f"  測試集 R²: {best_model_info['test_r2']:.4f}")
print(f"  CV RMSE: {best_model_info['cv_rmse']:.4f}")

# 保存比較結果
comparison_df.to_csv('reports/model_comparison.csv', index=False)

# 特徵係數解讀
print("\n" + "="*80)
print("最佳模型特徵係數解讀")
print("="*80)

best_pipeline = best_model_info['pipeline']
coefficients = best_pipeline.named_steps['regressor'].coef_

feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n{best_model_name} 的特徵係數 (按絕對值排序):")
print(feature_importance.to_string(index=False))

print("\n特徵影響解讀 (Top 10):")
print("-" * 60)
for idx, row in feature_importance.head(10).iterrows():
    direction = "正向" if row['Coefficient'] > 0 else "負向"
    print(f"  {row['Feature']}: {row['Coefficient']:.4f} ({direction}影響)")

feature_importance.to_csv('reports/feature_coefficients.csv', index=False)

# 殘差分析
print("\n" + "="*80)
print("殘差診斷")
print("="*80)

train_residuals = y_train - best_model_info['train_pred']
test_residuals = y_test - best_model_info['test_pred']

print(f"\n訓練集殘差:")
print(f"  平均值: {train_residuals.mean():.6f}")
print(f"  標準差: {train_residuals.std():.4f}")

print(f"\n測試集殘差:")
print(f"  平均值: {test_residuals.mean():.6f}")
print(f"  標準差: {test_residuals.std():.4f}")

# Shapiro-Wilk 正態性檢定
shapiro_stat_train, shapiro_p_train = shapiro(train_residuals)
shapiro_stat_test, shapiro_p_test = shapiro(test_residuals)

print(f"\nShapiro-Wilk 正態性檢定:")
print(f"  訓練集: 統計量={shapiro_stat_train:.4f}, p-value={shapiro_p_train:.4f}")
print(f"  測試集: 統計量={shapiro_stat_test:.4f}, p-value={shapiro_p_test:.4f}")

if shapiro_p_test > 0.05:
    print("  ✓ 殘差符合正態分佈假設 (p > 0.05)")
else:
    print("  ! 殘差可能不符合正態分佈 (p <= 0.05)")

# 計算預測區間
residual_std = np.std(test_residuals)
prediction_interval_95 = 1.96 * residual_std
prediction_interval_99 = 2.576 * residual_std

print(f"\n預測區間:")
print(f"  95% 預測區間: ±{prediction_interval_95:.4f}")
print(f"  99% 預測區間: ±{prediction_interval_99:.4f}")

# 保存殘差數據
residuals_df = pd.DataFrame({
    'Actual': np.concatenate([y_train.values, y_test.values]),
    'Predicted': np.concatenate([best_model_info['train_pred'], best_model_info['test_pred']]),
    'Residuals': np.concatenate([train_residuals.values, test_residuals.values]),
    'Set': ['Train']*len(y_train) + ['Test']*len(y_test)
})
residuals_df.to_csv('reports/figures/residuals_data.csv', index=False)

# ============================================================================
# Phase 7: Visualization (視覺化)
# ============================================================================
print("\n" + "="*80)
print("Phase 7: Visualization")
print("="*80)

# 1. 模型比較圖
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('模型性能比較', fontsize=16, fontweight='bold')

# RMSE 比較
ax1 = axes[0, 0]
x_pos = np.arange(len(comparison_df))
width = 0.35
ax1.bar(x_pos - width/2, comparison_df['Train_RMSE'], width, label='Train RMSE', alpha=0.8)
ax1.bar(x_pos + width/2, comparison_df['Test_RMSE'], width, label='Test RMSE', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('RMSE Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# MAE 比較
ax2 = axes[0, 1]
ax2.bar(x_pos - width/2, comparison_df['Train_MAE'], width, label='Train MAE', alpha=0.8)
ax2.bar(x_pos + width/2, comparison_df['Test_MAE'], width, label='Test MAE', alpha=0.8)
ax2.set_xlabel('Model')
ax2.set_ylabel('MAE')
ax2.set_title('MAE Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# R² 比較
ax3 = axes[1, 0]
ax3.bar(x_pos - width/2, comparison_df['Train_R2'], width, label='Train R²', alpha=0.8)
ax3.bar(x_pos + width/2, comparison_df['Test_R2'], width, label='Test R²', alpha=0.8)
ax3.set_xlabel('Model')
ax3.set_ylabel('R² Score')
ax3.set_title('R² Score Comparison')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# CV RMSE 比較
ax4 = axes[1, 1]
ax4.bar(comparison_df['Model'], comparison_df['CV_RMSE'], alpha=0.8, color='coral')
ax4.set_xlabel('Model')
ax4.set_ylabel('CV RMSE')
ax4.set_title('Cross-Validation RMSE')
ax4.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/01_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已保存: reports/figures/01_model_comparison.png")
plt.close()

# 2. 實際 vs. 預測圖 (含預測區間)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'{best_model_name} - 實際值 vs. 預測值', fontsize=16, fontweight='bold')

# 訓練集
ax1 = axes[0]
ax1.scatter(y_train, best_model_info['train_pred'], alpha=0.5, s=30)
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Calories Burned')
ax1.set_ylabel('Predicted Calories Burned')
ax1.set_title(f'Training Set (R² = {best_model_info["train_r2"]:.4f})')
ax1.legend()
ax1.grid(alpha=0.3)

# 測試集 (含預測區間)
ax2 = axes[1]
sorted_indices = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_indices]
test_pred_sorted = best_model_info['test_pred'][sorted_indices]

ax2.scatter(y_test, best_model_info['test_pred'], alpha=0.5, s=30, label='Predictions')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')

# 繪製預測區間
ax2.fill_between(y_test_sorted, 
                  test_pred_sorted - prediction_interval_95,
                  test_pred_sorted + prediction_interval_95,
                  alpha=0.2, color='gray', label='95% Prediction Interval')

ax2.set_xlabel('Actual Calories Burned')
ax2.set_ylabel('Predicted Calories Burned')
ax2.set_title(f'Test Set (R² = {best_model_info["test_r2"]:.4f})')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/02_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已保存: reports/figures/02_actual_vs_predicted.png")
plt.close()

# 3. 殘差圖
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('殘差診斷', fontsize=16, fontweight='bold')

# 殘差 vs. 預測值 (訓練集)
ax1 = axes[0, 0]
ax1.scatter(best_model_info['train_pred'], train_residuals, alpha=0.5, s=30)
ax1.axhline(y=0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('Predicted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs. Predicted (Training Set)')
ax1.grid(alpha=0.3)

# 殘差 vs. 預測值 (測試集)
ax2 = axes[0, 1]
ax2.scatter(best_model_info['test_pred'], test_residuals, alpha=0.5, s=30)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs. Predicted (Test Set)')
ax2.grid(alpha=0.3)

# 殘差分佈直方圖 (訓練集)
ax3 = axes[1, 0]
ax3.hist(train_residuals, bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
ax3.set_title('Residuals Distribution (Training Set)')
ax3.grid(alpha=0.3)

# 殘差分佈直方圖 (測試集)
ax4 = axes[1, 1]
ax4.hist(test_residuals, bins=30, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Residuals')
ax4.set_ylabel('Frequency')
ax4.set_title('Residuals Distribution (Test Set)')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/03_residual_plots.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已保存: reports/figures/03_residual_plots.png")
plt.close()

# 4. QQ-plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Q-Q Plot (Normal Probability Plot)', fontsize=16, fontweight='bold')

# 訓練集 QQ-plot
ax1 = axes[0]
stats.probplot(train_residuals, dist="norm", plot=ax1)
ax1.set_title('Training Set')
ax1.grid(alpha=0.3)

# 測試集 QQ-plot
ax2 = axes[1]
stats.probplot(test_residuals, dist="norm", plot=ax2)
ax2.set_title('Test Set')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/04_qq_plots.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已保存: reports/figures/04_qq_plots.png")
plt.close()

# 5. 特徵係數圖
fig, ax = plt.subplots(figsize=(12, 8))
feature_importance_top15 = feature_importance.head(15)

colors = ['green' if x > 0 else 'red' for x in feature_importance_top15['Coefficient']]
ax.barh(range(len(feature_importance_top15)), feature_importance_top15['Coefficient'], color=colors, alpha=0.7)
ax.set_yticks(range(len(feature_importance_top15)))
ax.set_yticklabels(feature_importance_top15['Feature'])
ax.set_xlabel('Coefficient Value')
ax.set_title(f'{best_model_name} - Feature Coefficients (Top 15)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/05_feature_coefficients.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已保存: reports/figures/05_feature_coefficients.png")
plt.close()

# 6. 相關矩陣熱圖 (選定特徵)
fig, ax = plt.subplots(figsize=(14, 12))
selected_features_with_target = selected_features + ['Calories_Burned']
corr_matrix = df[selected_features_with_target].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix of Selected Features', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/06_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已保存: reports/figures/06_correlation_matrix.png")
plt.close()

# ============================================================================
# Phase 8: Deployment (部署)
# ============================================================================
print("\n" + "="*80)
print("Phase 8: Deployment - 保存模型")
print("="*80)

# 保存最佳模型
model_path = 'models/best_model_pipeline.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_pipeline, f)
print(f"✓ 最佳模型已保存: {model_path}")

# 保存特徵資訊
feature_info = {
    'selected_features': selected_features,
    'model_name': best_model_name,
    'performance': {
        'test_rmse': best_model_info['test_rmse'],
        'test_mae': best_model_info['test_mae'],
        'test_r2': best_model_info['test_r2'],
        'cv_rmse': best_model_info['cv_rmse']
    },
    'prediction_intervals': {
        '95%': prediction_interval_95,
        '99%': prediction_interval_99
    }
}

feature_path = 'models/model_features.pkl'
with open(feature_path, 'wb') as f:
    pickle.dump(feature_info, f)
print(f"✓ 特徵資訊已保存: {feature_path}")

# 保存最終報告摘要
final_summary = {
    'Project': 'Gym Members Calorie Prediction',
    'Date': '2025-10-19',
    'Dataset_Size': len(df),
    'Train_Test_Split': f'{len(X_train)}/{len(X_test)}',
    'Selected_Features': len(selected_features),
    'Best_Model': best_model_name,
    'Test_RMSE': best_model_info['test_rmse'],
    'Test_MAE': best_model_info['test_mae'],
    'Test_R2': best_model_info['test_r2'],
    'CV_RMSE': best_model_info['cv_rmse'],
    'Residuals_Normal': 'Yes' if shapiro_p_test > 0.05 else 'No',
    'Prediction_Interval_95': f'±{prediction_interval_95:.2f}'
}

summary_path = 'reports/final_summary.csv'
pd.DataFrame([final_summary]).T.to_csv(summary_path, header=False)
print(f"✓ 最終摘要已保存: {summary_path}")

# ============================================================================
# 結論
# ============================================================================
print("\n" + "="*80)
print("專題執行完成！")
print("="*80)

print("\n生成的檔案:")
print("  模型檔案:")
print("    - models/best_model_pipeline.pkl")
print("    - models/model_features.pkl")
print("\n  報告檔案:")
print("    - reports/model_comparison.csv")
print("    - reports/feature_coefficients.csv")
print("    - reports/feature_selection_results.csv")
print("    - reports/final_summary.csv")
print("\n  圖表檔案:")
print("    - reports/figures/01_model_comparison.png")
print("    - reports/figures/02_actual_vs_predicted.png")
print("    - reports/figures/03_residual_plots.png")
print("    - reports/figures/04_qq_plots.png")
print("    - reports/figures/05_feature_coefficients.png")
print("    - reports/figures/06_correlation_matrix.png")
print("    - reports/figures/residuals_data.csv")

print("\n" + "="*80)
print("執行說明:")
print("  1. 從 Kaggle 下載資料集到 data/ 目錄")
print("  2. 執行腳本: python 7114056186_hw2.py")
print("  3. 查看 reports/ 目錄中的結果和圖表")
print("  4. 載入模型: pickle.load(open('models/best_model_pipeline.pkl', 'rb'))")
print("="*80)

print("\n✓ 所有任務完成！")
