import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import os

def handle_outliers(ds, cols):
    for col in cols:
        if col not in ds.columns:
            continue
        Q1 = ds[col].quantile(0.25)
        Q3 = ds[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        ds[col] = ds[col].clip(lower, upper)
    return ds

def train_gym_models(csv_file_stream, tune_xgboost=False):
    """
    Trains SVM, Random Forest, and XGBoost on the gym/fitness churn dataset.
    Features: age, gender, membership_type, membership_duration_months,
              attendance_frequency_per_week, last_checkin_days_ago, monthly_fee,
              payment_delay_count, workout_sessions_completed, personal_trainer,
              group_class_participation, bmi, distance_from_gym_km, app_usage_hours,
              renewal_count, goal_type, upi_payment_usage
    Target: churn_status (0/1)
    """
    ds = pd.read_csv(csv_file_stream)
    ds.columns = ds.columns.str.lower().str.strip()

    # Handle binary categoricals first to avoid NaNs
    for col in ['personal_trainer', 'upi_payment_usage']:
        if col in ds.columns:
            ds[col] = ds[col].astype(str).str.strip().str.lower().map(
                {"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0}
            ).fillna(0).astype(int)

    # Handle missing values for other categoricals
    cat_cols = ['gender', 'membership_type', 'goal_type']
    for col in cat_cols:
        if col in ds.columns:
            ds[col] = ds[col].fillna(ds[col].mode()[0])

    num_cols = ['age', 'membership_duration_months', 'attendance_frequency_per_week',
                'last_checkin_days_ago', 'monthly_fee', 'payment_delay_count',
                'workout_sessions_completed', 'group_class_participation', 'bmi',
                'distance_from_gym_km', 'app_usage_hours', 'renewal_count']
    for col in num_cols:
        if col in ds.columns:
            ds[col] = pd.to_numeric(ds[col], errors='coerce')
            ds[col] = ds[col].fillna(ds[col].median())

    # Handle duplicates
    if all(col in ds.columns for col in ['member_id', 'name', 'churn_status']):
        ds = ds.drop_duplicates(subset=['member_id', 'name', 'churn_status'], keep='first')
    elif all(col in ds.columns for col in ['member_id', 'churn_status']):
        ds = ds.drop_duplicates(subset=['member_id', 'churn_status'], keep='first')

    # Extract features
    expected_cols = [
        'age', 'gender', 'membership_type', 'membership_duration_months',
        'attendance_frequency_per_week', 'last_checkin_days_ago', 'monthly_fee',
        'payment_delay_count', 'workout_sessions_completed', 'personal_trainer',
        'group_class_participation', 'bmi', 'distance_from_gym_km', 'app_usage_hours',
        'renewal_count', 'goal_type', 'upi_payment_usage', 'churn_status'
    ]
    missing = [col for col in expected_cols if col not in ds.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    ds = ds[expected_cols]

    if 'churn_status' not in ds.columns:
        raise ValueError("Target column 'churn_status' is missing from the dataset.")

    ds["churn_status"] = ds["churn_status"].replace(
        {"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0, "y": 1, "n": 0, '  Yes ': 1, '  No ': 0}
    )
    ds["churn_status"] = pd.to_numeric(ds["churn_status"], errors='coerce').fillna(0).astype(int)

    counts = ds['churn_status'].value_counts()
    total = len(ds)
    churn_rate = float((counts.get(1, 0) / total) * 100) if total > 0 else 0.0
    retention_rate = float((counts.get(0, 0) / total) * 100) if total > 0 else 0.0

    # Encoding
    cat_encode = [c for c in ['gender', 'membership_type', 'goal_type'] if c in ds.columns]
    if cat_encode:
        ds = pd.get_dummies(data=ds, columns=cat_encode, drop_first=True, dtype=int)

    X = ds.drop(columns=['churn_status'])
    y = ds['churn_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    outlier_cols = [c for c in num_cols if c in X_train.columns]
    X_train = handle_outliers(X_train, outlier_cols)

    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if tune_xgboost:
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        random_search = RandomizedSearchCV(
            estimator=xgb, param_distributions=xgb_params,
            n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1
        )
        random_search.fit(X_train_scaled, y_train)
        best_xgb = random_search.best_estimator_
        xgb_score = best_xgb.score(X_test_scaled, y_test)
        scores = {"Tuned XGBoost": xgb_score}
        best_model_name = "Tuned XGBoost"
        best_model = best_xgb
        best_score = xgb_score
    else:
        base_svm = LinearSVC(C=1, max_iter=5000, random_state=42)
        model_svm = CalibratedClassifierCV(base_svm, cv=3)
        model_svm.fit(X_train_scaled, y_train)
        svm_score = model_svm.score(X_test_scaled, y_test)

        model_rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', random_state=42
        )
        model_rf.fit(X_train_scaled, y_train)
        rf_score = model_rf.score(X_test_scaled, y_test)

        model_xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        model_xgb.fit(X_train_scaled, y_train)
        xgb_score = model_xgb.score(X_test_scaled, y_test)

        scores = {"SVM": svm_score, "Random Forest": rf_score, "XGBoost": xgb_score}
        models = {"SVM": model_svm, "Random Forest": model_rf, "XGBoost": model_xgb}
        best_model_name = max(scores, key=scores.get)
        best_model = models[best_model_name]
        best_score = scores[best_model_name]

    os.makedirs('dump models', exist_ok=True)
    joblib.dump(best_model, os.path.join('dump models', 'gym_best_model.pkl'))
    joblib.dump(scaler, os.path.join('dump models', 'gym_scaler.pkl'))
    joblib.dump(list(X.columns), os.path.join('dump models', 'gym_features.pkl'))

    return {
        'scores': scores,
        'best_model': best_model_name,
        'best_score': best_score,
        'churn_rate': round(churn_rate, 2),
        'retention_rate': round(retention_rate, 2)
    }
