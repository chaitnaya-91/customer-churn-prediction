import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

def handle_outliers(ds, cols):
    for col in cols:
        Q1 = ds[col].quantile(0.25)
        Q3 = ds[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        ds[col] = ds[col].clip(lower, upper)
    return ds

def train_telecom_models(csv_file_stream, tune_xgboost=False):
    """
    Trains models for the telecom industry from the uploaded CSV stream.
    If tune_xgboost is True, only performs hyperparameter tuning for XGBoost.
    Returns: dict with scores, best model name, best score.
    """
    # 1. Load data
    ds = pd.read_csv(csv_file_stream)
    if len(ds) > 1500:
        ds = ds.sample(n=1500, random_state=42)
    ds.columns = ds.columns.str.lower()

    # 2. Handling Missing Values
    cat_cols = ['internetservice', 'autopay', 'churn']
    for col in cat_cols:
        if col in ds.columns:
            ds[col] = ds[col].fillna(ds[col].mode()[0])

    num_cols = ['index', 'tenure', 'monthlycharges', 'totalcharges', 'datausage', 
                'customersupportcalls', 'complaints', 'lastrechargedaysago']
    for col in num_cols:
        if col in ds.columns:
            ds[col] = ds[col].fillna(ds[col].mode()[0])

    # 3. Handling Duplicates
    if all(col in ds.columns for col in ['customerid', 'name', 'churn']):
        ds = ds.drop_duplicates(subset=['customerid', 'name', 'churn'], keep='first')
    elif all(col in ds.columns for col in ['customerid', 'churn']):
        ds = ds.drop_duplicates(subset=['customerid', 'churn'], keep='first')
    elif all(col in ds.columns for col in ['tenure', 'monthlycharges', 'autopay', 'churn']):
        ds = ds.drop_duplicates(subset=['tenure', 'monthlycharges', 'autopay', 'churn'], keep='first')

    # 4. Remove Extra columns
    expected_cols = [
        'tenure', 'monthlycharges', 'totalcharges',
        'internetservice', 'datausage', 'customersupportcalls',
        'complaints', 'autopay', 'lastrechargedaysago', 'churn'
    ]
    # Ensure all expected columns are present (except any id/name columns handled elsewhere)
    missing = [col for col in expected_cols if col not in ds.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    ds = ds[expected_cols]

    # datatype correction
    if 'monthlycharges' in ds.columns:
        ds['monthlycharges'] = pd.to_numeric(ds['monthlycharges'], errors='coerce')

    # Extracting Features
    if 'churn' in ds.columns:
        ds["churn"] = ds["churn"].replace(
            {"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0, "y": 1, "n": 0, '  Yes ': 1, '  No ': 0}
        ).astype(int)

    # Calculate Churn & Retention Rate
    churn_rate = 0.0
    retention_rate = 0.0
    if 'churn' in ds.columns:
        counts = ds['churn'].value_counts()
        total = len(ds)
        if total > 0:
            churn_rate = float((counts.get(1, 0) / total) * 100)
            retention_rate = float((counts.get(0, 0) / total) * 100)

    if 'autopay' in ds.columns:
        ds["autopay"] = (ds['autopay'].astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int))

    # Encoding
    if 'internetservice' in ds.columns:
        ds["internetservice"] = ds["internetservice"].astype(str).str.strip()
        ds = pd.get_dummies(data=ds, columns=['internetservice'], drop_first=True, dtype=int)

    # 6. Train Test Split
    if 'churn' not in ds.columns:
        raise ValueError("Target column 'churn' is missing from the dataset.")

    X = ds.drop(columns=['churn'])
    y = ds['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. Outlier remove
    outlier_cols = [
        'tenure', 'monthlycharges', 'totalcharges',
        'datausage', 'customersupportcalls',
        'complaints', 'lastrechargedaysago'
    ]
    outlier_cols = [col for col in outlier_cols if col in X_train.columns]
    X_train = handle_outliers(X_train, outlier_cols)

    # Fill any remaining NaNs
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # 8. Scaling data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 9. Model Training
    
    if tune_xgboost:
        # Perform hyperparameter tuning for XGBoost only
        xgb_params = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
        
        xgb = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=xgb_params,
            n_iter=10,
            cv=3,
            scoring='accuracy',
            random_state=42,
            n_jobs=1
        )
        
        random_search.fit(X_train_scaled, y_train)
        best_xgb = random_search.best_estimator_
        xgb_score = best_xgb.score(X_test_scaled, y_test)
        
        scores = {
            "Tuned XGBoost": xgb_score
        }
        best_model_name = "Tuned XGBoost"
        best_model = best_xgb
        best_score = xgb_score
        
    else:
        # Train all models
        
        # SVM
        model_svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
        model_svm.fit(X_train_scaled, y_train)
        svm_score = model_svm.score(X_test_scaled, y_test)

        # Random Forest
        model_rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, min_samples_split=10, 
            min_samples_leaf=5, max_features='sqrt', random_state=42
        )
        model_rf.fit(X_train_scaled, y_train)
        rf_score = model_rf.score(X_test_scaled, y_test)

        # XGBoost
        model_xgb = XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        model_xgb.fit(X_train_scaled, y_train)
        xgb_score = model_xgb.score(X_test_scaled, y_test)

        scores = {
            "SVM": svm_score,
            "Random Forest": rf_score,
            "XGBoost": xgb_score
        }

        models = {
            "SVM": model_svm,
            "Random Forest": model_rf,
            "XGBoost": model_xgb
        }

        # Best model
        best_model_name = max(scores, key=scores.get)
        best_model = models[best_model_name]
        best_score = scores[best_model_name]

    # Save best model and scaler
    os.makedirs('dump models', exist_ok=True)
    joblib.dump(best_model, os.path.join('dump models', 'telecom_best_model.pkl'))
    joblib.dump(scaler, os.path.join('dump models', 'telecom_scaler.pkl'))
    
    # Also save the columns used for training so prediction knows what features to expect
    joblib.dump(list(X.columns), os.path.join('dump models', 'telecom_features.pkl'))

    return {
        'scores': scores,
        'best_model': best_model_name,
        'best_score': best_score,
        'churn_rate': round(churn_rate, 2),
        'retention_rate': round(retention_rate, 2)
    }
