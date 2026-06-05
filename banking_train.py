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

def train_banking_models(csv_file_stream, tune_xgboost=False):
    """
    Trains models for the banking industry from the uploaded CSV stream.
    If tune_xgboost is True, only performs hyperparameter tuning for XGBoost.
    Returns: dict with scores, best model name, best score, churn rate, retention rate.
    """
    # 1. Load data
    ds = pd.read_csv(csv_file_stream)
    ds.columns = ds.columns.str.lower()

    # 2. Handling Missing Values
    cat_cols = ['geography', 'gender', 'hascrcard', 'isactivemember', 'exited']
    for col in cat_cols:
        if col in ds.columns:
            ds[col] = ds[col].fillna(ds[col].mode()[0])

    num_cols = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary']
    for col in num_cols:
        if col in ds.columns:
            ds[col] = ds[col].fillna(ds[col].mode()[0])

    # 3. Handling Duplicates
    if all(col in ds.columns for col in ['customerid', 'balance']):
        ds = ds.drop_duplicates(subset=['customerid', 'balance'], keep='first')
    elif all(col in ds.columns for col in ['tenure', 'balance', 'estimatedsalary', 'exited']):
        ds = ds.drop_duplicates(subset=['tenure', 'balance', 'estimatedsalary', 'exited'], keep='first')

    # 4. Remove Extra columns
    expected_cols = [
        'creditscore', 'geography', 'gender', 'age', 'tenure',
        'balance', 'numofproducts', 'hascrcard', 'isactivemember',
        'estimatedsalary', 'exited'
    ]
    # Enforce presence of all expected columns (except id/name which are excluded)
    missing = [col for col in expected_cols if col not in ds.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    ds = ds[expected_cols]

    # datatype correction
    for col in num_cols:
        if col in ds.columns:
            ds[col] = pd.to_numeric(ds[col], errors='coerce')

    # Extracting Target
    if 'exited' in ds.columns:
        ds["exited"] = ds["exited"].replace(
            {"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0, "y": 1, "n": 0, '  Yes ': 1, '  No ': 0}
        ).fillna(0).astype(int)

    # Calculate Churn & Retention Rate
    churn_rate = 0.0
    retention_rate = 0.0
    if 'exited' in ds.columns:
        counts = ds['exited'].value_counts()
        total = len(ds)
        if total > 0:
            churn_rate = float((counts.get(1, 0) / total) * 100)
            retention_rate = float((counts.get(0, 0) / total) * 100)

    # Boolean representation correction
    for col in ['hascrcard', 'isactivemember']:
        if col in ds.columns:
            ds[col] = (ds[col].astype(str).str.strip().str.lower().map(
                {"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
            ).fillna(0).astype(int))

    # Encoding
    if 'gender' in ds.columns:
        ds['gender'] = pd.Categorical(ds['gender'].fillna('Female'), categories=['Female', 'Male', 'Other'])
        ds = pd.get_dummies(data=ds, columns=['gender'], drop_first=True, dtype=int)
        
    if 'geography' in ds.columns:
        ds['geography'] = pd.Categorical(ds['geography'].fillna('France'), categories=['France', 'Germany', 'Spain'])
        ds = pd.get_dummies(data=ds, columns=['geography'], drop_first=True, dtype=int)

    # 6. Train Test Split
    if 'exited' not in ds.columns:
        raise ValueError("Target column 'exited' is missing from the dataset.")

    X = ds.drop(columns=['exited'])
    y = ds['exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. Outlier remove (only on numerical columns)
    outlier_cols = [c for c in num_cols if c in X_train.columns]
    X_train = handle_outliers(X_train, outlier_cols)

    # Fill any remaining NaNs
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # 8. Scaling data (only for continuous numerical features to match the notebook)
    scaler = StandardScaler()
    active_num_cols = [c for c in num_cols if c in X_train.columns]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if active_num_cols:
        X_train_scaled[active_num_cols] = scaler.fit_transform(X_train[active_num_cols])
        X_test_scaled[active_num_cols] = scaler.transform(X_test[active_num_cols])

    # 9. Model Training
    if tune_xgboost:
        # Perform hyperparameter tuning for XGBoost only
        xgb_params = {
            'n_estimators': [100, 200, 300],
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
            n_jobs=-1
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
        # SVM (using standard SVC with probability support)
        model_svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
        model_svm.fit(X_train_scaled, y_train)
        svm_score = model_svm.score(X_test_scaled, y_test)

        # Random Forest (with notebook parameters)
        model_rf = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=10, 
            min_samples_leaf=5, class_weight='balanced', random_state=42
        )
        model_rf.fit(X_train_scaled, y_train)
        rf_score = model_rf.score(X_test_scaled, y_test)

        # XGBoost (with notebook parameters)
        model_xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
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
    joblib.dump(best_model, os.path.join('dump models', 'banking_best_model.pkl'))
    joblib.dump(scaler, os.path.join('dump models', 'banking_scaler.pkl'))
    
    # Save target numerical columns list so scaler knows what to scale
    joblib.dump(active_num_cols, os.path.join('dump models', 'banking_scaler_cols.pkl'))
    
    # Save the columns used for training so prediction knows what features to expect
    joblib.dump(list(X.columns), os.path.join('dump models', 'banking_features.pkl'))

    return {
        'scores': scores,
        'best_model': best_model_name,
        'best_score': best_score,
        'churn_rate': round(churn_rate, 2),
        'retention_rate': round(retention_rate, 2)
    }
