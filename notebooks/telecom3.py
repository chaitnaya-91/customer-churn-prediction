#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


df= pd.read_csv('telecom_churn_dataset.csv')
df.head()

# In[3]:


ds= df.copy()

# # 1. Understanding data

# In[4]:


ds.info()

# In[5]:


ds.describe()

# In[6]:


ds.columns= ds.columns.str.lower()

# In[7]:


ds.head(2)

# # 2. Handling Missing Values

# In[8]:


ds.isnull().sum()

# In[9]:


# Categorical columns
ds['internetservice'] = ds['internetservice'].fillna(ds['internetservice'].mode()[0])
ds['autopay'] = ds['autopay'].fillna(ds['autopay'].mode()[0])
ds['churn'] = ds['churn'].fillna(ds['churn'].mode()[0])

# Numeric columns
ds['index'] = ds['index'].fillna(ds['index'].mode()[0])
ds['tenure'] = ds['tenure'].fillna(ds['tenure'].mode()[0])
ds['monthlycharges'] = ds['monthlycharges'].fillna(ds['monthlycharges'].mode()[0])
ds['totalcharges'] = ds['totalcharges'].fillna(ds['totalcharges'].mode()[0])
ds['datausage'] = ds['datausage'].fillna(ds['datausage'].mode()[0])
ds['customersupportcalls'] = ds['customersupportcalls'].fillna(ds['customersupportcalls'].mode()[0])
ds['complaints'] = ds['complaints'].fillna(ds['complaints'].mode()[0])
ds['lastrechargedaysago'] = ds['lastrechargedaysago'].fillna(ds['lastrechargedaysago'].mode()[0])

# In[10]:


ds.isnull().sum()

# # 3. Handling Duplicates

# In[11]:


if all(col in ds.columns for col in ['customerid', 'name', 'churn']):
    ds = ds.drop_duplicates(subset=['customerid', 'name', 'churn'], keep='first')
elif all(col in ds.columns for col in ['customerid', 'churn']):
    ds = ds.drop_duplicates(subset=['customerid', 'churn'], keep='first')
else:
    ds = ds.drop_duplicates(subset=['tenure', 'monthlycharges', 'autopay', 'churn'], keep='first')

# # 4. Remove Extra

# In[12]:


ds.columns

# In[13]:


cols = [
    'tenure', 'monthlycharges', 'totalcharges',
    'internetservice', 'datausage', 'customersupportcalls',
    'complaints', 'autopay', 'lastrechargedaysago', 'churn'
]

ds = ds[cols]
ds.head(2)

# In[14]:


ds.head()

# # datatype correction

# In[15]:


ds['monthlycharges'] = pd.to_numeric(ds['monthlycharges'], errors='coerce')

# In[16]:


df.head()

# In[ ]:




# # 4. Extracting Features

# In[17]:


ds['churn'].unique()

# In[18]:


ds["churn"] = ds["churn"].replace(
    {"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0, "y": 1, "n": 0, '  Yes ': 1, '  No ': 0}
).astype(int)

# In[19]:


ds['churn'].unique()

# In[20]:


ds['churn'].value_counts()

# In[ ]:




# In[21]:


ds["autopay"].unique()

# In[22]:


# str.lower() necessary
ds["autopay"] = (ds['autopay'].str.strip().str.lower().map(
    {"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0, "y": 1, "n": 0}
).astype(int))

# In[23]:


ds.head()

# In[ ]:




# # 5. Encoding

# In[24]:


ds['internetservice'].unique()

# In[25]:


ds["internetservice"] = ds["internetservice"].str.strip()

# In[26]:


ds['internetservice'].unique()

# In[27]:


ds['internetservice'].value_counts()

# In[28]:


enc= pd.get_dummies(data=ds, columns=['internetservice'], drop_first=True, dtype=int)  # drop categorical col 1st instead in alphabetical order
ds=enc

# In[29]:


ds.head()

# In[30]:


print(ds.dtypes)
print(ds.head())

# In[ ]:




# # 6. Train Test Split

# In[31]:


X= ds.drop(columns=['churn'])
# y= ds[['churn']]
y= ds['churn']

# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# In[33]:


X_train.shape, X_test.shape

# In[34]:


y_train.shape, y_test.shape

# # 7. Outlier remove

# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns

for col in ['totalcharges', 'monthlycharges', 'datausage']:
    sns.boxplot(x=X_train[col])
    plt.title(col)
    plt.show()

# In[36]:


def handle_outliers(ds, cols):
    for col in cols:
        Q1 = ds[col].quantile(0.25)
        Q3 = ds[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # clip values
        ds[col] = ds[col].clip(lower, upper)

    return ds

# In[37]:


num_cols = [
    'tenure', 'monthlycharges', 'totalcharges',
    'datausage', 'customersupportcalls',
    'complaints', 'lastrechargedaysago'
]

X_train = handle_outliers(X_train, num_cols)

# # 8. Scaling data

# In[38]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # converted into array
X_test = scaler.transform(X_test)

# # 9. Model Training

# In[39]:


"""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
"""

# In[40]:


print(pd.DataFrame(X_train).isnull().sum())

# In[41]:


ds.head()

# In[ ]:




# In[42]:


X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# In[ ]:




# In[43]:


print(X_train.isnull().sum())

# In[44]:


X_train['monthlycharges'].fillna(X_train['monthlycharges'].median(), inplace=True)
X_test['monthlycharges'].fillna(X_train['monthlycharges'].median(), inplace=True)

# In[45]:


print(X_train.isnull().sum())

# In[ ]:




# # again train model

# In[66]:


from sklearn.svm import SVC

model_svm = SVC(
    kernel='rbf',
    C=1,
    gamma='scale',
    probability=True,
    random_state=42
)

model_svm.fit(X_train, y_train)

# In[ ]:




# In[67]:


from sklearn.ensemble import RandomForestClassifier
model_rf  = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
model_rf .fit(X_train, y_train)

# In[ ]:




# In[70]:


model_svm.score(X_train, y_train)

# In[71]:


model_svm.score(X_test, y_test)

# In[ ]:




# In[72]:


model_rf.score(X_train, y_train)

# In[73]:


model_rf.score(X_test, y_test)

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # hyper parameter tuning

# In[56]:


""""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Step 1: Define base model
model1 = RandomForestClassifier(random_state=42)

# Step 2: Define parameter grid
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}

# Step 3: Grid Search
grid = GridSearchCV(
    estimator=model1,
    param_grid=params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1   # use all CPU cores
)

# Step 4: Train
grid.fit(X_train, y_train)

# Step 5: Best parameters
print("Best Params:", grid.best_params_)

# Step 6: Best model
best_model = grid.best_estimator_

# Step 7: Evaluate
print("Train Score:", best_model.score(X_train, y_train))
print("Test Score:", best_model.score(X_test, y_test))
"""

# In[ ]:




# In[57]:


from xgboost import XGBClassifier
model_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model_xgb.fit(X_train, y_train)

# In[58]:


model_xgb.score(X_train, y_train)

# In[59]:


model_xgb.score(X_test, y_test)

# In[ ]:




# In[80]:


models = {
    "SVM": model_svm,
    "Random Forest": model_rf,
    "XGBoost": model_xgb
}

best_model = None
best_score = 0

print("\n===== MODEL COMPARISON =====\n")

for name, model in models.items():

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"{name}")
    print(f"Training Accuracy : {train_score:.4f}")
    print(f"Testing Accuracy  : {test_score:.4f}")
    print("-" * 35)

    # Select Best Model
    if test_score > best_score:
        best_score = test_score
        best_model = name

print(f"\nBest Model : {best_model}")
print(f"Best Accuracy : {best_score:.4f}")

# In[ ]:




# In[ ]:




# In[60]:


new_customer = pd.DataFrame(0, index=[0], columns=X.columns)

# Numeric values
new_customer['tenure'] = 40
new_customer['monthlycharges'] = 799
new_customer['totalcharges'] = 32000
new_customer['datausage'] = 25.5
new_customer['customersupportcalls'] = 3
new_customer['complaints'] = 1
new_customer['lastrechargedaysago'] = 20

# Encoded categorical values
new_customer['autopay'] = 1
new_customer['internetservice_DSL'] = 0
new_customer['internetservice_Fiber'] = 1

# In[61]:


new_customer_scaled = scaler.transform(new_customer)

# In[ ]:




# In[81]:


best_model_object = models[best_model]
prediction = best_model_object.predict(new_customer_scaled)

print(f"\nBest Model Selected : {best_model_name}")
print(f"Best Accuracy       : {best_accuracy:.4f}")

if prediction[0] == 1:
    print("\nPrediction : Customer will leave")
else:
    print("\nPrediction : Customer will stay")

# In[84]:


prob = best_model_object.predict_proba(new_customer_scaled)

stay_prob = prob[0][0] * 100
leave_prob = prob[0][1] * 100

print(f"Stay Probability  : {stay_prob:.2f}%")
print(f"Leave Probability : {leave_prob:.2f}%")

# In[ ]:




# In[86]:


# =========================================
# Hyperparameter Tuning For XGBoost
# =========================================

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# =========================================
# XGBoost Parameter Grid
# =========================================

xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# =========================================
# Base XGBoost Model
# =========================================

xgb = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# =========================================
# Randomized Search CV
# =========================================

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=xgb_params,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# =========================================
# Train Tuned Model
# =========================================

random_search.fit(X_train, y_train)

# =========================================
# Best Parameters
# =========================================

print("\nBest Parameters:\n")
print(random_search.best_params_)

# =========================================
# Best Tuned XGBoost Model
# =========================================

best_xgb = random_search.best_estimator_

# =========================================
# Accuracy Scores
# =========================================

train_score = best_xgb.score(X_train, y_train)
test_score = best_xgb.score(X_test, y_test)

print("\nTraining Accuracy :", train_score)
print("Testing Accuracy  :", test_score)

# =========================================
# Prediction Using Tuned XGBoost
# =========================================

new_customer_scaled = scaler.transform(new_customer)

prediction = best_xgb.predict(new_customer_scaled)

if prediction[0] == 1:
    print("\nCustomer will leave")
else:
    print("\nCustomer will stay")

# =========================================
# Probability Prediction
# =========================================

prob = best_xgb.predict_proba(new_customer_scaled)

stay_prob = prob[0][0] * 100
leave_prob = prob[0][1] * 100

print(f"\nStay Probability  : {stay_prob:.2f}%")
print(f"Leave Probability : {leave_prob:.2f}%")

# In[ ]:




# In[88]:


import joblib

# Save SVM Model
joblib.dump(model_svm, "telecom_svm_model.pkl")

# Save Scaler
joblib.dump(scaler, "telecom_scaler.pkl")

print("SVM Model Saved Successfully")
print("Scaler Saved Successfully")

# In[ ]:




# In[64]:


ds

# In[ ]:




# In[ ]:




# In[ ]:




# # accuracy

# In[65]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Train
    model1.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("🔹 Train Accuracy:", train_acc)
    print("🔹 Test Accuracy:", test_acc)
    
    # Classification report
    print("\n🔹 Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("\n🔹 Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    return model
