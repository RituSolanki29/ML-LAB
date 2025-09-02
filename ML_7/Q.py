import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Optional imports (XGBoost, CatBoost)
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
    print("xgboost not found — skipping.")

try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False
    print("catboost not found — skipping.")

# Load dataset
df = pd.read_excel("SMART_dataset.xlsx")   

# Drop datetime columns
datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
if len(datetime_cols) > 0:
    df = df.drop(columns=datetime_cols)

# Features & target
X = df.drop(columns=["failure"])
y = df["failure"]

# Keep numeric only
X = X.select_dtypes(include=[np.number])

# Drop all-NaN cols
X = X.dropna(axis=1, how="all")

# Impute
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print("Train class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError("Train/test split has only 1 class. Need both 0/1 for classification.")

# Metrics function
def compute_metrics(y_true, y_pred, y_prob=None, average='binary'):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    try:
        if y_prob is not None:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        else:
            metrics['roc_auc'] = np.nan
    except Exception:
        metrics['roc_auc'] = np.nan

    return metrics

# Classifiers & hyperparam search space
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_iter_search = 20

models_and_spaces = []

# 1) SVC
models_and_spaces.append((
    "SVC",
    Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
    {"clf__C": np.logspace(-3, 2, 10), "clf__gamma": np.logspace(-4, 0, 10), "clf__kernel": ["rbf"]}
))

# 2) Decision Tree
models_and_spaces.append((
    "DecisionTree",
    Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=42))]),
    {"clf__max_depth": [None, 5, 10, 20, 50], "clf__min_samples_split": [2, 5, 10], "clf__criterion": ["gini", "entropy"]}
))

# 3) Random Forest
models_and_spaces.append((
    "RandomForest",
    Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))]),
    {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5, 10]}
))

# 4) AdaBoost
models_and_spaces.append((
    "AdaBoost",
    Pipeline([("scaler", StandardScaler()), ("clf", AdaBoostClassifier(random_state=42))]),
    {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.01, 0.1, 1.0]}
))

# 5) XGBoost
if HAVE_XGB:
    models_and_spaces.append((
        "XGBoost",
        Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))]),
        {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [3, 6, 10], "clf__learning_rate": [0.01, 0.1, 0.2], "clf__subsample": [0.6, 0.8, 1.0]}
    ))

# 6) CatBoost
if HAVE_CAT:
    models_and_spaces.append((
        "CatBoost",
        Pipeline([("scaler", StandardScaler()), ("clf", CatBoostClassifier(verbose=0, random_state=42))]),
        {"clf__iterations": [100, 200, 400], "clf__depth": [3, 6, 10], "clf__learning_rate": [0.01, 0.1, 0.2]}
    ))

# 7) Naive Bayes
models_and_spaces.append(("GaussianNB", Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]), {}))

# 8) MLP
models_and_spaces.append((
    "MLP",
    Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=500, random_state=42))]),
    {"clf__hidden_layer_sizes": [(50,), (100,), (100, 50)], "clf__alpha": [1e-4, 1e-3, 1e-2], "clf__learning_rate_init": [1e-3, 1e-2]}
))

# Run models
results = []

for name, pipeline, param_dist in models_and_spaces:
    print(f"\n---- Running {name} ----")
    if param_dist is None or len(param_dist) == 0:
        pipeline.fit(X_train, y_train)
        best_estimator = pipeline
        best_params = {}
        best_score = np.nan
    else:
        rs = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1,
            refit=True
        )
        try:
            rs.fit(X_train, y_train)
            best_estimator = rs.best_estimator_
            best_params = rs.best_params_
            best_score = rs.best_score_
            print(f"Best CV score (accuracy): {best_score:.4f}")
            print("Best params:", best_params)
        except Exception as e:
            print(f"RandomizedSearchCV failed for {name} -> {e}")
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline
            best_params = {}
            best_score = np.nan

    # Predictions
    y_train_pred = best_estimator.predict(X_train)
    y_test_pred = best_estimator.predict(X_test)

    # Probabilities
    y_train_prob, y_test_prob = None, None
    try:
        y_train_prob = best_estimator.predict_proba(X_train)
        y_test_prob = best_estimator.predict_proba(X_test)
    except Exception:
        try:
            dtf_train = best_estimator.decision_function(X_train)
            dtf_test = best_estimator.decision_function(X_test)
            if len(dtf_train.shape) == 1:
                y_train_prob = np.vstack([1 - (dtf_train - dtf_train.min())/(dtf_train.max()-dtf_train.min()+1e-12),
                                          (dtf_train - dtf_train.min())/(dtf_train.max()-dtf_train.min()+1e-12)]).T
                y_test_prob = np.vstack([1 - (dtf_test - dtf_test.min())/(dtf_test.max()-dtf_test.min()+1e-12),
                                         (dtf_test - dtf_test.min())/(dtf_test.max()-dtf_test.min()+1e-12)]).T
        except Exception:
            pass

    average_mode = 'binary' if y_train.nunique() == 2 else 'weighted'
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob, average=average_mode)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, average=average_mode)

    results.append({
        "Model": name,
        "CV_BestScore": best_score,
        "Train_Acc": train_metrics['accuracy'],
        "Train_Prec": train_metrics['precision'],
        "Train_Recall": train_metrics['recall'],
        "Train_F1": train_metrics['f1'],
        "Train_ROC_AUC": train_metrics['roc_auc'],
        "Test_Acc": test_metrics['accuracy'],
        "Test_Prec": test_metrics['precision'],
        "Test_Recall": test_metrics['recall'],
        "Test_F1": test_metrics['f1'],
        "Test_ROC_AUC": test_metrics['roc_auc'],
    })

# Display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Test_F1", ascending=False)

print("\n\n===== Summary results (sorted by Test F1) =====")
print(results_df.to_string(index=False))
