# ============================================================
# CUSTOMER DELINQUENCY PREDICTION
# Model Comparison: Logistic Regression vs Random Forest vs XGBoost
# ============================================================

import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "Delinquency_prediction_dataset.xlsx"
if not os.path.exists(file_path):
    file_path = "/content/Delinquency_prediction_dataset.xlsx"

df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["Delinquent_Account"].value_counts())

print("\nTarget percentage:")
print(
    df["Delinquent_Account"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ============================================================
# 2. REMOVE IDENTIFIER
# ============================================================

X = df.drop(
    columns=["Delinquent_Account", "Customer_ID"]
)

y = df["Delinquent_Account"]

# ============================================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

# ============================================================
# 4. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numerical_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 6. DEFINE MODELS
# ============================================================

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    eval_metric="logloss",
    random_state=42
)

# ============================================================
# 7. CREATE PIPELINES
# ============================================================

models = {

    "Logistic Regression":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", logistic_model)
        ]),

    "Random Forest":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", rf_model)
        ]),

    "XGBoost":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", xgb_model)
        ])
}

# ============================================================
# 8. STRATIFIED CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():

    print("\n================================")
    print(name)
    print("================================")

    # Out-of-fold predictions
    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    auc = roc_auc_score(
        y_train,
        probabilities
    )

    pr_auc = average_precision_score(
        y_train,
        probabilities
    )

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": name,
        "ROC-AUC": auc,
        "PR-AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(
    results_df.sort_values(
        "PR-AUC",
        ascending=False
    )
)

# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .sort_values("PR-AUC", ascending=False)
    .iloc[0]["Model"]
)

print(
    "\nSelected Champion Model:",
    best_model_name
)

best_model = models[best_model_name]

# ============================================================
# 10. TRAIN CHAMPION MODEL
# ============================================================

best_model.fit(
    X_train,
    y_train
)

# ============================================================
# 11. TEST SET EVALUATION
# ============================================================

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= 0.50
).astype(int)

print("\n================================")
print("FINAL TEST PERFORMANCE")
print("================================")

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            test_probabilities
        ),
        4
    )
)

print(
    "PR-AUC:",
    round(
        average_precision_score(
            y_test,
            test_probabilities
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0
        ),
        4
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# ============================================================
# 12. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    test_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"{best_model_name}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Delinquency Prediction")

plt.legend()
plt.grid()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 13. PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_test,
        test_probabilities
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(
    "Precision-Recall Curve - Delinquency Prediction"
)

plt.grid()
plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.show()