import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score

# Load the processed data
X = joblib.load(r"E:\mini-project01\data\X_train_scaled.pkl")
y = joblib.load(r"E:\mini-project01\data\y_train.pkl")

# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

# Hyperparameter grids
thrs = [0.2, 0.3, 0.4, 0.5]
ks = [3, 5, 7, 10, 15]
Depths = [2, 5, 10, 15, None]
alphas = [0.05, 0.01, 0.005, 0.001]

#  Logistic Regression 
for thr in thrs:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        y_proba_lr = lr.predict_proba(X_test)[:, 1]
        y_pred_lr = (y_proba_lr >= thr).astype(int)

        recalls.append(recall_score(y_test, y_pred_lr))
        precisions.append(precision_score(y_test, y_pred_lr))
        f1_scores.append(f1_score(y_test, y_pred_lr))

    results.append({
        "Model": "Logistic Regression",
        "Params": f"threshold={thr}",
        "Recall": np.mean(recalls),
        "Precision": np.mean(precisions),
        "F1": np.mean(f1_scores)
    })

#  KNN
for k in ks:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        recalls.append(recall_score(y_test, y_pred_knn))
        precisions.append(precision_score(y_test, y_pred_knn))
        f1_scores.append(f1_score(y_test, y_pred_knn))

    results.append({
        "Model": "KNN",
        "Params": f"k={k}",
        "Recall": np.mean(recalls),
        "Precision": np.mean(precisions),
        "F1": np.mean(f1_scores)
    })

# Decision Tree  
for depth in Depths:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)

        recalls.append(recall_score(y_test, y_pred_dt))
        precisions.append(precision_score(y_test, y_pred_dt))
        f1_scores.append(f1_score(y_test, y_pred_dt))

    results.append({
        "Model": "Decision Tree",
        "Params": f"max_depth={depth}",
        "Recall": np.mean(recalls),
        "Precision": np.mean(precisions),
        "F1": np.mean(f1_scores)
    })

# MLP 
for alpha in alphas:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            learning_rate_init=alpha,
            random_state=42,
            max_iter=300
        )
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)

        recalls.append(recall_score(y_test, y_pred_mlp))
        precisions.append(precision_score(y_test, y_pred_mlp))
        f1_scores.append(f1_score(y_test, y_pred_mlp))

    results.append({
        "Model": "MLP",
        "Params": f"lr_init={alpha}",
        "Recall": np.mean(recalls),
        "Precision": np.mean(precisions),
        "F1": np.mean(f1_scores)
    })

#  Final results table for CV (MODEL SELECTIONS)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))