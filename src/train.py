import joblib
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

X = joblib.load(r"E:\mini-project01\data\X_train_scaled.pkl")
y = joblib.load(r"E:\mini-project01\data\y_train.pkl")

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

thrs = [0.2, 0.3, 0.4, 0.5]
ks = [3, 5, 7, 10, 15]
Depths = [2, 5, 10, 15, None]
alphas = [0.05, 0.01, 0.005, 0.001]

for thr in thrs:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # due to overfitting, we add regularization term: C=0.1
        lr = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
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

for depth in Depths:

    recalls = []
    precisions = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # due to overfitting, we add min_samples_leaf=10, min_samples_split=20
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_leaf=10, min_samples_split=20)
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

for alpha in alphas:
    for thr in thrs:

        recalls = []
        precisions = []
        f1_scores = []

        for train_idx, test_idx in skf.split(X, y):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # due to overfitting, smaller hidden layers, add L2, and early stopping
            mlp = MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation='relu',
                alpha=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                solver='adam',
                learning_rate_init=alpha,
                random_state=42,
                max_iter=300
            )
            mlp.fit(X_train, y_train)
            y_proba_mlp = mlp.predict_proba(X_test)[:, 1]
            y_pred_mlp = (y_proba_mlp >= thr).astype(int)

            recalls.append(recall_score(y_test, y_pred_mlp))
            precisions.append(precision_score(y_test, y_pred_mlp))
            f1_scores.append(f1_score(y_test, y_pred_mlp))

        results.append({
            "Model": "MLP",
            "Params": f"lr_init={alpha}, threshold={thr}",
            "Recall": np.mean(recalls),
            "Precision": np.mean(precisions),
            "F1": np.mean(f1_scores)
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


# Hyperparameters Selected, Train on Whole Training Data
# best hyperparameters for MLP:   lr_init=0.001, threshold=0.3
# best hyperparameters for LR:    threshold=0.2
# best hyperparameters for KNN:   k = 3
# best hyperparameters for DT:    max_depth=5

# Saving paths
# model_dir = r"E:\mini-project01\models"
# report_dir = r"E:\mini-project01\reports"

# # best models based on cv
# best_models = {

#     "Logistic Regression": {
#         "model": LogisticRegression(
#             random_state=42,
#             max_iter=1000
#         ),
#         "threshold": 0.2,
#         "filename": "logistic_regression.pkl"
#     },

#     "KNN": {
#         "model": KNeighborsClassifier(
#             n_neighbors=3
#         ),
#         "threshold": None,
#         "filename": "knn.pkl"
#     },

#     "Decision Tree": {
#         "model": DecisionTreeClassifier(
#             max_depth=5,
#             random_state=42
#         ),
#         "threshold": None,
#         "filename": "decision_tree.pkl"
#     },

#     "MLP": {
#         "model": MLPClassifier(
#             hidden_layer_sizes=(32, 16),
#             activation="relu",
#             solver="adam",
#             learning_rate_init=0.001,
#             random_state=42,
#             max_iter=300
#         ),
#         "threshold": 0.3,
#         "filename": "mlp.pkl"
#     }
# }

# train and evaluate

# final_results = []

# for name, info in best_models.items():

#     model = info["model"]
#     model.fit(X, y)

#     # prediction
#     if info["threshold"] is None:
#         y_pred = model.predict(X)
#     else:
#         y_prob = model.predict_proba(X)[:, 1]
#         y_pred = (y_prob >= info["threshold"]).astype(int)

#     TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()

#     final_results.append({
#         "Model": name,
#         "Recall": recall_score(y, y_pred),
#         "Precision": precision_score(y, y_pred),
#         "F1": f1_score(y, y_pred),
#         "TP": TP,
#         "TN": TN,
#         "FP": FP,
#         "FN": FN
#     })

#     # saving model
#     joblib.dump(
#         model,
#         os.path.join(model_dir, info["filename"])
#     )

# # Results Table
# final_df = pd.DataFrame(final_results)

# print(final_df)

# # Save the table 
# markdown = "# Final Training Models Performance\n\n"
# markdown += final_df.to_markdown(index=False)

# with open(
#     os.path.join(report_dir, "experiments.md"),
#     "w",
#     encoding="utf-8"
# ) as f:
#     f.write(markdown)

# print("\nResults saved to:")
# print(os.path.join(report_dir, "experiments.md"))