import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import os


X = joblib.load(r"E:\mini-project01\data\X_test_scaled.pkl")
y = joblib.load(r"E:\mini-project01\data\y_test.pkl")

# Load trained models
lr_model = joblib.load(r"E:\mini-project01\models\logistic_regression.pkl")
knn_model = joblib.load(r"E:\mini-project01\models\knn.pkl")
dt_model = joblib.load(r"E:\mini-project01\models\decision_tree.pkl")
mlp_model = joblib.load(r"E:\mini-project01\models\mlp.pkl")

# prediction
pred_lr = lr_model.predict(X)
pred_knn = knn_model.predict(X)
pred_dt = dt_model.predict(X)
pred_mlp = mlp_model.predict(X)

# Calculate metrics
models = {
    'Logistic Regression': pred_lr,
    'KNN': pred_knn,
    'Decision Tree': pred_dt,
    'MLP': pred_mlp
}

results = []

for name, pred in models.items():
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    
    recall = recall_score(y, pred)
    precision = precision_score(y, pred)
    f1 = f1_score(y, pred)
    
    results.append({
        'Model': name,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Recall': round(recall, 4),
        'Precision': round(precision, 4),
        'F1-Score': round(f1, 4)
    })

# Create DataFrame
df_results = pd.DataFrame(results)
print(df_results)

# Save
report_dir = r"E:\mini-project01\reports"
markdown = "# Final Testing Models Performance\n\n"
markdown += df_results.to_markdown(index=False)

with open(
    os.path.join(report_dir, "experiments.md"),
    "a",
    encoding="utf-8"
) as f:
    f.write(markdown)

print("\nResults saved to:")
print(os.path.join(report_dir, "experiments.md"))