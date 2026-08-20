import joblib
import json
import sys
import pandas as pd

model = joblib.load(r"E:\mini-project01\models\mlp.pkl")
scaler = joblib.load(r"E:\mini-project01\models\scaler.pkl")


def predict(input_path, threshold=0.3):

    with open(input_path) as f:
        data = json.load(f)

    df = pd.DataFrame([data])

    scaled = scaler.transform(df)

    proba = model.predict_proba(scaled)[0][1]

    pred_class = int(proba >= threshold)

    return {
        "prediction": "Fraud" if pred_class == 1 else "Legitimate",
        "class_id": pred_class,
        "probability": round(float(proba), 4),
        "threshold": threshold,
        "status": "success"
    }


if __name__ == "__main__":

    result = predict(sys.argv[1])

    output_path = r"E:\mini-project01\reports\fraud_prediction.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {output_path}")

