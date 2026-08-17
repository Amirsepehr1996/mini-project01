import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib



data = pd.read_csv(r"E:\mini-project01\data\creditcard.csv")
print(data.shape)

# Quick Show
data.info()
print(data.head())
print(data.describe())
print(data.isnull().sum())
print(data['Class'].value_counts())
print(data['Class'].value_counts(normalize=True) * 100)
print(data.duplicated().sum())

# Preprocessing
data = data.drop_duplicates()
print(data['Class'].value_counts())
print(data['Class'].value_counts(normalize=True) * 100)

data['Hour'] = (data['Time'] % (24 * 3600)) / 3600
data = data.drop('Time', axis=1)
print(data.describe())

X = data.drop(columns=['Class'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


joblib.dump(scaler, r"E:\mini-project01\models\scaler.pkl")

joblib.dump(X_train_scaled, r"E:\mini-project01\data\X_train_scaled.pkl")
joblib.dump(X_test_scaled, r"E:\mini-project01\data\X_test_scaled.pkl")
joblib.dump(y_train, r"E:\mini-project01\data\y_train.pkl")
joblib.dump(y_test, r"E:\mini-project01\data\y_test.pkl")