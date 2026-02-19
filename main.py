# ===============================
# Student Performance Prediction
# ===============================

import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Downloading dataset...")

path = kagglehub.dataset_download("kundanbedmutha/student-performance-dataset")
print("Dataset folder:", path)

print("Files:", os.listdir(path))

file = os.path.join(path, "Student_Performance.csv")

df = pd.read_csv(file)

print("\nFirst rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Use simple numeric columns
X = df[['study_hours',
        'attendance_percentage',
        'math_score',
        'science_score',
        'english_score']]

y = df['overall_score']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nMAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

print("\nFeature Importance:")
for name, score in zip(X.columns, model.feature_importances_):
    print(name, ":", score)
