import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# 1. Load Data
iris = load_iris()
X = iris.data
y = iris.target

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Save Model and Scaler
os.makedirs("model", exist_ok=True)
with open("model/iris_model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)

print("✅ Model and scaler saved in model/iris_model.pkl")
