import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def load_and_preprocess(path="data/raw/cloudwatch_logs.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df.to_csv("data/processed/features.csv", index=False)
    return df

def train_model(df):
    features = [
        "cpu_utilization", "memory_utilization", "duration_ms",
        "invocations", "hour_sin", "hour_cos", "day_sin", "day_cos",
        "service", "region"
    ]
    target = "carbon_gco2"
    X = df[features]
    y = df[target]

    numeric_features = [
        "cpu_utilization", "memory_utilization", "duration_ms",
        "invocations", "hour_sin", "hour_cos", "day_sin", "day_cos"
    ]
    categorical_features = ["service", "region"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    models = {
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")

        results[name] = {"r2": r2, "mae": mae, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
        print(f"{name}: R²={r2:.4f}, MAE={mae:.6f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_model_name = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/carbon_model.pkl")
    print(f"\n✅ Best model: {best_model_name} (R²={best_score:.4f}) saved to models/carbon_model.pkl")
    return best_model, results, X_test, y_test

if __name__ == "__main__":
    df = load_and_preprocess()
    train_model(df)