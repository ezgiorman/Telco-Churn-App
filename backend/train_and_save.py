import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


CSV_PATH = "Telco-Customer-Churn.csv"


df = pd.read_csv(CSV_PATH)


if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)


df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns=["Churn"])
y = df["Churn"]


cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# ✅ 8) Preprocess: kategorik -> OneHot, numeric -> olduğu gibi bırak
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)


gb = GradientBoostingClassifier(random_state=42)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", gb)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)

joblib.dump(pipe, "churn_pipeline.joblib")
print("✅ Saved: churn_pipeline.joblib")
