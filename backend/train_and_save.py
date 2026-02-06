import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# ✅ 1) CSV dosya adı (gerekirse değiştir)
CSV_PATH = "Telco-Customer-Churn.csv"

# ✅ 2) Veriyi oku
df = pd.read_csv(CSV_PATH)

# ✅ 3) customerID varsa sil
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# ✅ 4) TotalCharges sayıya çevir (boşluklar NaN olur)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# ✅ 5) Hedefi 0/1'e çevir
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# ✅ 6) X / y
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ✅ 7) Kategorik ve sayısal kolonlar
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# ✅ 8) Preprocess: kategorik -> OneHot, numeric -> olduğu gibi bırak
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# ✅ 9) Model (Gradient Boosting)
gb = GradientBoostingClassifier(random_state=42)

# ✅ 10) Pipeline: preprocess + model
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", gb)
])

# ✅ 11) Split (stratify önemli)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ 12) Train
pipe.fit(X_train, y_train)

# ✅ 13) Kaydet
joblib.dump(pipe, "churn_pipeline.joblib")
print("✅ Saved: churn_pipeline.joblib")
