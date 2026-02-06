import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Telco Churn API")

# Frontend'den istek gelebilsin diye CORS açıyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # sonra istersen sadece localhost'a indirgeriz
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline'ı yükle
pipe = joblib.load("churn_pipeline.joblib")


# ✅ Frontend formundan gelecek alanlar (Telco dataset kolonları)
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: ChurnInput):
    # JSON -> DataFrame (pipeline bunu bekliyor)
    df = pd.DataFrame([data.model_dump()])

    # churn olasılığı (class 1)
    proba = float(pipe.predict_proba(df)[:, 1][0])
    label = int(proba >= 0.5)

    return {
        "churn_probability": proba,
        "churn_label": label
    }
