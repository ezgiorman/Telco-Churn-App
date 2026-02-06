# Telco Customer Churn Prediction (Full-Stack)

This project is a full-stack machine learning application that predicts  
customer churn for a telecom company.

## Features

- Trained **Decision Tree–based machine learning pipeline**
- **FastAPI backend** serving the trained model
- **Web frontend** (HTML + JavaScript)
- Interactive **dashboard with charts**
- Fully decoupled frontend & backend
- Ready for **cloud deployment**

---

## Project Structure

```text
churn_app/
├── backend/
│   ├── app.py                  # FastAPI backend
│   ├── requirements.txt         # Python dependencies
│   └── churn_pipeline.joblib    # Trained ML pipeline
│
├── frontend/
│   ├── index.html               # Prediction UI
│   └── dashboard.html           # Charts & analytics
│
└── README.md


## Model Information

- **Task:** Binary classification (Churn / Not Churn)
- **Algorithm:** Decision Tree Classifier
- **Preprocessing:**
  - One-Hot Encoding for categorical features
  - Numerical features passed directly
- **Outputs:**
  - Churn probability
  - Binary churn label (threshold = 0.5)

The trained model is saved as a `.joblib` pipeline and loaded by the backend at application startup.

---

## How the System Works

1. User fills in customer information in the frontend form
2. Frontend sends the input data to the backend via `/predict` API
3. Backend loads the trained ML pipeline
4. Model generates churn probability and prediction
5. Result is returned to frontend and displayed
6. Predictions are stored locally in the browser for dashboard visualization

---

## Local Run (Development)

### Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload


http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

Frontend Setup
-- cd frontend
-- python -m http.server 5173

Open in browser:
http://127.0.0.1:5173/index.html

Deployment
- This project can be deployed to the cloud as follows:

-- Backend --
Render: https://render.com
Railway: https://railway.app

-- Frontend --
Netlify: https://www.netlify.com
Vercel: https://vercel.com

Notes
No database is used
- Prediction history is stored in browser localStorage
- Backend and frontend are completely independent
The model can be replaced by updating:
- backend/churn_pipeline.joblib
