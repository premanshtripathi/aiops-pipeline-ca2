import os
import sys

# Add project root directory to sys.path so 'main.py' can be imported from 'tests/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "RandomForestClassifier"
    assert "n_estimators" in data["saved_hyperparameters"]
    assert len(data["expected_features"]) == 8


def test_predict_high_risk_patient():
    payload = {
        "Pregnancies": 6,
        "Glucose": 168,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 36.6,
        "DiabetesPedigreeFunction": 0.672,
        "Age": 52
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["diagnosis"] in ["Diabetic", "Non-Diabetic"]
    assert 0.0 <= data["diabetes_risk_probability"] <= 1.0


def test_predict_low_risk_patient():
    payload = {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["diagnosis"] == "Non-Diabetic"