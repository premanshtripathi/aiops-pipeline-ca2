from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import os

# 1. Initialize FastAPI App
app = FastAPI(
    title="Pima Diabetes Prediction API",
    description="FastAPI microservice serving a Random Forest model trained on Kaggle Pima Indians Diabetes Dataset.",
    version="1.0.0"
)

# 2. Load the saved model artifact (model + parameters) on startup from 'models/' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "model.joblib"))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure model.joblib is inside the 'models/' directory.")

artifact = joblib.load(MODEL_PATH)

# Unpack model, parameters, and feature names from the Stage 3 bundle
model = artifact["model"]
model_params = artifact["params"]
feature_names = artifact["feature_names"]
test_accuracy = artifact["test_accuracy"]


# 3. Define Request Schema using Pydantic
class PatientMetrics(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    Glucose: int = Field(..., ge=0, description="Plasma glucose concentration")
    BloodPressure: int = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: int = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    Insulin: int = Field(..., ge=0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0.0, description="Body mass index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, description="Diabetes pedigree function score")
    Age: int = Field(..., ge=1, description="Age in years")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Pregnancies": 2,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }
    )


# 4. Health Check Endpoint
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Diabetes Prediction API",
        "model_loaded": True
    }


# 5. Model & Hyperparameters Info Endpoint (Proves Stage 3 parameter saving)
@app.get("/model-info")
def get_model_info():
    return {
        "model_type": type(model).__name__,
        "saved_hyperparameters": model_params,
        "expected_features": feature_names,
        "colab_test_accuracy": test_accuracy
    }


# 6. Prediction Endpoint
@app.post("/predict")
def predict_diabetes(patient: PatientMetrics):
    try:
        # Convert incoming Pydantic object to a DataFrame matching exact training feature order
        input_dict = patient.model_dump()
        input_df = pd.DataFrame([input_dict], columns=feature_names)

        # Perform prediction and probability estimation
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        return {
            "prediction": prediction,
            "diagnosis": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "diabetes_risk_probability": round(probability, 4),
            "diabetes_risk_percentage": f"{round(probability * 100, 2)}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")