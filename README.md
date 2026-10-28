# End-to-End Diabetes Prediction MLOps Pipeline (`aiops-pipeline-ca2`)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)

An end-to-end Machine Learning & MLOps pipeline built for binary classification on the **Pima Indians Diabetes Dataset**. This project demonstrates the complete lifecycle from model training and artifact serialization (`model.joblib`) to serving predictions via **FastAPI**, containerization with **Docker**, and automated Continuous Integration (**CI**) using **GitHub Actions**.

---

## 1. Case Study & Dataset Overview

* **Problem Statement:** Predict whether a patient has diabetes (`1` = Diabetic, `0` = Non-Diabetic) based on diagnostic medical measurements.
* **Dataset:** Pima Indians Diabetes Database (`768` rows, `8` numerical input features, `1` binary target column `Outcome`).
* **Input Features:**
  1. `Pregnancies`: Number of times pregnant (`int`)
  2. `Glucose`: Plasma glucose concentration (`int`)
  3. `BloodPressure`: Diastolic blood pressure in mm Hg (`int`)
  4. `SkinThickness`: Triceps skin fold thickness in mm (`int`)
  5. `Insulin`: 2-Hour serum insulin in mu U/ml (`int`)
  6. `BMI`: Body mass index in $kg/m^2$ (`float`)
  7. `DiabetesPedigreeFunction`: Genetic diabetes likelihood score (`float`)
  8. `Age`: Age in years (`int`)

---

## 2. Repository Structure

```text
aiops-pipeline-ca2/
├── .github/
│   └── workflows/
│       └── ci.yaml          # GitHub Actions CI pipeline (Pytest + Docker smoke test)
├── data/
│   └── diabetes.csv         # Pima Indians Diabetes dataset
├── models/
│   └── model.joblib         # Serialized Random Forest model + hyperparameters + metadata
├── scripts/
│   └── train.ipynb          # Jupyter/Colab notebook for EDA, training, evaluation & export
├── tests/
│   └── test_main.py         # Automated Pytest suite for FastAPI endpoints
├── .dockerignore            # Excludes local virtual environments & caches from Docker build
├── .gitignore               # Excludes __pycache__, .pytest_cache, and .venv
├── Dockerfile               # Container definition for the FastAPI inference server
├── main.py                  # FastAPI application serving /, /model-info, and /predict
├── requirements.txt         # Pinned Python dependencies
└── README.md                # Project documentation
```

---

## 3. Model Architecture & Serialized Artifact (`model.joblib`)

The model is trained in `scripts/train.ipynb` using an 80/20 stratified train-test split (`random_state=42`).

* **Algorithm:** `RandomForestClassifier` (scale-invariant ensemble decision trees)
* **Saved Hyperparameters:**
  * `n_estimators`: `100`
  * `max_depth`: `6`
  * `min_samples_split`: `5`
  * `random_state`: `42`
* **Artifact Bundle Structure:** Instead of saving only the raw estimator, `models/model.joblib` stores a structured dictionary containing:
  * `model`: Trained `RandomForestClassifier` instance
  * `params`: Tuned hyperparameter dictionary
  * `feature_names`: Ordered list of expected input columns
  * `test_accuracy`: Hold-out test accuracy (`~75.97%`)
  * `sklearn_version`: Version tracking for reproducibility

---

## 4. FastAPI Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check verifying service status and model readiness |
| `GET` | `/model-info` | Returns model type, saved hyperparameters, feature list, and test accuracy |
| `POST` | `/predict` | Validates patient metrics via Pydantic and returns class prediction + risk probability |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### Sample `POST /predict` Request
```json
{
  "Pregnancies": 2,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

### Sample Response
```json
{
  "prediction": 1,
  "diagnosis": "Diabetic",
  "diabetes_risk_probability": 0.7214,
  "diabetes_risk_percentage": "72.14%"
}
```

---

## 5. Local Setup & Execution (VS Code)

### 1. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
pytest -v
```

### 3. Start the FastAPI Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Open **`http://127.0.0.1:8000/docs`** in your browser to test the endpoints interactively.

---

## 6. Docker Execution

### 1. Build the Docker Image
```bash
docker build -t diabetes-fastapi-app:v1 .
```

### 2. Run the Container
```bash
docker run -d -p 8000:8000 --name diabetes-api-container diabetes-fastapi-app:v1
```

### 3. Test Containerized Endpoints via CLI
```bash
# Health Check
curl -X GET "http://localhost:8000/"

# Model Info & Saved Hyperparameters
curl -X GET "http://localhost:8000/model-info"

# Live Prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Pregnancies": 2, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}'
```

---

## 7. Continuous Integration (GitHub Actions CI)

The workflow defined in `.github/workflows/ci.yaml` triggers automatically on every `push` and `pull_request` to `main`/`master` (and supports manual runs via `workflow_dispatch`):

1. **Checkout & Python Setup:** Provisions `ubuntu-latest` with Python `3.11` and `pip` caching.
2. **Artifact & Unit Testing:** Verifies `models/model.joblib` exists and runs the 4-test `pytest` suite (`tests/test_main.py`).
3. **Docker Build & Smoke Test:** Builds the production Docker image, starts the container on port `8000`, and executes live `curl --fail` requests against `/`, `/model-info`, and `/predict`.
4. **Automated Teardown:** Stops and removes the test container cleanly using `if: always()`.