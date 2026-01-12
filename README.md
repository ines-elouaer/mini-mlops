# Mini MLOps Project – Breast Cancer Prediction

## Stack
- Python, scikit-learn
- MLflow
- DVC
- FastAPI
- Docker
- GitHub Actions (CI)
- Kubernetes (minikube)

## How to run locally
docker compose up --build

## API
- GET /health
- GET /features
- POST /predict

## CI/CD
- GitHub Actions workflow in `.github/workflows/ci.yml`

## Kubernetes
kubectl apply -f k8s/
minikube service mini-mlops-api-svc
![CI](https://github.com/ines-elouaer/mini-mlops/actions/workflows/ci.yml/badge.svg)
