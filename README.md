# 🧠 Mini MLOps Project — Breast Cancer Prediction

## 📌 Objectif
Ce projet implémente un **pipeline MLOps complet** pour un cas d’usage de **classification médicale (Breast Cancer)**, couvrant l’ensemble du cycle de vie ML : entraînement, suivi d’expériences, déploiement, CI/CD et orchestration.

---

## 🧰 Stack technique
- Python 3.11
- scikit-learn
- MLflow
- DVC
- ZenML
- Optuna
- FastAPI
- Docker & Docker Compose
- GitHub Actions (CI/CD)
- Kubernetes (minikube)




---

## 📊 Données & Modèle
- Dataset : Breast Cancer Wisconsin
- 30 features numériques
- Modèle :  
  - StandardScaler  
  - Logistic Regression
- Métriques :
  - Accuracy
  - F1-score
  - Confusion Matrix

Les données sont **versionnées avec DVC** (pas stockées directement dans Git).

---

## 📈 Experiment Tracking — MLflow
MLflow est utilisé pour :
- le suivi des paramètres
- le suivi des métriques
- la sauvegarde des modèles et artefacts

### Lancer MLflow


API : http://localhost:8000/docs

MLflow UI : http://localhost:5000

--- 
### Pipeline ML — ZenML

Un pipeline ZenML permet :

le chargement des données

l’entraînement du modèle

l’évaluation des performances

python -m src.run_pipeline

---
### Optimisation — Optuna

Une optimisation des hyperparamètres est réalisée avec Optuna.

python scripts/optuna_search.py


->Les résultats sont enregistrés dans MLflow.

---
### Docker & Docker Compose

L’application est entièrement containerisée.

docker compose up --build


---
### CI/CD — GitHub Actions

Un pipeline CI/CD est défini dans :

.github/workflows/ci.yml

---
### Kubernetes 

Déploiement local sur Kubernetes via minikube.


Déployer


kubectl apply -f k8s/

Vérifier

kubectl get pods
kubectl get svc


Accès à l’API

kubectl port-forward svc/mini-mlops-api-svc 8000:8000


lien : 

http://localhost:8000/health

http://localhost:8000/docs

---

### Interface utilisateur (UI)

Une interface web simple permet :

de charger automatiquement les features,

de remplir un exemple,

d’envoyer une requête à l’API,

de visualiser la prédiction et la probabilité.

