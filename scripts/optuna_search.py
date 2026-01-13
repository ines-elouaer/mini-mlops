# scripts/optuna_search.py
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "data/raw/breast_cancer.csv"
STUDY_NAME = "breast-cancer-optuna"
STORAGE = "sqlite:///optuna.db"

def objective(trial):
    C = trial.suggest_float("C", 0.001, 100.0, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 1000)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="liblinear"))
    ])

    with mlflow.start_run(nested=True):
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1 = f1_score(y_test, preds)
        mlflow.log_metric("f1", f1)

    return f1

if __name__ == "__main__":
    mlflow.set_experiment("breast-cancer-optuna")

    # Run parent = résumé
    with mlflow.start_run(run_name="optuna-summary"):
        study = optuna.create_study(
            direction="maximize",
            study_name=STUDY_NAME,
            storage=STORAGE,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=10)

        # log best trial in parent
        mlflow.log_param("best_C", study.best_params["C"])
        mlflow.log_param("best_max_iter", study.best_params["max_iter"])
        mlflow.log_metric("best_f1", study.best_value)

        print("Best params:", study.best_params)
        print("Best f1:", study.best_value)
