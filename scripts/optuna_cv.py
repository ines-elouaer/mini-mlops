import optuna
import mlflow
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/raw/breast_cancer.csv"
STUDY_NAME = "breast-cancer-optuna-cv"
STORAGE = "sqlite:///optuna.db"


def objective(trial):
    model_name = trial.suggest_categorical("model", ["logreg", "rf"])

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_name == "logreg":
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 800)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="liblinear"))
        ])
        params_to_log = {"model": model_name, "C": C, "max_iter": max_iter}

    else:
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        params_to_log = {
            "model": model_name,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
        }

    # CV score
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    f1_mean = float(f1_scores.mean())
    f1_std = float(f1_scores.std())

    with mlflow.start_run(nested=True):
        for k, v in params_to_log.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("f1_mean", f1_mean)
        mlflow.log_metric("f1_std", f1_std)

    return f1_mean


if __name__ == "__main__":
    mlflow.set_experiment("breast-cancer-optuna-cv")

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=10)

    print("Best params:", study.best_params)
    print("Best f1_mean:", study.best_value)
