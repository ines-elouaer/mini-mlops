import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn

def main(data_path: str, test_size: float, C: float, max_iter: int, seed: int,
         solver: str, run_name: str, run_type: str):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=seed
        )),
    ])

    mlflow.set_experiment("breast-cancer-baseline")

    with mlflow.start_run(run_name=run_name):
        # Tags pour filtrer dans MLflow
        mlflow.set_tag("run_type", run_type)
        mlflow.set_tag("model", "logreg")

        # Params
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("seed", seed)
        mlflow.log_param("solver", solver)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        out_dir = Path("models")
        out_dir.mkdir(exist_ok=True)

        run_id = mlflow.active_run().info.run_id
        model_path = out_dir / f"model_{run_id}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        cm = confusion_matrix(y_test, y_pred)
        cm_path = out_dir / f"confusion_matrix_{run_id}.txt"
        cm_path.write_text(str(cm))
        mlflow.log_artifact(str(cm_path))

        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        print(f"✅ Done. accuracy={acc:.4f}, f1={f1:.4f}")
        print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/breast_cancer.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", default="liblinear", choices=["liblinear", "lbfgs", "saga"])
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--run-type", default="baseline")  # baseline | manual_variation | optuna_best
    args = parser.parse_args()

    main(args.data_path, args.test_size, args.C, args.max_iter, args.seed,
         args.solver, args.run_name, args.run_type)
