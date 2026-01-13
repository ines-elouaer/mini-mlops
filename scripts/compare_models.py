import argparse
from pathlib import Path
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_model(model_name: str, seed: int, C: float, max_iter: int, n_estimators: int, max_depth: int | None):
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="liblinear", random_state=seed))
        ])
    elif model_name == "rf":
        # RF n'a pas besoin de scaler
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/breast_cancer.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # logreg params
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)

    # rf params
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=0, help="0 => None")

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    mlflow.set_experiment("breast-cancer-compare")

    for model_name in ["logreg", "rf"]:
        max_depth = None if args.max_depth == 0 else args.max_depth
        model = build_model(
            model_name=model_name,
            seed=args.seed,
            C=args.C,
            max_iter=args.max_iter,
            n_estimators=args.n_estimators,
            max_depth=max_depth
        )

        with mlflow.start_run(run_name=f"baseline_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("data_path", args.data_path)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("seed", args.seed)

            # params spécifiques
            if model_name == "logreg":
                mlflow.log_param("C", args.C)
                mlflow.log_param("max_iter", args.max_iter)
            else:
                mlflow.log_param("n_estimators", args.n_estimators)
                mlflow.log_param("max_depth", max_depth if max_depth is not None else "None")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)

            # sauvegarde du modèle
            out_dir = Path("models")
            out_dir.mkdir(exist_ok=True)
            run_id = mlflow.active_run().info.run_id
            model_path = out_dir / f"{model_name}_{run_id}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            # confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_path = out_dir / f"cm_{model_name}_{run_id}.txt"
            cm_path.write_text(str(cm))
            mlflow.log_artifact(str(cm_path))

            mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

            print(f"[{model_name}] accuracy={acc:.4f} f1={f1:.4f} saved={model_path}")


if __name__ == "__main__":
    main()
