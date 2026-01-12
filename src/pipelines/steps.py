from typing import Tuple
import pandas as pd
from zenml import step

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@step
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]  # nettoyage colonnes
    return df


# ✅ Si tu veux forcer l'entraînement à chaque run (pas de cache):
# @step(enable_cache=False)
@step(enable_cache=False)
def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    C: float = 1.0,
    max_iter: int = 200,
    seed: int = 42,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    # cible = label
    if "label" not in df.columns:
        raise ValueError(f"Colonne 'label' introuvable. Colonnes: {df.columns.tolist()}")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ✅ Pipeline = StandardScaler + LogisticRegression (stable)
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="liblinear",
            random_state=seed
        )),
    ])

    model.fit(X_train, y_train)

    return model, X_test, y_test


# ✅ Si tu veux éviter le cache aussi ici:
# @step(enable_cache=False)
@step(enable_cache=False)
def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, float]:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    return acc, f1
