from zenml import pipeline
from .steps import load_data, train_model, evaluate_model

@pipeline
def training_pipeline(
    data_path: str,
    test_size: float = 0.2,
    C: float = 1.0,
    max_iter: int = 200,
    seed: int = 42,
):
    df = load_data(data_path=data_path)
    model, X_test, y_test = train_model(df=df, test_size=test_size, C=C, max_iter=max_iter, seed=seed)
    accuracy, f1 = evaluate_model(model=model, X_test=X_test, y_test=y_test)
