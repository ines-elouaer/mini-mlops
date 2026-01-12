# src/run_pipeline.py
import argparse
from src.pipelines.pipeline import training_pipeline

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/raw/breast_cancer.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cache", action="store_true", help="disable cache for this run")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # (optionnel) désactiver le cache globalement pour ce run
    if args.no_cache:
        import os
        os.environ["ZENML_ENABLE_STEP_CACHE"] = "false"

    training_pipeline(
        data_path=args.data_path,
        test_size=args.test_size,
        C=args.C,
        max_iter=args.max_iter,
        seed=args.seed,
    )
