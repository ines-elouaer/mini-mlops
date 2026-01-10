from sklearn.datasets import load_breast_cancer
from pathlib import Path

def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})

    out_path = out_dir / "breast_cancer.csv"
    df.to_csv(out_path, index=False)

    print("✅ Saved:", out_path)
    print("Shape:", df.shape)
    print("Labels:", df["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
