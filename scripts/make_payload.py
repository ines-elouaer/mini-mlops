import json
import pandas as pd

df = pd.read_csv("data/raw/breast_cancer.csv")
df.columns = [c.strip() for c in df.columns]

# on prend une ligne d'exemple
row = df.drop(columns=["label"]).iloc[0].to_dict()

payload = {"features": {k: float(v) for k, v in row.items()}}

with open("payload.json", "w") as f:
    json.dump(payload, f, indent=2)

print("✅ payload.json généré (copie/colle dans Swagger /predict)")
