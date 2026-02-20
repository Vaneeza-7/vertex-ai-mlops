import argparse, os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from google.cloud import storage

def upload_dir_to_gcs(local_dir: str, gcs_dir: str):
    # gcs_dir like: gs://bucket/prefix
    assert gcs_dir.startswith("gs://")
    _, _, rest = gcs_dir.partition("gs://")
    bucket_name, _, prefix = rest.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_dir):
        for f in files:
            local_path = os.path.join(root, f)
            rel = os.path.relpath(local_path, local_dir)
            blob_path = f"{prefix.rstrip('/')}/{rel}"
            bucket.blob(blob_path).upload_from_filename(local_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)          # local
    parser.add_argument("--model_gcs_dir", type=str, required=True)      # gs://...
    parser.add_argument("--n_rows", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    x1 = rng.normal(size=args.n_rows)
    x2 = rng.normal(size=args.n_rows)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.3, size=args.n_rows) > 0).astype(int)

    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    X = df[["x1", "x2"]]
    y = df["y"]

    clf = LogisticRegression()
    clf.fit(X, y)

    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

    upload_dir_to_gcs(args.model_dir, args.model_gcs_dir)
    print("Uploaded model dir to:", args.model_gcs_dir)

if __name__ == "__main__":
    main()
