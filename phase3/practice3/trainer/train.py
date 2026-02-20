import argparse, os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=500)
    parser.add_argument("--model_dir", type=str, default=None)
    args = parser.parse_args()

    # Prefer pipeline-provided output dir; fallback to Vertex env if needed
    model_dir = args.model_dir or os.environ.get("AIP_MODEL_DIR", "/tmp/model")
    os.makedirs(model_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    x1 = rng.normal(size=args.n_rows)
    x2 = rng.normal(size=args.n_rows)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.3, size=args.n_rows) > 0).astype(int)

    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    X = df[["x1", "x2"]]
    y = df["y"]

    clf = LogisticRegression()
    clf.fit(X, y)

    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(clf, model_path)

    print("Model dir:", model_dir)
    print("Saved model to:", model_path)
    print("Model dir contents:", os.listdir(model_dir))

if __name__ == "__main__":
    main()