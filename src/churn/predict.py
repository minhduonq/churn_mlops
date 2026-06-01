import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.churn.config import load_config

def load_fold_models(model_dir: Path):
    model_paths = sorted(model_dir.glob("xgb_fold_*.pkl"))

    if len(model_paths) == 0:
        raise FileNotFoundError(
            f"No fold models found in {model_dir}."
            "Expected file like xgb_fold_1.pkl, ..."
        )
    models = [joblib.load(path) for path in model_paths]
    return models, model_paths

def predict(config: dict, input_path: str, output_path: str):
    model_dir = Path(config["training"]["model_dir"])
    target_col = config["data"]["target_col"]
    id_col = config["data"]["id_col"]

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_pipeline_path = model_dir / "feature_pipeline.pkl"

    if not feature_pipeline_path.exists():
        raise FileNotFoundError(
            f"Feature pipeline not found: {feature_pipeline_path}"
        )
    print(f'Loading input data from {input_path}')
    df = pd.read_csv(input_path)

    print(f"loading feature pipeline from {feature_pipeline_path}")
    feature_builder = joblib.load(feature_pipeline_path)

    print(f"Loading models from {model_dir}")
    models, model_paths = load_fold_models(model_dir)

    print("Building features....")
    X = feature_builder.transform(
        df.drop(columns=[target_col], errors='ignore')
    )

    print(f"Input shape after feature engineering: {X.shape}")

    preds = np.zeros(len(df))
    for model, model_path in zip(models, model_paths):
        print(f"Predicting with {model_path.name}")
        preds += model.predict_proba(X)[:,1] / len(models)
        
    result = pd.DataFrame()
    if id_col in df.columns:
        result[id_col] = df[id_col]
    else:
        result[id_col] = np.arange(len(df))

    result["churn_probability"] = preds
    result["prediction"] = (preds >= 0.5).astype(int)
    result.to_csv(output_path, index=False)
    print(f"Saved model to {output_path}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="configs/xgb_config.yaml"
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )

    parser.add_argument(
        "--output", type=str, default="data/predictions/predictions.csv", help="Path to output prediction CSV file"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    predict(config, input_path=args.input, output_path=args.output)

if __name__ == "__main__":
    main()
