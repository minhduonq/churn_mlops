import argparse
from pathlib import Path 
import json
import joblib

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.churn.config import load_config
from src.churn.data import load_data
from src.churn.features import ChurnFeatureBuilder

def train_cv(config:dict):
    seed = config["project"]["seed"]
    target_col = config["data"]["target_col"]
    id_col = config["data"]["id_col"]

    n_folds = config["training"]["n_folds"]
    model_dir = Path(config["training"]["model_dir"])
    prediction_dir = Path(config["training"]["prediction_dir"])

    model_dir.mkdir(parents=True, exist_ok=True)

    prediction_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train, test, original = load_data(config)

    feature_fit_df = pd.concat(
        [train.drop(columns=[target_col], errors="ignore"),
         original.drop(columns=[target_col], errors='ignore')],
         axis=0,
         ignore_index=True,
    )

    feature_builder = ChurnFeatureBuilder(config)
    feature_builder.fit(feature_fit_df)

    X = feature_builder.transform(train.drop(columns=[target_col], errors='ignore'))
    y = train[target_col].copy()

    X_test = feature_builder.transform(test)

    # print("Building features...")
    # train, test, original, feature_cols = build_features(train, test, original, config)

    # X = train[feature_cols].copy()
    # y = train[target_col].copy()
    # X_test = test[feature_cols].copy()

    feature_cols = feature_builder.feature_cols

    print(f"Train shape: {X.shape}")
    print(f"Test shape: {X_test.shape}")
    # print(f"Features: {len(feature_cols)}")

    xgb_params = config["xgboost"].copy()
    xgb_params['random_state'] = seed

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X,y), start=1):
        print(f"\n--------Fold {fold}/{n_folds}------")

        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()

        model = xgb.XGBClassifier(**xgb_params)

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=200)

        valid_pred = model.predict_proba(X_valid)[:,1]
        test_pred = model.predict_proba(X_test)[:,1]

        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(fold_auc)

        oof_preds[valid_idx] = valid_pred
        test_preds += test_pred/n_folds

        print(f"Fold {fold} AUC: {fold_auc:.6f}")
        model_path= model_dir / f"xgb_fold_{fold}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")
    overall_auc = roc_auc_score(y, oof_preds)

    metrics = {
        "oof_auc": float(overall_auc),
        "fold_auc": [float(score) for score in fold_scores],
        "mean_fold_auc": float(np.mean(fold_scores)),
        "std_fold_auc": float(np.std(fold_scores)),
        "n_folds": n_folds,
        "n_features": len(feature_cols),
    }

    feature_pipeline_path = model_dir / "feature_pipeline.pkl"
    joblib.dump(feature_builder, feature_pipeline_path)

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    
    with open(model_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    oof_df = pd.DataFrame({
        id_col: train[id_col],
        target_col: oof_preds,
    })

    oof_df.to_csv(prediction_dir / "oof_prediction.csv", index=False)

    submission_df = pd.DataFrame({
        id_col: test[id_col],
        target_col: test_preds
    })

    submission_df.to_csv(prediction_dir / "submission.csv", index=False)

    print("\n-------Training finished-------")
    print(f"OOF AUC: {overall_auc:.6f}")
    print(f"Mean fold AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/xgb_config.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_cv(config)

if __name__ == "__main__":
    main()







    