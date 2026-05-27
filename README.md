# Customer Churn Prediction - Production-style ML Pipeline

This project refactors a Kaggle customer churn notebook into a production-style machine learning pipeline.

## Features

- XGBoost-based churn prediction
- Feature engineering pipeline
- 5-fold stratified cross-validation
- Config-driven training with YAML
- Model artifact saving
- OOF prediction and submission generation

## Project Structure

```text
configs/        Training configuration
data/           Raw and processed data
models/         Saved models and metrics
notebooks/      Original Kaggle notebook
src/churn/      Source code