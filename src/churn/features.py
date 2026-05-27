import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold

def basic_preprocess(train, test, original, config):
    categortical_cols = config["features"]['categorical_cols']
    numerical_cols = config["features"]["numerical_cols"]

    for df in [train, test, original]:
        for col in categortical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return train, test, original

def add_frequency_features(train, test, original, categorical_cols):
    for col in categorical_cols:
        freq = pd.concat([train[col],original[col]]).value_counts(normalize=True)
        
        for df in [train, test, original]:
            mapped = df[col].map(freq).astype("float32")
            df[f"FREQ_{col}"] = mapped.fillna(0)
    
    return train, test, original

def add_ngram_features(train, test, top_cats):
    bigram_cols = []
    trigram_cols = []

    for c1, c2 in combinations(top_cats, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")
        bigram_cols.append(col_name)
    
    for c1, c2, c3 in combinations(top_cats[:4], 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype("category")
        trigram_cols.append(col_name)

    return train, test, bigram_cols, trigram_cols

def build_features(train, test, original, config):
    categorical_cols = config["features"]["categorical_cols"]
    numerical_cols = config["features"]["numerical_cols"]

    train, test, original = basic_preprocess(train, test, original, config)
    train, test, original = add_frequency_features(train, test, original, categorical_cols)

    top_cats_for_ngram = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
    ]

    train, test, bigram_cols, trigram_cols = add_ngram_features(train, test, top_cats_for_ngram)

    feature_cols = []
    feature_cols.extend(categorical_cols)
    feature_cols.extend(numerical_cols)

    freq_cols = [f"FREQ_{col}" for col in categorical_cols]
    feature_cols.extend(freq_cols)
    feature_cols.extend(bigram_cols)
    feature_cols.extend(trigram_cols)

    feature_cols = [col for col in feature_cols if col in train.columns]
    return train, test, original, feature_cols

