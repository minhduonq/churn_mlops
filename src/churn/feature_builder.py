import pandas as pd
from itertools import combinations

class ChurnFeatureBuilder:
    def __init__(self, config: dict):
        self.config = config

        self.categorical_cols = config["features"]["categorical_cols"]
        self.numerical_cols = config["features"]["numerical_cols"]
        self.freq_maps = {}
        self.feature_cols = []

        self.top_cat_for_ngram = [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "OnlineSecurity",
            "TechSuport",
        ]
    
    def fit(self, df: pd.DataFrame):
        df = df.copy()

        for col in self.categorical_cols:
            if col in df.columns:
                self.freq_maps[col] = df[col].value_counts(normalize=True)
        
        transformed = self.transform(df)
        self.feature_cols = list(transformed.columns)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Frequency encoding
        for col in self.categorical_cols:
            if col in df.columns and col in self.freq_maps:
                df[f"FREQ_{col}"] = (df[col].map(self.freq_maps[col]).fillna(0).astype("float32"))
            
        ngram_cols = []
        available_top_cats = [col for col in self.top_cat_for_ngram if col in df.columns]

        for c1, c2 in combinations(available_top_cats,2):
            col_name = f"{c1}_{c2}"
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")
            ngram_cols.append(col_name)

        for c1, c2, c3 in combinations(available_top_cats, 3):
            col_name = f"{c1}_{c2}_{c3}"
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype("category")
            ngram_cols.append(col_name)
        
        feature_cols = []
        for col in self.categorical_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        for col in self.numerical_cols:
            if col in df.columns:
                feature_cols.append(col)

        freq_cols = [f"FREQ_{col}" 
                     for col in self.categorical_cols 
                     if f"FREQ_{col}" in df.columns]
        
        feature_cols.extend(freq_cols)
        feature_cols.extend(ngram_cols)
        
        output = df[feature_cols].copy()

        if self.feature_cols:
            for col in self.feature_cols:
                if col not in output.columns:
                    output[col] = 0
            output = output[self.feature_cols]
        
        return output
    
    def fit_transform(self, df:pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
    
    
        