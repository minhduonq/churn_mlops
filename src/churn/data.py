import pandas as pd

def load_data(config: dict):
    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    original_path = config["data"]["original_path"]
    target_col = config["data"]["target_col"]

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    original = pd.read_csv(original_path)

    train[target_col] = train[target_col].map({"Yes": 1, "No": 0}).astype(int)
    original[target_col] = original[target_col].map({"Yes": 1, "No": 0}).astype(int)

    return train, test, original