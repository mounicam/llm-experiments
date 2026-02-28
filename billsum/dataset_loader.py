import pandas as pd
from datasets import Dataset, DatasetDict


def load_splits(train_path, dev_path) -> DatasetDict:
    """
    Load training and validation datasets from JSONL files.

    Args:
        train_path (str): Path to training JSONL file
        dev_path (str): Path to validation JSONL file

    Returns:
        DatasetDict: Dictionary containing 'train' and 'validation' datasets
    """
    train_df = pd.read_json(train_path, lines=True)
    train_df = train_df.drop(columns=["__index_level_0__"])
    dev_df = pd.read_json(dev_path, lines=True)
    dev_df = dev_df.drop(columns=["__index_level_0__"])

    # Drop rows with missing fields
    for df in (train_df, dev_df):
        df.dropna(subset=["text", "summary", "level"], inplace=True)

    train = Dataset.from_pandas(train_df)
    dev = Dataset.from_pandas(dev_df)

    dataset = DatasetDict({"train": train, "validation": dev})
    print(dataset)
    return dataset
