"""Data preparation: clean, drop identifiers and split into train and test sets.

Run:  python -m src.prepare_data
Encoding and scaling are NOT done here; they live inside the model pipeline (src/features.py)
so that the API applies exactly the same steps to new customers.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.features import load_raw


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for col in ["Geography", "Gender"]:
        df[col] = df[col].str.strip().str.title()
    if df[config.RAW_FEATURES + [config.TARGET]].isna().any().any():
        # None are present in this dataset, but fill defensively so the pipeline never breaks on new data.
        for col in config.RAW_FEATURES:
            df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == object else df[col].median())
        df = df.dropna(subset=[config.TARGET])
    return df[config.RAW_FEATURES + [config.TARGET]]  # drops RowNumber, CustomerId, Surname


def main() -> None:
    df = clean(load_raw())
    train, test = train_test_split(df, test_size=0.2, stratify=df[config.TARGET],
                                   random_state=config.RANDOM_STATE)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)
    print(f"Saved {len(train)} train rows ({train[config.TARGET].mean():.1%} churn) and "
          f"{len(test)} test rows ({test[config.TARGET].mean():.1%} churn) to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
