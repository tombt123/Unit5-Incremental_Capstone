"""Shared data loading and feature engineering.

The feature engineering step is the FIRST step of the saved model pipeline, so the API, the monitoring
scripts and training all apply exactly the same transformations (no training/serving skew).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def load_raw(path=config.RAW_DATA) -> pd.DataFrame:
    """Load Churn_Modeling.csv.

    In the provided file the first column is headed 'CustomerId' but it is really the row number (1..10000);
    the true customer id is in a column headed 'CustomerId.1'. We rename them so the names are honest.
    """
    df = pd.read_csv(path)
    if "CustomerId.1" in df.columns:
        df = df.rename(columns={"CustomerId": "RowNumber", "CustomerId.1": "CustomerId"})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered columns from the raw customer columns."""
    df = df.copy()
    df["HasBalance"] = (df["Balance"] > 0).astype(int)                         # 36% of customers hold a zero balance
    df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"].clip(lower=1)
    df["TenureByAge"] = df["Tenure"] / df["Age"].clip(lower=1)                 # share of life spent as a customer
    return df


def build_pipeline(classifier) -> Pipeline:
    """feature engineering -> scaling/encoding -> classifier, as ONE object."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("bin", "passthrough", config.BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", preprocessor),
        ("model", classifier),
    ])
