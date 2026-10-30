"""Simulated 'drift' used to demonstrate monitoring and retraining.

Real customer outcomes arrive over time; here we imitate that by gradually changing the test data:
  * covariate drift: customers get older and balances grow (the input data shifts)
  * concept drift: a competitor campaign makes younger customers outside Germany start churning
    (the relationship between inputs and churn changes, so the old model loses accuracy)
severity runs from 0 (no change) to 1 (strongest change).
"""
import numpy as np
import pandas as pd

from src import config


def drift_world(df: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    d = df.copy()
    # concept drift (uses the ORIGINAL age so the targeted segment stays the same customers)
    segment = (d["Age"] < 45) & (d["Geography"] != "Germany") & (d[config.TARGET] == 0)
    flip = segment & (rng.random(len(d)) < 0.9 * severity)
    d.loc[flip, config.TARGET] = 1
    # covariate drift
    d["Age"] = d["Age"] + int(round(12 * severity))
    d["Balance"] = d["Balance"] * (1 + 0.3 * severity)
    return d
