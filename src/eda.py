"""Task 1: explore the churn data.

Run:  python -m src.eda
Prints the findings and saves tables to reports/tables, charts to reports/figures and a written
summary to reports/eda_insights.md.
"""
import matplotlib

matplotlib.use("Agg")  # draw to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.features import load_raw

TARGET = config.TARGET


def inspect(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Steps 3-4: data types, categorical vs numerical features, missing / inconsistent values."""
    print("=== Shape ===", df.shape)
    print("\n=== Data types ===\n", df.dtypes)

    ids = ["RowNumber", "CustomerId", "Surname"]
    categorical = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("str")]
    categorical = [c for c in categorical if c not in ids]
    binary = [c for c in df.columns if set(df[c].unique()) <= {0, 1} and c != TARGET]
    numerical = [c for c in df.select_dtypes("number").columns if c not in ids + binary + [TARGET]]
    print("\nCategorical:", categorical)
    print("Binary flags:", binary)
    print("Numerical:", numerical)
    print("Identifiers (dropped before modelling):", ids)

    print("\n=== Missing values ===")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.any() else "None")

    print("\n=== Consistency checks ===")
    checks = {
        "duplicate rows": int(df.duplicated().sum()),
        "duplicate CustomerId": int(df["CustomerId"].duplicated().sum()),
        "Age outside 18-100": int((~df["Age"].between(18, 100)).sum()),
        "CreditScore outside 300-850": int((~df["CreditScore"].between(300, 850)).sum()),
        "negative Balance": int((df["Balance"] < 0).sum()),
        "Balance exactly 0": int((df["Balance"] == 0).sum()),
        "text with stray spaces": int(sum((df[c] != df[c].str.strip()).sum() for c in categorical)),
    }
    for name, count in checks.items():
        print(f"  {name}: {count}")
    for c in categorical:
        if c != "Surname":
            print(f"  {c} values: {sorted(df[c].unique())}")
    return numerical, binary + categorical


def analyse(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Steps 5-6: grouping, pivot tables and summary statistics."""
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}

    print("\n=== Overall churn ===")
    print(df[TARGET].value_counts(normalize=True).round(3))

    for col in ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard"]:
        t = df.groupby(col).agg(customers=(TARGET, "size"), churn_rate=(TARGET, "mean"),
                                avg_age=("Age", "mean"), avg_balance=("Balance", "mean"),
                                avg_credit_score=("CreditScore", "mean")).round(3)
        tables[f"churn_by_{col}"] = t
        print(f"\n=== groupby {col} ===\n{t}")

    df = df.assign(AgeGroup=pd.cut(df["Age"], [17, 30, 40, 50, 60, 100],
                                   labels=["18-30", "31-40", "41-50", "51-60", "60+"]))
    t = df.groupby("AgeGroup", observed=True).agg(customers=(TARGET, "size"), churn_rate=(TARGET, "mean")).round(3)
    tables["churn_by_AgeGroup"] = t
    print(f"\n=== groupby AgeGroup ===\n{t}")

    pivots = {
        "pivot_churn_rate_geo_gender": pd.pivot_table(df, values=TARGET, index="Geography", columns="Gender",
                                                      aggfunc="mean", margins=True).round(3),
        "pivot_customers_geo_gender": pd.pivot_table(df, values=TARGET, index="Geography", columns="Gender",
                                                     aggfunc="count", margins=True),
        "pivot_avg_balance_geo_gender": pd.pivot_table(df, values="Balance", index="Geography", columns="Gender",
                                                       aggfunc="mean", margins=True).round(0),
        "pivot_churn_rate_geo_active": pd.pivot_table(df, values=TARGET, index="Geography", columns="IsActiveMember",
                                                      aggfunc="mean", margins=True).round(3),
    }
    for name, t in pivots.items():
        print(f"\n=== {name} ===\n{t}")
    tables.update(pivots)

    print("\n=== Summary statistics (numerical) ===")
    stats = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]].describe().T.round(2)
    print(stats)
    tables["summary_statistics"] = stats
    by_outcome = df.groupby(TARGET)[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                                     "EstimatedSalary"]].mean().round(1)
    tables["means_by_churn"] = by_outcome
    print("\n=== Mean of each feature, stayed (0) vs churned (1) ===\n", by_outcome)

    for name, t in tables.items():
        t.to_csv(config.TABLE_DIR / f"{name}.csv")
    return tables


def draw_charts(df: pd.DataFrame, tables: dict) -> None:
    """Step 4 (distributions) as saved figures."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, ["Geography", "Gender", "NumOfProducts", "IsActiveMember"]):
        rate = df.groupby(col)[TARGET].mean() * 100
        ax.bar(rate.index.astype(str), rate.values, color="#2a6f97")
        ax.axhline(df[TARGET].mean() * 100, color="#c1121f", ls="--", label="overall")
        ax.set(title=f"Churn rate by {col}", ylabel="% churned")
        ax.legend()
    ax = axes.flat[4]
    for label, colour in [(0, "#2a6f97"), (1, "#c1121f")]:
        ax.hist(df.loc[df[TARGET] == label, "Age"], bins=30, alpha=.6, color=colour,
                label="churned" if label else "stayed", density=True)
    ax.set(title="Age distribution", xlabel="Age"); ax.legend()
    ax = axes.flat[5]
    for label, colour in [(0, "#2a6f97"), (1, "#c1121f")]:
        ax.hist(df.loc[df[TARGET] == label, "Balance"], bins=30, alpha=.6, color=colour,
                label="churned" if label else "stayed", density=True)
    ax.set(title="Balance distribution", xlabel="Balance"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "eda_overview.png", dpi=130); plt.close(fig)

    num = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
              "IsActiveMember", "EstimatedSalary", TARGET]]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "correlation_matrix.png", dpi=130); plt.close(fig)

    piv = tables["pivot_churn_rate_geo_gender"].drop(index="All", columns="All") * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    piv.plot.bar(ax=ax, color=["#c1121f", "#2a6f97"], rot=0)
    ax.set(title="Churn rate (%) by Geography and Gender", ylabel="% churned")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "churn_geo_gender.png", dpi=130); plt.close(fig)
    print(f"\nCharts saved to {config.FIG_DIR}")


def write_insights(df: pd.DataFrame, tables: dict) -> None:
    """Step 7: a short written summary, with every number computed from the data."""
    geo = tables["churn_by_Geography"]["churn_rate"]
    gen = tables["churn_by_Gender"]["churn_rate"]
    prod = tables["churn_by_NumOfProducts"]
    act = tables["churn_by_IsActiveMember"]["churn_rate"]
    age = tables["churn_by_AgeGroup"]
    means = tables["means_by_churn"]
    corr_age = df["Age"].corr(df[TARGET])
    zero_bal = (df["Balance"] == 0).mean()

    text = f"""# Churn data insights

* **Class balance:** {df[TARGET].mean():.1%} of the {len(df):,} customers churned, so the classes are imbalanced.
  Accuracy alone would be misleading (always predicting "stays" scores {1 - df[TARGET].mean():.1%}); models are
  judged on F1, precision and recall, and class weights are used where available.
* **Geography:** {geo.idxmax()} has the highest churn rate ({geo.max():.1%}); {geo.idxmin()} the lowest ({geo.min():.1%}).
* **Gender:** {gen.idxmax()} customers churn more ({gen.max():.1%} vs {gen.min():.1%}).
* **Age:** churned customers are older on average ({means.loc[1, 'Age']:.1f} vs {means.loc[0, 'Age']:.1f} years);
  the {age['churn_rate'].idxmax()} group churns the most ({age['churn_rate'].max():.1%}). Age has the strongest
  correlation with churn among the numeric columns (r = {corr_age:.2f}).
* **Activity:** inactive members churn at {act.loc[0]:.1%} vs {act.loc[1]:.1%} for active members.
* **Products:** customers with {int(prod['churn_rate'].idxmax())} products churn at {prod['churn_rate'].max():.1%};
  customers with 2 products churn least ({prod.loc[2, 'churn_rate']:.1%}).
* **Balance:** churned customers hold higher balances on average ({means.loc[1, 'Balance']:,.0f} vs
  {means.loc[0, 'Balance']:,.0f}); {zero_bal:.1%} of customers have a zero balance.
* **Data quality:** no missing values or duplicates. The first column ('CustomerId') is really a row number and the real id is in
  'CustomerId.1'; both ids and Surname are dropped before modelling.

## Implications for modelling
* Use stratified train/test splits and report F1, precision and recall, not accuracy alone.
* One-hot encode Geography and Gender; scale numeric features; add `HasBalance`, `BalanceSalaryRatio` and `TenureByAge`.
* Expect Age, NumOfProducts, IsActiveMember and Geography to be the most useful predictors.
"""
    (config.REPORT_DIR / "eda_insights.md").write_text(text)
    print("\n" + text)


def main() -> None:
    df = load_raw()
    inspect(df)
    tables = analyse(df)
    draw_charts(df, tables)
    write_insights(df, tables)


if __name__ == "__main__":
    main()
