"""HW2: Wine quality multiple linear regression.

Colab setup:
    !pip install kagglehub pandas numpy matplotlib seaborn scikit-learn statsmodels joblib
    !python 5115056030_hw2.py
"""

from pathlib import Path

import joblib
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
DATASET_SLUG = "yasserh/wine-quality-dataset"
OUTPUT_DIR = Path("wine_quality_outputs")


def locate_dataset() -> Path:
    """Use a local CSV when available; otherwise download the public Kaggle dataset."""
    local_candidates = [Path("WineQT.csv"), Path("winequality-red.csv")]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate

    dataset_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    candidates = list(dataset_dir.rglob("WineQT.csv"))
    if not candidates:
        candidates = list(dataset_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found in KaggleHub directory: {dataset_dir}")
    return candidates[0]


def make_pipeline(selected: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if selected:
        steps.append(("select", SelectKBest(score_func=f_regression, k=5)))
    steps.extend([("reg", LinearRegression())])
    return Pipeline(steps)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_path = locate_dataset()
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)
    df = df.drop(columns=["Id", "id"], errors="ignore")

    if "quality" not in df.columns:
        raise KeyError("The dataset must contain a 'quality' target column.")

    X = df.drop(columns=["quality"])
    y = df["quality"]
    if X.shape[1] != 11:
        raise ValueError(f"Expected 11 input features, found {X.shape[1]}.")

    print(f"Dataset: {data_path}")
    print(f"Shape: {df.shape}")
    print("Missing values:\n", df.isna().sum())
    print("Quality distribution:\n", y.value_counts().sort_index())

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Wine quality feature correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_correlation_heatmap.png", dpi=180)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    models = {
        "A_Simple_alcohol": (["alcohol"], make_pipeline()),
        "B_Multiple_all11": (list(X.columns), make_pipeline()),
        "C_Selected_top5": (list(X.columns), make_pipeline(selected=True)),
    }

    results = []
    predictions = {}
    for name, (columns, model) in models.items():
        model.fit(X_train[columns], y_train)
        prediction = model.predict(X_test[columns])
        predictions[name] = prediction
        mse = mean_squared_error(y_test, prediction)
        selected_cols = columns
        if name == "C_Selected_top5":
            selected_cols = X_train[columns].columns[
                model.named_steps["select"].get_support()
            ].tolist()
        results.append(
            {
                "Model": name,
                "Features": ", ".join(selected_cols),
                "N_features": len(selected_cols),
                "MAE": mean_absolute_error(y_test, prediction),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_test, prediction),
            }
        )
        joblib.dump(model, OUTPUT_DIR / f"{name}.joblib")

    metrics = pd.DataFrame(results).sort_values("RMSE")
    print("\nModel metrics:\n", metrics.to_string(index=False))
    metrics.to_csv(OUTPUT_DIR / "02_model_metrics.csv", index=False, encoding="utf-8-sig")

    # Compare every model on the same held-out observations.
    comparison_order = np.argsort(y_test.to_numpy())
    plt.figure(figsize=(12, 6))
    plt.scatter(
        np.arange(len(y_test)), y_test.to_numpy()[comparison_order],
        color="black", s=16, alpha=0.55, label="Actual quality",
    )
    colors = {"A_Simple_alcohol": "#2563eb", "B_Multiple_all11": "#dc2626", "C_Selected_top5": "#059669"}
    labels = {
        "A_Simple_alcohol": "Simple alcohol",
        "B_Multiple_all11": "Multiple all 11",
        "C_Selected_top5": "SelectKBest top 5",
    }
    for name, prediction in predictions.items():
        plt.plot(
            np.arange(len(y_test)), prediction[comparison_order],
            linewidth=1.4, color=colors[name], label=labels[name],
        )
    plt.xlabel("Test samples (sorted by actual quality)")
    plt.ylabel("Quality score")
    plt.title("Comparison of Linear Regression Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_model_comparison.png", dpi=200)
    plt.close()

    chart_metrics = metrics.sort_values("N_features")
    plt.figure(figsize=(7, 5))
    plt.plot(chart_metrics["N_features"], chart_metrics["RMSE"], marker="o", linewidth=2)
    plt.xticks(chart_metrics["N_features"])
    plt.xlabel("Number of features")
    plt.ylabel("RMSE")
    plt.title("RMSE by Number of Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_rmse_by_features.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(chart_metrics["N_features"], chart_metrics["R2"], marker="o", linewidth=2)
    plt.xticks(chart_metrics["N_features"])
    plt.xlabel("Number of features")
    plt.ylabel("R-squared")
    plt.title("R-squared by Number of Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_r2_by_features.png", dpi=200)
    plt.close()

    selected_pipe = models["C_Selected_top5"][1]
    selector = selected_pipe.named_steps["select"]
    selected_names = X_train.columns[selector.get_support()].tolist()
    imputer = selected_pipe.named_steps["imputer"]
    tr_selected = selector.transform(imputer.transform(X_train))
    te_selected = selector.transform(imputer.transform(X_test))

    X_ols_train = sm.add_constant(tr_selected, has_constant="add")
    X_ols_test = sm.add_constant(te_selected, has_constant="add")
    ols = sm.OLS(y_train.to_numpy(), X_ols_train).fit()
    frame = ols.get_prediction(X_ols_test).summary_frame(alpha=0.05)

    order = np.argsort(frame["mean"].to_numpy())
    actual = y_test.to_numpy()[order]
    predicted = frame["mean"].to_numpy()[order]
    lower = frame["obs_ci_lower"].to_numpy()[order]
    upper = frame["obs_ci_upper"].to_numpy()[order]
    x_axis = np.arange(len(order))

    plt.figure(figsize=(13, 6))
    plt.fill_between(x_axis, lower, upper, alpha=0.28, label="95% prediction interval")
    plt.plot(x_axis, predicted, linewidth=1.6, label="Predicted quality")
    plt.scatter(x_axis, actual, s=12, alpha=0.6, label="Actual quality")
    plt.xlabel("Test samples (sorted by prediction)")
    plt.ylabel("Quality score")
    plt.title("Selected Linear Regression: Prediction with 95% PI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_prediction_interval.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, frame["mean"], alpha=0.5)
    limits = [min(y_test.min(), frame["mean"].min()), max(y_test.max(), frame["mean"].max())]
    plt.plot(limits, limits, "r--", label="Ideal")
    plt.xlabel("Actual quality")
    plt.ylabel("Predicted quality")
    plt.title("Actual vs Predicted (Selected Linear Regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_actual_vs_predicted.png", dpi=200)
    plt.close()

    coverage = np.mean(
        (y_test.to_numpy() >= frame["obs_ci_lower"].to_numpy())
        & (y_test.to_numpy() <= frame["obs_ci_upper"].to_numpy())
    )
    interval_summary = pd.DataFrame(
        {
            "selected_features": [", ".join(selected_names)],
            "prediction_interval_coverage": [coverage],
            "mean_prediction_interval_width": [np.mean(upper - lower)],
        }
    )
    interval_summary.to_csv(
        OUTPUT_DIR / "05_prediction_interval_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nSelected top-5 features:", selected_names)
    print("95% PI coverage:", round(coverage, 4))
    print("Mean PI width:", round(float(np.mean(upper - lower)), 4))
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
