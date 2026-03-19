from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from pandas_datareader import data as web
except ImportError as exc:
    raise ImportError(
        "pandas_datareader is required to download FRED mortgage rates. "
        "Install dependencies from requirements.txt before running mbs_NN.py."
    ) from exc

try:
    from patsy import build_design_matrices, dmatrix
except ImportError as exc:
    raise ImportError(
        "patsy is required to build spline basis features. "
        "It is usually installed with statsmodels."
    ) from exc


EPSILON = 1e-6
TARGET_SECURITY_TYPES = [
    "15yr TBA Eligible",
    "20yr TBA Eligible",
    "30yr TBA Eligible",
]
def make_spline_formula(lower_bound: float, upper_bound: float) -> str:
    return (
        "bs(spread, df=5, degree=3, include_intercept=False, "
        f"lower_bound={lower_bound:.12g}, upper_bound={upper_bound:.12g}) - 1"
    )


def load_prepay_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="|", low_memory=False)
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    )

    numeric_cols = [
        "Cohort_Current_UPB",
        "Cohort_WA_Current_Interest_Rate",
        "Cohort_WA_Current_Loan_Age",
        "SMM",
        "Cumulative_SMM",
        "CPR",
        "Cumulative_CPR",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")

    df["date_parsed"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df["period"] = df["date_parsed"].dt.to_period("M")
    return df


def download_market_rates(df: pd.DataFrame) -> pd.DataFrame:
    start_year = int(df["date_parsed"].min().year) if df["date_parsed"].notna().any() else 2010
    end_year = int(df["date_parsed"].max().year) if df["date_parsed"].notna().any() else 2025

    mr30 = web.DataReader("MORTGAGE30US", "fred", f"{start_year}-01-01", f"{end_year}-12-31").rename(
        columns={"MORTGAGE30US": "mr30"}
    )
    mr15 = web.DataReader("MORTGAGE15US", "fred", f"{start_year}-01-01", f"{end_year}-12-31").rename(
        columns={"MORTGAGE15US": "mr15"}
    )

    market_monthly = mr30.join(mr15, how="inner").resample("ME").mean()
    market_monthly.index = market_monthly.index.to_period("M")
    market_monthly["mr20"] = market_monthly["mr15"] + (1.0 / 3.0) * (
        market_monthly["mr30"] - market_monthly["mr15"]
    )
    return market_monthly[["mr15", "mr20", "mr30"]]


def assign_market_rate(row: pd.Series) -> float:
    sec = str(row["Type_of_Security"]).lower()
    if "15" in sec:
        return row["mr15"]
    if "20" in sec:
        return row["mr20"]
    return row["mr30"]


def build_observation_panel(df: pd.DataFrame, market_monthly: pd.DataFrame) -> pd.DataFrame:
    panel = df.merge(market_monthly, left_on="period", right_index=True, how="left")
    panel["market_rate"] = panel.apply(assign_market_rate, axis=1)
    panel["loan_age"] = pd.to_numeric(panel["Cohort_WA_Current_Loan_Age"], errors="coerce")
    panel["coupon"] = pd.to_numeric(panel["Cohort_WA_Current_Interest_Rate"], errors="coerce")
    panel["spread"] = panel["coupon"] - pd.to_numeric(panel["market_rate"], errors="coerce")
    panel["log_UPB"] = np.log(pd.to_numeric(panel["Cohort_Current_UPB"], errors="coerce") + 1.0)
    panel["SMM_monthly"] = pd.to_numeric(panel["SMM"], errors="coerce")
    panel["actual_smm"] = panel["SMM_monthly"]
    panel["smm_adj"] = panel["actual_smm"].clip(lower=EPSILON, upper=1.0 - EPSILON)
    panel["logit_smm"] = np.log(panel["smm_adj"] / (1.0 - panel["smm_adj"]))
    panel["actual_cpr_pct"] = 100.0 * (1.0 - (1.0 - panel["actual_smm"]) ** 12.0)

    panel = panel[panel["Type_of_Security"].isin(TARGET_SECURITY_TYPES)].copy()
    panel = panel.dropna(
        subset=[
            "period",
            "Type_of_Security",
            "market_rate",
            "coupon",
            "spread",
            "loan_age",
            "Cohort_Current_UPB",
            "SMM_monthly",
            "log_UPB",
            "logit_smm",
        ]
    ).reset_index(drop=True)
    return panel


def assign_time_splits(
    panel: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float
) -> pd.DataFrame:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    periods = sorted(panel["period"].dropna().unique())
    n_periods = len(periods)
    if n_periods < 3:
        raise ValueError("Need at least 3 unique periods for train/validation/test split.")

    train_end = max(1, int(np.floor(n_periods * train_ratio)))
    val_end = max(train_end + 1, int(np.floor(n_periods * (train_ratio + val_ratio))))
    val_end = min(val_end, n_periods - 1)

    train_periods = set(periods[:train_end])
    val_periods = set(periods[train_end:val_end])
    test_periods = set(periods[val_end:])

    if not val_periods or not test_periods:
        raise ValueError("Time split produced an empty validation or test period block.")

    panel = panel.copy()
    panel["split"] = np.where(
        panel["period"].isin(train_periods),
        "train",
        np.where(panel["period"].isin(val_periods), "validation", "test"),
    )
    return panel


def fit_upb_standardizer(train_df: pd.DataFrame) -> tuple[float, float]:
    mean = float(train_df["log_UPB"].mean())
    std = float(train_df["log_UPB"].std())
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return mean, std


def build_security_feature_frames(
    security_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_df = security_panel[security_panel["split"] == "train"].copy()
    val_df = security_panel[security_panel["split"] == "validation"].copy()
    test_df = security_panel[security_panel["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Each security type needs non-empty train, validation, and test subsets.")

    log_upb_mean, log_upb_std = fit_upb_standardizer(train_df)
    for subset in [train_df, val_df, test_df]:
        subset["log_UPB_z"] = (subset["log_UPB"] - log_upb_mean) / log_upb_std

    spread_lower = float(security_panel["spread"].min())
    spread_upper = float(security_panel["spread"].max())
    if not np.isfinite(spread_lower) or not np.isfinite(spread_upper):
        raise ValueError("Spread bounds must be finite for spline feature construction.")
    if spread_lower >= spread_upper:
        spread_lower -= 1e-6
        spread_upper += 1e-6

    spline_formula = make_spline_formula(spread_lower, spread_upper)
    train_basis = dmatrix(spline_formula, train_df[["spread"]], return_type="dataframe")
    design_info = train_basis.design_info
    val_basis = build_design_matrices([design_info], val_df[["spread"]], return_type="dataframe")[0]
    test_basis = build_design_matrices([design_info], test_df[["spread"]], return_type="dataframe")[0]

    train_features = assemble_feature_frame(train_df, pd.DataFrame(train_basis, index=train_df.index))
    val_features = assemble_feature_frame(val_df, pd.DataFrame(val_basis, index=val_df.index))
    test_features = assemble_feature_frame(test_df, pd.DataFrame(test_basis, index=test_df.index))
    feature_columns = train_features.columns.tolist()
    feature_scaler = StandardScaler()
    train_features = pd.DataFrame(
        feature_scaler.fit_transform(train_features[feature_columns]),
        index=train_features.index,
        columns=feature_columns,
    )
    val_features = pd.DataFrame(
        feature_scaler.transform(val_features[feature_columns]),
        index=val_features.index,
        columns=feature_columns,
    )
    test_features = pd.DataFrame(
        feature_scaler.transform(test_features[feature_columns]),
        index=test_features.index,
        columns=feature_columns,
    )

    metadata = {
        "log_upb_mean": log_upb_mean,
        "log_upb_std": log_upb_std,
        "design_info": design_info,
        "feature_columns": feature_columns,
        "feature_scaler": feature_scaler,
    }
    return train_features, val_features, test_features, metadata


def assemble_feature_frame(base_df: pd.DataFrame, spline_basis: pd.DataFrame) -> pd.DataFrame:
    feature_df = pd.DataFrame(index=base_df.index)
    spline_cols: list[str] = []

    for idx, col in enumerate(spline_basis.columns, start=1):
        out_col = f"spline_{idx}"
        feature_df[out_col] = pd.to_numeric(spline_basis[col], errors="coerce")
        spline_cols.append(out_col)

    feature_df["loan_age"] = pd.to_numeric(base_df["loan_age"], errors="coerce")
    feature_df["log_UPB_z"] = pd.to_numeric(base_df["log_UPB_z"], errors="coerce")

    for col in spline_cols:
        feature_df[f"{col}_x_loan_age"] = feature_df[col] * feature_df["loan_age"]

    return feature_df


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class SmallMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int] = (64, 32, 16),
        dropout_rates: tuple[float, float] = (0.1, 0.1),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)
        self.params = self._init_params()
        self.adam_m = {key: np.zeros_like(val) for key, val in self.params.items()}
        self.adam_v = {key: np.zeros_like(val) for key, val in self.params.items()}
        self.adam_t = 0

    def _init_params(self) -> dict[str, np.ndarray]:
        d0, d1, d2, d3 = self.input_dim, *self.hidden_dims
        return {
            "W1": self.rng.normal(0.0, np.sqrt(2.0 / d0), size=(d0, d1)),
            "b1": np.zeros((1, d1)),
            "W2": self.rng.normal(0.0, np.sqrt(2.0 / d1), size=(d1, d2)),
            "b2": np.zeros((1, d2)),
            "W3": self.rng.normal(0.0, np.sqrt(2.0 / d2), size=(d2, d3)),
            "b3": np.zeros((1, d3)),
            "W4": self.rng.normal(0.0, np.sqrt(2.0 / d3), size=(d3, 1)),
            "b4": np.zeros((1, 1)),
        }

    def get_state(self) -> dict[str, np.ndarray]:
        return {key: val.copy() for key, val in self.params.items()}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.params = {key: val.copy() for key, val in state.items()}

    def _dropout_mask(self, shape: tuple[int, ...], rate: float) -> np.ndarray:
        if rate <= 0:
            return np.ones(shape, dtype=float)
        keep_prob = 1.0 - rate
        mask = (self.rng.random(shape) < keep_prob).astype(float)
        return mask / keep_prob

    def forward(self, X: np.ndarray, training: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cache: dict[str, np.ndarray] = {"X": X}

        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = relu(z1)
        if training:
            mask1 = self._dropout_mask(a1.shape, self.dropout_rates[0])
            a1 = a1 * mask1
            cache["mask1"] = mask1
        cache["z1"] = z1
        cache["a1"] = a1

        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = relu(z2)
        if training:
            mask2 = self._dropout_mask(a2.shape, self.dropout_rates[1])
            a2 = a2 * mask2
            cache["mask2"] = mask2
        cache["z2"] = z2
        cache["a2"] = a2

        z3 = a2 @ self.params["W3"] + self.params["b3"]
        a3 = relu(z3)
        cache["z3"] = z3
        cache["a3"] = a3

        pred = a3 @ self.params["W4"] + self.params["b4"]
        cache["pred"] = pred
        return pred, cache

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred, _ = self.forward(X, training=False)
        return pred.reshape(-1)

    def backward(self, cache: dict[str, np.ndarray], y: np.ndarray, sample_weight: np.ndarray) -> dict[str, np.ndarray]:
        pred = cache["pred"].reshape(-1, 1)
        y = y.reshape(-1, 1)
        weights = sample_weight.reshape(-1, 1)
        denom = float(np.sum(weights))
        if denom <= 0:
            weights = np.ones_like(weights)
            denom = float(len(weights))

        dz4 = 2.0 * weights * (pred - y) / denom

        grads: dict[str, np.ndarray] = {}
        grads["W4"] = cache["a3"].T @ dz4 + self.weight_decay * self.params["W4"]
        grads["b4"] = np.sum(dz4, axis=0, keepdims=True)

        da3 = dz4 @ self.params["W4"].T
        dz3 = da3 * (cache["z3"] > 0)
        grads["W3"] = cache["a2"].T @ dz3 + self.weight_decay * self.params["W3"]
        grads["b3"] = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.params["W3"].T
        if "mask2" in cache:
            da2 = da2 * cache["mask2"]
        dz2 = da2 * (cache["z2"] > 0)
        grads["W2"] = cache["a1"].T @ dz2 + self.weight_decay * self.params["W2"]
        grads["b2"] = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.params["W2"].T
        if "mask1" in cache:
            da1 = da1 * cache["mask1"]
        dz1 = da1 * (cache["z1"] > 0)
        grads["W1"] = cache["X"].T @ dz1 + self.weight_decay * self.params["W1"]
        grads["b1"] = np.sum(dz1, axis=0, keepdims=True)
        return grads

    def update_params(self, grads: dict[str, np.ndarray], beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.adam_t += 1
        for key in self.params:
            self.adam_m[key] = beta1 * self.adam_m[key] + (1.0 - beta1) * grads[key]
            self.adam_v[key] = beta2 * self.adam_v[key] + (1.0 - beta2) * (grads[key] ** 2)

            m_hat = self.adam_m[key] / (1.0 - beta1**self.adam_t)
            v_hat = self.adam_v[key] / (1.0 - beta2**self.adam_t)
            self.params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def compute_rmse(pred: np.ndarray, actual: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    err2 = (pred - actual) ** 2
    if sample_weight is None:
        return float(np.sqrt(np.mean(err2)))
    w = np.asarray(sample_weight, dtype=float)
    w = np.where(w > 0, w, 0.0)
    denom = float(np.sum(w))
    if denom <= 0:
        return float(np.sqrt(np.mean(err2)))
    return float(np.sqrt(np.sum(w * err2) / denom))


def compute_mse(pred: np.ndarray, actual: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    err2 = (pred - actual) ** 2
    if sample_weight is None:
        return float(np.mean(err2))
    w = np.asarray(sample_weight, dtype=float)
    w = np.where(w > 0, w, 0.0)
    denom = float(np.sum(w))
    if denom <= 0:
        return float(np.mean(err2))
    return float(np.sum(w * err2) / denom)


def train_single_security_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    upb_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    upb_val: np.ndarray,
    weighted_loss: bool,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> tuple[SmallMLP, pd.DataFrame]:
    model = SmallMLP(
        input_dim=X_train.shape[1],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    best_state = model.get_state()
    best_val_rmse = np.inf
    best_epoch = -1
    history_rows: list[dict[str, float | int]] = []
    train_weights = upb_train if weighted_loss else np.ones_like(y_train)

    for epoch in range(1, max_epochs + 1):
        order = model.rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            weight_batch = train_weights[batch_idx]
            _, cache = model.forward(X_batch, training=True)
            grads = model.backward(cache, y_batch, weight_batch)
            model.update_params(grads)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_loss = compute_mse(train_pred, y_train, train_weights if weighted_loss else None)
        val_loss = compute_mse(val_pred, y_val, upb_val if weighted_loss else None)
        train_rmse = compute_rmse(train_pred, y_train)
        val_rmse = compute_rmse(val_pred, y_val)

        if val_rmse < best_val_rmse - 1e-8:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = model.get_state()

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "best_val_rmse_so_far": best_val_rmse,
                "best_epoch_so_far": best_epoch,
            }
        )

        if epoch - best_epoch >= patience:
            break

    model.set_state(best_state)
    return model, pd.DataFrame(history_rows)


def get_spread_grid(subset: pd.DataFrame, q_low: float, q_high: float, num_points: int = 80) -> np.ndarray:
    lo = float(subset["spread"].quantile(q_low))
    hi = float(subset["spread"].quantile(q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(subset["spread"].quantile(0.05))
        hi = float(subset["spread"].quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(subset["spread"].min())
        hi = float(subset["spread"].max())
    return np.linspace(lo, hi, num_points)


def build_counterfactual_features(
    subset: pd.DataFrame,
    spread_value: float,
    design_info,
    log_upb_mean: float,
    log_upb_std: float,
    feature_columns: list[str],
    feature_scaler: StandardScaler,
) -> pd.DataFrame:
    cf = subset.copy()
    cf["spread"] = float(spread_value)
    cf["log_UPB_z"] = (cf["log_UPB"] - log_upb_mean) / log_upb_std
    spline_basis = build_design_matrices([design_info], cf[["spread"]], return_type="dataframe")[0]
    raw_features = assemble_feature_frame(cf, pd.DataFrame(spline_basis, index=cf.index))
    raw_features = raw_features[feature_columns]
    scaled_features = pd.DataFrame(
        feature_scaler.transform(raw_features),
        index=raw_features.index,
        columns=feature_columns,
    )
    return scaled_features


def slugify_security_type(security_type: str) -> str:
    return security_type.lower().replace(" ", "_")


def fit_security_models(
    panel: pd.DataFrame,
    weighted_loss: bool,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, dict[str, object]], dict[str, pd.DataFrame]]:
    fitted_models: dict[str, dict[str, object]] = {}
    history_map: dict[str, pd.DataFrame] = {}

    for sec_idx, sec_type in enumerate(TARGET_SECURITY_TYPES):
        security_panel = panel[panel["Type_of_Security"] == sec_type].copy()
        if security_panel.empty:
            continue

        train_features, val_features, test_features, metadata = build_security_feature_frames(security_panel)
        feature_cols = metadata["feature_columns"]

        train_df = security_panel[security_panel["split"] == "train"].copy()
        val_df = security_panel[security_panel["split"] == "validation"].copy()
        test_df = security_panel[security_panel["split"] == "test"].copy()
        train_df["log_UPB_z"] = ((train_df["log_UPB"] - metadata["log_upb_mean"]) / metadata["log_upb_std"]).to_numpy()
        val_df["log_UPB_z"] = ((val_df["log_UPB"] - metadata["log_upb_mean"]) / metadata["log_upb_std"]).to_numpy()
        test_df["log_UPB_z"] = ((test_df["log_UPB"] - metadata["log_upb_mean"]) / metadata["log_upb_std"]).to_numpy()

        X_train = train_features[feature_cols].to_numpy(dtype=float)
        X_val = val_features[feature_cols].to_numpy(dtype=float)
        X_test = test_features[feature_cols].to_numpy(dtype=float)
        y_train = train_df["logit_smm"].to_numpy(dtype=float)
        y_val = val_df["logit_smm"].to_numpy(dtype=float)
        upb_train = train_df["Cohort_Current_UPB"].to_numpy(dtype=float)
        upb_val = val_df["Cohort_Current_UPB"].to_numpy(dtype=float)

        if len(train_df) < 50:
            continue

        print(
            f"[NN][{sec_type}] train actual_smm mean/std/min/max="
            f"{train_df['actual_smm'].mean():.6f}/{train_df['actual_smm'].std():.6f}/"
            f"{train_df['actual_smm'].min():.6f}/{train_df['actual_smm'].max():.6f}"
        )
        print(
            f"[NN][{sec_type}] train logit_smm mean/std/min/max="
            f"{train_df['logit_smm'].mean():.6f}/{train_df['logit_smm'].std():.6f}/"
            f"{train_df['logit_smm'].min():.6f}/{train_df['logit_smm'].max():.6f}"
        )
        feature_summary = pd.DataFrame(
            {
                "feature": feature_cols,
                "mean": train_features[feature_cols].mean().to_numpy(),
                "std": train_features[feature_cols].std(ddof=0).to_numpy(),
            }
        )
        print(f"[NN][{sec_type}] scaled X_train feature summary:")
        print(feature_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        model, history_df = train_single_security_model(
            X_train=X_train,
            y_train=y_train,
            upb_train=upb_train,
            X_val=X_val,
            y_val=y_val,
            upb_val=upb_val,
            weighted_loss=weighted_loss,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            seed=seed + sec_idx,
        )

        fitted_models[sec_type] = {
            "model": model,
            "feature_columns": feature_cols,
            "design_info": metadata["design_info"],
            "log_upb_mean": metadata["log_upb_mean"],
            "log_upb_std": metadata["log_upb_std"],
            "feature_scaler": metadata["feature_scaler"],
            "train_features": train_features,
            "val_features": val_features,
            "test_features": test_features,
            "train_panel": train_df,
            "val_panel": val_df,
            "test_panel": test_df,
            "X_test": X_test,
        }
        history_map[sec_type] = history_df

    return fitted_models, history_map


def build_prediction_panel(fitted_models: dict[str, dict[str, object]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sec_type, artifacts in fitted_models.items():
        model = artifacts["model"]
        feature_cols = artifacts["feature_columns"]

        for split_name, panel_key, feature_key in [
            ("train", "train_panel", "train_features"),
            ("validation", "val_panel", "val_features"),
            ("test", "test_panel", "test_features"),
        ]:
            subset = artifacts[panel_key].copy()
            feature_df = artifacts[feature_key]
            if subset.empty:
                continue
            pred_logit_smm = model.predict(feature_df[feature_cols].to_numpy(dtype=float))
            subset["pred_logit_smm"] = pred_logit_smm
            pred_smm = sigmoid(pred_logit_smm)
            subset["pred_smm"] = np.clip(pred_smm, EPSILON, 1.0 - EPSILON)
            subset["pred_cpr_pct"] = 100.0 * (1.0 - (1.0 - subset["pred_smm"]) ** 12.0)
            subset["actual_cpr_pct"] = 100.0 * (1.0 - (1.0 - subset["actual_smm"]) ** 12.0)
            subset["Type_of_Security"] = sec_type
            subset["split"] = split_name
            print(
                f"[NN][{sec_type}][{split_name}] pred_logit_smm mean/std/min/max="
                f"{subset['pred_logit_smm'].mean():.6f}/{subset['pred_logit_smm'].std():.6f}/"
                f"{subset['pred_logit_smm'].min():.6f}/{subset['pred_logit_smm'].max():.6f}"
            )
            print(
                f"[NN][{sec_type}][{split_name}] pred_smm mean/std/min/max="
                f"{subset['pred_smm'].mean():.6f}/{subset['pred_smm'].std():.6f}/"
                f"{subset['pred_smm'].min():.6f}/{subset['pred_smm'].max():.6f}"
            )
            frames.append(subset)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_oos_predictions(prediction_panel: pd.DataFrame) -> pd.DataFrame:
    oos = prediction_panel[prediction_panel["split"].isin(["validation", "test"])].copy()
    oos["period"] = oos["period"].astype(str)
    return oos[
        [
            "Type_of_Security",
            "period",
            "spread",
            "loan_age",
            "log_UPB_z",
            "actual_smm",
            "pred_logit_smm",
            "pred_smm",
            "actual_cpr_pct",
            "pred_cpr_pct",
            "split",
            "Cohort_Current_UPB",
        ]
    ]


def build_oos_metrics(oos_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    def metric_row(df: pd.DataFrame, level: str, security_type: str) -> dict[str, float | int | str]:
        err = df["pred_smm"] - df["actual_smm"]
        w = pd.to_numeric(df["Cohort_Current_UPB"], errors="coerce").fillna(0.0)
        w = np.where(w > 0, w, 0.0)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones(len(df), dtype=float)
            w_sum = float(np.sum(w))
        return {
            "level": level,
            "Type_of_Security": security_type,
            "nobs": int(len(df)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "upb_weighted_rmse": float(np.sqrt(np.sum(w * (err**2)) / w_sum)),
            "upb_weighted_mae": float(np.sum(w * np.abs(err)) / w_sum),
        }

    if not oos_predictions.empty:
        rows.append(metric_row(oos_predictions, "overall", "ALL"))
    for sec_type in TARGET_SECURITY_TYPES:
        subset = oos_predictions[oos_predictions["Type_of_Security"] == sec_type]
        if subset.empty:
            continue
        rows.append(metric_row(subset, "security_type", sec_type))
    return pd.DataFrame(rows)


def build_calibration_dataset(prediction_panel: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for sec_type in TARGET_SECURITY_TYPES:
        subset = prediction_panel[
            (prediction_panel["Type_of_Security"] == sec_type) & (prediction_panel["split"] == "test")
        ].copy()
        if subset.empty:
            continue

        n_bins_eff = min(n_bins, max(2, subset["spread"].nunique()))
        subset["spread_bin"] = pd.qcut(subset["spread"], q=n_bins_eff, duplicates="drop")
        grouped = (
            subset.groupby("spread_bin", observed=False)
            .agg(
                mean_spread=("spread", "mean"),
                actual_cpr_mean=("actual_cpr_pct", "mean"),
                predicted_cpr_mean=("pred_cpr_pct", "mean"),
                actual_cpr_std=("actual_cpr_pct", "std"),
                total_upb=("Cohort_Current_UPB", "sum"),
                n_obs=("spread", "size"),
            )
            .dropna(subset=["mean_spread", "actual_cpr_mean", "predicted_cpr_mean"])
            .reset_index()
            .sort_values("mean_spread")
        )

        grouped["actual_cpr_se"] = grouped["actual_cpr_std"] / np.sqrt(grouped["n_obs"].clip(lower=1))
        grouped["actual_cpr_ci95"] = 1.96 * grouped["actual_cpr_se"].fillna(0.0)
        grouped["bin_sq_error"] = (grouped["actual_cpr_mean"] - grouped["predicted_cpr_mean"]) ** 2
        upb_sum = grouped["total_upb"].sum()
        grouped["upb_weighted_bin_sq_error"] = np.where(
            upb_sum > 0,
            grouped["bin_sq_error"] * grouped["total_upb"] / upb_sum,
            np.nan,
        )

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "Type_of_Security": sec_type,
                    "spread_bin": str(row["spread_bin"]),
                    "mean_spread": float(row["mean_spread"]),
                    "actual_cpr_mean": float(row["actual_cpr_mean"]),
                    "predicted_cpr_mean": float(row["predicted_cpr_mean"]),
                    "actual_cpr_std": float(row["actual_cpr_std"]) if pd.notna(row["actual_cpr_std"]) else np.nan,
                    "actual_cpr_se": float(row["actual_cpr_se"]) if pd.notna(row["actual_cpr_se"]) else np.nan,
                    "actual_cpr_ci95": float(row["actual_cpr_ci95"]),
                    "n_obs": int(row["n_obs"]),
                    "bin_sq_error": float(row["bin_sq_error"]),
                    "upb_weighted_bin_sq_error": (
                        float(row["upb_weighted_bin_sq_error"])
                        if pd.notna(row["upb_weighted_bin_sq_error"])
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_theoretical_curves(
    fitted_models: dict[str, dict[str, object]],
    spread_q_low: float,
    spread_q_high: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for sec_type, artifacts in fitted_models.items():
        test_panel = artifacts["test_panel"].copy()
        if test_panel.empty:
            continue

        spread_grid = get_spread_grid(test_panel, spread_q_low, spread_q_high)
        model = artifacts["model"]
        rows: list[dict[str, float | int | str]] = []

        for spread_value in spread_grid:
            feature_df = build_counterfactual_features(
                subset=test_panel,
                spread_value=float(spread_value),
                design_info=artifacts["design_info"],
                log_upb_mean=artifacts["log_upb_mean"],
                log_upb_std=artifacts["log_upb_std"],
                feature_columns=artifacts["feature_columns"],
                feature_scaler=artifacts["feature_scaler"],
            )
            pred_logit_smm = model.predict(feature_df[artifacts["feature_columns"]].to_numpy(dtype=float))
            pred_smm = np.clip(sigmoid(pred_logit_smm), EPSILON, 1.0 - EPSILON)
            pred_cpr_pct = 100.0 * (1.0 - (1.0 - pred_smm) ** 12.0)
            rows.append(
                {
                    "Type_of_Security": sec_type,
                    "spread": float(spread_value),
                    "predicted_cpr_mean": float(np.mean(pred_cpr_pct)),
                    "predicted_cpr_p25": float(np.percentile(pred_cpr_pct, 25)),
                    "predicted_cpr_p75": float(np.percentile(pred_cpr_pct, 75)),
                    "n_obs": int(len(test_panel)),
                }
            )

        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_slope_diagnostics(theoretical_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if theoretical_df.empty:
        return pd.DataFrame()

    for sec_type in theoretical_df["Type_of_Security"].drop_duplicates():
        sec_df = theoretical_df[theoretical_df["Type_of_Security"] == sec_type].sort_values("spread").copy()
        slopes = np.gradient(sec_df["predicted_cpr_mean"].to_numpy(), sec_df["spread"].to_numpy())
        sec_df["dCPR_dspread"] = slopes
        rows.extend(sec_df[["Type_of_Security", "spread", "predicted_cpr_mean", "dCPR_dspread"]].to_dict("records"))
    return pd.DataFrame(rows)


def plot_calibration_curves(calibration_df: pd.DataFrame, output_root: Path, show_plots: bool) -> None:
    sec_types = calibration_df["Type_of_Security"].drop_duplicates().tolist()
    if not sec_types:
        return

    fig, axes = plt.subplots(nrows=len(sec_types), ncols=1, figsize=(12, 5 * max(1, len(sec_types))))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, sec_type in zip(axes, sec_types):
        sec_df = calibration_df[calibration_df["Type_of_Security"] == sec_type].sort_values("mean_spread")
        ax.errorbar(
            sec_df["mean_spread"],
            sec_df["actual_cpr_mean"],
            yerr=sec_df["actual_cpr_ci95"],
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.2,
            capsize=3,
            label="Actual bin mean",
        )
        ax.plot(
            sec_df["mean_spread"],
            sec_df["predicted_cpr_mean"],
            color="#d62728",
            linewidth=2.2,
            marker="o",
            label="Predicted bin mean",
        )
        ax.set_title(sec_type)
        ax.set_xlabel("Spread (WAC - Market Rate) in %")
        ax.set_ylabel("CPR (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    fig.suptitle("NN Calibration Curves by Security Type", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_root / "calibration_curves.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_theoretical_curves(theoretical_df: pd.DataFrame, output_root: Path, show_plots: bool) -> None:
    if theoretical_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for sec_type in theoretical_df["Type_of_Security"].drop_duplicates():
        sec_df = theoretical_df[theoretical_df["Type_of_Security"] == sec_type].sort_values("spread")
        ax.plot(sec_df["spread"], sec_df["predicted_cpr_mean"], linewidth=2.5, label=sec_type)
        ax.fill_between(sec_df["spread"], sec_df["predicted_cpr_p25"], sec_df["predicted_cpr_p75"], alpha=0.12)

    ax.set_title("NN Theoretical Counterfactual S-Curves")
    ax.set_xlabel("Spread (WAC - Market Rate) in %")
    ax.set_ylabel("Average Predicted CPR (%)")
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_root / "theoretical_scurve.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_slope_diagnostics(slope_df: pd.DataFrame, output_root: Path, show_plots: bool) -> None:
    if slope_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for sec_type in slope_df["Type_of_Security"].drop_duplicates():
        sec_df = slope_df[slope_df["Type_of_Security"] == sec_type].sort_values("spread")
        ax.plot(sec_df["spread"], sec_df["dCPR_dspread"], linewidth=2.0, label=sec_type)

    ax.set_title("NN Theoretical S-Curve Slope Diagnostics")
    ax.set_xlabel("Spread (WAC - Market Rate) in %")
    ax.set_ylabel("dCPR / dspread")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_root / "theoretical_slope_diagnostics.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def save_model_weights(output_root: Path, fitted_models: dict[str, dict[str, object]]) -> None:
    for sec_type, artifacts in fitted_models.items():
        model = artifacts["model"]
        state = model.get_state()
        lines = [
            f"Security Type: {sec_type}",
            f"Feature Count: {len(artifacts['feature_columns'])}",
            "Feature Columns:",
            ", ".join(artifacts["feature_columns"]),
            f"log_upb_mean: {artifacts['log_upb_mean']:.12g}",
            f"log_upb_std: {artifacts['log_upb_std']:.12g}",
            "",
            "Parameter Summary:",
        ]
        for key, value in state.items():
            arr = np.asarray(value, dtype=float)
            lines.append(
                f"{key}: shape={arr.shape}, mean={arr.mean():.6g}, std={arr.std():.6g}, "
                f"min={arr.min():.6g}, max={arr.max():.6g}"
            )
        (output_root / f"model_weights_{slugify_security_type(sec_type)}.txt").write_text("\n".join(lines) + "\n")


def save_training_histories(output_root: Path, history_map: dict[str, pd.DataFrame]) -> None:
    for sec_type, history_df in history_map.items():
        history_df.to_csv(output_root / f"training_history_{slugify_security_type(sec_type)}.csv", index=False)


def save_outputs(
    output_root: Path,
    fitted_models: dict[str, dict[str, object]],
    history_map: dict[str, pd.DataFrame],
    oos_predictions: pd.DataFrame,
    oos_metrics: pd.DataFrame,
    calibration_df: pd.DataFrame,
    theoretical_df: pd.DataFrame,
    slope_df: pd.DataFrame,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    save_training_histories(output_root, history_map)
    save_model_weights(output_root, fitted_models)
    oos_predictions.drop(columns=["Cohort_Current_UPB"], errors="ignore").to_csv(
        output_root / "oos_predictions.csv", index=False
    )
    oos_metrics.to_csv(output_root / "oos_metrics.csv", index=False)
    calibration_df.to_csv(output_root / "calibration_curve_bins.csv", index=False)
    theoretical_df.to_csv(output_root / "theoretical_scurve.csv", index=False)
    slope_df.to_csv(output_root / "theoretical_slope_diagnostics.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MBS neural-network pipeline aligned with the OLS benchmark feature set and OOS framework."
    )
    parser.add_argument("--data-path", type=str, default="PrepayData.txt")
    parser.add_argument("--output-dir", type=str, default="outputs/nn")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--spread-q-low", type=float, default=0.01)
    parser.add_argument("--spread-q-high", type=float, default=0.99)
    parser.add_argument("--show-plots", action="store_true", default=True)
    parser.add_argument("--no-show-plots", action="store_false", dest="show_plots")
    parser.add_argument("--weighted-loss", action="store_true", dest="weighted_loss")
    parser.add_argument("--unweighted-loss", action="store_false", dest="weighted_loss")
    parser.set_defaults(weighted_loss=False)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_dir)

    df = load_prepay_data(Path(args.data_path))
    market_monthly = download_market_rates(df)
    panel = build_observation_panel(df, market_monthly)
    panel = assign_time_splits(panel, args.train_ratio, args.val_ratio, args.test_ratio)

    fitted_models, history_map = fit_security_models(
        panel=panel,
        weighted_loss=args.weighted_loss,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    prediction_panel = build_prediction_panel(fitted_models)
    oos_predictions = build_oos_predictions(prediction_panel)
    oos_metrics = build_oos_metrics(oos_predictions)
    calibration_df = build_calibration_dataset(prediction_panel, n_bins=args.n_bins)
    theoretical_df = build_theoretical_curves(
        fitted_models=fitted_models,
        spread_q_low=args.spread_q_low,
        spread_q_high=args.spread_q_high,
    )
    slope_df = build_slope_diagnostics(theoretical_df)

    save_outputs(
        output_root=output_root,
        fitted_models=fitted_models,
        history_map=history_map,
        oos_predictions=oos_predictions,
        oos_metrics=oos_metrics,
        calibration_df=calibration_df,
        theoretical_df=theoretical_df,
        slope_df=slope_df,
    )
    plot_calibration_curves(calibration_df, output_root, args.show_plots)
    plot_theoretical_curves(theoretical_df, output_root, args.show_plots)
    plot_slope_diagnostics(slope_df, output_root, args.show_plots)


if __name__ == "__main__":
    main()
