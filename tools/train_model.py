import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import os
from sklearn.model_selection import train_test_split


def simulate_gbm_path(num_steps: int, drift: float, volatility: float, s0: float, dt: float = 1.0) -> np.ndarray:
    """Simulates one geometric Brownian motion price path. Uses the standard GBM discretization

    Args:
        num_steps (int): Total length of the returned price path
        drift (float): Drift term in the GBM SDE (mu)
        volatility (float): Volatility term in the GBM SDE (sigma)
        s0 (float): Starting price at t = 0
        dt (float, optional): Time step size. Defaults to 1.0.

    Returns:
        np.ndarray: The simulated price path
    """
    z = np.random.normal(size=num_steps - 1) 
    log_returns = (drift - 0.5 * volatility ** 2) * \
        dt + volatility * np.sqrt(dt) * z
    log_path = np.cumsum(log_returns)
    prices = s0 * np.exp(log_path)
    return np.concatenate(([s0], prices))


def compute_features(prices: np.ndarray) -> pd.DataFrame:
    """Derive rolling return, realized volatility, and momentum features
    at windows [5, 10, 20] from a price path. All three are backward-looking, so the 
    first max(windows) rows are NaN (insufficient history). Momentum specifically has NaN in its
    first w rows (via shift(w)) versus w-1 for the rolling stats, so
    callers should drop at least max(windows) rows before training.

    Args:
        prices (np.ndarray): 1D array of prices

    Returns:
        pd.DataFrame: Data frame with rolling returns, realised volatility and momentum
    """
    price_series = pd.Series(prices)
    log_ret = np.log(price_series / price_series.shift(1))
    windows = [5, 10, 20]
    features = {}
    for w in windows:
        features[f"roll_return_{w}"] = log_ret.rolling(w).mean()
        features[f"realized_vol_{w}"] = log_ret.rolling(w).std()
        features[f"momentum_{w}"] = price_series / price_series.shift(w) - 1.0
    features = pd.DataFrame(features)
    features.index.name = "t"
    return features


def label_regimes(prices: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    """Label each point as trending-up (2), sideways (1), or
    trending-down (0) based on the forward return over `horizon` steps. This is forward-looking, 
    so the last `horizon` rows are NaN (no future price to compare against) so callers should drop 
    these before training, same as compute_features' leading NaNs.

    Args:
        prices (np.ndarray): 1D array of prices
        horizon (int): Number of steps forward to measure the return over
        threshold (float): Cutoff on the forward return; above -> trending-up, below its negation -> trending-down, otherwise sideways

    Returns:
        np.ndarray: Labels (0, 1, 2, or NaN) for the last horizon ows, same length as prices.
    """
    price_series = pd.Series(prices)
    forward_return = price_series.shift(-horizon) / price_series - 1.0
    labels = pd.Series(np.nan, index=price_series.index)
    labels[forward_return > threshold] = 2
    labels[forward_return < -threshold] = 0
    labels[(forward_return >= -threshold) & (forward_return <= threshold)] = 1
    return labels.to_numpy()


def inject_missing_values(test_df: pd.DataFrame, num_rows: int, num_features_each: int) -> pd.DataFrame:
    """"Return a copy of test_df with NaN injected into a random subset
    of rows and features, to exercise the missing-value (default_left)
    traversal path in TreeInfer. Only ever applied to test
    data, never training data, so the model itself is trained dense.

    Args:
        test_df (pd.DataFrame): Test-set feature DataFrame to inject NaNs into
        num_rows (int): Number of rows to modify (clamped to len(test_df))
        num_features_each (int): Number of feature columns to null out per selected row

    Returns:
        pd.DataFrame: A copy of test_df with NaN values injected
    """
    out = test_df.copy()
    num_rows = min(num_rows, len(out))
    num_features_each = min(num_features_each, out.shape[1])
    row_idx = np.random.choice(out.index, size=num_rows, replace=False)
    for r in row_idx:
        cols = np.random.choice(
            out.columns, size=num_features_each, replace=False)
        out.loc[r, cols] = np.nan
    return out

def generate_sweep_models(clf, round_counts, output_dir):
    """Slices the trained booster into ten smaller ensembles by boosting round, and saves each as its own model file.
    Our generated data has 3 classes and we get one tree per class so the round counts below produce tree counts roughly from 100 to 1000 in increments of 100
    Args:
        clf (XGBClassifier): scikit-learn wrapper
        round_counts (list[int]): List of boosting-round counts to slice at [33, 67, 100, 133, 167, 200, 233, 267, 300, 333]
        output_dir (str): The directory path sliced model files get written to
    """
    os.makedirs(output_dir, exist_ok=True)
    booster = clf.get_booster()
    for count in round_counts:
        sliced = booster[0:count] 
        sliced.save_model(f"{output_dir}/model_{count}rounds.json")
        

def main():
    os.makedirs("ground_truth", exist_ok=True)
    num_steps = 3000
    horizon = 5
    threshold = 0.01
    prices = simulate_gbm_path(num_steps=num_steps, drift=0.0002, volatility=0.02, s0=100.0)
    features = compute_features(prices)
    labels = label_regimes(prices, horizon=horizon, threshold=threshold)
    features["label"] = labels
    aligned = features.dropna()
    y = aligned.pop("label").astype(int)
    x = aligned
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    clf = XGBClassifier(objective="multi:softmax", num_class=3, n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="mlogloss")
    clf.fit(x_train, y_train)
    booster = clf.get_booster()
    booster.save_model("ground_truth/model.json")
    x_test_missing = inject_missing_values(x_test, num_rows=20, num_features_each=2)
    margins = clf.predict(x_test_missing, output_margin=True)
    out = x_test_missing.copy()
    for c in range(margins.shape[1]):
        out[f"margin_{c}"] = margins[:, c]
    out.to_csv("ground_truth/test_cases.csv", index=False)


if __name__ == "__main__":
    main()
