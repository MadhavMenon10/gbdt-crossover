import numpy as np
import pandas as pd
from train_model import simulate_gbm_path, compute_features

WINDOWS = [5, 10, 20]  # must match compute_features' own windows

def generate_sample_pool(target_pool_size: int, drift: float, volatility: float, s0: float) -> pd.DataFrame:
    """_summary_

    Args:
        target_pool_size (int): The number of samples we want in our sample pool
        drift (float): The drift of our generated samples
        volatility (float): The volatility of our generated samples
        s0 (float): Starting price

    Returns:
        pd.DataFrame: Data frame of sample pool
    """
    # generous headroom beyond target_pool_size, since option B samples
    # target_pool_size rows out of a larger trimmed set — needs real
    # surplus to draw from, not just exactly enough to survive the trim
    num_steps = target_pool_size * 3 + max(WINDOWS)

    prices = simulate_gbm_path(num_steps=num_steps, drift=drift, volatility=volatility, s0=s0)
    features = compute_features(prices)

    # leading-edge trim only i.e. no label_regimes call so no trailing trim
    trimmed = features.iloc[max(WINDOWS):]

    # random subsample so we break serial correlation between adjacent rows
    pool = trimmed.sample(n=target_pool_size, random_state=42)
    return pool

def main():
    pool = generate_sample_pool(target_pool_size=750, drift=0.0002, volatility=0.02, s0=100.0)
    pool.to_csv("ground_truth/bench_samples.csv", index=False)

if __name__ == "__main__":
    main()
