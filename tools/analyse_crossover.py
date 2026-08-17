import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    """Loads a benchmark results CSV and cleans up its column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"h2d (ms)": "h2d_ms", "kernel (ms)": "kernel_ms",  "d2h (ms)": "d2h_ms", "predict (ms)": "predict_ms",})
    return df


def aggregate_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates GPU results to one median total latency per (ensemble_size, batch_size).

    Sums h2d/kernel/d2h per first, then takes the median of those sums. A per-trial sum reflects
    what a single real request actually experienced while summing separately-computed
    medians would construct a total latency that never occurred on any real trial,
    since an unusually slow h2d on one trial could coincide with a fast kernel/d2h
    on that same trial, and vice versa."""
    df = df.copy()
    df["gpu_total_ms"] = df["h2d_ms"] + df["kernel_ms"] + df["d2h_ms"]
    return df.groupby(["ensemble_size", "batch_size"])["gpu_total_ms"].median().reset_index()


def aggregate_cpu(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates CPU results to one median latency per (ensemble_size, batch_size)"""
    return df.groupby(["ensemble_size", "batch_size"])["predict_ms"].median().reset_index()


def find_crossover(gpu_agg: pd.DataFrame, cpu_agg: pd.DataFrame) -> dict:
    """For each ensemble_size, walks batch sizes in ascending order and returns
    the first batch size where gpu_total_ms < predict_ms.
    If GPU never beats CPU within the tested batch range, that ensemble_size
    maps to None.
    """
    merged = pd.merge(gpu_agg, cpu_agg, on=["ensemble_size", "batch_size"])
    crossovers = {}
    for ensemble_size, group in merged.groupby("ensemble_size"):
        group = group.sort_values("batch_size")
        found = group[group["gpu_total_ms"] < group["predict_ms"]]
        crossovers[ensemble_size] = found["batch_size"].iloc[0] if not found.empty else None
    return crossovers


def main():
    gpu_df = load_results("ground_truth/gpu_bench_results.csv")
    cpu_df = load_results("ground_truth/cpu_bench_results.csv")

    gpu_agg = aggregate_gpu(gpu_df)
    cpu_agg = aggregate_cpu(cpu_df)

    crossovers = find_crossover(gpu_agg, cpu_agg)
    for ensemble_size, batch_size in sorted(crossovers.items()):
        if batch_size is None:
            print(f"ensemble_size={ensemble_size}: no crossover found in tested range")
        else:
            print(f"ensemble_size={ensemble_size}: crossover at batch_size={batch_size}")


if __name__ == "__main__":
    main()
