import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    """Loads a benchmark results CSV and cleans up its column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"h2d (ms)": "h2d_ms", "kernel (ms)": "kernel_ms",
                   "d2h (ms)": "d2h_ms", "predict (ms)": "predict_ms", })
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


def aggregate_fil(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates FIL results, split by optimised mode.
    """
    return df.groupby(["ensemble_size", "batch_size", "optimised"])["predict_ms"].median().reset_index().rename(columns={"predict_ms": "fil_ms"})


def find_crossover(baseline_agg: pd.DataFrame, baseline_col: str, comparison_agg: pd.DataFrame, comparison_col: str) -> dict:
    """Generalized from the original two-argument version. Now takes any two
    aggregated frames and their value-column names, so this same function
    serves GPU-vs-CPU and GPU-vs-FIL without duplicating the logic.
    Returns the first batch_size, per ensemble_size, where comparison_col
    beats baseline_col.
    """
    merged = pd.merge(baseline_agg, comparison_agg, on=[
                      "ensemble_size", "batch_size"])
    crossovers = {}
    for ensemble_size, group in merged.groupby("ensemble_size"):
        group = group.sort_values("batch_size")
        found = group[group[comparison_col] < group[baseline_col]]
        crossovers[ensemble_size] = found["batch_size"].iloc[0] if not found.empty else None
    return crossovers


def aggregate_gpu_p99(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates GPU Bench results to one p99 total latency per (ensemble_size, batch_size).
    """
    df = df.copy()
    df["gpu_total_ms"] = df["h2d_ms"] + df["kernel_ms"] + df["d2h_ms"]
    return df.groupby(["ensemble_size", "batch_size"])["gpu_total_ms"].quantile(0.99).reset_index()


def aggregate_cpu_p99(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates CPU Bench results to one p99 total latency per (ensemble_size, batch_size).
    """
    """Aggregates CPU results to one p99 latency per (ensemble_size, batch_size)."""
    return df.groupby(["ensemble_size", "batch_size"])["predict_ms"].quantile(0.99).reset_index()


def aggregate_fil_p99(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates FIL Bench results to one p99 total latency per (ensemble_size, batch_size).
    """
    return df.groupby(["ensemble_size", "batch_size", "optimised"])["predict_ms"].quantile(0.99).reset_index().rename(columns={"predict_ms": "fil_ms"})


def main():
    gpu_df = load_results("ground_truth/gpu_bench_results.csv")
    cpu_df = load_results("ground_truth/cpu_bench_results.csv")
    fil_df = load_results("ground_truth/fil_bench_results.csv")
    gpu_agg = aggregate_gpu(gpu_df)
    cpu_agg = aggregate_cpu(cpu_df)
    fil_agg = aggregate_fil(fil_df)

    gpu_vs_cpu = find_crossover(cpu_agg, "predict_ms", gpu_agg, "gpu_total_ms")
    fil_unopt = fil_agg[fil_agg["optimised"]
                        == False].drop(columns="optimised")
    gpu_vs_fil = find_crossover(fil_unopt, "fil_ms", gpu_agg, "gpu_total_ms")
    print("=== GPU vs CPU ===")
    for ensemble_size, batch_size in sorted(gpu_vs_cpu.items()):
        print(
            f"ensemble_size={ensemble_size}: {'crossover at batch_size=' + str(batch_size) if batch_size else 'no crossover found'}")
    print("=== GPU vs FIL ===")
    for ensemble_size, batch_size in sorted(gpu_vs_fil.items()):
        print(
            f"ensemble_size={ensemble_size}: {'crossover at batch_size=' + str(batch_size) if batch_size else 'no crossover found'}")
    gpu_loses_to_fil = find_crossover(
        gpu_agg, "gpu_total_ms", fil_unopt, "fil_ms")

    print("=== Where GPU stops winning (FIL becomes faster) ===")
    for ensemble_size, batch_size in sorted(gpu_loses_to_fil.items()):
        print(
            f"ensemble_size={ensemble_size}: {'FIL overtakes at batch_size=' + str(batch_size) if batch_size else 'GPU wins across entire tested range'}")
    gpu_agg_p99 = aggregate_gpu_p99(gpu_df)
    cpu_agg_p99 = aggregate_cpu_p99(cpu_df)
    fil_agg_p99 = aggregate_fil_p99(fil_df)
    fil_unopt_p99 = fil_agg_p99[fil_agg_p99["optimised"] == False].drop(columns="optimised")

    gpu_vs_cpu_p99 = find_crossover(cpu_agg_p99, "predict_ms", gpu_agg_p99, "gpu_total_ms")
    gpu_vs_fil_p99 = find_crossover(fil_unopt_p99, "fil_ms", gpu_agg_p99, "gpu_total_ms")
    gpu_loses_to_fil_p99 = find_crossover(gpu_agg_p99, "gpu_total_ms", fil_unopt_p99, "fil_ms")

    print("=== [p99] GPU vs CPU ===")
    for ensemble_size, batch_size in sorted(gpu_vs_cpu_p99.items()):
        print(f"ensemble_size={ensemble_size}: {'crossover at batch_size=' + str(batch_size) if batch_size else 'no crossover found'}")

    print("=== [p99] GPU vs FIL ===")
    for ensemble_size, batch_size in sorted(gpu_vs_fil_p99.items()):
        print(f"ensemble_size={ensemble_size}: {'crossover at batch_size=' + str(batch_size) if batch_size else 'no crossover found'}")

    print("=== [p99] Where GPU stops winning (FIL becomes faster) ===")
    for ensemble_size, batch_size in sorted(gpu_loses_to_fil_p99.items()):
        print(f"ensemble_size={ensemble_size}: {'FIL overtakes at batch_size=' + str(batch_size) if batch_size else 'GPU wins across entire tested range'}")

if __name__ == "__main__":
    main()
