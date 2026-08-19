import os
import json
import random
import time
import argparse
import pandas as pd
import nvforest  
# Same fixed batch-size list as gpu_bench/cpu_bench, for direct comparability.
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 800, 1600, 3200, 6400, 10000]
NUM_TRIALS = 50
SEED = 42

def get_ensemble_size(model_path: str) -> int:
    """Reads the tree count directly from the model JSON
    """
    with open(model_path) as f:
        data = json.load(f)
    return len(data["learner"]["gradient_booster"]["model"]["trees"])


def run_trial(fil_model, batch) -> float:
    """Times a single FIL predict call. batch is already flattened/typed
    before this runs; only the library call itself is timed
    """
    start = time.perf_counter()
    fil_model.predict(batch)
    end = time.perf_counter()
    return (end - start) * 1000  # seconds to ms


def run_batch_trials(fil_model, sample_pool: pd.DataFrame, batch_size: int, num_trials: int, optimised: bool) -> list:
    """Draws a fixed random subset of size batch_size onc which is reused across
    every trial
    """
    random.seed(SEED)
    indices = random.sample(range(len(sample_pool)), batch_size)
    batch = sample_pool.iloc[indices].to_numpy(dtype = "float32")
    if optimised:
        fil_model.optimize(batch_size = batch_size)
    return [run_trial(fil_model, batch) for _ in range(num_trials)]


def write_results(output_path: str, ensemble_size: int, timing_map: dict, optimised: bool):
    """Writes results to output csv
    """
    file_exists = os.path.exists(output_path)
    with open(output_path, "a") as f:
        if not file_exists:
            f.write("ensemble_size,batch_size,trial_num,predict_ms,optimised\n")
        for batch_size, trials in timing_map.items():
            for trial_num, predict_ms in enumerate(trials):
                f.write(f"{ensemble_size},{batch_size},{trial_num},{predict_ms},{optimised}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_directory")
    parser.add_argument("csv_path")
    parser.add_argument("output_path")
    parser.add_argument("--optimised", action="store_true", help="Call FIL's .optimize(batch_size=...) before each batch size's trials") # So we can pass in the optimised flag as --optimised
    args = parser.parse_args()
    sample_pool = pd.read_csv(args.csv_path)
    warmed_up = False
    for filename in os.listdir(args.json_directory):
        model_path = os.path.join(args.json_directory, filename)
        ensemble_size = get_ensemble_size(model_path)
        fil_model = nvforest.load_model(model_path, device="gpu")  
        if not warmed_up:
            # RAPIDS also pays extra overhead on the first call so we need to cold start
            toy_batch = sample_pool.iloc[[0]].to_numpy(dtype = "float32")
            fil_model.predict(toy_batch)
            warmed_up = True

        timing_map = {}
        for batch_size in BATCH_SIZES:
            timing_map[batch_size] = run_batch_trials(fil_model, sample_pool, batch_size, NUM_TRIALS, args.optimised)
        write_results(args.output_path, ensemble_size, timing_map, args.optimised)


if __name__ == "__main__":
    main()
