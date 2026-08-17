#include "xgb_booster.hpp"
#include "rapidcsv.h"
#include "sample.hpp"
#include "model.hpp"
#include "model_loader.hpp"
#include <iostream>
#include "c_api.h" // Includes the XGBoost C API
#include <chrono>
#include <vector>
#include <stdexcept>
#include <random>
#include <numeric>
#include <algorithm>
#include <map>
#include <filesystem>
#include <fstream>

namespace {
    float run_trial_host(const XGBBooster& xgb_booster, const std::vector<float>& feature_values, size_t num_rows, size_t num_cols) {
        DMatrixHandle dmatrix_handle;
        bst_ulong const* out_shape;
        bst_ulong out_dim;
        float const* out_result;
        auto start = std::chrono::steady_clock::now();
        int create_matrix_success {XGDMatrixCreateFromMat(feature_values.data(), num_rows, num_cols, NAN, &dmatrix_handle)};
        if (create_matrix_success == -1) {
            throw std::runtime_error(std::string("Could not create DMatrix: ") + XGBGetLastError());
        }
        int predict_success {XGBoosterPredictFromDMatrix(xgb_booster.handle(), dmatrix_handle, "{\"type\": 1, \"iteration_begin\": 0, \"iteration_end\": 0, \"training\": false, \"strict_shape\": true}", &out_shape, &out_dim, &out_result)}; // We're scoring an already trained model so training false
        if (predict_success == -1) {
            throw std::runtime_error(std::string("Could not predict: ") + XGBGetLastError());
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<float, std::milli> elapsed_ms = end - start;
        XGDMatrixFree(dmatrix_handle);
        return elapsed_ms.count();
    }
    std::vector<float> run_batch_trials_host(const XGBBooster& xgb_booster, const rapidcsv::Document& sample_pool, size_t batch_size, size_t num_trials) {
        /*
         * Draws a fixed, random subset of size batch_size from the sample pool once before the
         * trial loop starts, using the same subset for every one of the num_trials repeated runs.
         * This is deliberate: repeated trials exist to mitigate timing noise
         * (OS scheduling jitter, thermal drift, etc.), and that only works if the input is held
         * constant across trials 
         */
        std::vector<size_t> indices(sample_pool.GetRowCount()); // Vector of indices we can randomly sample from
        std::iota(indices.begin(), indices.end(), 0); // Fills it in sequentially
        std::mt19937 random_engine {42};
        std::vector<size_t> chosen_indices;
        std::vector<TreeInfer::Sample> samples;
        std::sample(indices.begin(), indices.end(), std::back_inserter(chosen_indices), batch_size, random_engine);
        for (auto idx : chosen_indices) {
            TreeInfer::Sample sample {sample_pool.GetRow<float>(idx)};
            samples.push_back(sample);
        }
        std::vector<float> feature_values;
        for (const auto& sample : samples) {
            feature_values.insert(feature_values.end(), sample.values.begin(), sample.values.end());
        }
        std::vector<float> results;
        for (size_t i {}; i < num_trials; ++i) {
            float time {run_trial_host(xgb_booster, feature_values, batch_size, samples[0].values.size())};
            results.push_back(time);
        }
        return results;
    }
    std::map<size_t, std::vector<float>> run_sweep_host(const XGBBooster& xgb_booster, const rapidcsv::Document& sample_pool, size_t num_trials) {
        const std::vector<size_t> batch_sizes {1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 800, 1600, 3200, 6400, 10000};
        std::map<size_t, std::vector<float>> timing_map;
        for (auto batch_size : batch_sizes) {
            std::vector<float> trial_result {run_batch_trials_host(xgb_booster, sample_pool, batch_size, num_trials)};
            timing_map[batch_size] = trial_result;
        }
        return timing_map;
    }


    void write_results(const std::filesystem::path& output_path, size_t ensemble_size, const std::map<size_t, std::vector<float>>& timing_map) {
        bool file_exists {std::filesystem::exists(output_path)};
        std::ofstream out_file(output_path, std::ios::app);
        if (!file_exists) {
            out_file << "ensemble_size, batch_size, trial_num, predict (ms)\n";
        }
        for (const auto& [batch_size, trial_results] : timing_map) {
            for (size_t trial_num {}; trial_num < trial_results.size(); ++trial_num) {
                out_file << ensemble_size << "," << batch_size << "," << trial_num << "," << trial_results[trial_num] << "\n";
            }
        }
    }
}

// We pass in the directory of JSON files to create models from, CSV path of samples, and a file path to write output times into 
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Insufficient amount of arguments passed\n";
        return 1;
    }
    try {
        std::filesystem::path json_dir {argv[1]};
        std::filesystem::path csv_path {argv[2]};
        std::filesystem::path output_path {argv[3]};
        rapidcsv::Document bench_samples(csv_path.string());
        constexpr size_t num_trials {50};
        for (const auto& file : std::filesystem::directory_iterator(json_dir)) {
            auto file_path {file.path()};
            TreeInfer::Model model {TreeInfer::load_model(file_path)};
            XGBBooster xgb_booster(file_path);
            std::map<size_t, std::vector<float>> timing_map {run_sweep_host(xgb_booster, bench_samples, num_trials)};
            size_t ensemble_size {model.trees().size()};
            write_results(output_path, ensemble_size, timing_map);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
