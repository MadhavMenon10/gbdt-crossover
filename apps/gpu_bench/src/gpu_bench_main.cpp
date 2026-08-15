#include "launch_tree_traversal_device.cuh"
#include "model_loader.hpp"
#include "model.hpp"
#include "sample.hpp"
#include "dense_model.hpp"
#include "rapidcsv.h"
#include "trial_results.hpp"
#include "utils.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <random>
#include <numeric>
#include <algorithm>
#include <vector>
#include <map>
#include <utility>
#include <optional>
#include <fstream>


namespace {
    TreeInfer::TrialResults run_batch_trials_device(const TreeInfer::DenseModel& dense_model, const rapidcsv::Document& sample_pool, size_t batch_size, size_t num_trials) {
        // This section randomly chooses samples 
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
        TreeInfer::TrialResults trial_results;
        for (size_t i {}; i < num_trials; ++i) {
           auto [results, h2d_time, kernel_time, d2h_time] = TreeInfer::launch_dense_tree_traversal_kernel(dense_model, samples);
           trial_results.add_trial(h2d_time, kernel_time, d2h_time);
        }
        return trial_results;
    }
    std::map<size_t, TreeInfer::TrialResults> coarse_run_device(const TreeInfer::DenseModel& dense_model, const rapidcsv::Document& sample_pool, size_t num_trials) {
        std::map<size_t, TreeInfer::TrialResults> timing_map;
        std::vector<size_t> coarse_batch_size {1, 2, 4, 8, 16, 32, 64, 100};
        for (auto batch_size : coarse_batch_size) {
            TreeInfer::TrialResults trial_result {run_batch_trials_device(dense_model, sample_pool, batch_size, num_trials)};
            timing_map[batch_size] = trial_result;
        }
        return timing_map;
    }
    std::optional<std::pair<size_t, size_t>> find_bracket_device(const std::map<size_t, TreeInfer::TrialResults>& timing_map) {
        size_t prev_batch_size {0};
        bool prev_condition {false};
        for (const auto& [batch_size, trial_result] :  timing_map) {
            float h2d_time_median {Utils::median(trial_result.h2d_times())};
            float kernel_time_median {Utils::median(trial_result.kernel_times())};
            float d2h_time_median {Utils::median(trial_result.d2h_times())};
            bool curr_condition {kernel_time_median > h2d_time_median + d2h_time_median};
            if (curr_condition && !prev_condition) {
                return std::pair<size_t, size_t> {prev_batch_size, batch_size};
            } 
            prev_condition = curr_condition;
            prev_batch_size = batch_size;
        }
        return std::nullopt;
    }
    void dense_run_device(const TreeInfer::DenseModel& dense_model, const rapidcsv::Document& sample_pool, size_t num_trials, size_t lower_bound, size_t upper_bound, std::map<size_t, TreeInfer::TrialResults>& timing_map) {
        for (size_t batch_size {lower_bound + 1}; batch_size < upper_bound; ++batch_size) { // So we don't overwrite at lower_bound and upper_bound
            TreeInfer::TrialResults trial_result {run_batch_trials_device(dense_model, sample_pool, batch_size, num_trials)};
            timing_map[batch_size] = trial_result;
        }
    }
    void write_results(const std::filesystem::path& output_path, size_t ensemble_size, const std::map<size_t, TreeInfer::TrialResults>& timing_map) {
        bool file_exists {std::filesystem::exists(output_path)};
        std::ofstream out_file(output_path, std::ios::app);
        if (!file_exists) {
            out_file << "ensemble_size, batch_size, trial_num, h2d (ms), kernel (ms), d2h (ms)\n";
        }
        for (const auto& [batch_size, trial_result] : timing_map) {
            for (size_t trial_num {}; trial_num < trial_result.h2d_times().size(); ++trial_num) {
                out_file << ensemble_size << ", " << batch_size << ", " << trial_num << ", " << trial_result.h2d_times()[trial_num]  << ", " << trial_result.kernel_times()[trial_num] << ", " << trial_result.d2h_times()[trial_num] << "\n";
            }
        }
    }
    std::map<size_t, TreeInfer::TrialResults> run_sweep(const TreeInfer::DenseModel& dense_model, const rapidcsv::Document& sample_pool, size_t num_trials) {
        std::map<size_t, TreeInfer::TrialResults> timing_map {coarse_run_device(dense_model, sample_pool, num_trials)};
        std::optional<std::pair<size_t, size_t>> bracket {find_bracket_device(timing_map)};
        if (bracket.has_value()) {
            auto [lower_bound, upper_bound] = bracket.value();
            dense_run_device(dense_model, sample_pool, num_trials, lower_bound, upper_bound, timing_map);
        } else {
            std::cerr << "No transition found for this ensemble size within the coarse range\n"; // For us to see if this ensemble size didn't work
        }
        return timing_map;
    }
}

// We pass in the directory of JSON files to create models from, CSV path of samples,  max_depth for our trees, and a file path to write output times into 
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Insufficient amount of arguments passed\n";
        return 1;
    }
    try {
        std::filesystem::path json_dir {argv[1]};
        std::filesystem::path csv_path {argv[2]};
        std::uint16_t max_depth {static_cast<std::uint16_t>(std::stoi(argv[3]))};
        std::filesystem::path output_path {argv[4]};
        rapidcsv::Document bench_samples(csv_path.string());
        bool cold_start {true};
        size_t num_trials {50}; // Subject to change
        for (const auto& file : std::filesystem::directory_iterator(json_dir)) {
            auto file_path {file.path()};
            TreeInfer::DenseModel dense_model(TreeInfer::load_model(file_path), max_depth);
            if (cold_start) {
                TreeInfer::launch_dense_tree_traversal_kernel(dense_model, std::vector<TreeInfer::Sample>{TreeInfer::Sample{std::vector<float>(dense_model.num_features(), 0.0f)}}); // A toy sample we throw-away to get the cold-start
                cold_start = false;
            }
            std::map<size_t, TreeInfer::TrialResults> timing_map {run_sweep(dense_model, bench_samples, num_trials)};
            size_t ensemble_size {dense_model.dense_trees().size()};
            write_results(output_path, ensemble_size, timing_map);    
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
