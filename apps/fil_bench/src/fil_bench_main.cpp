#include <treelite/model_loader.h>
#include <treelite/tree.h>
#include <nvforest/constants.hpp>
#include <nvforest/treelite_importer.hpp>
#include <nvforest/detail/index_type.hpp> 
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <raft/core/handle.hpp>
#include <nvforest/detail/raft_proto/handle.hpp>
#include <tuple>


namespace {
    auto load_nvforest_model(const std::filesystem::path& model_path) {
        auto treelite_model = treelite::model_loader::LoadXGBoostModelJSON(model_path.c_str(), "{}");
        return nvforest::import_from_treelite_model(*treelite_model, nvforest::preferred_tree_layout, nvforest::index_type{}, std::nullopt, raft_proto::device_type::gpu);
    }
    //model type is templated since load_nvforest_model's return type is only known via `auto`
    template <typename ModelType>
    std::tuple<float, float, float> run_trial_fil(ModelType& model, raft_proto::handle_t& handle, const std::vector<float>& feature_values, size_t num_rows, size_t num_cols) {
        // Runs a FIL trial and does timing
        float* input_d {nullptr};
        float* output_d {nullptr};
        size_t input_size {sizeof(float) * num_rows * num_cols};
        size_t output_size {sizeof(float) * num_rows * model.num_outputs()};
        cudaEvent_t start_event, h2d_done_event, predict_done_event, stop_event;
        CUDAUtils::check_cuda_error(cudaEventCreate(&start_event), "Create start event");
        CUDAUtils::check_cuda_error(cudaEventCreate(&h2d_done_event), "Create h2d_done event");
        CUDAUtils::check_cuda_error(cudaEventCreate(&predict_done_event), "Create predict_done event");
        CUDAUtils::check_cuda_error(cudaEventCreate(&stop_event), "Create stop event");

        CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&input_d), input_size), "Allocate input_d");
        CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&output_d), output_size), "Allocate output_d");

        CUDAUtils::check_cuda_error(cudaEventRecord(start_event, 0), "Record start event");
        CUDAUtils::check_cuda_error(cudaMemcpy(input_d, feature_values.data(), input_size, cudaMemcpyHostToDevice), "Copy input to device");
        CUDAUtils::check_cuda_error(cudaEventRecord(h2d_done_event, 0), "Record h2d_done event");

        model.predict(handle, output_d, input_d, num_rows, raft_proto::device_type::gpu, raft_proto::device_type::gpu, nvforest::infer_kind::default_kind);
        CUDAUtils::check_cuda_error(cudaEventRecord(predict_done_event, 0), "Record predict_done event");

        std::vector<float> output_vector(num_rows * model.num_outputs());
        CUDAUtils::check_cuda_error(cudaMemcpy(output_vector.data(), output_d, output_size, cudaMemcpyDeviceToHost), "Copy output to host");
        CUDAUtils::check_cuda_error(cudaEventRecord(stop_event, 0), "Record stop event");
        CUDAUtils::check_cuda_error(cudaEventSynchronize(stop_event), "Synchronize on stop event");

        float h2d_ms{}, kernel_ms{}, d2h_ms{}; // kernel_ms is the time it takes to carry out a prediction
        CUDAUtils::check_cuda_error(cudaEventElapsedTime(&h2d_ms, start_event, h2d_done_event), "Compute h2d time");
        CUDAUtils::check_cuda_error(cudaEventElapsedTime(&kernel_ms, h2d_done_event, predict_done_event), "Compute predict time");
        CUDAUtils::check_cuda_error(cudaEventElapsedTime(&d2h_ms, predict_done_event, stop_event), "Compute d2h time");
        // Free
        CUDAUtils::check_cuda_error(cudaEventDestroy(start_event), "Destroy start event");
        CUDAUtils::check_cuda_error(cudaEventDestroy(h2d_done_event), "Destroy h2d_done event");
        CUDAUtils::check_cuda_error(cudaEventDestroy(predict_done_event), "Destroy predict_done event");
        CUDAUtils::check_cuda_error(cudaEventDestroy(stop_event), "Destroy stop event");
        CUDAUtils::check_cuda_error(cudaFree(input_d), "Free input_d");
        CUDAUtils::check_cuda_error(cudaFree(output_d), "Free output_d");
        return std::tuple<float, float, float>{h2d_ms, kernel_ms, d2h_ms};
    }
    template <typename ModelType>
    TreeInfer::TrialResults run_batch_trials_fil(ModelType& model, raft_proto::handle_t& handle, const rapidcsv::Document& sample_pool, size_t batch_size, size_t num_trials) {
        /*
         * Draws a fixed, random subset of size batch_size from the sample pool once before the
         * trial loop starts, using the same subset for every one of the num_trials repeated runs.
         * This is deliberate: repeated trials exist to mitigate timing noise
         * (OS scheduling jitter, thermal drift, etc.), and that only works if the input is held
         * constant across trials 
         */ 
        std::vector<size_t> indices(sample_pool.GetRowCount());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 random_engine {42};
        std::vector<size_t> chosen_indices;
        std::vector<TreeInfer::Sample> samples;
        std::sample(indices.begin(), indices.end(), std::back_inserter(chosen_indices), batch_size, random_engine);
        for (auto idx : chosen_indices) {
            TreeInfer::Sample sample {sample_pool.GetRow<float>(idx)};
            samples.push_back(sample);
        }
        size_t num_cols {samples[0].values.size()};
        std::vector<float> feature_values;
        for (const auto& sample : samples) {
            feature_values.insert(feature_values.end(), sample.values.begin(), sample.values.end());
        }
        TreeInfer::TrialResults trial_results;
        for (size_t i {}; i < num_trials; ++i) {
            auto [h2d_ms, predict_ms, d2h_ms] = run_trial_fil(model, handle, feature_values, batch_size, num_cols);
            trial_results.add_trial(h2d_ms, predict_ms, d2h_ms);
        }
        return trial_results;
    }
    void write_results(const std::filesystem::path& output_path, size_t ensemble_size, const std::map<size_t, TreeInfer::TrialResults>& timing_map) {
        /*
         * Appends timing_map's rows to the results CSV right after that
         * model's sweep (coarse plus dense if a bracket was found) completes. Writes incrementally
         * rather than accumulating all ten models' results and writing once at the end because a full
         * sweep across ten ensemble sizes is a long-running program, and if it crashes partway
         * through (which it can since I'm SSHing into a GPU), 
         * incremental writes mean only the current model's data is lost, not every model that already finished successfully.
         */
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
}


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
        // Constructed once because raft::handle_t manages CUDA streams/library handles
        // meant to be reused, not recreated on every call.
        raft::handle_t raft_handle{};
        raft_proto::handle_t handle{raft_handle};
        std::vector<size_t> batch_sizes {1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 800, 1600, 3200, 6400, 10000};
        for (const auto& file : std::filesystem::directory_iterator(json_dir)) {
            auto file_path {file.path()};
            TreeInfer::Model model {TreeInfer::load_model(file_path)};
            size_t ensemble_size {model.trees().size()};
            auto nvforest_model = load_nvforest_model(file_path);
            std::map<size_t, TreeInfer::TrialResults> timing_map;
            for (auto batch_size : batch_sizes) {
                timing_map[batch_size] = run_batch_trials_fil(nvforest_model, handle, bench_samples, batch_size, num_trials);
            }
            write_results(output_path, ensemble_size, timing_map);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

