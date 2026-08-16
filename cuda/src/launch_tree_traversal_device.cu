#include "launch_tree_traversal_device.cuh"
#include "tree_traversal_device.cuh"
#include "device_model.cuh"
#include "cuda_utils.cuh"

std::tuple<std::vector<float>, float, float, float> TreeInfer::launch_dense_tree_traversal_kernel(const DeviceModel& device_model, const std::vector<TreeInfer::Sample>& samples) {
    size_t num_samples {samples.size()};
    size_t features_per_sample {samples[0].values.size()};
    size_t nodes_per_tree {device_model.nodes_per_tree()};
    size_t num_trees {device_model.num_trees()};

    std::vector<float> feature_values {};
    for (const auto& sample : samples) {
        // Accumulates all the feature values across every sample in one vector
        feature_values.insert(feature_values.end(), sample.values.begin(), sample.values.end());
    }
    cudaEvent_t start_event, h2d_done_event, kernel_done_event, stop_event;
    CUDAUtils::check_cuda_error(cudaEventCreate(&start_event), "Create start event");
    CUDAUtils::check_cuda_error(cudaEventCreate(&h2d_done_event), "Create h2d_done event");
    CUDAUtils::check_cuda_error(cudaEventCreate(&kernel_done_event), "Create kernel_done event");
    CUDAUtils::check_cuda_error(cudaEventCreate(&stop_event), "Create stop event");
    DenseNode* dense_nodes_d {device_model.dense_nodes_device_ptr()};
    float* feature_values_d {nullptr};
    size_t overall_feature_values_size {sizeof(float) * num_samples * features_per_sample};
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&feature_values_d), overall_feature_values_size), "Allocate memory for feature_values_d array");
    CUDAUtils::check_cuda_error(cudaEventRecord(start_event, 0), "Record start event");
    CUDAUtils::check_cuda_error(cudaMemcpy(feature_values_d, feature_values.data(), overall_feature_values_size, cudaMemcpyHostToDevice), "Copy feature_values_d array onto device");
    CUDAUtils::check_cuda_error(cudaEventRecord(h2d_done_event, 0), "Record end event for h2d transfer");
    float* output_array_d {nullptr};
    size_t num_output_elements {num_samples * num_trees};
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&output_array_d), sizeof(float) * num_output_elements), "Allocate memory for output_array_d");
    size_t block_size {256};
    TreeInfer::traverse_dense_tree_device<<<(num_output_elements + block_size - 1) / block_size, block_size>>>(dense_nodes_d, feature_values_d, nodes_per_tree, num_samples, num_trees, features_per_sample, output_array_d);
    CUDAUtils::check_cuda_error(cudaEventRecord(kernel_done_event, 0), "Record end event for kernel execution");
    std::vector<float> output_vector(num_output_elements);
    // Move output of the kernel back into the output vector
    CUDAUtils::check_cuda_error(cudaMemcpy(output_vector.data(), output_array_d, sizeof(float) * num_output_elements, cudaMemcpyDeviceToHost), "Move output elements into the output vector");
    CUDAUtils::check_cuda_error(cudaEventRecord(stop_event, 0), "Record end event for d2h transfer");
    CUDAUtils::check_cuda_error(cudaEventSynchronize(stop_event), "Synchronize on stop event");
    float h2d_time {};
    float kernel_time {};
    float d2h_time {};
    CUDAUtils::check_cuda_error(cudaEventElapsedTime(&h2d_time, start_event, h2d_done_event), "Compute h2d elapsed time");
    CUDAUtils::check_cuda_error(cudaEventElapsedTime(&kernel_time, h2d_done_event, kernel_done_event), "Compute kernel elapsed time");
    CUDAUtils::check_cuda_error(cudaEventElapsedTime(&d2h_time, kernel_done_event, stop_event), "Compute d2h elapsed time");
    // Destroy CUDA events
    CUDAUtils::check_cuda_error(cudaEventDestroy(start_event), "Destroy start event");
    CUDAUtils::check_cuda_error(cudaEventDestroy(h2d_done_event), "Destroy h2d_done event");
    CUDAUtils::check_cuda_error(cudaEventDestroy(kernel_done_event), "Destroy kernel_done event");
    CUDAUtils::check_cuda_error(cudaEventDestroy(stop_event), "Destroy stop event");
    // Free pointers
    CUDAUtils::check_cuda_error(cudaFree(feature_values_d), "Free feature_values_d");
    CUDAUtils::check_cuda_error(cudaFree(output_array_d), "Free output_array_d");
    feature_values_d = nullptr;
    output_array_d = nullptr;
    return std::tuple<std::vector<float>, float, float, float>{std::move(output_vector), h2d_time, kernel_time, d2h_time};
}   
