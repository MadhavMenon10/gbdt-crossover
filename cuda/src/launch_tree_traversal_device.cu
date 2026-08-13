#include "launch_tree_traversal_device.cuh"
#include "cuda_utils.cuh"

std::vector<float> TreeInfer::launch_dense_tree_traversal_kernel(const DenseModel& dense_model, const std::vector<TreeInfer::Sample>& samples) {
    size_t num_samples {samples.size()};
    size_t num_trees {dense_model.dense_trees().size()};
    size_t nodes_per_tree {dense_model.dense_trees()[0].num_nodes()};
    std::vector<DenseNode> tree_nodes {};
    for (const auto& dense_tree : dense_model.dense_trees()) {
        tree_nodes.insert(tree_nodes.end(), dense_tree.nodes().begin(), dense_tree.nodes().end());
    }
    size_t features_per_sample {samples[0].values.size()};
    std::vector<float> feature_values {};
    for (const auto& sample : samples) {
        feature_values.insert(feature_values.end(), sample.values.begin(), sample.values.end());
    }
    DenseNode* dense_nodes_d {nullptr};
    size_t overall_dense_nodes_size {sizeof(DenseNode) * nodes_per_tree * num_trees};
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&dense_nodes_d), overall_dense_nodes_size), "Allocate memory for dense_nodes_d array");
    float* feature_values_d {nullptr};
    size_t overall_feature_values_size {sizeof(float) * num_samples * features_per_sample};
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&feature_values_d), overall_feature_values_size), "Allocate memory for feature_values_d array");
    CUDAUtils::check_cuda_error(cudaMemcpy(dense_nodes_d, tree_nodes.data(), overall_dense_nodes_size, cudaMemcpyHostToDevice), "Copy dense_nodes_d array onto device");
    CUDAUtils::check_cuda_error(cudaMemcpy(feature_values_d, feature_values.data(), overall_feature_values_size, cudaMemcpyHostToDevice), "Copy feature_values_d array onto device");
    float* output_array_d {nullptr};
    size_t num_output_elements {num_samples * num_trees};
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&output_array_d), sizeof(float) * num_output_elements), "Allocate memory for output_array_d");
    size_t block_size {256};
    traverse_dense_tree_device<<<(num_output_elements + block_size - 1) / block_size, block_size>>>(dense_nodes_d, feature_values_d, nodes_per_tree, num_samples, num_trees, features_per_sample, output_array_d);
    std::vector<float> output_vector(num_output_elements);
    CUDAUtils::check_cuda_error(cudaMemcpy(output_vector.data(), output_array_d, sizeof(float) * num_output_elements, cudaMemcpyDeviceToHost), "Move output elements into the output vector");
    CUDAUtils::check_cuda_error(cudaFree(dense_nodes_d), "Free dense_nodes_d");
    CUDAUtils::check_cuda_error(cudaFree(feature_values_d), "Free feature_values_d");
    CUDAUtils::check_cuda_error(cudaFree(output_array_d), "Free output_array_d");
    dense_nodes_d = nullptr;
    feature_values_d = nullptr;
    output_array_d = nullptr;
    return output_vector;
}   
