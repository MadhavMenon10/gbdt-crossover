#include "launch_tree_traversal_device.cuh"
#include "model_loader.hpp"
#include "model.hpp"
#include "dense_model.hpp"
#include "device_model.cuh"
#include "sample.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Insufficient arguments passed\n";
        return 1;
    }
    try {
        std::filesystem::path model_path {argv[1]};
        std::uint16_t max_depth {static_cast<std::uint16_t>(std::stoi(argv[2]))};
        size_t batch_size {static_cast<size_t>(std::stoi(argv[3]))};
        TreeInfer::DenseModel dense_model(TreeInfer::load_model(model_path), max_depth);
        TreeInfer::DeviceModel device_model(dense_model);
        std::vector<TreeInfer::Sample> samples(batch_size, TreeInfer::Sample{std::vector<float>(dense_model.num_features(), 0.0f)});
        TreeInfer::launch_dense_tree_traversal_kernel(device_model, samples);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
