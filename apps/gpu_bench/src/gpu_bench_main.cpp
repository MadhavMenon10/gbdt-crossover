#include "launch_tree_traversal_device.cuh"
#include "model_loader.hpp"
#include "model.hpp"
#include "sample.hpp"
#include "dense_model.hpp"
#include "rapidcsv.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <cstdint>


// We pass in the directory of JSON files to create models from, CSV path of samples and max_depth for our trees
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Insufficient amount of arguments passed\n";
        return 1;
    }
    try {
        std::filesystem::path json_dir {argv[1]};
        std::filesystem::path csv_path {argv[2]};
        std::uint16_t max_depth {static_cast<std::uint16_t>(std::stoi(argv[3]))};
        rapidcsv::Document bench_samples(csv_path.string());
        for (const auto& file : std::filesystem::directory_iterator(json_dir)) {
            auto file_path {file.path()};
            TreeInfer::DenseModel dense_model(TreeInfer::load_model(file_path), max_depth);

        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
