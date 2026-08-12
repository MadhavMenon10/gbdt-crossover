#include "model_loader.hpp"
#include "sample.hpp"
#include "model.hpp"
#include "rapidcsv.h"
#include "cpu_reference.hpp"
#include "dense_builder.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <cmath>

namespace {
    void check_row(const TreeInfer::Model& model, const TreeInfer::Sample& sample, const std::vector<float>& expected_margins, size_t row) {
        std::vector<float> prediction_result {predict(model, sample)};
        float eps {1e-4f};
        for (size_t i {}; i < prediction_result.size(); ++i) {
            if (std::abs(prediction_result[i] - expected_margins[i]) > eps) {
                std::cerr << "Error at index " << i << " for row: " << row << "\n";
                std::cerr << "Prediction Result: " << prediction_result[i] << "\n";
                std::cerr << "Expected Result: " << expected_margins[i] << "\n";
            }
        }
    }
    void compare_dense_tree(const TreeInfer::Model& model, const TreeInfer::Sample& sample, size_t row, std::uint16_t max_depth) {
        float eps {1e-4f};
        for (size_t i {}; i < model.trees().size(); ++i) {
            const auto& tree {model.trees()[i]};
            float tree_traversal_res {TreeInfer::traverse_tree(tree, sample)};
            float dense_tree_traversal_res {TreeInfer::traverse_dense_tree(TreeInfer::generate_dense_tree(tree, max_depth), sample)};
            if (std::abs(tree_traversal_res - dense_tree_traversal_res) > eps) {
                std::cerr << "Dense tree traversal and regular tree traversal results do not match for tree " << i << "\n";
                std::cerr << "Regular tree traversal: " << tree_traversal_res << "\n";
                std::cerr << "Dense tree traversal: " << dense_tree_traversal_res << "\n";
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
        std::filesystem::path model_path(argv[1]);
        std::filesystem::path csv_path(argv[2]);
        TreeInfer::Model model {TreeInfer::load_model(model_path)};
        std::uint16_t max_depth {static_cast<std::uint16_t>(std::stoi(argv[3]))};
        // Avoids throwing on empty cells which inject_missing_values deliberately creates
        rapidcsv::Document csv_doc(csv_path.string(), rapidcsv::LabelParams(), rapidcsv::SeparatorParams(), rapidcsv::ConverterParams(true));
        for (size_t row {}; row < csv_doc.GetRowCount(); ++row) {
            std::vector<float> csv_row {csv_doc.GetRow<float>(row)};
            std::vector<float> feature_values(csv_row.begin(), csv_row.begin() + model.num_features());
            std::vector<float> expected_margins(csv_row.begin() + model.num_features(), csv_row.end());
            TreeInfer::Sample sample {std::move(feature_values)};
            check_row(model, sample, expected_margins, row);
            compare_dense_tree(model, sample, row, max_depth);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
