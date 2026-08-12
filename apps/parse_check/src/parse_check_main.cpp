#include "model_loader.hpp"
#include "sample.hpp"
#include "model.hpp"
#include "rapidcsv.h"
#include "cpu_reference.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <cmath>

namespace {
    void check_row(const TreeInfer::Model& model, const rapidcsv::Document& csv_doc, size_t row) {
        std::vector<float> csv_row {csv_doc.GetRow<float>(row)};
        std::vector<float> feature_values(csv_row.begin(), csv_row.begin() + model.num_features());
        std::vector<float> expected_margins(csv_row.begin() + model.num_features(), csv_row.end());
        TreeInfer::Sample sample {std::move(feature_values)};
        std::vector<float> prediction_result {predict(model, sample)};
        float eps {1e-4f};
        for (size_t i {}; i < prediction_result.size(); ++i) {
            if (std::abs(prediction_result[i] - expected_margins[i]) > eps) {
                std::cerr << "Error at index " << i << " for row: " << std::to_string(row) << "\n";
                std::cerr << "Prediction Result: " << prediction_result[i] << "\n";
                std::cerr << "Expected Result: " << expected_margins[i] << "\n";
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Insufficient amount of arguments passed\n";
        return 1;
    }
    try {
        std::filesystem::path model_path(argv[1]);
        std::filesystem::path csv_path(argv[2]);
        TreeInfer::Model model {TreeInfer::load_model(model_path)};
        // Avoids throwing on empty cells which inject_missing_values deliberately creates
        rapidcsv::Document csv_doc(csv_path.string(), rapidcsv::LabelParams(), rapidcsv::SeparatorParams(), rapidcsv::ConverterParams(true));
        for (size_t row {}; row < csv_doc.GetRowCount(); ++row) {
            check_row(model, csv_doc, row);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
