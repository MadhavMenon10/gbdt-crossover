#include "model_loader.hpp"
#include "tree.hpp"
#include <fstream>
#include "json.hpp"
#include <charconv>
#include <iostream>

namespace {
    TreeInfer::Tree generate_tree(const nlohmann::json& data, std::uint16_t class_index) {
        std::vector<TreeInfer::Node> tree_vector;
        // We don't use brace initialisation as a quirk of JSON parsing is that if we {} a singular value it is treated as an array
        auto left_children = data["left_children"];
        auto right_children = data["right_children"];
        auto split_indices = data["split_indices"];
        auto split_conditions = data["split_conditions"];
        auto base_weights = data["base_weights"];
        auto default_left = data["default_left"];
        size_t num_nodes {left_children.size()};
        for (size_t i {}; i < num_nodes; ++i) {
            auto left_child = left_children[i];
            auto right_child = right_children[i];
            auto feature_index = split_indices[i];
            auto threshold = split_conditions[i];
            auto value = base_weights[i];
            auto default_left_i = static_cast<bool>(default_left[i].get<int>());
            TreeInfer::Node node;
            // Checks if there is actually a child. XGBoost encodes no child with a -1
            left_child == -1 ? node.left_child = node.k_no_child : node.left_child = left_child;
            right_child == -1 ? node.right_child = node.k_no_child : node.right_child = right_child;
            node.feature_index = feature_index;
            node.threshold = threshold;
            node.value = value;
            node.default_left = default_left_i;
            tree_vector.push_back(node);
        }
        return TreeInfer::Tree(std::move(tree_vector), 0, class_index); // XGBoost defaults root index to be 0
    }
}

TreeInfer::Model TreeInfer::load_model(const std::filesystem::path& file_path) {
    std::ifstream ifs(file_path);
    ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    nlohmann::json data = nlohmann::json::parse(ifs);
    // Extracts the json data into a string
    std::string num_features_str = data["learner"]["learner_model_param"]["num_feature"];
    std::uint16_t num_features;
    auto num_features_result {std::from_chars(num_features_str.data(), num_features_str.data() + num_features_str.length(), num_features)};
    // Throws if the extraction into num_features_failed
    if (num_features_result.ec != std::errc{}) {
        throw std::invalid_argument("num_features could not be extracted from the JSON file");
    }
    std::string base_score_str = data["learner"]["learner_model_param"]["base_score"];
    auto base_score_json = nlohmann::json::parse(base_score_str);
    std::vector<float> base_scores;
    for (const auto& val : base_score_json) {
        base_scores.push_back(val);
    }

    std::string num_classes_str = data["learner"]["learner_model_param"]["num_class"];
    std::uint16_t num_classes;
    auto num_classes_result {std::from_chars(num_classes_str.data(), num_classes_str.data() + num_classes_str.length(), num_classes)};
    if (num_classes_result.ec != std::errc{}) {
        throw std::invalid_argument("num_classes could not be extracted from the JSON file");
    }
    // XGBoost stores binary classes with 0 so we update it to 1 to maintain intuitiveness
    if (num_classes == 0) {
        num_classes = 1;
    }
    std::vector<Tree> trees;
    auto tree_data = data["learner"]["gradient_booster"]["model"]["trees"];
    for (size_t i {}; i < tree_data.size(); ++i) {
        /* A quirk of the new version of XGBoost.
         * Rather than storing an explicit "tree_info" array in this schema version,
         * the setup is such that tree_index i belongs to class_index i % num_classes
        */
        std::uint16_t class_index {static_cast<std::uint16_t>(i % num_classes)}; 
        trees.push_back(generate_tree(tree_data[i], class_index));
    }
    Model model(std::move(trees), num_features, num_classes, std::move(base_scores));
    return model;
}
