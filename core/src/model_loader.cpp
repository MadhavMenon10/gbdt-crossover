#include "model_loader.hpp"
#include "tree.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <charconv>

namespace {
    TreeInfer::Tree generate_tree(const nlohmann::json& data) {
        std::vector<TreeInfer::Node> tree_vector;
        auto left_children {data["left_children"]};
        auto right_children {data["right_children"]};
        auto split_indices {data["split_indices"]};
        auto split_conditions {data["split_conditions"]};
        auto base_weights {data["base_weights"]};
        auto default_left {data["default_left"]};
        size_t num_nodes {left_children.size()};
        for (size_t i {}; i < num_nodes; ++i) {
            auto left_child {left_children[i]};
            auto right_child {right_children[i]};
            auto feature_index {split_indices[i]};
            auto threshold {split_conditions[i]};
            auto value {base_weights[i]};
            auto default_left_i {static_cast<bool>(default_left[i])};
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
        return Tree(std::move(tree_vector), 0); // XGBoost defaults root index to be 0
    }
}

TreeInfer::Model TreeInfer::load_model(const std::filesystem::path& file_path) {
    std::ifstream ifs(file_path);
    ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    nlohmann::json data {nlohmann::json::parse(ifs)};
    // Extracts the json data into a string
    std::string num_features_str{data["learner"]["learner_model_param"]["num_feature"]};
    std::uint16_t num_features;
    auto num_features_result {std::from_chars(num_features_str.data(), num_features_str.data() + num_features_str.length(), num_features)};
    // Throws if the extraction into num_features_failed
    if (num_features_result.ec != std::errc{}) {
        throw std::invalid_argument("num_features could not be extracted from the JSON file");
    }
    std::string base_score_str{data["learner"]["learner_model_param"]["base_score"]};
    float base_score;
    auto base_score_result {std::from_chars(base_score_str.data(), base_score_str.data() + base_score_str.length(), base_score, std::chars_format::general)};
    if (base_score_result.ec != std::errc{}) {
        throw std::invalid_argument("base_score could not be extracted from the JSON");
    }
    std::vector<Tree> trees;
    for (const auto& tree_json : data["learner"]["gradient_booster"]["model"]["trees"]) {
        trees.push_back(generate_tree(tree_json));
    }
    Model model(std::move(trees), num_features, base_score);
    return model;
}
