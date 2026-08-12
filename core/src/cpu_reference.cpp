#include "cpu_reference.hpp"
#include "sample.hpp"
#include "tree.hpp"
#include "dense_tree.hpp"
#include "dense_builder.hpp"
#include <cmath>
#include <stdexcept>



float TreeInfer::traverse_tree(const Tree& tree, const Sample& sample) {
    auto curr_node {tree.root()};
    while (!curr_node.is_leaf()) {
        auto curr_value {sample.values[curr_node.feature_index]};
        if (std::isnan(curr_value)) {
            curr_node = (curr_node.default_left ? tree.nodes()[curr_node.left_child] : tree.nodes()[curr_node.right_child]);
        } else if (curr_value >= curr_node.threshold) {
            curr_node = tree.nodes()[curr_node.right_child];
        } else {
            curr_node = tree.nodes()[curr_node.left_child];
        }
    }
    return curr_node.value;
}

float TreeInfer::traverse_dense_tree(const DenseTree& tree, const Sample& sample) {
    auto curr_node {tree.root()};
    auto curr_idx {tree.root_index()};
    while (!curr_node.is_leaf) {
        auto curr_value {sample.values[curr_node.feature_index]};
        auto left_idx {TreeInfer::left_child_index(curr_idx)};
        auto right_idx {TreeInfer::right_child_index(curr_idx)};
        if (std::isnan(curr_value)) {
            curr_idx = (curr_node.default_left ? left_idx : right_idx);
            curr_node = (curr_node.default_left ? tree.nodes()[left_idx] : tree.nodes()[right_idx]);
        } else if (curr_value >= curr_node.threshold) {
            curr_node = tree.nodes()[right_idx];
            curr_idx = right_idx;
        } else {
            curr_node = tree.nodes()[left_idx];
            curr_idx = left_idx;
        }
    }
    return curr_node.value;
}



std::vector<float> TreeInfer::predict(const Model& model, const Sample& sample) {
    if (sample.values.size() != model.num_features()) {
        throw std::invalid_argument("The number of values in the sample do not match the number of features in the model");
    }
    std::vector<float> totals {model.base_scores()};
    for (const auto& tree : model.trees()) {
        totals[tree.class_index()] += traverse_tree(tree, sample);
    }
    return totals;
}


std::vector<float> TreeInfer::predict_dense(const Model& model, const Sample& sample, std::uint16_t max_depth) {
    if (sample.values.size() != model.num_features()) {
        throw std::invalid_argument("The number of values in the sample do not match the number of features in the model");
    }
    std::vector<float> totals {model.base_scores()};
    for (const auto& tree : model.trees()) {
        totals[tree.class_index()] += traverse_dense_tree(TreeInfer::generate_dense_tree(tree, max_depth), sample);
    }
    return totals;
}
