#include "model.hpp"
#include <string>
#include <stdexcept>

namespace {
    // If validate_model fails, then the entire Model construction aborts (because we use list initialisation) so a caller can never receive a broken Model
    void validate_model(std::uint16_t num_features, std::uint16_t num_classes, const std::vector<TreeInfer::Tree>& trees) {
        for (const auto& tree : trees) {
            if (tree.class_index() >= num_classes) {
                throw std::invalid_argument("The class index for a tree exceeds the number of classes in the model");
            }
            for (const auto& node : tree.nodes()) {
                // We check only split nodes as feature_index has no meaning on a leaf node. Thus, it may hold a garbage value that we should not check
                if (node.is_leaf()) {
                    continue;
                }
                if (node.feature_index >= num_features) {
                    throw std::out_of_range("Node has a feature index of " + std::to_string(node.feature_index) + " which is greater than " + std::to_string(num_features));
                }
            }
        }
    }
}

TreeInfer::Model::Model(std::vector<Tree> trees, std::uint16_t num_features, std::uint16_t num_classes, std::vector<float> base_score) : 
    trees_(std::move(trees)), 
    num_features_(num_features), 
    num_classes_(num_classes),
    base_scores_(std::move(base_score))
    {
        validate_model(num_features, num_classes, trees_); 
        if (base_scores_.size() != num_classes_) {
            throw std::invalid_argument("The number of base scores does not match the number of classes");
        }
    }
