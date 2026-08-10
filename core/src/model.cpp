#include "model.hpp"
#include <string>
#include <stdexcept>

namespace {
    // If validate_trees fails, then the entire Model construction aborts (because we use list initialisation) so a caller can never receive a broken Model
    std::uint16_t validate_trees(std::uint16_t num_features, const std::vector<TreeInfer::Tree>& trees) {
        for (const auto& tree : trees) {
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
        return num_features;
    }
}

TreeInfer::Model::Model(std::vector<Tree> trees, std::uint16_t num_features, float base_score) : 
    trees_(std::move(trees)), 
    num_features_(validate_trees(num_features, trees_)), 
    base_score_(base_score)
    {}
