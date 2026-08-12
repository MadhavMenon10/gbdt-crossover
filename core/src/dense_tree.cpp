#include "dense_tree.hpp"
#include <vector>
#include <stdexcept>

namespace {
    // If validate_root_index fails, then the entire Tree construction aborts (because we use list initialisation) so a caller can never receive a broken tree
    void validate_dense_tree(std::uint16_t max_depth, const std::vector<TreeInfer::DenseNode>& dense_nodes) {
        size_t num_nodes {(static_cast<size_t>(1) << (max_depth + 1)) - 1};
        if (dense_nodes.size() != num_nodes) {
            throw std::invalid_argument("The dense tree does not meet the maximum depth");
        }
    }
}

TreeInfer::DenseTree::DenseTree(std::vector<DenseNode> dense_tree_nodes, std::uint16_t max_depth) : 
    dense_tree_nodes_(std::move(dense_tree_nodes)),
    max_depth_(max_depth)
    {
        validate_dense_tree(max_depth_, dense_tree_nodes_);
    }

