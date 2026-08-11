#include "tree.hpp"
#include <stdexcept>
#include <string>

namespace {
    // If validate_root_index fails, then the entire Tree construction aborts (because we use list initialisation) so a caller can never receive a broken tree
    void validate_root_index(std::uint32_t root_index, const std::vector<TreeInfer::Node> &nodes) {
        if (root_index >= nodes.size()) {
            throw std::out_of_range("Root index " + std::to_string(root_index) + " is not less than the number of nodes in the tree");
        }
    }
}

TreeInfer::Tree::Tree(std::vector<Node> tree_nodes, std::uint32_t root_index, std::uint16_t class_index) : 
    tree_nodes_(std::move(tree_nodes)), 
    root_index_(root_index),
    class_index_(class_index) 
    {
        validate_root_index(root_index_, tree_nodes_);
    }

