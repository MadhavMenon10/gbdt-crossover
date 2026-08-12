#include "dense_builder.hpp"

namespace {
    void place_dense_node(const TreeInfer::Tree& tree, std::uint32_t source_index, size_t destination_index, std::vector<TreeInfer::DenseNode>& dense_node_vect) {
        const TreeInfer::Node& source_node {tree.nodes()[source_index]};
        TreeInfer::DenseNode dense_node;
        dense_node.feature_index = source_node.feature_index;
        dense_node.threshold = source_node.threshold;
        dense_node.default_left = source_node.default_left;
        dense_node.value = source_node.value;
        dense_node.is_leaf = source_node.is_leaf();
        dense_node_vect[destination_index] = dense_node;
        if (source_node.is_leaf()) {
            return;
        }
        auto left_child_src_idx {source_node.left_child};
        auto right_child_src_idx {source_node.right_child};
        auto left_child_dest_idx {TreeInfer::left_child_index(destination_index)};
        auto right_child_dest_idx {TreeInfer::right_child_index(destination_index)};
        place_dense_node(tree, left_child_src_idx, left_child_dest_idx, dense_node_vect);
        place_dense_node(tree, right_child_src_idx, right_child_dest_idx, dense_node_vect);
    }
}

TreeInfer::DenseTree TreeInfer::generate_dense_tree(const Tree& tree, std::uint16_t max_depth) {
    size_t num_nodes {(static_cast<size_t>(1) << (max_depth + 1)) - 1}; // A complete binary tree of depth d has 2^(d+1) -1 nodes
    std::vector<TreeInfer::DenseNode> dense_nodes(num_nodes);
    auto source_idx {tree.root_index()};
    place_dense_node(tree, source_idx, 0, dense_nodes); // We want to place the root at position 0 in the dense tree
    DenseTree dense_tree(std::move(dense_nodes), max_depth);
    return dense_tree;
}
