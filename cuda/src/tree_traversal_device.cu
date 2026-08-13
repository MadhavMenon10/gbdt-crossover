#include "tree_traversal_device.cuh"


__global__ void TreeInfer::traverse_dense_tree_device(const DenseNode* dense_nodes, const float* feature_values, size_t nodes_per_tree, size_t num_samples, size_t num_trees, size_t num_features, float* output_values) {
    /*
    * dense_nodes is every tree's nodes stored contiguously in one buffer.
    * Tree tree_idx occupies positions [tree_idx * nodes_per_tree, (tree_idx + 1) * nodes_per_tree),
    * so tree_idx * nodes_per_tree is the offset where that tree's data starts.
    *
    * feature_values is every sample's feature vector stored contiguously the same way.
    * Sample sample_idx's values start at sample_idx * num_features.
    *
    * global_id maps to (sample_idx, tree_idx) via global_id % num_trees and global_id / num_trees,
    * so tree_idx varies fastest across consecutive thread ids. This keeps a warp fully occupied
    * even at batch size 1, since 32 consecutive threads get 32 different trees on the same sample
    * instead of stalling on a single valid sample.
    *
    * i tracks the current node's position within its own tree (0 at the root, doubling via
    * left_child_index/right_child_index). curr_idx is that same position in the flat dense_nodes
    * buffer (i plus the tree's base offset). i is recovered from curr_idx after each move by
    * subtracting the base offset back out, so left_child_index/right_child_index always operate
    * on a local, within-tree index rather than a global one.
    *
    * output_values is indexed by global_id, not curr_idx, since curr_idx only encodes a position
    * within one tree and would collide across threads that land on the same local index in
    * different trees.
    */
    size_t global_id {threadIdx.x + blockIdx.x * blockDim.x}; // Assume the tree is laid out flat
    if (global_id >= num_samples * num_trees) {
        return;
    }
    size_t tree_idx {global_id % num_trees}; // We want this to vary fast per tree
    size_t sample_idx {global_id / num_trees}; // We want this to vary slow per tree so we parallelise with multiple trees running on the same sample
    size_t i {0};
    auto curr_idx {tree_idx * nodes_per_tree + i};
    auto curr_node {dense_nodes[curr_idx]};
    while (!curr_node.is_leaf) {
        auto curr_value {feature_values[sample_idx * num_features + curr_node.feature_index]};
        auto left_idx {tree_idx * nodes_per_tree + TreeInfer::left_child_index(i)};
        auto right_idx {tree_idx * nodes_per_tree + TreeInfer::right_child_index(i)};
        if (isnan(curr_value)) { // CUDA specific isnan
            curr_idx = (curr_node.default_left ? left_idx : right_idx);
            curr_node = (curr_node.default_left ? dense_nodes[left_idx] : dense_nodes[right_idx]);
        } else if (curr_value >= curr_node.threshold) {
            curr_node = dense_nodes[right_idx];
            curr_idx = right_idx;
        } else {
            curr_node = dense_nodes[left_idx];
            curr_idx = left_idx;
        }
        i = curr_idx - tree_idx * nodes_per_tree;
    }
    output_values[global_id] = curr_node.value;
}

