#ifndef TREEINFER_TREE_TRAVERSAL_DEVICE_CUH
#define TREEINFER_TREE_TRAVERSAL_DEVICE_CUH
#include "dense_tree.hpp"

namespace TreeInfer {
    /**
     * @brief Traverses a dense tree on the GPU
     * @param dense_nodes A contiguous array of DenseNodes 
     * @param feature_values A contiguous array of feature values
     * @param nodes_per_tree The number of nodes per dense tree (determined by max_depth)
     * @param num_samples The number of samples we are using per tree
     * @param num_trees The number of dense trees in our model
     * @param num_features The number of feature values per sample
     * @param output_values A contiguous array of values to write into based on the results of the traversal
     */ 
    __global__ void traverse_dense_tree_device(const DenseNode* dense_nodes, const float* feature_values, size_t nodes_per_tree, size_t num_samples, size_t num_trees, size_t num_features, float* output_values);
}

#endif
