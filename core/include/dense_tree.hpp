#ifndef TREEINFER_DENSE_TREE_H
#define TREEINFER_DENSE_TREE_H
#include <cstdint>
#include <cstddef>


namespace TreeInfer{
    /**
     * @brief Represents a node in a 0-indexed dense (complete) binary tree that we pass into the GPU
     * Unlike node, it doesn't store children members as we can calculate its children indices using 
     * index arithmetic. 
     */
    struct DenseNode {
        /// @brief Stores the index of a given sample's feature vector
        std::uint16_t feature_index; 
        /// @brief Stores whether we traverse to the left node or right node in the event a sample has a missing feature
        bool default_left;
        /// @brief Stores whether the node is a leaf or not. Used to determine if we can access child indices
        bool is_leaf;
        /// @brief Stores the actual threshold value used for a split decision. Only valid for non-leaf nodes. Stored as float as we don't need the extra precision a double grants. Further, this halves the memory that has to move per node which is important as such trees are generally memory bound not compute bound
        float threshold;
        /// @brief Stores the actual value of a given node. Only valid for leaf nodes. Stored as float as we don't need the extra precision a double grants. Further, this halves the memory that has to move per node which is important as such trees are generally memory bound not compute bound
        float value;
    };
    /// @brief Returns the index of the left child of a dense node
    /// @param index 
    /// @return size_t
    inline size_t left_child_index(size_t index) {
        return 2 * index + 1;
    }
    /// @brief Returns the index of the right child of a dense node
    /// @param index 
    /// @return size_t
    inline size_t right_child_index(size_t index) {
        return 2 * index + 2;
    }
}





#endif
