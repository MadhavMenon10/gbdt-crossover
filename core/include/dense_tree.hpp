#ifndef TREEINFER_DENSE_TREE_H
#define TREEINFER_DENSE_TREE_H
// We need this so that left_child_index and right_child_index compile on both the host and device
#ifdef __CUDACC__
#define TREEINFER_HOST_DEVICE __host__ __device__
#else
#define TREEINFER_HOST_DEVICE
#endif

#include <cstdint>
#include <cstddef>
#include <vector>


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
    TREEINFER_HOST_DEVICE inline size_t left_child_index(size_t index) {
        return 2 * index + 1;
    }
    /// @brief Returns the index of the right child of a dense node
    /// @param index 
    /// @return size_t
    TREEINFER_HOST_DEVICE inline size_t right_child_index(size_t index) {
        return 2 * index + 2;
    }

    /**
     * @brief Represents a tree made of dense nodes that we pass into the GPU. 
     * We need the regular Tree and Node class as DenseTree and DenseNode cannot directly be constructed
     * from XGBoost. Tree and Node are not optimised for the GPU as the nodes live in an arbitrary order in memory
     * and they are all of different depth
     */
    class DenseTree {
        public:
            DenseTree(std::vector<DenseNode> dense_tree_nodes, std::uint16_t max_depth);
            /// @brief Returns the root node of the dense tree
            /// @return DenseNode
            inline DenseNode root() const {return dense_tree_nodes_[0];}
            /// @brief Returns the nodes in the dense tree
            /// @return const std::vector<DenseNode>&
            inline const std::vector<DenseNode>& nodes() const {return dense_tree_nodes_;}
            /// @brief Returns the index of the root node
            /// @return std::uint32_t
            inline std::uint32_t root_index() const {return 0;}
            /// @brief Returns the max depth of the dense tree
            /// @return std::uint16_t
            inline std::uint16_t max_depth() const {return max_depth_;}
            inline size_t num_nodes() const {return (static_cast<size_t>(1) << (max_depth_ + 1)) - 1;}
        private:
            /// @brief Stores the dense nodes
            std::vector<DenseNode> dense_tree_nodes_;
            /// @brief Stores the maximum depth of the tree
            std::uint16_t max_depth_;

    };
}





#endif
