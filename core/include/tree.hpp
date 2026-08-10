#ifndef TREEINFER_TREE_H
#define TREEINFER_TREE_H
#include <cstdint>
#include <limits>

namespace treeinfer {
    /**
     * @brief Represents one node in a decision tree. It can either be a split node or a leaf node
     * It is kept as a single flat type rather than be pointer-based to mirror XGBoost's own array-based node format
     * The types of each field are determined by a realistic upper-bound for the values it can take
     */
    struct Node {
        /// @brief Stores the index of the left child of the node. Stored as uint32_t as this gives us ~2000x headroom over a tree with 2 million nodes
        std::uint32_t left_child;     
        /// @brief Stores the index of the right child of the node. Stored as uint32_t as this gives us ~2000x headroom over a tree with 2 million nodes
        std::uint32_t right_child;   
        /// @brief Stores the index of a given sample's feature vector. 
        std::uint16_t feature_index; 
        /// @brief Stores the actual threshold value used for a split decision. Only valid for non-leaf nodes. Stored as float as we don't need the extra precision a double grants. Further, this halves the memory that has to move per node which is important as such trees are generally memory bound not compute bound
        float threshold;
        /// @brief Stores the actual value of a given node. Only valid for leaf nodes. Stored as float as we don't need the extra precision a double grants. Further, this halves the memory that has to move per node which is important as such trees are generally memory bound not compute bound
        float value;
        ///  Sentinel value used to determine if a Node is a leaf node. This value is determined by the fact that
        /// XGBoost has a default max tree depth of 6 so we use the numeric limit of std::uint32_t as it is overkill, even
        /// for an unusually tree that has say 2 million nodes. Declared as static so that it is a property of Node
        /// but not a part of each instance of Node that is created
        ///
        static constexpr std::uint32_t k_no_child {std::numeric_limits<std::uint32_t>::max()};  
        bool is_leaf() const {return (left_child == k_no_child && right_child == k_no_child);}
    };
}

#endif
