#ifndef TREEINFER_TREE_H
#define TREEINFER_TREE_H
#include <cstdint>
#include <limits>
#include <vector>

namespace TreeInfer {
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
        /// @brief Stores the index of a given sample's feature vector 
        std::uint16_t feature_index; 
        /// @brief  Stores whether we traverse to the left node or right node in the event a sample has a missing feature
        bool default_left;
        /// @brief Stores the actual threshold value used for a split decision. Only valid for non-leaf nodes. Stored as float as we don't need the extra precision a double grants. Further, this halves the memory that has to move per node which is important as such trees are generally memory bound not compute bound
        float threshold;
        /// @brief Stores the actual value of a given node. Only valid for leaf nodes. Stored as float as we don't need the extra precision a double grants. Further, this halves the memory that has to move per node which is important as such trees are generally memory bound not compute bound
        float value;
        ///  Sentinel value used to determine if a Node is a leaf node. This value is determined by the fact that
        /// XGBoost has a default max tree depth of 6 so we use the numeric limit of std::uint32_t as it is overkill, even
        /// for an unusually tree that has say 2 million nodes. Declared as static so that it is a property of Node
        /// but not a part of each instance of Node that is created
        static constexpr std::uint32_t k_no_child {std::numeric_limits<std::uint32_t>::max()};  
        bool is_leaf() const {return (left_child == k_no_child && right_child == k_no_child);}
    };
    
    /**
     * @brief Represents one tree in the ensemble. 
     * 
     */
    class Tree {
        public:
            /// @brief Constructs a tree. Takes tree_nodes by value rather than const&, so that the common case where a caller passing a temporary during parsing costs less as only the move constructor is called and 0 Node objects are copied. const& on the other hand would have copied every node
            /// @param tree_nodes The nodes used to construct the tree
            /// @param root_index The index of the root of the tree (typically 0 but may change for non XGBoost formats)
            /// @param class_index The index of the class the tree belongs to
            /// @throws std::out_of_range Throws if the index of the root is greater than the number of nodes in the tree
            Tree(std::vector<Node> tree_nodes, std::uint32_t root_index, std::uint16_t class_index);
            /// @brief Returns the nodes in the tree
            /// @return const std::vector<Node>&
            inline const std::vector<Node>& nodes() const {return tree_nodes_;}
            /// @brief Returns the root of the Tree
            /// @return Node
            inline Node root() const {return tree_nodes_[root_index_];}
            /// @brief Returns which class the tree belongs to 
            /// @return std::uint16_t
            inline std::uint16_t class_index() const {return class_index_;}
        private:
            /// @brief Stores all the nodes in the tree in a vector, mirrors XGBoost's array node structure
            std::vector<Node> tree_nodes_;
            /// @brief Stores the root index of the tree
            std::uint32_t root_index_;
            /// @brief Stores the class the tree belongs to
            std::uint16_t class_index_;
    };
}

#endif
