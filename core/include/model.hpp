#ifndef TREEINFER_MODEL_H
#define TREEINFER_MODEL_H
#include "tree.hpp"
#include <vector>
#include <cstdint>

namespace TreeInfer {
    /**
     * @brief Represents one ensemble of trees and its associated metadata. Stored as a class as we want
     * to ensure that every node in each tree does not have a feature index greater than the number of features
     * stored in this model. This is validated in the constructor
     */
    class Model {
        public:
            /// @brief Constructs a Model. Takes trees by value rather than const&, so that the common case where a caller passing a temporary during parsing costs less as only the move constructor is called and 0 Tree objects are copied. const& on the other hand would have copied every Tree
            /// @param trees The vector of trees for this model
            /// @param num_features The number of features a sample needs to have to be scored by this model
            /// @param base_score The base score of the model which is then added to the sum of the leaf nodes of the trees
            Model(std::vector<Tree> trees, std::uint16_t num_features, float base_score);
            /// @brief Getter for the trees used in this model
            /// @return const std::vector<Tree>&
            inline const std::vector<Tree>& trees() const {return trees_;}
            /// @brief Getter for the number of features a sample needs to be scored by this model
            /// @return std::uint16_t
            inline std::uint16_t num_features() const {return num_features_;}
            /// @brief Getter for the base score of this model
            /// @return float
            inline float base_score() const  {return base_score_;}
        private:
            /// @brief Vector storing the trees used in the model
            std::vector<Tree> trees_;
            /// @brief The number of features a sample needs to have to be scored by this model
            std::uint16_t num_features_;
            /// @brief The base score of the model which is then added to the sum of the leaf nodes of the trees
            float base_score_;
    };
}

#endif
