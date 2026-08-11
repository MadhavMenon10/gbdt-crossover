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
            /// @param num_classes The number of classes in this model
            /// @param base_score The base score of the model which is then added to the sum of the leaf nodes of the trees
            /// @throws std::invalid_argument Throws if the class index of a tree exceeds the number of classes or if the number of base scores don't match the number of classes
            /// @throws std::out_of_range Throws if a tree node has a feature index greater than the number of features
            Model(std::vector<Tree> trees, std::uint16_t num_features, std::uint16_t num_classes, std::vector<float> base_scores);
            /// @brief Getter for the trees used in this model
            /// @return const std::vector<Tree>&
            inline const std::vector<Tree>& trees() const {return trees_;}
            /// @brief Getter for the number of features a sample needs to be scored by this model
            /// @return std::uint16_t
            inline std::uint16_t num_features() const {return num_features_;}
            /// @brief Getter for the base score of this model
            /// @return std::vector<float>
            inline const std::vector<float>& base_scores() const  {return base_scores_;}
            /// @brief Returns the number of classes the model stores
            /// @return std::uint16_t 
            inline std::uint16_t num_classes() const {return num_classes_;}

        private:
            /// @brief Vector storing the trees used in the model
            std::vector<Tree> trees_;
            /// @brief The number of features a sample needs to have to be scored by this model
            std::uint16_t num_features_;
            /// @brief The number of classes the Model has 
            std::uint16_t num_classes_;
            /// @brief The base score of the model which is then added to the sum of the leaf nodes of the trees
            std::vector<float> base_scores_;
    };
}

#endif
