#ifndef TREEINFER_CPU_REFERENCE_H
#define TREEINFER_CPU_REFERENCE_H
#include "model.hpp"
#include "sample.hpp"
#include "dense_tree.hpp"
#include <vector>


namespace TreeInfer {
    /**
     * @brief Traverses a regular tree and returns a leaf value
     * 
     * @param tree The tree to traverse
     * @param sample The sample used in our decision tree
     * @return float 
     */
    float traverse_tree(const Tree& tree, const Sample& sample);
    /**
     * @brief Traverses a dense tree and returns a lead value
     * 
     * @param tree The dense tree to traverse
     * @param sample The sample used in our decision tree
     * @return float 
     */
    float traverse_dense_tree(const DenseTree& tree, const Sample& sample);
    /**
     * @brief Predicts a score for the sample using the Model by walking through every tree in the ensemble. 
     * The prediction is computed as the base score of the model plus the sum of the leaf node values 
     * across every tree
     * 
     * @param model The model we want to score our sample from
     * @param sample The set of feature values we want to score
     * @return std::vector<float> 
     * @throws std::invalid_argument Throws if the number of features in the model does not correspond to the 
     * number of feature values in the sample
     */
    std::vector<float> predict(const Model& model, const Sample& sample);
    /**
     * @brief Predicts a score for the sample using the Model by walking through every dense tree in the ensemble. 
     * The prediction is computed as the base score of the model plus the sum of the leaf node values 
     * across every dense tree
     * 
     * @param model The model we want to score our sample from
     * @param sample The set of feature values we want to score
     * @param max_depth The max depth of the dense tree
     * @return std::vector<float> 
     * @throws std::invalid_argument Throws if the number of features in the model does not correspond to the 
     * number of feature values in the sample.
     */
    std::vector<float> predict_dense(const Model& model, const Sample& sample, std::uint16_t max_depth);
}


#endif
