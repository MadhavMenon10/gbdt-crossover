#ifndef TREEINFER_CPU_REFERENCE_H
#define TREEINFER_CPU_REFERENCE_H
#include "model.hpp"
#include "sample.hpp"
#include <vector>


namespace TreeInfer {
    /**
     * @brief Predicts a score for the sample using the Model by walking through every tree in the ensemble. 
     * The prediction is computed as the base score of the model plus the sum of the leaf node values 
     * across every tree
     * 
     * @param model The model we want to score our sample from
     * @param sample The set of feature values we want to score
     * @return std::vector<float> 
     * @throws std::invalid_argument Throws if the number of features in the model does not correspond to the 
     * number of feature values in the sample.
     */
    std::vector<float> predict(const Model& model, const Sample& sample);
}


#endif
