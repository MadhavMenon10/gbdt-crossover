#ifndef TREEINFER_SAMPLE_H
#define TREEINFER_SAMPLE_H
#include <vector>

namespace TreeInfer {
    /**
     * @brief A vector of feature values for the sample to score. Every index i holds the value for whatever feature feature_index == i refers to in every Node across the model
     */
    struct Sample {
        /// @brief Vector that holds the feature values for the sample to score
        std::vector<float> values;
    };
}

#endif
