#ifndef TREE_INFER_SAMPLE_H
#define TREE_INFER_SAMPLE_H
#include <vector>

namespace TreeInfer {
    /**
     * @brief A vector of feature values for the sample to score
     */
    struct Sample {
        std::vector<float> values;
    };
}

#endif
