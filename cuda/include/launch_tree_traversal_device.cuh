#ifndef TREEINFER_LAUNCH_TREE_TRAVERSAL_DEVICE_CUH
#define TREEINFER_LAUNCH_TREE_TRAVERSAL_DEVICE_CUH
#include "dense_tree.hpp"
#include "sample.hpp"
#include "dense_model.hpp"
#include <vector>

namespace TreeInfer {
    /**
     * @brief Launches the traverse_dense_tree_device kernel from the host
     * @param dense_model The ensemble of dense trees
     * @param samples The vector of samples our model checks against
     * @return std::vector<float>
    **/
    std::vector<float> launch_dense_tree_traversal_kernel(const DenseModel& dense_model, const std::vector<TreeInfer::Sample>& samples);

}

#endif
