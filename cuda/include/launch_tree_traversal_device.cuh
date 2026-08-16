#ifndef TREEINFER_LAUNCH_TREE_TRAVERSAL_DEVICE_CUH
#define TREEINFER_LAUNCH_TREE_TRAVERSAL_DEVICE_CUH
#include "dense_tree.hpp"
#include "sample.hpp"
#include "device_model.cuh"
#include <vector>
#include <tuple>

namespace TreeInfer {
    /**
     * @brief Launches the traverse_dense_tree_device kernel from the host, returns the output and the time it takes to:
     * 1. Copy data from host to device [only sample transfer]
     * 2. Launch the kernel
     * 3. Copy data from device to host
     * @param device_model The ensemble of dense trees on the device
     * @param samples The vector of samples our model checks against
     * @return std::tuple<std::vector<float>, float, float, float> [traversal_results, h2d_time, kernel_execution, d2h_time]
    **/
    std::tuple<std::vector<float>, float, float, float> launch_dense_tree_traversal_kernel(const DeviceModel& device_model, const std::vector<TreeInfer::Sample>& samples);

}

#endif
