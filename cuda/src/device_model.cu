#include "device_model.cuh"
#include "cuda_utils.cuh"

TreeInfer::DeviceModel::DeviceModel(const DenseModel& dense_model): num_trees_(dense_model.dense_trees().size()), nodes_per_tree_(dense_model.dense_trees()[0].num_nodes()) {
    std::vector<DenseNode> tree_nodes {};
    for (const auto& dense_tree : dense_model.dense_trees()) {
        // Accumulates all the nodes across every dense tree in one vector
        tree_nodes.insert(tree_nodes.end(), dense_tree.nodes().begin(), dense_tree.nodes().end());

    }
    cudaEvent_t start_event, h2d_done_event;
    size_t overall_dense_nodes_size {sizeof(DenseNode) * nodes_per_tree_ * num_trees_};
    CUDAUtils::check_cuda_error(cudaEventCreate(&start_event), "Create start event");
    CUDAUtils::check_cuda_error(cudaEventCreate(&h2d_done_event), "Create h2d_done event");
    CUDAUtils::check_cuda_error(cudaMalloc(reinterpret_cast<void**>(&dense_node_d_), overall_dense_nodes_size), "Allocate memory for dense_nodes_d array");
    CUDAUtils::check_cuda_error(cudaEventRecord(start_event, 0), "Record start event");
    CUDAUtils::check_cuda_error(cudaMemcpy(dense_node_d_, tree_nodes.data(), overall_dense_nodes_size, cudaMemcpyHostToDevice), "Copy dense_nodes_d array onto device");
    CUDAUtils::check_cuda_error(cudaEventRecord(h2d_done_event, 0), "Record end event for h2d transfer");
    CUDAUtils::check_cuda_error(cudaEventSynchronize(h2d_done_event), "Synchronize on h2d_done event");
    CUDAUtils::check_cuda_error(cudaEventElapsedTime(&upload_time_ms_, start_event, h2d_done_event), "Compute upload time");
    CUDAUtils::check_cuda_error(cudaEventDestroy(start_event), "Destroy start event");
    CUDAUtils::check_cuda_error(cudaEventDestroy(h2d_done_event), "Destroy h2d_done event");
}

TreeInfer::DeviceModel::~DeviceModel() {
    CUDAUtils::check_cuda_error(cudaFree(dense_node_d_), "Free dense_node_d_");
    dense_node_d_ = nullptr;
}
