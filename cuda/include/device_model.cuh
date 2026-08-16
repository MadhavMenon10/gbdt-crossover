#ifndef TREEINFER_DEVICE_MODEL_CUH
#define TREEINFER_DEVICE_MODEL_CUH
#include "dense_model.hpp"
#include "dense_tree.hpp"

namespace TreeInfer {
    /**
     * @brief Holds one model's dense tree data already uploaded and resident on the device.
     * This exists to fix a  measurement bug: the original design re-uploaded the full
     * tree ensemble to the device on every single trial, inside the per-trial kernel-launch
     * function. That meant H2D timing was dominated almost entirely by re-sending the same,
     * unchanging tree data (roughly 150KB even for the smallest ensemble) rather than by the
     * actual batch of new samples being scored (under 4KB even at batch=100). 
     * DeviceModel separates the two: the tree upload happens once, here, at construction so H2D timing
     *  finally reflects the cost of sending new data
     *  not the cost of re-sending the whole model every time.
     */
    class DeviceModel {
        public:
            /// @brief Uploads dense_model's flattened tree data to the device once, timing the
            /// transfer separately from any per-trial measurement.
            /// @param dense_model The host-side dense model to upload
            DeviceModel(const DenseModel& dense_model);
            /// @brief Frees the device-resident tree data.
            ~DeviceModel();
            /// @brief Deleted: copying would duplicate the device pointer, and both copies'
            /// destructors would then try to free the same memory (double free).
            DeviceModel(const DeviceModel&) = delete;
            DeviceModel& operator=(const DeviceModel&) = delete;
            /// @brief Deleted: DeviceModel is only ever passed by reference in this project,
            /// so move support is unnecessary
            DeviceModel(DeviceModel&&) = delete;
            DeviceModel& operator=(DeviceModel&&) = delete;
            /// @brief Getter for the number of trees in the uploaded model.
            /// @return size_t
            inline size_t num_trees() const {return num_trees_;}
            /// @brief Getter for the number of nodes per tree in the uploaded model.
            /// @return size_t
            inline size_t nodes_per_tree() const {return nodes_per_tree_;}
            /// @brief Getter for the device pointer to the uploaded, flattened tree data.
            /// @return DenseNode*
            inline DenseNode* dense_nodes_device_ptr() const {return dense_node_d_;}
            /// @brief Getter for how long the one-time model upload took, in milliseconds.
            /// @return float
            inline float upload_time_ms() const {return upload_time_ms_;}
        private:
            /// @brief Number of trees in the uploaded model.
            size_t num_trees_;
            /// @brief Number of nodes per tree in the uploaded model.
            size_t nodes_per_tree_;
            /// @brief Device pointer to the flattened, uploaded tree data.
            DenseNode* dense_node_d_;
            /// @brief Time taken to upload the tree data to the device, in milliseconds.
            float upload_time_ms_;
    };
}


#endif
