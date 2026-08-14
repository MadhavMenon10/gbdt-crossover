#ifndef TREEINFER_TRIAL_RESULTS_H
#define TREEINFER_TRIAL_RESULTS_H
#include <vector>

namespace TreeInfer {
    /**
     * @brief Class that stores timing results in order to benchmark the GPU.
     * Stored this way so we have three parallel vectors kept in sync as add_trial is the only way to touch them, so no code path can desync one vector from the others
     * 
     */
    class TrialResults {
        public:
            /// @brief  Getter for vector holding the time it takes to transfer data from host to device
            /// @return const std::vector<float>&
            inline const std::vector<float>& h2d_times() const {return h2d_times_;}
            /// @brief Getter for vector holding kernel execution times
            /// @return const std::vector<float>&
            inline const std::vector<float>& kernel_times() const {return kernel_times_;}
            /// @brief Getter for vector holding the time it takes to transfer data from device to host 
            /// @return const std::vector<float>&
            inline const std::vector<float>& d2h_times() const {return d2h_times_;}
            /// @brief Adds results of one trial for storage
            /// @param h2d_time Time taken to transfer data from host to device
            /// @param kernel_time Kernel execution time
            /// @param d2h_time Time taken to transfer data from device to host
            inline void add_trial(float h2d_time, float kernel_time, float d2h_time) {
                h2d_times_.push_back(h2d_time);
                kernel_times_.push_back(kernel_time);
                d2h_times_.push_back(d2h_time);
            }
            
        private:
            /// @brief Vector holding host to device times
            std::vector<float> h2d_times_;
            /// @brief Vector holding kernel execution times
            std::vector<float> kernel_times_;
            /// @brief Vector holding device to host times
            std::vector<float> d2h_times_;
    };
}


#endif
