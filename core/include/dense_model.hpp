#ifndef TREEINFER_DENSE_MODEL_H
#define TREEINFER_DENSE_MODEL_H
#include "model.hpp"
#include "dense_tree.hpp"
#include <vector>
#include <cstdint>


namespace TreeInfer {
    class DenseModel {
        public:
            DenseModel(const Model& model, std::uint16_t max_depth);
            inline const std::vector<DenseTree>& dense_trees() const {return dense_trees_;}
            inline std::uint16_t max_depth() const {return max_depth_;}
            inline std::uint16_t num_classes() const {return num_classes_;}
            inline const std::vector<float>& base_scores() const {return base_scores_;}
        private:
            std::vector<DenseTree> dense_trees_;
            std::uint16_t max_depth_;
            std::uint16_t num_classes_;
            std::vector<float> base_scores_;
    };
}

#endif
