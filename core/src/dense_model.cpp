#include "dense_model.hpp"
#include "dense_builder.hpp"

TreeInfer::DenseModel::DenseModel(const Model& model, std::uint16_t max_depth): max_depth_(max_depth), num_classes_(model.num_classes()), base_scores_(model.base_scores()) {
    for (const auto& tree : model.trees()) {
        dense_trees_.push_back(generate_dense_tree(tree, max_depth_));
    }
}
