#ifndef TREEINFER_DENSE_BUILDER_H
#define TREEINFER_DENSE_BUILDER_H
#include "tree.hpp"
#include "dense_tree.hpp"
#include <vector>

namespace TreeInfer {
    DenseTree generate_dense_tree(const Tree& tree, std::uint16_t max_depth);
}

#endif
