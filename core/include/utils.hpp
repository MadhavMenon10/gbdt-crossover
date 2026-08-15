#ifndef GBDT_UTILS_H
#define GBDT_UTILS_H
#include <vector>
#include <algorithm>

namespace Utils {
    inline float median(std::vector<float> vect) {
        size_t size {vect.size()};
        std::sort(vect.begin(), vect.end());
        if (size % 2 != 0) {
            return vect[size / 2];
        } else {
            return (vect[size / 2 - 1] + vect[size / 2]) / 2;
        }
    }
}

#endif
