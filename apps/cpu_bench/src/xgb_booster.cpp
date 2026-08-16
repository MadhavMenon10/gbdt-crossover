#include "xgb_booster.hpp"
#include <stdexcept>

XGBBooster::XGBBooster(const std::filesystem::path& model_path) {
    int success_create {XGBoosterCreate(nullptr, 0, &handle_)}; // Returns 0 for success, -1 for failure
    if (success_create == -1) {
        throw std::runtime_error(std::string("XGBBooster could not be created: ") + XGBGetLastError());
    }
    int success_load {XGBoosterLoadModel(handle_, model_path.c_str())};
    if (success_load == -1) {
        throw std::runtime_error(std::string("XGBBooster could not be loaded: ") + XGBGetLastError());
    }
}

XGBBooster::~XGBBooster() {
    XGBoosterFree(handle_);
}
