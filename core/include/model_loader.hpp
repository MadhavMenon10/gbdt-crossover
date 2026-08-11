#ifndef TREEINFER_MODEL_LOADER_H
#define TREEINFER_MODEL_LOADER_H
#include "model.hpp"
#include <filesystem>


namespace TreeInfer {
    /**
     * @brief Loads a model from the file path given. We want the format booster.save_model('model.json') and not booster.dump_model(dump_format='json') as the former is completely serialised with parallel arrays and the latter is documented as not meant to be reloaded.
     * 
     * @param file_path The file path we create the model from
     * @return Model The model we create from the file path
     * @throws std::ios_base::failure Throws if the file won't open or can't be read
     * @throws nlohmann::json::parse_error Throws if the JSON is syntactically malformed
     * @throws std::invalid_argument Throws if num_feature or base_score isn't valid
     * @throws std::out_of_range Throws if the Model is structurally invalid. This is thrown from either the Model or Tree's constructor
     */
    Model load_model(const std::filesystem::path& file_path);
}

#endif

