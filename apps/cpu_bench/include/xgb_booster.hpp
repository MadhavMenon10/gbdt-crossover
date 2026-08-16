#ifndef XGBBOOSTER_H
#define XGBBOOSTER_H
#include <filesystem>
#include "xgboost/c_api.h"


class XGBBooster {
    public:
        XGBBooster(const std::filesystem::path& model_path);
        ~XGBBooster();
        XGBBooster(const XGBBooster&) = delete;
        XGBBooster& operator=(const XGBBooster&) = delete;
        XGBBooster(XGBBooster&&) = delete;
        XGBBooster& operator=(XGBBooster&&) = delete;
        inline BoosterHandle handle() const {return handle_;}

    private:
        BoosterHandle handle_;
};


#endif
