// p100e-server: empty web UI asset table (see ui.h)
#include "ui.h"

const llama_ui_asset * llama_ui_find_asset(const std::string &) {
    return nullptr;
}

bool llama_ui_use_gzip() {
    return false;
}

const std::array<llama_ui_asset, 0> & llama_ui_get_assets() {
    static const std::array<llama_ui_asset, 0> empty{};
    return empty;
}
