// p100e-server: stand-in for the fork's generated tools/ui/ui.h (llama-ui-embed). No web UI is embedded, so the
// asset table is empty and server-http.cpp serves only the API routes.
#pragma once

#include <array>
#include <string>

struct llama_ui_asset {
    std::string           name;
    const unsigned char * data;
    std::size_t           size;
    std::string           etag;
    std::string           type;
};

const llama_ui_asset * llama_ui_find_asset(const std::string & name);
bool llama_ui_use_gzip();
const std::array<llama_ui_asset, 0> & llama_ui_get_assets();
