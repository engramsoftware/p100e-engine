# The llama.cpp fork as a subdirectory of p100e (see cmake/server.cmake). This wrapper directory scopes one compile
# definition to the fork's targets: ggml's ggml_abort is renamed to fork_ggml_abort, because p100e_core defines its own
# ggml_abort (src/types.cu, for the vendored dequant kernels) and both end up in p100e-server.exe (LNK2005). Renaming
# on the fork side leaves every p100e engine object untouched. Targets that include the fork's ggml.h (p100e-server's
# own sources) get the same definition (server.cmake).
add_compile_definitions(ggml_abort=fork_ggml_abort)
# GCC: some fork sources use std::max / std::min on an initializer list without including <algorithm> (MSVC's
# headers pull it in transitively). Force it for the fork's C++ files instead of patching the fork tree.
if(NOT MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-includealgorithm>)
endif()
add_subdirectory("${P100E_FORK_DIR}" "${CMAKE_BINARY_DIR}/fork" EXCLUDE_FROM_ALL)
