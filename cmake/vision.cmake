# Vision: llama.cpp mtmd / clip from the fork, linked against the fork build's import
# libraries (P100E_FORK_BUILD, the commit P100E_FORK_COMMIT names). The DLLs are
# delay-loaded (src/vision/vision.cpp points the loader at them), so text-only runs never need them.
set(P100E_FORK_SRC "${CMAKE_SOURCE_DIR}/third_party/llama.cpp-fork" CACHE PATH "llama.cpp fork source (at P100E_FORK_COMMIT)")
set(P100E_FORK_BUILD "${CMAKE_SOURCE_DIR}/third_party/llama.cpp-fork/build" CACHE PATH "llama.cpp fork build (BUILD_SHARED_LIBS, GGML_CUDA)")
option(P100E_VISION "build the image encoder (mtmd / clip from the llama.cpp fork)" ON)

if(MSVC)
    set(_fork_libs
        "${P100E_FORK_BUILD}/tools/mtmd/mtmd.lib"
        "${P100E_FORK_BUILD}/src/llama.lib"
        "${P100E_FORK_BUILD}/ggml/src/ggml.lib"
        "${P100E_FORK_BUILD}/ggml/src/ggml-base.lib")
else()
    # Linux: the fork built with BUILD_SHARED_LIBS=ON GGML_CUDA=ON puts its .so files in <build>/bin
    set(_fork_libs
        "${P100E_FORK_BUILD}/bin/libmtmd.so"
        "${P100E_FORK_BUILD}/bin/libllama.so"
        "${P100E_FORK_BUILD}/bin/libggml.so"
        "${P100E_FORK_BUILD}/bin/libggml-base.so")
endif()
set(_have_fork ON)
foreach(f ${_fork_libs} "${P100E_FORK_SRC}/tools/mtmd/mtmd.h" "${P100E_FORK_SRC}/include/llama.h")
    if(NOT EXISTS "${f}")
        set(_have_fork OFF)
    endif()
endforeach()

if(P100E_VISION AND MSVC AND _have_fork)
    add_library(p100e_vision STATIC src/vision/vision.cpp)
    target_include_directories(p100e_vision PUBLIC src/vision
        PRIVATE "${P100E_FORK_SRC}/include" "${P100E_FORK_SRC}/ggml/include" "${P100E_FORK_SRC}/tools/mtmd")
    target_compile_definitions(p100e_vision PRIVATE LLAMA_SHARED GGML_SHARED "P100E_FORK_BIN=\"${P100E_FORK_BUILD}/bin\"")
    target_link_libraries(p100e_vision PUBLIC ${_fork_libs} delayimp)
    target_link_options(p100e_vision INTERFACE
        "/DELAYLOAD:mtmd.dll" "/DELAYLOAD:llama.dll" "/DELAYLOAD:ggml.dll" "/DELAYLOAD:ggml-base.dll")
    target_link_libraries(p100e-run p100e_vision)
    target_compile_definitions(p100e-run PRIVATE P100E_HAVE_VISION)
    message(STATUS "p100e: vision encoder ON (fork ${P100E_FORK_SRC})")
    # host-side checks of the image path (mm_track, GGML_* pinning, mmproj width / deepstack, encoder locking)
    add_executable(p100e-vision-test tests/vision_test.cpp)
    target_include_directories(p100e-vision-test PRIVATE "${P100E_FORK_SRC}/ggml/include")
    target_compile_definitions(p100e-vision-test PRIVATE GGML_SHARED)
    target_link_libraries(p100e-vision-test p100e_vision)

    # p100e-server: the same encoder as a DLL (p100e-vision.dll next to the exe, itself delay-loaded, so a server
    # without --mmproj never loads it or the fork DLLs). Not the static p100e_vision: the server links the fork's
    # llama / mtmd / ggml statically (CPU-only, cmake/server.cmake), and the fork import libraries here export the same
    # names - in one module the linker would bind one copy for both (vision.h, P100E_VISION_API).
    if(TARGET p100e_server_obj)
        add_library(p100e_vision_dll SHARED src/vision/vision.cpp)
        set_target_properties(p100e_vision_dll PROPERTIES OUTPUT_NAME p100e-vision)
        target_include_directories(p100e_vision_dll PUBLIC src/vision
            PRIVATE "${P100E_FORK_SRC}/include" "${P100E_FORK_SRC}/ggml/include" "${P100E_FORK_SRC}/tools/mtmd")
        target_compile_definitions(p100e_vision_dll PRIVATE LLAMA_SHARED GGML_SHARED P100E_VISION_BUILD_DLL
            "P100E_FORK_BIN=\"${P100E_FORK_BUILD}/bin\"" INTERFACE P100E_VISION_DLL)
        target_link_libraries(p100e_vision_dll PRIVATE ${_fork_libs} delayimp)
        target_link_options(p100e_vision_dll PRIVATE
            "/DELAYLOAD:mtmd.dll" "/DELAYLOAD:llama.dll" "/DELAYLOAD:ggml.dll" "/DELAYLOAD:ggml-base.dll")
        target_link_libraries(p100e_server_obj PUBLIC p100e_vision_dll)
        target_compile_definitions(p100e_server_obj PRIVATE P100E_HAVE_VISION)
        foreach(t p100e-server p100e-server-fake)
            if(TARGET ${t})
                target_link_libraries(${t} PRIVATE delayimp)
                target_link_options(${t} PRIVATE "/DELAYLOAD:p100e-vision.dll")
            endif()
        endforeach()
        message(STATUS "p100e: p100e-server image input ON (p100e-vision.dll)")
    endif()
elseif(P100E_VISION AND _have_fork)
    # Linux: no delay loading. The fork's .so files are ordinary dependencies found through RPATH; vision.cpp's loader
    # is a no-op. The one hazard is ELF symbol interposition: p100e-server already holds a static CPU-only copy of the
    # fork (cmake/server.cmake) and p100e_core holds ggml_* names of its own (third_party/ggml), and by default a symbol
    # an executable exports wins over the same name in a shared library. --exclude-libs,ALL keeps every symbol that
    # comes from a static archive out of the dynamic symbol table (executables and libp100e-vision.so alike), and
    # -Bsymbolic binds libp100e-vision.so's own references inside itself, so each copy only ever calls itself.
    set(_fork_inc "${P100E_FORK_SRC}/include" "${P100E_FORK_SRC}/ggml/include" "${P100E_FORK_SRC}/tools/mtmd")
    set(_fork_rpath "${P100E_FORK_BUILD}/bin")
    add_library(p100e_vision STATIC src/vision/vision.cpp)
    target_include_directories(p100e_vision PUBLIC src/vision PRIVATE ${_fork_inc})
    target_link_libraries(p100e_vision PUBLIC ${_fork_libs})
    target_link_libraries(p100e-run p100e_vision)
    target_compile_definitions(p100e-run PRIVATE P100E_HAVE_VISION)
    target_link_options(p100e-run PRIVATE "LINKER:--exclude-libs,ALL")
    message(STATUS "p100e: vision encoder ON (fork ${P100E_FORK_SRC}, libs ${_fork_rpath})")
    add_executable(p100e-vision-test tests/vision_test.cpp)
    target_include_directories(p100e-vision-test PRIVATE "${P100E_FORK_SRC}/ggml/include")
    target_link_libraries(p100e-vision-test p100e_vision)
    target_link_options(p100e-vision-test PRIVATE "LINKER:--exclude-libs,ALL")

    if(TARGET p100e_server_obj)
        add_library(p100e_vision_dll SHARED src/vision/vision.cpp)
        set_target_properties(p100e_vision_dll PROPERTIES OUTPUT_NAME p100e-vision
            BUILD_RPATH "${_fork_rpath}" INSTALL_RPATH "${_fork_rpath}")
        target_include_directories(p100e_vision_dll PUBLIC src/vision PRIVATE ${_fork_inc})
        target_compile_definitions(p100e_vision_dll PRIVATE P100E_VISION_BUILD_DLL INTERFACE P100E_VISION_DLL)
        target_link_libraries(p100e_vision_dll PRIVATE ${_fork_libs})
        target_link_options(p100e_vision_dll PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:-Bsymbolic")
        target_link_libraries(p100e_server_obj PUBLIC p100e_vision_dll)
        target_compile_definitions(p100e_server_obj PRIVATE P100E_HAVE_VISION)
        foreach(t p100e-server p100e-server-fake)
            if(TARGET ${t})
                target_link_options(${t} PRIVATE "LINKER:--exclude-libs,ALL")
            endif()
        endforeach()
        message(STATUS "p100e: p100e-server image input ON (libp100e-vision.so)")
    endif()
else()
    message(STATUS "p100e: vision encoder OFF (P100E_VISION=${P100E_VISION}, fork found: ${_have_fork})")
endif()

# interleaved M-RoPE kernels (images): bit-exactness vs the text kernels / per-op path, and vs ggml formula
add_executable(p100e-mrope-test tests/mrope_test.cu)
target_link_libraries(p100e-mrope-test p100e_core)
