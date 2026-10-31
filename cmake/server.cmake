# p100e-server: llama.cpp-compatible HTTP server on the p100e CUDA engine.
#
# Build choice (documented in server/README.md): the text layer reuses the llama.cpp fork
# (P100E_FORK_DIR, at P100E_FORK_COMMIT), compiled from source INTO this build
# with add_subdirectory(... EXCLUDE_FROM_ALL):
#   - ggml CPU-only (GGML_CUDA OFF): libllama is only loaded vocab-only (tokenizer + GGUF metadata), so ggml never
#     touches the GPUs and creates no CUDA context next to the engine's; static libs, so p100e-server.exe is one file
#     (a single file to deploy)
#   - libllama + libcommon (llama-common: common_chat_* templates with the fork's Jinja engine, tool-call / reasoning
#     parsers, sampler chain, reasoning budget, arg parser) + mtmd (server_tokens needs its headers / types; no mmproj
#     is loaded yet) + cpp-httplib / nlohmann (vendored in the fork)
#   - the fork's own server sources that hold no llama_context logic (server-common / -chat / -task / -queue / -schema /
#     -stream / -http.cpp) are compiled into p100e-server unchanged; server/server_context.cpp replaces the fork's
#     server-context.cpp (same server_context / server_routes declarations, slot loop on the p100e engine)
# The p100e engine (CUDA) stays a separate static library (p100e_engine, server/engine_tp.cpp) behind the narrow
# interface server/engine.h, so no p100e header (third_party/ggml/ggml.h!) is ever on the server's include path.
# Not reusing a shared GGML_CUDA=ON build of the fork (ggml-cuda.dll would register both
# P100s and allocate a second CUDA context per GPU).
# Vision does NOT switch GGML_CUDA on here: the vision code delay-loads the fork's mtmd / ggml-cuda DLLs
# at run time, so this build stays static and CPU-only. A static ggml-cuda in the same exe would collide
# with p100e_core's ggml_cuda_* / ggml_abort / conversion symbols, and MSVC may silently link one copy (server/README.md).

option(P100E_SERVER "build p100e-server (needs the llama.cpp fork sources, P100E_FORK_DIR)" ON)
set(P100E_FORK_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp-fork" CACHE PATH
    "llama.cpp fork source tree (at P100E_FORK_COMMIT)")

if(P100E_SERVER AND NOT EXISTS "${P100E_FORK_DIR}/tools/server/server-common.cpp")
    message(WARNING "p100e-server: fork sources not found at P100E_FORK_DIR=${P100E_FORK_DIR}; skipping the server")
    set(P100E_SERVER OFF)
endif()

# The server's behaviour matches llama-server (templates,
# tokenizer, parsers, response shapes) only at the verified commit. third_party/llama.cpp-fork is a copy of that commit
# and has no git history of its own, so the check is off by default. For an outside fork checkout, turn it on: it fails
# unless P100E_FORK_DIR is at P100E_FORK_COMMIT with no local changes under the paths the build compiles
# (cmake/check_fork.cmake). Checked at configure time and again before every build of the server (target
# p100e_fork_check), since the tree can move after the configure.
set(P100E_FORK_COMMIT "98ab7e5" CACHE STRING "fork commit (prefix) p100e-server must be built from")
option(P100E_FORK_CHECK "fail unless P100E_FORK_DIR is at P100E_FORK_COMMIT with no local changes in the files the server compiles" OFF)
if(P100E_SERVER AND P100E_FORK_CHECK)
    set(FORK_DIR "${P100E_FORK_DIR}")
    set(FORK_COMMIT "${P100E_FORK_COMMIT}")
    include("${CMAKE_CURRENT_LIST_DIR}/check_fork.cmake")
    add_custom_target(p100e_fork_check
        COMMAND "${CMAKE_COMMAND}" "-DFORK_DIR=${P100E_FORK_DIR}" "-DFORK_COMMIT=${P100E_FORK_COMMIT}"
                -P "${CMAKE_CURRENT_LIST_DIR}/check_fork.cmake"
        COMMENT "p100e-server: checking the fork tree (${P100E_FORK_COMMIT}, clean)"
        VERBATIM)
endif()

if(P100E_SERVER)
    # fork options: static, CPU-only, only the libraries we link (EXCLUDE_FROM_ALL builds just those)
    set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA             OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP           OFF CACHE BOOL "" FORCE)
    set(GGML_RPC              OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_MTMD      ON  CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
    set(LLAMA_SUBPROCESS      OFF CACHE BOOL "" FORCE)
    set(LLAMA_LLGUIDANCE      OFF CACHE BOOL "" FORCE)
    set(LLAMA_TOOLS_INSTALL   OFF CACHE BOOL "" FORCE)
    set(MTMD_VIDEO            OFF CACHE BOOL "" FORCE)
    add_subdirectory(server/fork "${CMAKE_BINARY_DIR}/fork_wrap")   # adds the fork with ggml_abort renamed

    # the engine behind server/engine.h (CUDA, p100e include paths stay private)
    add_library(p100e_engine STATIC server/engine_tp.cpp server/engine_fn.cpp)
    target_link_libraries(p100e_engine PRIVATE p100e_core)

    set(FORK_SRV "${P100E_FORK_DIR}/tools/server")
    # everything but main() and the engine, compiled once: p100e-server links it with the engine (p100e_engine),
    # p100e-server-fake with the CPU fake engine (tests/server/fake_engine.cpp)
    add_library(p100e_server_obj OBJECT
        server/server_context.cpp
        server/srv_sampler.cpp
        server/selftest.cpp
        server/parse_test.cpp
        server/stub/ui.cpp
        ${FORK_SRV}/server-common.cpp
        ${FORK_SRV}/server-chat.cpp
        ${FORK_SRV}/server-task.cpp
        ${FORK_SRV}/server-queue.cpp
        ${FORK_SRV}/server-schema.cpp
        ${FORK_SRV}/server-stream.cpp
        ${FORK_SRV}/server-http.cpp
    )
    target_link_libraries(p100e_server_obj PUBLIC llama-common mtmd cpp-httplib)
    function(p100e_server_opts t)
        # server/stub first: its ui.h (no web UI assets) replaces the generated tools/ui/ui.h
        target_include_directories(${t} PRIVATE server/stub server "${FORK_SRV}" "${P100E_FORK_DIR}/tools/mtmd"
                                   "${P100E_FORK_DIR}")
        target_compile_definitions(${t} PRIVATE P100E_FORK_DIR_STR="${P100E_FORK_DIR}" ggml_abort=fork_ggml_abort)
        if(MSVC)
            target_compile_options(${t} PRIVATE /bigobj /wd4244 /wd4267 /wd4996)
        endif()
        if(TARGET p100e_fork_check)
            add_dependencies(${t} p100e_fork_check)
        endif()
    endfunction()
    p100e_server_opts(p100e_server_obj)

    add_executable(p100e-server server/server_main.cpp)
    p100e_server_opts(p100e-server)
    target_link_libraries(p100e-server PRIVATE p100e_server_obj p100e_engine)

    # engine_tp's batched draft phase (draft_batch) vs each sequence alone, bit-exact (tests/engine_draft_test.cpp)
    add_executable(p100e-engine-draft-test tests/engine_draft_test.cpp)
    target_include_directories(p100e-engine-draft-test PRIVATE server)
    target_link_libraries(p100e-engine-draft-test PRIVATE p100e_engine)

    # engine_tp's prompt cache on chat-shaped prompts vs cache-off runs (tests/engine_cache_test.cpp)
    add_executable(p100e-engine-cache-test tests/engine_cache_test.cpp)
    target_include_directories(p100e-engine-cache-test PRIVATE server)
    target_link_libraries(p100e-engine-cache-test PRIVATE p100e_engine)

    # p100e-server on a deterministic CPU fake of server/engine.h: slot loop / queue / MTP bookkeeping tests without a
    # GPU (tests/server/fake_engine_test.py). Never deployed.
    option(P100E_SERVER_FAKE "build p100e-server-fake (tests)" ON)
    if(P100E_SERVER_FAKE)
        add_executable(p100e-server-fake server/server_main.cpp tests/server/fake_engine.cpp)
        p100e_server_opts(p100e-server-fake)
        target_link_libraries(p100e-server-fake PRIVATE p100e_server_obj)
    endif()
endif()
